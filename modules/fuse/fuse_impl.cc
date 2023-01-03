/** Copyright 2020-2023 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "fuse/fuse_impl.h"

#include <limits>
#include <unordered_map>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/macros.h"

#include "basic/ds/array.h"
#include "basic/ds/dataframe.h"
#include "boost/algorithm/string/predicate.hpp"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/core_types.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"
#include "common/util/uuid.h"
#include "fuse/adaptors/arrow_ipc/deserializer_registry.h"
#include "fuse/adaptors/arrow_ipc/serializer_registry.h"

namespace vineyard {

namespace fuse {

struct fs::fs_state_t fs::state {};

std::string name_from_path(std::string const& path) {
  return path.substr(1, path.length() - 6 /* .arrow */ - 1);
}

std::shared_ptr<arrow::Buffer> generate_fuse_view(
    std::shared_ptr<Object> object) {
  if (auto dataframe = std::dynamic_pointer_cast<vineyard::DataFrame>(object)) {
    return fuse::arrow_view(dataframe);
  } else if (auto recordbatch =
                 std::dynamic_pointer_cast<vineyard::RecordBatch>(object)) {
    return fuse::arrow_view(recordbatch);
  } else if (auto table = std::dynamic_pointer_cast<vineyard::Table>(object)) {
    return fuse::arrow_view(table);
  }
  VINEYARD_ASSERT(
      false, "Unsupported vineyard data type: " + object->meta().GetTypeName());
  return nullptr;
}

int fs::fuse_getattr(const char* path, struct stat* stbuf,
                     struct fuse_file_info*) {
  DLOG(INFO) << "fuse: getattr on " << path;
  std::lock_guard<std::mutex> guard(state.mtx_);

  memset(stbuf, 0, sizeof(struct stat));
  if (strcmp(path, "/") == 0) {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    return 0;
  }

  stbuf->st_mode = S_IFREG | 0444;
  stbuf->st_nlink = 1;

  {
    auto iter = state.views.find(path);
    if (iter != state.views.end()) {
      DLOG(INFO) << "fuse: getattr found " << path;

      stbuf->st_size = iter->second->size();
      return 0;
    }
  }
  {
    auto iter = state.mutable_views.find(path);
    if (iter != state.mutable_views.end()) {
      stbuf->st_size = iter->second->length();
      return 0;
    }
  }

  {
    std::string path_string(path);
    if (!boost::algorithm::ends_with(path_string, ".arrow")) {
      DLOG(INFO) << path_string << "should end with arrow";
      return -ENOENT;
    }
    ObjectID target = InvalidObjectID();
    auto prefix = name_from_path(path);
    auto status = state.client->GetName(prefix, target);
    if (status.ok()) {
      DLOG(INFO) << "get attr name not exists";
      bool exists = false;
      VINEYARD_CHECK_OK(state.client->Exists(target, exists));
      if (!exists) {
        return -ENOENT;
      }
      auto obj = state.client->GetObject(target);
      DLOG(INFO) << "tryting to deserialize" << obj->meta().GetTypeName();
      auto d =
          fs::state.ipc_desearilizer_registry.at(obj->meta().GetTypeName());
      auto buffer = d(obj);
      state.views[path_string] = buffer;

      stbuf->st_size = buffer->size();
    } else {
      auto obj = state.client->GetObject(ObjectIDFromString(prefix));
      DLOG(INFO) << "trying to deserialize " << obj->meta().GetTypeName();

      if (obj == nullptr) {
        return -ENOENT;
      }
      auto d =
          fs::state.ipc_desearilizer_registry.at(obj->meta().GetTypeName());
      auto buffer = d(obj);
      state.views[path_string] = buffer;
      stbuf->st_size = buffer->size();
    }
    return 0;
  }
}

int fs::fuse_open(const char* path, struct fuse_file_info* fi) {
  DLOG(INFO) << "fuse: open " << path << " with mode " << fi->flags;
  if (((fi->flags & O_ACCMODE) & (O_RDONLY))) {
    return -EACCES;
  }

  std::lock_guard<std::mutex> guard(state.mtx_);

  // check for the existence todo: switch the name store from the vineyard to
  // fuse

  // the opened file referenced by the user-defined name
  auto filename = name_from_path(path);
  auto target = InvalidObjectID();
  auto loc = state.views.find(path);
  std::string path_string(path);
  std::shared_ptr<vineyard::Object> object = nullptr;
  if (loc == state.views.end()) {
    if (state.client->GetName(filename, target).ok()) {
      object = state.client->GetObject(target);
    }

    if (object == nullptr) {
      object = state.client->GetObject(ObjectIDFromString(filename.substr(1)));
    }
    if (object == nullptr) {
      return -ENOENT;
    } else {
      auto d =
          fs::state.ipc_desearilizer_registry.at(object->meta().GetTypeName());
      state.views[path_string] = d(object);
    }
  }

  // bypass kernel's page cache to avoid knowing the size in `getattr`.
  // see also:
  // https://stackoverflow.com/questions/46267972/fuse-avoid-calculating-size-in-getattr
  fi->direct_io = 1;
  return 0;
}

int fs::fuse_read(const char* path, char* buf, size_t size, off_t offset,
                  struct fuse_file_info* fi) {
  DLOG(INFO) << "fuse: read " << path << " from " << offset << ", expect "
             << size << " bytes";

  std::unordered_map<
      std::string, std::shared_ptr<internal::ChunkBuffer>>::const_iterator loc;
  {
    std::lock_guard<std::mutex> guard(state.mtx_);
    loc = state.views.find(path);
  }
  if (loc == state.views.end()) {
    return -ENOENT;
  }
  auto cbuffer = loc->second;
  auto readByte = cbuffer->readAt(offset, size, buf);
  return readByte;
}

int fs::fuse_write(const char* path, const char* buf, size_t size, off_t offset,
                   struct fuse_file_info* fi) {
  DLOG(INFO) << "fuse: write " << path << " from " << offset << ", expect "
             << size << " bytes";
  std::unordered_map<std::string,
                     std::shared_ptr<arrow::BufferBuilder>>::const_iterator loc;

  {
    std::lock_guard<std::mutex> guard(state.mtx_);

    loc = state.mutable_views.find(path);
  }
  if (loc == state.mutable_views.end()) {
    return -ENOENT;
  }
  auto buffer = loc->second;
  if (static_cast<int64_t>(offset + size) >= buffer->capacity()) {
    VINEYARD_CHECK_OK(buffer->Reserve(offset + size));
  }
  memcpy(buffer->mutable_data() + offset, buf, size);
  buffer->UnsafeAdvance(size);
  return size;
}

int fs::fuse_statfs(const char* path, struct statvfs*) {
  DLOG(INFO) << "fuse: statfs " << path;
  return 0;
}

int fs::fuse_flush(const char* path, struct fuse_file_info*) {
  DLOG(INFO) << "fuse: flush " << path;
  return 0;
}

int fs::fuse_release(const char* path, struct fuse_file_info*) {
  DLOG(INFO) << "fuse: release " << path;

  {
      // TODO: ref count should be used
      //
      // auto loc = state.views.find(path);
      // if (loc != state.views.end()) {
      //   state.views.erase(loc);
      //   return 0;
      // }
  }

  {
    auto loc = state.mutable_views.find(path);
    if (loc != state.mutable_views.end()) {
      fuse::from_arrow_view(state.client.get(), loc->first, loc->second);
      loc->second->Reset();
      state.mutable_views.erase(loc);
      return 0;
    }
  }

  return -ENOENT;
}

int fs::fuse_getxattr(const char* path, const char* name, char*, size_t) {
  DLOG(INFO) << "fuse: getxattr " << path << ": name";
  return 0;
}

int fs::fuse_opendir(const char* path, struct fuse_file_info* info) {
  DLOG(INFO) << "fuse: opendir " << path;
  return 0;
}

int fs::fuse_readdir(const char* path, void* buf, fuse_fill_dir_t filler,
                     off_t offset, struct fuse_file_info* fi,
                     enum fuse_readdir_flags flags) {
  DLOG(INFO) << "fuse: readdir " << path;

  if (strcmp(path, "/") != 0) {
    return -ENOENT;
  }

  filler(buf, ".", NULL, 0, fuse_fill_dir_flags::FUSE_FILL_DIR_PLUS);
  filler(buf, "..", NULL, 0, fuse_fill_dir_flags::FUSE_FILL_DIR_PLUS);

  std::unordered_map<ObjectID, json> metas{};

  VINEYARD_CHECK_OK(state.client->ListData(
      ".*", true, std::numeric_limits<size_t>::max(), metas));
  for (auto const& item : metas) {
    if (item.second.contains("__name")) {
      std::string base = item.second["__name"].get<std::string>() + ".arrow";
      DLOG(INFO) << "open object with name" << base;
      filler(buf, base.c_str(), NULL, 0,
             fuse_fill_dir_flags::FUSE_FILL_DIR_PLUS);
    } else {
      std::string base = ObjectIDToString(item.first).c_str();
      base.append(".arrow");
      DLOG(INFO) << "open object without name" << base;

      filler(buf, base.c_str(), NULL, 0,
             fuse_fill_dir_flags::FUSE_FILL_DIR_PLUS);
    }
  }
  return 0;
}

void* fs::fuse_init(struct fuse_conn_info* conn, struct fuse_config* cfg) {
  DLOG(INFO) << "fuse: initfs with vineyard socket " << state.vineyard_socket;

  state.client.reset(new vineyard::Client());
  state.client->Connect(state.vineyard_socket);

  fuse_apply_conn_info_opts(state.conn_opts, conn);
  // conn->max_read = conn->max_readahead;

  cfg->kernel_cache = 0;
  return NULL;
}

void fs::fuse_destroy(void* private_data) {
  DLOG(INFO) << "fuse: destroy";

  state.views.clear();
  state.mutable_views.clear();
  state.client->Disconnect();
}

int fs::fuse_access(const char* path, int mode) {
  DLOG(INFO) << "fuse: access " << path << " with mode " << mode;

  return mode;
}

int fs::fuse_create(const char* path, mode_t mode, struct fuse_file_info*) {
  DLOG(INFO) << "fuse: create " << path << " with mode " << mode;
  if (state.mutable_views.find(path) != state.mutable_views.end()) {
    LOG(ERROR) << "fuse: create: file already exists" << path;
    return EEXIST;
  }
  DLOG(INFO) << "creating " << path;
  state.mutable_views.emplace(
      path, std::shared_ptr<arrow::BufferBuilder>(new arrow::BufferBuilder()));
  return 0;
}

}  // namespace fuse

}  // namespace vineyard
