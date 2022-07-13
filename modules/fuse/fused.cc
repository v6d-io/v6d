/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include "fuse/fused.h"
#include "fuse/adaptors/formats.h"

#include <limits>
#include <unordered_map>

#include "arrow/status.h"
#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/macros.h"

#include "basic/ds/array.h"
#include "basic/ds/dataframe.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/core_types.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"
#include "common/util/uuid.h"
#include <iostream>
#include <unordered_map>
 

namespace vineyard {

namespace fuse {
template<typename K, typename V>
std::ostream &operator<<(std::ostream &os,
                        const std::unordered_map<K, V> &m) {
        os<< "{";
    for (const std::pair<K, V> &p: m) {
        os << p.first;
    }
     os<< "}";
    return os;
}
struct fs::fs_state_t fs::state {};

int fs::fuse_getattr(const char* path, struct stat* stbuf,
                     struct fuse_file_info*) {
  VLOG(2) << "fuse: getattr on " << path;

  memset(stbuf, 0, sizeof(struct stat));
  if (strcmp(path, "/") == 0) {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    return 0;
  }
  auto id = ObjectIDFromString(path + 1);
  bool exists = false;
  VINEYARD_CHECK_OK(state.client->Exists(id, exists));
  if (!exists) {
    return -ENOENT;
  }

  stbuf->st_mode = S_IFREG | 0444;
  stbuf->st_nlink = 1;
  stbuf->st_size = 0;  // not used, as we use `direct_io` in `open`.
  return 0;
}
#ifdef sdf
#define FUSE_CHECK_OK(status)                                            \
  do {                                                                   \
    auto _ret = (status);                                                \
    if (!_ret.ok()) {                                                    \
      VLOG(0) << "[error] Check failed: " << _ret.ToString() << " in \"" \
              << #status << "\""                                         \
              << ", in function " << __PRETTY_FUNCTION__ << ", file "    \
              << __FILE__ << ", line " << VINEYARD_TO_STRING(__LINE__)   \
              << std::endl;                                              \
      return -ECANCELED;                                                 \
    }                                                                   \
  }while (0)                                                            
// ECANCELED 125 Operation canceled

#define FUSE_ASSIGN_OR_RAISE_IMPL(result_name, lhs, rexpr) \
  auto&& result_name = (rexpr);                            \
  FUSE_CHECK_OK((result_name).status());                   \
  lhs = std::move(result_name).ValueUnsafe();

#define FUSE_ASSIGN_OR_RAISE(lhs, rexpr) \
  FUSE_ASSIGN_OR_RAISE_IMPL(             \
      ARROW_ASSIGN_OR_RAISE_NAME(_error_or_value, __COUNTER__), lhs, rexpr);
#endif
int fs::fuse_open(const char* path, struct fuse_file_info* fi) {
  VLOG(2) << "fuse: open " << path << " with mode " << fi->flags;
  if ((fi->flags & O_ACCMODE) != O_RDONLY) {
    return -EACCES;
  }
  auto target = ObjectIDFromString(path + 1);
  VLOG(2) << "converted objectID" << target;
  std::unique_ptr<ObjectMeta> omp(new ObjectMeta{});
  VINEYARD_CHECK_OK(state.client->GetMetaData(target, *omp));
  auto typeName = omp.get()->GetTypeName();
 

  VLOG(2)<<"initialized the variables";

  auto loc = state.views.find(target);
  std::shared_ptr<arrow::Buffer> buffer_;
  VLOG(2)<<"state view"<<state.views;
  if (loc == state.views.end()) {
      VLOG(2)<<"taregt: "<<target<<" ObjectID is not Found";

    if (typeName == "vineyard::Dataframe") {
      VLOG(1)<<"access a datarame";
      auto object = state.client->GetObject<vineyard::DataFrame>(target);
      if (object == nullptr) {
        return -ENOENT;
      }
      buffer_ = fuse::parquet_view(object);

      
    } else {
      VLOG(1)<<"access a numpy array";
        auto arr = std::dynamic_pointer_cast<vineyard::NumericArray<int64_t>>(
      state.client->GetObject(target));
      buffer_ = fuse::arrow_ipc_view(arr);
      
    }
    
    state.views[target] = buffer_;
  }else{
      VLOG(2)<<"taregt: "<<target<<" ObjectID is Found";

  }

  // bypass kernel's page cache to avoid knowing the size in `getattr`.
  //
  // see also:
  // https://stackoverflow.com/questions/46267972/fuse-avoid-calculating-size-in-getattr
  fi->direct_io = 1;
  return 0;
}

int fs::fuse_read(const char* path, char* buf, size_t size, off_t offset,
                  struct fuse_file_info* fi) {
  VLOG(2) << "fuse: read " << path << " from " << offset << ", expect " << size
          << " bytes";

  auto target = ObjectIDFromString(path + 1);
  VLOG(3) <<" read target "<<target;
  VLOG(2)<<"state view"<<state.views;

  auto loc = state.views.find(target);
  if (loc == state.views.end()) {
    VLOG(3) <<" target "<<target <<" is not found";

    return -ENOENT;
  }

  auto buffer = loc->second;
    VLOG(3) <<" target size: " << buffer->size();

  if (offset >= buffer->size()) {
    VLOG(2) << "finished transferring" ;

    return 0;
  } else {
    size_t slice = size;
    if (slice > static_cast<size_t>(buffer->size() - offset)) {
      slice = buffer->size() - offset;
    }
    memcpy(buf, buffer->data() + offset, slice);
    VLOG(2) << "copied data"<<slice;
    return slice;
  }
}

int fs::fuse_release(const char* path, struct fuse_file_info*) {
  VLOG(2) << "fuse: release " << path;

  auto target = ObjectIDFromString(path + 1);
  auto loc = state.views.find(target);
  if (loc == state.views.end()) {
    return -ENOENT;
  }
  state.views.erase(loc);
  return 0;
}

int fs::fuse_statfs(const char* path, struct statvfs*) {
  VLOG(2) << "fuse: statfs " << path;
  return 0;
}

int fs::fuse_opendir(const char* path, struct fuse_file_info* info) {
  VLOG(2) << "fuse: opendir " << path;
  return 0;
}

int fs::fuse_readdir(const char* path, void* buf, fuse_fill_dir_t filler,
                     off_t offset, struct fuse_file_info* fi,
                     enum fuse_readdir_flags flags) {
  VLOG(2) << "fuse: readdir " << path;

  if (strcmp(path, "/") != 0) {
    return -ENOENT;
  } 

  filler(buf, ".", NULL, 0, fuse_fill_dir_flags::FUSE_FILL_DIR_PLUS);
  filler(buf, "..", NULL, 0, fuse_fill_dir_flags::FUSE_FILL_DIR_PLUS);

  std::unordered_map<ObjectID, json> metas{};
  // VINEYARD_CHECK_OK(state.client->ListData(
  //     "vineyard::DataFrame", false, std::numeric_limits<size_t>::max(),
  //     metas));
  VINEYARD_CHECK_OK(state.client->ListData(
      ".*", true, std::numeric_limits<size_t>::max(), metas));
  for (auto const& item : metas) {
    std::string base = ObjectIDToString(item.first).c_str();
    filler(buf, base.c_str(), NULL, 0, fuse_fill_dir_flags::FUSE_FILL_DIR_PLUS);
  }
  return 0;
}

void* fs::fuse_init(struct fuse_conn_info* conn, struct fuse_config* cfg) {
  VLOG(2) << "fuse: initfs with vineyard socket " << state.vineyard_socket;

  state.client.reset(new vineyard::Client());
  state.client->Connect(state.vineyard_socket);

  (void) conn;
  cfg->kernel_cache = 0;
  return NULL;
}

void fs::fuse_destroy(void* private_data) {
  VLOG(2) << "fuse: destroy";

  state.views.clear();
  state.client->Disconnect();
}

int fs::fuse_access(const char* path, int mode) {
  VLOG(2) << "fuse: access " << path << " with mode " << mode;

  return mode;
}

}  // namespace fuse

}  // namespace vineyard
