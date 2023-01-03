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

#ifndef MODULES_FUSE_FUSE_IMPL_H_
#define MODULES_FUSE_FUSE_IMPL_H_

#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define FUSE_USE_VERSION 32
#include "fuse3/fuse.h"
#include "fuse3/fuse_common.h"
#include "fuse3/fuse_lowlevel.h"

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "client/client.h"

#include "adaptors/arrow_ipc/deserializer_registry.h"
#include "adaptors/chunk_buffer/chunk_buffer.h"
namespace arrow {
class Buffer;
}

namespace vineyard {

namespace fuse {

struct fs {
  static struct fs_state_t {
    struct fuse_conn_info_opts* conn_opts;
    std::string vineyard_socket;
    std::shared_ptr<Client> client;
    std::mutex mtx_;
    std::unordered_map<std::string, std::shared_ptr<internal::ChunkBuffer>>
        views;
    std::unordered_map<std::string, std::shared_ptr<arrow::BufferBuilder>>
        mutable_views;
    std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt>
        ipc_desearilizer_registry;
  } state;

  static int fuse_getattr(const char* path, struct stat* stbuf,
                          struct fuse_file_info*);

  static int fuse_open(const char* path, struct fuse_file_info* fi);

  static int fuse_read(const char* path, char* buf, size_t size, off_t offset,
                       struct fuse_file_info* fi);

  static int fuse_write(const char* path, const char* buf, size_t size,
                        off_t offset, struct fuse_file_info* fi);

  static int fuse_statfs(const char* path, struct statvfs*);

  static int fuse_flush(const char* path, struct fuse_file_info*);

  static int fuse_release(const char* path, struct fuse_file_info*);

  static int fuse_getxattr(const char* path, const char* name, char*, size_t);

  static int fuse_opendir(const char* path, struct fuse_file_info* info);

  static int fuse_readdir(const char* path, void* buf, fuse_fill_dir_t filler,
                          off_t offset, struct fuse_file_info* fi,
                          enum fuse_readdir_flags flags);

  static void* fuse_init(struct fuse_conn_info* conn, struct fuse_config* cfg);

  static void fuse_destroy(void* private_data);

  static int fuse_access(const char* path, int mode);

  static int fuse_create(const char* path, mode_t, struct fuse_file_info*);
};

}  // namespace fuse

}  // namespace vineyard

#endif  // MODULES_FUSE_FUSE_IMPL_H_
