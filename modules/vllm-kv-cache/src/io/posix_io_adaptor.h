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

#ifndef MODULES_VLLM_KV_CACHE_SRC_IO_POSIX_IO_ADAPTOR_H_
#define MODULES_VLLM_KV_CACHE_SRC_IO_POSIX_IO_ADAPTOR_H_

#include <string>

#include "common/util/status.h"
#include "vllm-kv-cache/src/io/io_adaptor.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

class PosixIOAdaptor : public IIOAdaptor {
 public:
  explicit PosixIOAdaptor(const std::string& location, bool direct_io = false)
      : location_(location), fd_(-1) {}

  Status Open(std::string mode, bool direct_io = false) override;

  Status Read(void* data, size_t size) override;

  Status Write(void* data, size_t size) override;

  Status GetFileSize(size_t& size) override;

  Status Close() override;

  Status FileTruncate(size_t size) override;

 private:
  std::string location_;
  int fd_ = -1;
};

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_IO_POSIX_IO_ADAPTOR_H_
