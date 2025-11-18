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

#ifndef MODULES_VLLM_KV_CACHE_SRC_IO_MOCK_IO_ADAPTER_H_
#define MODULES_VLLM_KV_CACHE_SRC_IO_MOCK_IO_ADAPTER_H_

#include <memory>
#include <string>
#include <vector>

#include "common/util/callback.h"
#include "common/util/env.h"
#include "common/util/status.h"
#include "vllm-kv-cache/src/io/aio_adaptor.h"
#include "vllm-kv-cache/src/io/error_injection.h"
#include "vllm-kv-cache/src/io/io_adaptor.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

// Factory function for creating MockIOAdaptor instances
IOAdaptorFactory GetMockIOAdaptorFactory();

class MockIOAdaptor : public io::IIOAdaptor {
 public:
  explicit MockIOAdaptor(const std::string& location);

  Status Open(std::string mode, bool direct_io = false) override;

  Status Read(void* data, size_t size) override;

  Status Write(void* data, size_t size) override;

  Status Close() override;

  Status GetFileSize(size_t& size) override;

  Status FileTruncate(size_t size) override;

  std::future<Status> AsyncWrite(void* data, size_t size,
                                 size_t offset) override;

  std::future<Status> AsyncRead(void* data, size_t size,
                                size_t offset) override;

  Status BatchAsyncRead(std::vector<void*>& data_vec,
                        std::vector<size_t>& size_vec,
                        std::vector<size_t>& offset_vec,
                        std::vector<std::future<Status>>& results) override;

  Status BatchAsyncWrite(std::vector<void*>& data_vec,
                         std::vector<size_t>& size_vec,
                         std::vector<size_t>& offset_vec,
                         std::vector<std::future<Status>>& results) override;

  Status WriteDataAtOffset(const std::string& location, void* data, size_t size,
                           size_t offset);

 private:
  std::string location_;
  std::shared_ptr<io::AIOAdaptor> aio_adaptor_;
};

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_IO_MOCK_IO_ADAPTER_H_
