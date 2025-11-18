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

#ifndef MODULES_VLLM_KV_CACHE_SRC_IO_IO_ADAPTOR_H_
#define MODULES_VLLM_KV_CACHE_SRC_IO_IO_ADAPTOR_H_

#include <future>
#include <memory>
#include <string>
#include <vector>

#include "common/util/status.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

class IIOAdaptor {
 public:
  virtual Status Open(std::string mode, bool direct_io = false) = 0;

  virtual Status Read(void* data, size_t size) = 0;

  virtual Status Write(void* data, size_t size) = 0;

  virtual Status Close() = 0;

  virtual Status GetFileSize(size_t& size) = 0;

  virtual Status FileTruncate(size_t size) = 0;

  virtual std::future<Status> AsyncRead(void* data, size_t size,
                                        size_t offset) {
    return std::async(std::launch::async,
                      [this, data, size]() { return this->Read(data, size); });
  }

  virtual std::future<Status> AsyncWrite(void* data, size_t size,
                                         size_t offset) {
    return std::async(std::launch::async,
                      [this, data, size]() { return this->Write(data, size); });
  }

  virtual Status BatchAsyncRead(std::vector<void*>& data_vec,
                                std::vector<size_t>& size_vec,
                                std::vector<size_t>& offset_vec,
                                std::vector<std::future<Status>>& results) {
    for (size_t i = 0; i < data_vec.size(); ++i) {
      results.push_back(
          std::async(std::launch::async, [this, &data_vec, &size_vec, i]() {
            return this->Read(data_vec[i], size_vec[i]);
          }));
    }
    return Status::OK();
  }

  virtual Status BatchAsyncWrite(std::vector<void*>& data_vec,
                                 std::vector<size_t>& size_vec,
                                 std::vector<size_t>& offset_vec,
                                 std::vector<std::future<Status>>& results) {
    for (size_t i = 0; i < data_vec.size(); ++i) {
      results.push_back(
          std::async(std::launch::async, [this, &data_vec, &size_vec, i]() {
            return this->Write(data_vec[i], size_vec[i]);
          }));
    }
    return Status::OK();
  }

  virtual std::future<Status> AsyncRead(std::shared_ptr<void> data, size_t size,
                                        size_t offset) {
    return std::async(std::launch::async, []() {
      return Status::NotImplemented(
          "AsyncRead with shared_ptr<void> is not implemented in IIOAdaptor");
    });
  }

  virtual std::future<Status> AsyncWrite(std::shared_ptr<void> data,
                                         size_t size, size_t offset) {
    return std::async(std::launch::async, []() {
      return Status::NotImplemented(
          "AsyncWrite with shared_ptr<void> is not implemented in IIOAdaptor");
    });
  }

  virtual Status BatchAsyncRead(std::vector<std::shared_ptr<void>>& data_vec,
                                std::vector<size_t>& size_vec,
                                std::vector<size_t>& offset_vec,
                                std::vector<std::future<Status>>& results) {
    return Status::NotImplemented(
        "BatchAsyncRead with shared_ptr<void> is not implemented in "
        "IIOAdaptor");
  }

  virtual Status BatchAsyncWrite(std::vector<std::shared_ptr<void>>& data_vec,
                                 std::vector<size_t>& size_vec,
                                 std::vector<size_t>& offset_vec,
                                 std::vector<std::future<Status>>& results) {
    return Status::NotImplemented(
        "BatchAsyncWrite with shared_ptr<void> is not implemented in "
        "IIOAdaptor");
  }
};

using IOAdaptorFactory =
    std::function<std::shared_ptr<IIOAdaptor>(const std::string& path)>;

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_IO_IO_ADAPTOR_H_
