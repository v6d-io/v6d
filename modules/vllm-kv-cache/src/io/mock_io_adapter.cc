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

#include "vllm-kv-cache/src/io/mock_io_adapter.h"
#include "vllm-kv-cache/src/io/mock_aio_operations.h"

#include <fcntl.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/util/logging.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

IOAdaptorFactory GetMockIOAdaptorFactory() {
  LOG(INFO) << "Using mock IO adaptor";
  return [](const std::string& path)
             -> std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> {
    auto adaptor = std::make_shared<MockIOAdaptor>(path);
    return adaptor;
  };
}

// MockIOAdaptor using MockAIOOperations
MockIOAdaptor::MockIOAdaptor(const std::string& location)
    : location_(location),
      aio_adaptor_(std::make_shared<io::AIOAdaptor>(
          location,
          std::make_shared<vineyard::vllm_kv_cache::io::MockAIOOperations>())) {
}

Status MockIOAdaptor::Open(std::string mode, bool direct_io) {
  return aio_adaptor_->Open(mode, direct_io);
}

Status MockIOAdaptor::Read(void* data, size_t size) {
  // Check if we need to inject a read error
  if (global_mock_io_read_error) {
    return Status::IOError("Injected read error");
  }

  // Check if we need to simulate a read timeout
  if (global_mock_io_read_timeout) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_io_timeout_ms));
  }

  return aio_adaptor_->Read(data, size);
}

Status MockIOAdaptor::Write(void* data, size_t size) {
  // Check if we need to inject a write error
  if (global_mock_io_write_error) {
    return Status::IOError("Injected write error");
  }

  // Check if we need to simulate a write timeout
  if (global_mock_io_write_timeout) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_io_timeout_ms));
  }

  // need to write data to disk(cpfs) to serve for get file size
  WriteDataAtOffset(location_, data, size, 0);

  return aio_adaptor_->Write(data, size);
}

Status MockIOAdaptor::Close() { return aio_adaptor_->Close(); }

Status MockIOAdaptor::GetFileSize(size_t& size) {
  return aio_adaptor_->GetFileSize(size);
}

Status MockIOAdaptor::FileTruncate(size_t size) { return Status::OK(); }

std::future<Status> MockIOAdaptor::AsyncWrite(void* data, size_t size,
                                              size_t offset) {
  if (global_mock_io_write_error) {
    std::promise<Status> promise;
    promise.set_value(Status::IOError("Injected async write error"));
    return promise.get_future();
  }

  if (global_mock_io_write_timeout) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_io_timeout_ms));
  }

  // need to write data to disk(cpfs) to serve for get file size
  WriteDataAtOffset(location_, data, size, offset);

  return aio_adaptor_->AsyncWrite(data, size, offset);
}

std::future<Status> MockIOAdaptor::AsyncRead(void* data, size_t size,
                                             size_t offset) {
  if (global_mock_io_read_error) {
    std::promise<Status> promise;
    promise.set_value(Status::IOError("Injected async read error"));
    return promise.get_future();
  }

  if (global_mock_io_read_timeout) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_io_timeout_ms));
  }

  return aio_adaptor_->AsyncRead(data, size, offset);
}

Status MockIOAdaptor::BatchAsyncRead(
    std::vector<void*>& data_vec, std::vector<size_t>& size_vec,
    std::vector<size_t>& offset_vec,
    std::vector<std::future<Status>>& results) {
  if (global_mock_io_batch_read_error) {
    for (size_t i = 0; i < data_vec.size(); ++i) {
      std::promise<Status> promise;
      promise.set_value(
          Status::IOError("Injected batch async read error for operation " +
                          std::to_string(i)));
      results.push_back(promise.get_future());
    }
    return Status::OK();
  }

  if (global_mock_io_batch_read_timeout) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_io_timeout_ms));
  }

  return aio_adaptor_->BatchAsyncRead(data_vec, size_vec, offset_vec, results);
}

Status MockIOAdaptor::BatchAsyncWrite(
    std::vector<void*>& data_vec, std::vector<size_t>& size_vec,
    std::vector<size_t>& offset_vec,
    std::vector<std::future<Status>>& results) {
  if (global_mock_io_batch_write_error) {
    for (size_t i = 0; i < data_vec.size(); ++i) {
      std::promise<Status> promise;
      promise.set_value(
          Status::IOError("Injected batch async write error for operation " +
                          std::to_string(i)));
      results.push_back(promise.get_future());
    }
    return Status::OK();
  }

  if (global_mock_io_batch_write_timeout) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_io_timeout_ms));
  }

  // need to write data to disk(cpfs) to serve for get file size
  for (size_t i = 0; i < data_vec.size(); ++i) {
    WriteDataAtOffset(location_, data_vec[i], size_vec[i], offset_vec[i]);
  }

  return aio_adaptor_->BatchAsyncWrite(data_vec, size_vec, offset_vec, results);
}

Status MockIOAdaptor::WriteDataAtOffset(const std::string& location, void* data,
                                        size_t size, size_t offset) {
  int fd = open(location.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd == -1) {
    LOG(ERROR) << "Failed to open file: " << location << ", errno: " << errno;
    return Status::IOError("Failed to open file: " + location +
                           ", errno: " + std::to_string(errno));
  }

  if (lseek(fd, offset, SEEK_SET) == -1) {
    close(fd);
    return Status::IOError("Failed to seek to offset: " +
                           std::to_string(offset));
  }

  ssize_t written = write(fd, data, size);
  if (written == -1) {
    close(fd);
    return Status::IOError("Failed to write data");
  }

  if (static_cast<size_t>(written) != size) {
    close(fd);
    return Status::IOError("Incomplete write: expected " +
                           std::to_string(size) + " bytes, wrote " +
                           std::to_string(written) + " bytes");
  }

  if (close(fd) == -1) {
    return Status::IOError("Failed to close file");
  }

  return Status::OK();
}

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard
