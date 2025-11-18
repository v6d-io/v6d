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

#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "common/util/status.h"
#include "vllm-kv-cache/src/io/posix_io_adaptor.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

Status PosixIOAdaptor::Open(std::string mode, bool direct_io) {
  int flags = 0;
  if (mode == "r") {
    flags = O_RDONLY;
  } else if (mode == "w") {
    flags = O_WRONLY | O_CREAT | O_TRUNC;
  } else {
    return Status::Invalid("Invalid mode: " + mode);
  }
  fd_ = open(location_.c_str(), flags, 0666);
  if (fd_ == -1) {
    return Status::IOError("Failed to open file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  return Status::OK();
}

Status PosixIOAdaptor::Read(void* data, size_t size) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  ssize_t bytes_read = read(fd_, data, size);
  if (bytes_read == -1) {
    return Status::IOError("Failed to read file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  if (static_cast<size_t>(bytes_read) < size) {
    return Status::EndOfFile();
  }
  return Status::OK();
}

Status PosixIOAdaptor::Write(void* data, size_t size) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_);
  }
  ssize_t bytes_written = write(fd_, data, size);
  if (bytes_written == -1) {
    return Status::IOError("Failed to write file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  if (static_cast<size_t>(bytes_written) < size) {
    return Status::IOError("Partial write to file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  return Status::OK();
}

Status PosixIOAdaptor::Close() {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  if (close(fd_) < 0) {
    return Status::IOError("Failed to close file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  fd_ = -1;
  return Status::OK();
}

Status PosixIOAdaptor::FileTruncate(size_t size) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  if (ftruncate(fd_, size) < 0) {
    return Status::IOError("Failed to truncate file: " + location_ +
                           " to size: " + std::to_string(size) +
                           " error:" + std::string(strerror(errno)));
  }
  return Status::OK();
}

Status PosixIOAdaptor::GetFileSize(size_t& size) {
  return Status::NotImplemented(
      "GetFileSize is not implemented in PosixIOAdaptor");
}

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard
