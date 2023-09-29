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

#include "server/util/file_io_adaptor.h"

#include <unistd.h>

#include <vector>

#if defined(BUILD_VINEYARDD_SPILLING)
#include "arrow/filesystem/filesystem.h"
#include "common/util/arrow.h"
#endif

namespace vineyard {
namespace io {

#if defined(BUILD_VINEYARDD_SPILLING)

FileIOAdaptor::FileIOAdaptor(const std::string& location) {
  // TODO(ZjuYTW): Maybe we should check the validation of dir_path
  location_ = location;
  fs_ = arrow::fs::FileSystemFromUriOrPath(location).ValueOrDie();
}

FileIOAdaptor::~FileIOAdaptor() {
  VINEYARD_DISCARD(Close());
  fs_.reset();
}

Status FileIOAdaptor::Open() { return this->Open("r"); }

Status FileIOAdaptor::Open(const char* mode) {
  if (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL) {
    int t = location_.find_last_of('/');
    if (t != -1) {
      std::string folder_path = location_.substr(0, t);
      if (access(folder_path.c_str(), 0) != 0) {
        RETURN_ON_ERROR(CreateDir(folder_path));
      }
    }

    if (strchr(mode, 'w') != NULL) {
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(ofp_, fs_->OpenOutputStream(location_));
    } else {
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(ofp_, fs_->OpenAppendStream(location_));
    }
  } else {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(ifp_, fs_->OpenInputFile(location_));
  }
  return Status::OK();
}

Status FileIOAdaptor::Write(const char* buffer, size_t size) {
  if (ofp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in write mode: " +
                           location_);
  }
  return ArrowError(ofp_->Write(buffer, size));
}

Status FileIOAdaptor::Flush() {
  if (ofp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in write mode: " +
                           location_);
  }
  return ArrowError(ofp_->Flush());
}

Status FileIOAdaptor::Close() {
  Status s1, s2;
  if (ifp_) {
    s1 = ArrowError(ifp_->Close());
  }
  if (ofp_) {
    auto status = ofp_->Flush();
    if (status.ok()) {
      s2 = ArrowError(ofp_->Close());
    } else {
      s2 = ArrowError(status);
    }
  }
  return s1 & s2;
}

Status FileIOAdaptor::Read(void* buffer, size_t size) {
  if (ifp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in read mode: " +
                           location_);
  }
  auto r = ifp_->Read(size, buffer);
  if (r.ok()) {
    if (r.ValueUnsafe() < static_cast<int64_t>(size)) {
      return Status::EndOfFile();
    } else {
      return Status::OK();
    }
  } else {
    return ArrowError(r.status());
  }
}

Status FileIOAdaptor::CreateDir(const std::string& path) {
  return ArrowError(fs_->CreateDir(path, true));
}

Status FileIOAdaptor::DeleteDir() {
  auto _ = fs_->DeleteDir(location_);  // discard the deletion error
  return Status::OK();
}

Status FileIOAdaptor::RemoveFiles(const std::vector<std::string>& paths) {
  auto _ = fs_->DeleteFiles(paths);  // discard the deletion error
  return Status::OK();
}

Status FileIOAdaptor::RemoveFile(const std::string& path) {
  auto _ = fs_->DeleteFile(path);  // discard the deletion error
  return Status::OK();
}

#else

FileIOAdaptor::FileIOAdaptor(const std::string& location) {
  // TODO(ZjuYTW): Maybe we should check the validation of dir_path
  location_ = location;
}

FileIOAdaptor::~FileIOAdaptor() {}

Status FileIOAdaptor::Open() { return this->Open("r"); }

Status FileIOAdaptor::Open(const char* mode) {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::Write(const char* buffer, size_t size) {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::Flush() {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::Close() {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::Read(void* buffer, size_t size) {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::CreateDir(const std::string& path) {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::DeleteDir() {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::RemoveFiles(const std::vector<std::string>& paths) {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

Status FileIOAdaptor::RemoveFile(const std::string& path) {
  return Status::NotImplemented(
      "The spilling functionality is not built into vineyardd");
}

#endif

}  // namespace io
}  // namespace vineyard
