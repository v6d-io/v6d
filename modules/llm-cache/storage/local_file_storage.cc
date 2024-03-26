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
#include <sys/file.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "common/util/logging.h"
#include "llm-cache/storage/local_file_storage.h"

namespace vineyard {
std::shared_ptr<FileDescriptor> LocalFileStorage::CreateFileDescriptor() {
  return std::make_shared<LocalFileDescriptor>();
}

Status LocalFileStorage::Open(std::string path,
                              std::shared_ptr<FileDescriptor>& fd,
                              FileOperationType fileOperationType) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);

  std::ios_base::openmode mode = std::ios_base::binary;
  if (fileOperationType & FileOperationType::READ) {
    mode |= std::ios_base::in;
  }
  if (fileOperationType & FileOperationType::WRITE) {
    mode |= std::ios_base::out;
  }
  lfd->fstream.open(path, mode);

  if (!lfd->fstream.is_open()) {
    VLOG(100) << "Failed to open file: " << path << " "
              << lfd->fstream.rdstate();
    return Status::IOError("Failed to open file: " + path);
  }
  return Status::OK();
}

Status LocalFileStorage::Seek(std::shared_ptr<FileDescriptor>& fd,
                              size_t offset) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  lfd->fstream.seekp(offset);
  lfd->fstream.seekg(offset);
  if (!lfd->fstream.good()) {
    lfd->fstream.clear();
    VLOG(100) << "Failed to seek file: ";
    return Status::IOError("Failed to seek file");
  }
  return Status::OK();
}

Status LocalFileStorage::Read(std::shared_ptr<FileDescriptor>& fd, void* data,
                              size_t size) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  lfd->fstream.read(reinterpret_cast<char*>(data), size);
  if (!lfd->fstream.good()) {
    VLOG(100) << "Failed to read file: ";
    VLOG(100) << "error code:" << lfd->fstream.rdstate();
    lfd->fstream.clear();
    return Status::IOError("Failed to read file");
  }
  return Status::OK();
}

Status LocalFileStorage::Write(std::shared_ptr<FileDescriptor>& fd,
                               const void* data, size_t size) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  lfd->fstream.write(reinterpret_cast<const char*>(data), size);
  if (!lfd->fstream.good()) {
    lfd->fstream.clear();
    VLOG(100) << "Failed to write file: ";
    VLOG(100) << "error code:" << lfd->fstream.rdstate();
    return Status::IOError("Failed to write file");
  }
  return Status::OK();
}

Status LocalFileStorage::Mkdir(std::string path) {
  // create the directory if it does not exist
  VLOG(100) << "Create directory:" << path;
  if (!std::filesystem::exists(path)) {
    if (!std::filesystem::create_directories(path)) {
      VLOG(100) << "Failed to create directory:" << path;
      return Status::IOError("Failed to create directory");
    }
  }
  return Status::OK();
}

Status LocalFileStorage::Flush(std::shared_ptr<FileDescriptor>& fd) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  lfd->fstream.flush();
  if (!lfd->fstream.good()) {
    lfd->fstream.clear();
    return Status::IOError("Failed to flush file");
  }
  return Status::OK();
}

Status LocalFileStorage::GetCurrentPos(std::shared_ptr<FileDescriptor>& fd,
                                       size_t& pos) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  pos = lfd->fstream.tellp();
  return Status::OK();
}

Status LocalFileStorage::Close(std::shared_ptr<FileDescriptor>& fd) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  lfd->fstream.close();
  if (lfd->fstream.is_open()) {
    VLOG(100) << "Failed to close";
    return Status::IOError("Failed to close file");
  }
  return Status::OK();
}

Status LocalFileStorage::GetFileSize(std::shared_ptr<FileDescriptor>& fd,
                                     size_t& size) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  size_t current_pos = lfd->fstream.tellp();
  lfd->fstream.seekp(0, std::ios_base::end);
  size = lfd->fstream.tellp();
  VLOG(100) << "read size:" << size;
  lfd->fstream.seekp(current_pos);
  if (size < 0) {
    return Status::IOError("Failed to get file size");
  }
  return Status::OK();
}

bool LocalFileStorage::IsFileExist(const std::string& path) {
  return std::filesystem::exists(path);
}

Status LocalFileStorage::Delete(std::string path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::remove_all(path);
  }
  return Status::OK();
}

std::string LocalFileStorage::GetTmpFileDir(std::string filePath) {
  pid_t pid = getpid();
  return this->tempFileDir + std::to_string(pid);
}

Status LocalFileStorage::MoveFileAtomic(std::string src, std::string dst) {
  if (renameat2(AT_FDCWD, src.c_str(), AT_FDCWD, dst.c_str(),
                RENAME_NOREPLACE)) {
    return Status::IOError("Failed to move file");
  }
  return Status::OK();
}

}  // namespace vineyard
