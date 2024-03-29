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
#include <stdio.h>
#include <sys/file.h>
#include <sys/stat.h>  // For stat
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "common/util/logging.h"
#include "llm-cache/storage/local_file_storage.h"
#include "llm-cache/thread_group.h"

namespace vineyard {
std::shared_ptr<FileDescriptor> LocalFileStorage::CreateFileDescriptor() {
  return std::make_shared<LocalFileDescriptor>();
}

static std::string formatIOError(std::string const& path) {
  std::stringstream ss;
  ss << " at " << path << ", errno: " << errno
     << ", error: " << strerror(errno);
  return ss.str();
}

Status LocalFileStorage::Open(std::string path,
                              std::shared_ptr<FileDescriptor>& fd,
                              FileOperationType fileOperationType) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);

  int flag = 0;
  if (fileOperationType & FileOperationType::READ) {
    flag |= O_RDONLY;
  } else {
    flag |= O_RDWR | O_CREAT;
  }
  lfd->fd = open(path.c_str(), flag, 0666);
  if (lfd->fd == -1) {
    return Status::IOError("Failed to open file: " + formatIOError(path));
  }
  return Status::OK();
}

Status LocalFileStorage::Seek(std::shared_ptr<FileDescriptor>& fd,
                              size_t offset) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  if (lseek(lfd->fd, offset, SEEK_SET) == -1) {
    return Status::IOError("Failed to seek file: " + formatIOError(lfd->path));
  }
  return Status::OK();
}

Status LocalFileStorage::Read(std::shared_ptr<FileDescriptor>& fd, void* data,
                              size_t size) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  if (read(lfd->fd, data, size) == -1) {
    return Status::IOError("Failed to read file: " + formatIOError(lfd->path));
  }
  return Status::OK();
}

Status LocalFileStorage::Write(std::shared_ptr<FileDescriptor>& fd,
                               const void* data, size_t size) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  if (write(lfd->fd, data, size) == -1) {
    return Status::IOError("Failed to write file: " + formatIOError(lfd->path));
  }
  return Status::OK();
}

Status LocalFileStorage::Mkdir(std::string path) {
  // create the directory if it does not exist
  if (!std::filesystem::exists(path)) {
    if (!std::filesystem::create_directories(path)) {
      if (!std::filesystem::exists(path)) {
        VLOG(100) << "directory exists" << path;
      } else {
        VLOG(100) << "Failed to create directory:" << path;
        return Status::IOError("Failed to create directory");
      }
    }
  }
  return Status::OK();
}

Status LocalFileStorage::Flush(std::shared_ptr<FileDescriptor>& fd) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  fdatasync(lfd->fd);
  return Status::OK();
}

Status LocalFileStorage::GetCurrentPos(std::shared_ptr<FileDescriptor>& fd,
                                       size_t& pos) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  pos = lseek(lfd->fd, 0, SEEK_CUR);
  return Status::OK();
}

Status LocalFileStorage::Close(std::shared_ptr<FileDescriptor>& fd) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  if (close(lfd->fd) == -1) {
    return Status::IOError("Failed to close file: " + formatIOError(lfd->path));
  }
  return Status::OK();
}

Status LocalFileStorage::GetFileSize(std::shared_ptr<FileDescriptor>& fd,
                                     size_t& size) {
  std::shared_ptr<LocalFileDescriptor> lfd =
      std::static_pointer_cast<LocalFileDescriptor>(fd);
  size_t current = lseek(lfd->fd, 0, SEEK_CUR);
  size = lseek(lfd->fd, 0, SEEK_END);
  lseek(lfd->fd, current, SEEK_SET);
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
  char* pod_name_str = getenv("POD_NAME");
  if (pod_name_str == nullptr or strlen(pod_name_str) == 0) {
    return this->tempFileDir + std::to_string(pid);
  }
  std::string pod_name = pod_name_str;
  return this->tempFileDir + pod_name + "/" + std::to_string(pid);
}

Status LocalFileStorage::MoveFileAtomic(std::string src, std::string dst) {
  // Use open and then rename to avoid the unsupported issue on NFS.
  int dst_fd = open(dst.c_str(), O_CREAT | O_RDWR, 0666);
  if (dst_fd == -1) {
    return Status::IOError("Failed to create file: " + formatIOError(dst));
  } else {
    close(dst_fd);
    if (rename(src.c_str(), dst.c_str())) {
      return Status::IOError("Failed to move file: " + formatIOError(src));
    }
  }
  return Status::OK();
}

}  // namespace vineyard
