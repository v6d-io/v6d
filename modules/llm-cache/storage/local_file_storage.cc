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
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/util/logging.h"
#include "gulrak/filesystem.hpp"
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
  lfd->path = path;

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
  if (!ghc::filesystem::exists(path)) {
    if (!ghc::filesystem::create_directories(path)) {
      if (ghc::filesystem::exists(path)) {
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
  int ret;
#ifdef __linux__
  ret = fdatasync(lfd->fd);
#else
  ret = fsync(lfd->fd);
#endif

  if (ret == -1) {
    return Status::IOError("Failed to flush file: " + formatIOError(lfd->path));
  }
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
  return ghc::filesystem::exists(path);
}

Status LocalFileStorage::Delete(std::string path) {
  if (ghc::filesystem::exists(path)) {
    ghc::filesystem::remove_all(path);
  }
  return Status::OK();
}

std::string LocalFileStorage::GetTmpFileDir(std::string surfix) {
  pid_t pid = getpid();
  char* pod_name_str = getenv("POD_NAME");
  if (pod_name_str == nullptr || strlen(pod_name_str) == 0) {
    return this->tempFileDir + std::to_string(pid);
  }
  std::string pod_name = pod_name_str;
  return this->tempFileDir + pod_name + "/" + std::to_string(pid) + surfix;
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

Status LocalFileStorage::GetFileAccessTime(
    const std::string& path,
    std::chrono::duration<int64_t, std::nano>& accessTime) {
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) == -1) {
    return Status::IOError("Failed to get file access time: " +
                           formatIOError(path));
  }
#ifdef __APPLE__
  accessTime = std::chrono::duration<int64_t, std::nano>(
      statbuf.st_atimespec.tv_sec * SECOND_TO_NANOSECOND +
      statbuf.st_atimespec.tv_nsec);
#else
  accessTime = std::chrono::duration<int64_t, std::nano>(
      statbuf.st_atim.tv_sec * SECOND_TO_NANOSECOND + statbuf.st_atim.tv_nsec);
#endif
  return Status::OK();
}

Status LocalFileStorage::TouchFile(const std::string& path) {
  VLOG(100) << "Before touch File:";
  PrintFileAccessTime(path);
  auto now = std::chrono::high_resolution_clock::now();
  auto now_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      now.time_since_epoch())
                      .count();
  struct timespec times[2] = {};
  times[0].tv_sec = now_nano / SECOND_TO_NANOSECOND;
  times[0].tv_nsec = now_nano % SECOND_TO_NANOSECOND;

#ifdef __APPLE__
  struct stat file_stat;
  if (stat(path.c_str(), &file_stat) == -1) {
    return Status::IOError("Failed to get file stat: " + formatIOError(path));
  }
  times[1].tv_sec = file_stat.st_mtime;
  times[1].tv_nsec = file_stat.st_mtimespec.tv_nsec;
#else
  times[1].tv_sec = UTIME_OMIT;
  times[1].tv_nsec = UTIME_OMIT;
#endif

  if (utimensat(AT_FDCWD, path.c_str(), times, 0) == -1) {
    return Status::IOError("Failed to touch file: " + formatIOError(path));
  }
  VLOG(100) << "After touch File:";
  PrintFileAccessTime(path);
  return Status::OK();
}

Status LocalFileStorage::GetFileList(std::string dirPath,
                                     std::vector<std::string>& fileList) {
  try {
    for (auto it = ghc::filesystem::recursive_directory_iterator(dirPath);
         it != ghc::filesystem::recursive_directory_iterator(); ++it) {
      if (ghc::filesystem::is_regular_file(*it)) {
        fileList.push_back(it->path().string());
      }
    }
  } catch (ghc::filesystem::filesystem_error& e) {
    if (e.code() == std::errc::no_such_file_or_directory) {
      return Status::OK();
    } else {
      return Status::IOError(e.what());
    }
  }
  return Status::OK();
}

}  // namespace vineyard
