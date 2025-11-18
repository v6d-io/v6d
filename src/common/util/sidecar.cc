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
#include <sys/mman.h>
#include <unistd.h>

#include <string>

#include "common/util/sidecar.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

#ifndef GET_BLOB_RECV_MEM_SIZE
#define GET_BLOB_RECV_MEM_SIZE (4096)
#endif  // GET_BLOB_RECV_MEM_SIZE

#ifndef ERROR_MSG_LENGTH
#define ERROR_MSG_LENGTH (256)
#endif  // ERROR_MSG_LENGTH

#ifndef MAX_METAS_FROM_NAME
#define MAX_METAS_FROM_NAME (1000)
#endif  // MAX_METAS_FROM_NAME

Status CreateMmapMemory(int& fd, size_t size, void*& base) {
  std::string file_name =
      std::string("/tmp/" + ObjectIDToString(GenerateObjectID(0))) + ".mmap";
  return CreateMmapMemory(file_name, fd, size, base);
}

Status CreateMmapMemory(std::string file_name, int& fd, size_t size,
                        void*& base) {
  fd = open(file_name.c_str(), O_RDWR | O_CREAT | O_NONBLOCK, 0666);
  if (fd < 0) {
    std::cout << "Failed to create mmap file: '" << file_name << "', "
              << strerror(errno);
    return Status::IOError("Failed to open file '" + file_name + "', " +
                           strerror(errno));
  }

  unlink(file_name.c_str());
  if (ftruncate64(fd, size) != 0) {
    std::cout << "Failed to ftruncate file " << file_name;
    close(fd);
    return Status::IOError("Failed to ftruncate file " + file_name);
  }

  base = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) {
    std::cout << "Failed to mmap file '" << file_name << "', "
              << strerror(errno);
    close(fd);
    return Status::IOError("Failed to mmap file '" + file_name + "', " +
                           strerror(errno));
  }

  memset(base, 0, size);
  return Status::OK();
}

Status WriteErrorMsg(Status error, void* base, size_t size) {
  if (base == nullptr) {
    std::cout << "Base pointer is null, cannot write error message."
              << std::endl;
    return Status::Invalid("Base pointer is null, cannot write error message.");
  }
  std::string error_str = error.ToString().substr(0, ERROR_MSG_LENGTH);
  memcpy(reinterpret_cast<unsigned char*>(base) +
             (size - ERROR_MSG_LENGTH - sizeof(unsigned char)),
         error_str.c_str(), error_str.size());
  reinterpret_cast<unsigned char*>(base)[size - sizeof(unsigned char)] =
      static_cast<unsigned char>(error.code());
  return Status::OK();
}

Status CheckBlobReceived(void* base, size_t size, int index, bool& finished) {
  if (base == nullptr) {
    std::cout << "Base pointer is null, cannot check blob received."
              << std::endl;
    return Status::Invalid("Base pointer is null, cannot check blob received.");
  }
  if (size >= static_cast<int>(GET_BLOB_RECV_MEM_SIZE - ERROR_MSG_LENGTH -
                               sizeof(unsigned char))) {
    return Status::Invalid("Size is too small to check blob received.");
  }

  finished = false;
  if (index == -1) {
    for (size_t i = 0; i < size; i++) {
      if (reinterpret_cast<unsigned char*>(base)[i] == 0) {
        return Status::OK();
      }
    }
    finished = true;
    return Status::OK();
  } else if (index > 0 && index < static_cast<int>(size)) {
    finished = reinterpret_cast<unsigned char*>(base)[index] == 1;
    return Status::OK();
  }
  return Status::Invalid("Index is out of bounds for checking blob received.");
}

Status SetBlobReceived(void* base, int index) {
  if (base == nullptr) {
    std::cout << "Base pointer is null, cannot set blob received." << std::endl;
    return Status::Invalid("Base pointer is null, cannot set blob received.");
  }
  if (index < 0 ||
      index >= static_cast<int>(GET_BLOB_RECV_MEM_SIZE - ERROR_MSG_LENGTH -
                                sizeof(unsigned char))) {
    return Status::Invalid("Index is out of bounds for setting blob received.");
  }

  reinterpret_cast<unsigned char*>(base)[index] = 1;  // mark as received
  return Status::OK();
}

Status ReleaseMmapMemory(int fd, void* base, size_t size) {
  if (base != nullptr) {
    if (munmap(base, size) != 0) {
      std::cout << "Failed to munmap memory, " << strerror(errno);
      return Status::IOError("Failed to munmap memory, " +
                             std::string(strerror(errno)));
    }
    int ret = close(fd);
    if (ret != 0) {
      std::cout << "Failed to close file descriptor, error:" << strerror(errno)
                << ", it may cause resource leak.";
      return Status::IOError("Failed to close file descriptor, " +
                             std::string(strerror(errno)));
    }
    std::cout << "Released mmap memory: fd = " << fd << ", base = " << base
              << ", size = " << size << std::endl;
  }
  return Status::OK();
}

}  // namespace vineyard
