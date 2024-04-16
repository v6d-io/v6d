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

#ifndef MODULES_LLM_CACHE_STORAGE_LOCAL_FILE_STORAGE_H_
#define MODULES_LLM_CACHE_STORAGE_LOCAL_FILE_STORAGE_H_

#include <fstream>
#include <list>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "llm-cache/storage/file_storage.h"

namespace vineyard {

struct LocalFileDescriptor : public FileDescriptor {
  int fd;
  std::string path;
};

class LocalFileStorage : public FileStorage {
 public:
  LocalFileStorage(int tensorBytes, int cacheCapacity, int layer, int batchSize,
                   int splitNumber, std::string rootPath,
                   int64_t clientGCInterval, int64_t ttl, bool enableGlobalGC,
                   int64_t globalGCInterval, int64_t globalTTL) {
    this->hashAlgorithm = std::make_shared<MurmurHash3Algorithm>();
    this->hasher = std::make_shared<Hasher>(hashAlgorithm.get());
    this->tensorBytes = tensorBytes;
    this->cacheCapacity = cacheCapacity;
    this->layer = layer;
    this->batchSize = batchSize;
    this->splitNumber = splitNumber;
    this->rootPath = std::regex_replace(rootPath + "/", std::regex("/+"), "/");
    this->tempFileDir =
        std::regex_replace(rootPath + "/__temp/", std::regex("/+"), "/");
    this->gcInterval = std::chrono::seconds(clientGCInterval);
    this->fileTTL = std::chrono::seconds(ttl);
    this->globalGCInterval = std::chrono::seconds(globalGCInterval);
    this->globalFileTTL = std::chrono::seconds(globalTTL);
    this->enableGlobalGC = enableGlobalGC;
  }

  ~LocalFileStorage() = default;

  Status Init() override {
    this->gcThread =
        std::thread(FileStorage::DefaultGCThread, shared_from_this());
    this->globalGCThread =
        std::thread(FileStorage::GlobalGCThread, shared_from_this());
    return Status::OK();
  }

  std::shared_ptr<FileDescriptor> CreateFileDescriptor() override;

  Status Open(std::string path, std::shared_ptr<FileDescriptor>& fd,
              FileOperationType fileOperationType) override;

  Status Seek(std::shared_ptr<FileDescriptor>& fd, size_t offset) override;

  Status Read(std::shared_ptr<FileDescriptor>& fd, void* data,
              size_t size) override;

  Status Write(std::shared_ptr<FileDescriptor>& fd, const void* data,
               size_t size) override;

  Status Mkdir(std::string path) override;

  Status GetFileSize(std::shared_ptr<FileDescriptor>& fd,
                     size_t& size) override;

  Status GetCurrentPos(std::shared_ptr<FileDescriptor>& fd,
                       size_t& pos) override;

  Status MoveFileAtomic(std::string src, std::string dst) override;

  bool IsFileExist(const std::string& path) override;

  Status Flush(std::shared_ptr<FileDescriptor>& fd) override;

  Status Close(std::shared_ptr<FileDescriptor>& fd) override;

  Status Delete(std::string path) override;

  Status GetFileList(std::string dirPath,
                     std::vector<std::string>& fileList) override;

  Status GetFileAccessTime(
      const std::string& path,
      std::chrono::duration<int64_t, std::nano>& accessTime) override;

  Status TouchFile(const std::string& path) override;

  std::string GetTmpFileDir() override;

  std::list<std::string>& GetGCList() { return this->gcList; }
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_STORAGE_LOCAL_FILE_STORAGE_H_
