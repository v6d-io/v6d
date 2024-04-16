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

#ifndef MODULES_LLM_CACHE_STORAGE_FILE_STORAGE_H_
#define MODULES_LLM_CACHE_STORAGE_FILE_STORAGE_H_

#include <chrono>
#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/util/status.h"
#include "llm-cache/hash/hasher.h"
#include "llm-cache/storage/storage.h"

#define SECOND_TO_MILLISECOND 1000
#define SECOND_TO_MICROSECOND 1000000
#define SECOND_TO_NANOSECOND 1000000000

namespace vineyard {

struct FileDescriptor {};

enum FilesystemType {
  LOCAL,
};

enum FileOperationType {
  READ = 1,
  WRITE = 1 << 1,
};

class FileStorage : public IStorage,
                    public std::enable_shared_from_this<FileStorage> {
 private:
  bool CompareTokenList(const std::vector<int>& tokenList,
                        const std::vector<int>& tokenList2, size_t length);

  virtual std::shared_ptr<FileDescriptor> CreateFileDescriptor() = 0;

  virtual Status Open(std::string path, std::shared_ptr<FileDescriptor>& fd,
                      FileOperationType fileOperationType) = 0;

  virtual Status Seek(std::shared_ptr<FileDescriptor>& fd, size_t offset) = 0;

  virtual Status Read(std::shared_ptr<FileDescriptor>& fd, void* data,
                      size_t size) = 0;

  virtual Status Write(std::shared_ptr<FileDescriptor>& fd, const void* data,
                       size_t size) = 0;

  virtual Status Mkdir(std::string path) = 0;

  virtual Status GetFileSize(std::shared_ptr<FileDescriptor>& fd,
                             size_t& size) = 0;

  virtual Status GetCurrentPos(std::shared_ptr<FileDescriptor>& fd,
                               size_t& pos) = 0;

  virtual Status MoveFileAtomic(std::string src, std::string dst) = 0;

  virtual Status Flush(std::shared_ptr<FileDescriptor>& fd) = 0;

  virtual Status Close(std::shared_ptr<FileDescriptor>& fd) = 0;

  virtual Status Delete(std::string path) = 0;

  virtual bool IsFileExist(const std::string& path) = 0;

  virtual Status GetFileAccessTime(
      const std::string& path,
      std::chrono::duration<int64_t, std::nano>& accessTime) = 0;

  virtual Status TouchFile(const std::string& path) = 0;

  virtual std::string GetTmpFileDir() = 0;

  Status DefaultGCFunc();

  Status GlobalGCFunc();

  virtual Status GetFileList(std::string dirPath,
                             std::vector<std::string>& fileList) = 0;

 protected:
  static void DefaultGCThread(std::shared_ptr<FileStorage> fileStorage);

  static void GlobalGCThread(std::shared_ptr<FileStorage> fileStorage);

  // for test
  void PrintFileAccessTime(std::string path);

  static std::string GetTimestamp(
      std::chrono::duration<int64_t, std::nano> time);

 public:
  FileStorage() = default;

  ~FileStorage() = default;

  Status Update(
      const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
      size_t& updated) override;

  Status Update(const std::vector<int>& tokenList, int nextToken,
                const std::vector<std::pair<LLMKV, LLMKV>>& kvState) override;

  Status Update(
      const std::vector<int>& prefix, const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
      size_t& updated) override;

  Status Query(const std::vector<int>& tokenList,
               std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
               size_t& matched) override;

  Status Query(const std::vector<int>& tokenList, int nextToken,
               std::vector<std::pair<LLMKV, LLMKV>>& kvState) override;

  void CloseCache() override;

  virtual Status Init() = 0;

  void StopGlobalGCThread() override { this->enableGlobalGC = false; }

  void StartGlobalGCThread() override { this->enableGlobalGC = true; }

 protected:
  size_t tensorBytes;
  size_t cacheCapacity;
  int layer;
  int batchSize;
  int splitNumber;
  std::string rootPath;
  std::string tempFileDir;
  std::shared_ptr<IHashAlgorithm> hashAlgorithm;
  std::shared_ptr<Hasher> hasher;

  std::chrono::duration<int64_t> gcInterval;
  std::chrono::duration<int64_t> globalGCInterval;
  std::chrono::duration<int64_t> fileTTL;
  std::chrono::duration<int64_t> globalFileTTL;

  bool exitFlag = false;
  bool enableGlobalGC = false;
  std::condition_variable cv;
  std::list<std::string> gcList;
  std::mutex gcMutex;
  std::thread gcThread;
  std::thread globalGCThread;
};

}  // namespace vineyard
#endif  // MODULES_LLM_CACHE_STORAGE_FILE_STORAGE_H_
