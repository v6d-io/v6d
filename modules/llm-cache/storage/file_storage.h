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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/util/status.h"
#include "llm-cache/hash/hasher.h"
#include "llm-cache/storage/storage.h"

namespace vineyard {

struct FileDescriptor {};

struct FileHeader {
  int prefixNum;
  int layer;
  int kvStateSize;
};

enum FilesystemType {
  LOCAL,
};

enum FileOperationType {
  READ = 1,
  WRITE = 1 << 1,
};

static std::mutex fileStorageLock;

class FileStorage : public IStorage {
 private:
  bool CompareTokenList(const std::vector<int>& tokenList,
                        const std::vector<int>& tokenList2, size_t length);

  void CloseCache() override {}

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

  virtual std::string GetTmpFileDir(std::string filePath) = 0;

 public:
  FileStorage() = default;

  ~FileStorage() = default;

  Status Update(const std::vector<int>& tokenList,
                const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>&
                    kvStateList) override;

  Status Update(const std::vector<int>& tokenList, int nextToken,
                const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) override;

  Status Update(
      const std::vector<int>& prefix, const std::vector<int>& tokenList,
      const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList);

  Status Query(const std::vector<int>& tokenList,
               std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList)
      override;

  Status Query(const std::vector<int>& tokenList, int nextToken,
               std::map<int, std::pair<LLMKV, LLMKV>>& kvState) override;

 protected:
  int tensorBytes;
  int cacheCapacity;
  int layer;
  int batchSize;
  int splitNumber;
  std::string rootPath;
  std::string tempFileDir;
  std::shared_ptr<IHashAlgorithm> hashAlgorithm;
  std::shared_ptr<Hasher> hasher;
};

}  // namespace vineyard
#endif  // MODULES_LLM_CACHE_STORAGE_FILE_STORAGE_H_
