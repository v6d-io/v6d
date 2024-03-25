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

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/util/logging.h"
#include "common/util/status.h"
#include "llm-cache/storage/file_storage.h"

namespace vineyard {

Status FileStorage::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  std::vector<std::string> pathList;
  std::string path;
  int tokenLength;

  RETURN_ON_ERROR(hasher->computePathForTokens(tokenList, batchSize,
                                               splitNumber, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }
  for (size_t i = 0; i < pathList.size(); i++) {
    tokenLength = (i + 1) * batchSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr = GetTmpFileDir(pathList[i]);
    std::filesystem::path tmpPath(tmpPathStr);
    std::string pathStr = this->rootPath + pathList[i];
    std::filesystem::path path(pathStr);

    RETURN_ON_ERROR(Mkdir(path.parent_path().string()));
    std::lock_guard<std::mutex> lock(fileStorageLock);

    if (Open(pathStr, fd, FileOperationType::READ).ok()) {
      int tokenLength;
      Read(fd, &tokenLength, sizeof(int));
      std::vector<int> tokens;
      tokens.resize(tokenLength);
      Read(fd, tokens.data(), tokenLength * sizeof(int));
      if (!CompareTokenList(tokenList, tokens, tokenLength)) {
        // Token list not match
        RETURN_ON_ERROR(Close(fd));
        return Status::OK();
      }
      // Skip this kv state
      RETURN_ON_ERROR(Close(fd));
      continue;
    }

    RETURN_ON_ERROR(Mkdir(tmpPath.parent_path().string()));
    if (!Open(tmpPathStr, fd, FileOperationType::WRITE).ok()) {
      return Status::OK();
    }

    // Currently we do not consider delete.

    RETURN_ON_ERROR(Write(fd, &tokenLength, sizeof(int)));
    RETURN_ON_ERROR(Write(fd, tokenList.data(), tokenLength * sizeof(int)));
    for (size_t currentTokenIndex = i * batchSize;
         currentTokenIndex < (i + 1) * batchSize; currentTokenIndex++) {
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        const void* k = kvStateList[currentTokenIndex]
                            .find(currentLayer)
                            ->second.first.data;
        const void* v = kvStateList[currentTokenIndex]
                            .find(currentLayer)
                            ->second.second.data;
        size_t length = kvStateList[currentTokenIndex]
                            .find(currentLayer)
                            ->second.first.length;
        RETURN_ON_ERROR(Write(fd, k, length));
        RETURN_ON_ERROR(Write(fd, v, length));
      }
    }

    RETURN_ON_ERROR(Close(fd));
    if (!MoveFileAtomic(tmpPathStr, pathStr).ok()) {
      // Move failed. There exists a file with the same name.
      Delete(tmpPathStr);
      return Status::OK();
    }
  }

  return Status::OK();
}

Status FileStorage::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  if (prefix.size() % batchSize != 0) {
    return Status::Invalid("Prefix size should be multiple of batch size!");
  }

  std::vector<std::string> pathList;
  std::string path;
  int tokenLength;
  std::vector<int> totalTokenList(prefix.begin(), prefix.end());
  totalTokenList.insert(totalTokenList.end(), tokenList.begin(),
                        tokenList.end());

  RETURN_ON_ERROR(hasher->computePathForTokens(totalTokenList, batchSize,
                                               splitNumber, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }
  size_t kvStateIndex = 0;
  for (size_t i = 0; i < pathList.size(); i++) {
    tokenLength = (i + 1) * batchSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr = GetTmpFileDir(pathList[i]);
    std::filesystem::path tmpPath(tmpPathStr);
    std::string pathStr = this->rootPath + pathList[i];
    std::filesystem::path path(pathStr);

    RETURN_ON_ERROR(Mkdir(path.parent_path().string()));
    std::lock_guard<std::mutex> lock(fileStorageLock);

    if (Open(pathStr, fd, FileOperationType::READ).ok()) {
      int tokenLength;
      Read(fd, &tokenLength, sizeof(int));
      std::vector<int> tokens;
      tokens.resize(tokenLength);
      Read(fd, tokens.data(), tokenLength * sizeof(int));
      if (!CompareTokenList(totalTokenList, tokens, tokenLength)) {
        // Token list not match
        RETURN_ON_ERROR(Close(fd));
        return Status::OK();
      }
      // Skip this kv state
      RETURN_ON_ERROR(Close(fd));
      continue;
    }

    RETURN_ON_ERROR(Mkdir(tmpPath.parent_path().string()));
    if (!Open(tmpPathStr, fd, FileOperationType::WRITE).ok()) {
      return Status::OK();
    }
    if (i * batchSize != kvStateIndex + prefix.size()) {
      /**
       * This can happen if someone else deletes a file that matches
       * the token halfway.
       */
      return Status::OK();
    }

    // Currently we do not consider delete.

    RETURN_ON_ERROR(Write(fd, &tokenLength, sizeof(int)));
    RETURN_ON_ERROR(
        Write(fd, totalTokenList.data(), tokenLength * sizeof(int)));
    for (size_t currentTokenIndex = i * batchSize;
         currentTokenIndex < (i + 1) * batchSize; currentTokenIndex++) {
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        const void* k =
            kvStateList[kvStateIndex].find(currentLayer)->second.first.data;
        const void* v =
            kvStateList[kvStateIndex].find(currentLayer)->second.second.data;
        size_t length =
            kvStateList[kvStateIndex].find(currentLayer)->second.first.length;
        RETURN_ON_ERROR(Write(fd, k, length));
        RETURN_ON_ERROR(Write(fd, v, length));
      }
    }
    kvStateIndex += batchSize;

    RETURN_ON_ERROR(Close(fd));
    if (!MoveFileAtomic(tmpPathStr, pathStr).ok()) {
      // Move failed. There exists a file with the same name.
      Delete(tmpPathStr);
      return Status::OK();
    }
  }

  return Status::OK();
}

Status FileStorage::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  // TBD
  return Status::NotImplemented();
}

Status FileStorage::Query(
    const std::vector<int>& tokenList,
    std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  std::vector<std::string> paths;
  std::string dir = rootPath;
  RETURN_ON_ERROR(
      hasher->computePathForTokens(tokenList, batchSize, splitNumber, paths));

  for (size_t i = 0; i < paths.size(); i++) {
    std::filesystem::path filePath(dir + paths[i]);
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();

    std::lock_guard<std::mutex> lock(fileStorageLock);
    // If open failed, it means the kv state is not in the cache(file not exist)
    if (!Open(filePath.string(), fd, FileOperationType::READ).ok()) {
      return Status::OK();
    }

    int tokenLength;
    Read(fd, &tokenLength, sizeof(int));
    std::vector<int> prefix;
    prefix.resize(tokenLength);
    Read(fd, prefix.data(), tokenLength * sizeof(int));

    if (!CompareTokenList(tokenList, prefix, prefix.size())) {
      VLOG(100) << "token list not match";
      RETURN_ON_ERROR(Close(fd));
      return Status::OK();
    } else {
      VLOG(100) << "token list match";
      for (int j = 0; j < batchSize; j++) {
        std::map<int, std::pair<LLMKV, LLMKV>> kvState;
        for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
          LLMKV k, v;
          k.length = v.length = tensorBytes;
          k.data = new uint8_t[k.length];
          v.data = new uint8_t[v.length];
          Read(fd, k.data, k.length);
          Read(fd, v.data, v.length);
          kvState.insert(std::make_pair(currentLayer, std::make_pair(k, v)));
        }
        kvStateList.push_back(kvState);
      }
    }

    RETURN_ON_ERROR(Close(fd));
  }

  return Status::OK();
}

Status FileStorage::Query(const std::vector<int>& tokenList, int nextToken,
                          std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  // TBD
  return Status::NotImplemented();
}

bool FileStorage::CompareTokenList(const std::vector<int>& tokenList,
                                   const std::vector<int>& tokenList2,
                                   size_t length) {
  if (tokenList.size() < length || tokenList2.size() < length) {
    return false;
  }
  for (size_t i = 0; i < length; i++) {
    if (tokenList[i] != tokenList2[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace vineyard
