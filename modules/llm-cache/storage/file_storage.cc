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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/util/logging.h"
#include "common/util/status.h"
#include "gulrak/filesystem.hpp"
#include "llm-cache/storage/file_storage.h"
#include "llm-cache/thread_group.h"

namespace vineyard {

/**
 * @brief Update the kv state with the given token list in the file storage.
 *
 * @param tokenList The token list to be updated.
 * @param kvCacheList The kv state list of the token list.
 *                    It's a 2D vector, the first dimension is the token index,
 *                    and the second dimension is the layer index.
 *                    The kv state is a pair of LLMKV, the first is the K tensor
 *                    and the second is the V tensor. It contains two fields:
 *                    data and length. The data is the pointer to the tensor
 *                    , and the length is the size of the tensor.
 * @param updated It's a return value to indicate the number of tokens that have
 *                been updated successfully.
 *
 *           *****************************************************************
 *           * Important, the kv state List must be                          *
 *           * initialized(pre-allocated) and released by the caller.        *
 *           *                                                               *
 *           * Assume the layer is 2, and the token list is [1,2] you should *
 *           * allocate the memory for the kv state like this:               *
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvCacheList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorNBytes);                     *
 *           *     value_state.data = malloc(tensorNBytes)                    *
 *           *     // Copy the k_state of LLM KV Cache to key_state.data     *
 *           *     // Copy the v_state of LLM KV Cache to value_state.data   *
 *           *     key_state.length = tensorNBytes;                           *
 *           *     value_state.length = tensorNBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvCacheList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, you must release(free) the       *
 *           * kv buffer of the kvCacheList manually                         *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The length of the token list should be as same as the length of the
 * kvCacheList.
 *
 *
 * @example Suppose the token list is [1, 2, 3, 4], the layer is 2,
 *          then the kvCacheList should be a 2D vector with size 4 * 2.
 *
 * @return Status
 */
Status FileStorage::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& updated) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }
  if (tokenList.size() % chunkSize != 0) {
    return Status::Invalid("Tokens size " + std::to_string(tokenList.size()) +
                           " should be multiple of batch size " +
                           std::to_string(chunkSize) + "!");
  }
  if (tokenList.size() > MAX_CACHE_TOKEN_LENGTH) {
    LOG(WARNING)
        << "The token list size is larger than the maximum cache token "
           "length. This token list will be ignored!";
    return Status::OK();
  }

  std::vector<std::string> pathList;
  std::set<std::string> createFileSet;
  std::mutex createFileSetMutex;
  RETURN_ON_ERROR(hasher->computePathForTokens(tokenList, chunkSize,
                                               hashChunkSize, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }

  std::vector<std::string> tempFilePaths(pathList.size());
  auto fn = [this, &tempFilePaths, &pathList, &tokenList, &kvCacheList,
             &createFileSet, &createFileSetMutex](int i) -> Status {
    int tokenLength = (i + 1) * chunkSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr = GetTmpFileDir("-" + std::to_string(i));
    tempFilePaths[i] = tmpPathStr;
    std::string pathStr = this->rootPath + pathList[i];
    ghc::filesystem::path path(pathStr);

    RETURN_ON_ERROR(Mkdir(path.parent_path().string()));

    if (Open(pathStr, fd, FileOperationType::READ).ok()) {
      int tokenLengthInFile;
      RETURN_ON_ERROR(Read(fd, &tokenLengthInFile, sizeof(int)));
      std::vector<int> tokens;
      tokens.resize(tokenLengthInFile);
      RETURN_ON_ERROR(Read(fd, tokens.data(), tokenLengthInFile * sizeof(int)));
      if (!CompareTokenList(tokenList, tokens, tokenLengthInFile)) {
        // Token list not match
        VINEYARD_DISCARD(Close(fd));
        return Status::ObjectExists("File exists for another token sequence");
      }
      // Skip this kv state
      VINEYARD_DISCARD(Close(fd));
      return Status::OK();
    }

    RETURN_ON_ERROR(Mkdir(ghc::filesystem::path(tmpPathStr + pathList[i])
                              .parent_path()
                              .string()));
    auto status = Open(tmpPathStr + pathList[i], fd, FileOperationType::WRITE);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to create temporary cache entry: "
                   << status.ToString();
      return Status::Wrap(status, "Failed to create temporary cache entry");
    }

    // Currently we do not consider delete.
    RETURN_ON_ERROR(Write(fd, &tokenLength, sizeof(int)));
    RETURN_ON_ERROR(Write(fd, tokenList.data(), tokenLength * sizeof(int)));
    for (int currentTokenIndex = i * chunkSize;
         currentTokenIndex < (i + 1) * chunkSize; currentTokenIndex++) {
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        const LLMKV& k = kvCacheList[currentTokenIndex][currentLayer].first;
        const LLMKV& v = kvCacheList[currentTokenIndex][currentLayer].second;
        RETURN_ON_ERROR(Write(fd, k.data, k.length));
        RETURN_ON_ERROR(Write(fd, v.data, k.length));
      }
    }

    VINEYARD_DISCARD(Flush(fd));
    VINEYARD_DISCARD(Close(fd));
    status = MoveFileAtomic(tmpPathStr + pathList[i], pathStr);
    if (!status.ok()) {
      // Move failed. There exists a file with the same name.
      LOG(WARNING) << "Failed to move cache entry: " << status.ToString();
      VINEYARD_SUPPRESS(Delete(tmpPathStr + pathList[i]));
      return Status::Wrap(status, "Failed to move cache entry");
    }
    std::lock_guard<std::mutex> lock(createFileSetMutex);
    createFileSet.insert(pathStr);
    return Status::OK();
  };

  parallel::ThreadGroup tg(
      std::min(pathList.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids(pathList.size());
  for (size_t i = 0; i < pathList.size(); ++i) {
    tids[i] = tg.AddTask(fn, i);
  }
  std::vector<Status> taskResults(pathList.size(), Status::OK());
  for (size_t i = 0; i < pathList.size(); ++i) {
    taskResults[i] = tg.TaskResult(tids[i]);
  }

  size_t upper_bound = 0;
  {
    for (size_t i = 0; i < pathList.size(); i++) {
      if (taskResults[i].ok()) {
        upper_bound += 1;
        if (createFileSet.find(this->rootPath + pathList[i]) !=
            createFileSet.end()) {
          VINEYARD_DISCARD(TouchFile(this->rootPath + pathList[i]));
        }
      } else {
        break;
      }
    }
  }
  updated = upper_bound * chunkSize;
  for (size_t i = upper_bound; i < pathList.size(); i++) {
    VINEYARD_SUPPRESS(Delete(this->rootPath + pathList[i]));
    VINEYARD_SUPPRESS(Delete(tempFilePaths[i]));
  }

  {
    std::lock_guard<std::mutex> lock(gcMutex);
    for (size_t i = 0; i < upper_bound; i++) {
      gcList.push_back(this->rootPath + pathList[i]);
    }
  }
  return Status::OK();
}

/**
 * @brief Update the kv state with the given prefix and token list in the file
 * storage.
 *
 * @param prefix The prefix token list. It should be a multiple of the batch
 * size.
 * @param tokenList The token list to be updated.
 * @param kvCacheList The kv state list of the token list.
 *                    It's a 2D vector, the first dimension is the token index,
 *                    and the second dimension is the layer index.
 *                    The kv state is a pair of LLMKV, the first is the K tensor
 *                    and the second is the V tensor. It contains two fields:
 *                    data and length. The data is the pointer to the tensor
 *                    , and the length is the size of the tensor.
 * @param updated It's a return value to indicate the number of tokens that have
 *                been updated successfully.
 *
 *           *****************************************************************
 *           * Important, the kv state List must be                          *
 *           * initialized(pre-allocated) and released by the caller.        *
 *           *                                                               *
 *           * Assume the layer is 2, and the token list is [1,2] you should *
 *           * allocate the memory for the kv state like this:               *
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvCacheList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorNBytes);                     *
 *           *     value_state.data = malloc(tensorNBytes)                    *
 *           *     // Copy the k_state of LLM KV Cache to key_state.data     *
 *           *     // Copy the v_state of LLM KV Cache to value_state.data   *
 *           *     key_state.length = tensorNBytes;                           *
 *           *     value_state.length = tensorNBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvCacheList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, you must release(free) the       *
 *           * kv buffer of the kvCacheList manually                         *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The length of the token list should be as same as the length of the
 * kvCacheList.
 *
 * @example Suppose the prefix is [1, 2], the token list is [3, 4], the layer is
 * 2, then the kvCacheList should be a 2D vector with size 2 * 2.
 *
 * @return Status
 */
Status FileStorage::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& updated) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }
  if (prefix.size() % chunkSize != 0) {
    return Status::Invalid("Prefix size " + std::to_string(prefix.size()) +
                           " should be multiple of batch size " +
                           std::to_string(chunkSize) + "!");
  }
  if (tokenList.size() % chunkSize != 0) {
    return Status::Invalid("Tokens size " + std::to_string(tokenList.size()) +
                           " should be multiple of batch size " +
                           std::to_string(chunkSize) + "!");
  }
  if (tokenList.size() > MAX_CACHE_TOKEN_LENGTH) {
    LOG(WARNING)
        << "The token list size is larger than the maximum cache token "
           "length. This token list will be ignored!";
    return Status::OK();
  }

  std::vector<std::string> pathList;
  std::set<std::string> createFileSet;
  std::mutex createFileSetMutex;
  std::vector<int> totalTokenList(prefix.begin(), prefix.end());
  totalTokenList.insert(totalTokenList.end(), tokenList.begin(),
                        tokenList.end());

  RETURN_ON_ERROR(hasher->computePathForTokens(totalTokenList, chunkSize,
                                               hashChunkSize, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }

  std::vector<std::string> tempFilePaths(pathList.size());
  auto fn = [this, &tempFilePaths, &pathList, &prefix, &totalTokenList,
             &kvCacheList, &createFileSet,
             &createFileSetMutex](size_t i) -> Status {
    int tokenLength = (i + 1) * chunkSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr = GetTmpFileDir("-" + std::to_string(i));
    tempFilePaths[i] = tmpPathStr;
    std::string pathStr = this->rootPath + pathList[i];
    ghc::filesystem::path path(pathStr);

    RETURN_ON_ERROR(Mkdir(path.parent_path().string()));

    if (Open(pathStr, fd, FileOperationType::READ).ok()) {
      int tokenLength;
      RETURN_ON_ERROR(Read(fd, &tokenLength, sizeof(int)));
      std::vector<int> tokens(tokenLength, -1);
      RETURN_ON_ERROR(Read(fd, tokens.data(), tokenLength * sizeof(int)));
      if (!CompareTokenList(totalTokenList, tokens, tokenLength)) {
        // Token list not match
        VINEYARD_DISCARD(Close(fd));
        return Status::ObjectExists("File exists for another token sequence");
      }
      // Skip this kv state
      VINEYARD_DISCARD(Close(fd));
      return Status::OK();
    }

    if ((i + 1) * chunkSize <= prefix.size()) {
      return Status::ObjectNotExists("The prefix is not in the file cache");
    }

    RETURN_ON_ERROR(Mkdir(ghc::filesystem::path(tmpPathStr + pathList[i])
                              .parent_path()
                              .string()));
    auto status = Open(tmpPathStr + pathList[i], fd, FileOperationType::WRITE);
    if (!status.ok()) {
      return Status::Wrap(status, "Failed to create temporary cache entry");
    }

    // Currently we do not consider delete.

    RETURN_ON_ERROR(Write(fd, &tokenLength, sizeof(int)));
    RETURN_ON_ERROR(
        Write(fd, totalTokenList.data(), tokenLength * sizeof(int)));
    size_t kvStatePos =
        (i * chunkSize) < prefix.size() ? 0 : (i * chunkSize) - prefix.size();
    for (size_t currentTokenIndex = kvStatePos;
         currentTokenIndex < kvStatePos + chunkSize; currentTokenIndex++) {
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        const LLMKV& k = kvCacheList[currentTokenIndex][currentLayer].first;
        const LLMKV& v = kvCacheList[currentTokenIndex][currentLayer].second;
        RETURN_ON_ERROR(Write(fd, k.data, k.length));
        RETURN_ON_ERROR(Write(fd, v.data, k.length));
      }
    }

    VINEYARD_DISCARD(Flush(fd));
    VINEYARD_DISCARD(Close(fd));
    if (!MoveFileAtomic(tmpPathStr + pathList[i], pathStr).ok()) {
      // Move failed. There exists a file with the same name.
      VINEYARD_SUPPRESS(Delete(tmpPathStr + pathList[i]));
      return Status::Wrap(status, "Failed to move cache entry");
    }
    std::lock_guard<std::mutex> lock(createFileSetMutex);
    createFileSet.insert(pathStr);
    return Status::OK();
  };

  parallel::ThreadGroup tg(
      std::min(pathList.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids(pathList.size());
  for (size_t i = 0; i < pathList.size(); ++i) {
    tids[i] = tg.AddTask(fn, i);
  }
  std::vector<Status> taskResults(pathList.size(), Status::OK());
  for (size_t i = 0; i < pathList.size(); ++i) {
    taskResults[i] = tg.TaskResult(tids[i]);
  }

  size_t upper_bound = 0;
  {
    for (size_t i = 0; i < pathList.size(); i++) {
      if (taskResults[i].ok()) {
        upper_bound += 1;
        if (upper_bound * chunkSize > prefix.size() &&
            createFileSet.find(this->rootPath + pathList[i]) !=
                createFileSet.end()) {
          // Only this part is created.
          VINEYARD_DISCARD(TouchFile(this->rootPath + pathList[i]));
        }
      } else {
        break;
      }
    }
  }
  updated = upper_bound * chunkSize <= prefix.size()
                ? 0
                : upper_bound * chunkSize - prefix.size();
  for (size_t i = upper_bound; i < pathList.size(); i++) {
    VINEYARD_SUPPRESS(Delete(this->rootPath + pathList[i]));
    VINEYARD_SUPPRESS(Delete(tempFilePaths[i]));
  }
  {
    std::lock_guard<std::mutex> lock(gcMutex);
    for (size_t i = 0; i < upper_bound; i++) {
      gcList.push_back(this->rootPath + pathList[i]);
    }
  }
  return Status::OK();
}

Status FileStorage::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  // TBD
  return Status::NotImplemented();
}

Status FileStorage::BatchedUpdate(
    const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& updated) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }
  if (tokenList.size() % chunkSize != 0) {
    return Status::Invalid("Tokens size " + std::to_string(tokenList.size()) +
                           " should be multiple of batch size " +
                           std::to_string(chunkSize) + "!");
  }
  if (tokenList.size() > MAX_CACHE_TOKEN_LENGTH) {
    LOG(WARNING)
        << "The token list size is larger than the maximum cache token "
           "length. This token list will be ignored!";
    return Status::OK();
  }

  std::vector<std::string> pathList;
  std::set<std::string> createFileSet;
  std::mutex createFileSetMutex;
  RETURN_ON_ERROR(hasher->computePathForTokens(tokenList, chunkSize,
                                               hashChunkSize, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }

  std::vector<std::shared_ptr<FileDescriptor>> read_fd_list;
  RETURN_ON_ERROR(BatchedOpen(pathList, read_fd_list, FileOperationType::READ));

  auto read_fn = [this, &read_fd_list, &tokenList](int i) -> Status {
    int tokenLength = (i + 1) * chunkSize;
    RETURN_ON_ERROR(Read(read_fd_list[i], &tokenLength, sizeof(int)));
    std::vector<int> tokens;
    tokens.resize(tokenLength);
    RETURN_ON_ERROR(
        Read(read_fd_list[i], tokens.data(), tokenLength * sizeof(int)));
    if (!CompareTokenList(tokenList, tokens, tokenLength)) {
      // Token list not match
      VINEYARD_DISCARD(Close(read_fd_list[i]));
      return Status::ObjectExists("File exists for another token sequence");
    }
    // Skip this kv state
    VINEYARD_DISCARD(Close(read_fd_list[i]));
    return Status::OK();
  };

  int lower_bound = 0;
  if (read_fd_list.size() > 0) {
    parallel::ThreadGroup tg(
        std::min(read_fd_list.size(),
                 static_cast<size_t>(std::thread::hardware_concurrency())));
    std::vector<parallel::ThreadGroup::tid_t> tids(read_fd_list.size());
    for (size_t i = 0; i < read_fd_list.size(); ++i) {
      tids[i] = tg.AddTask(read_fn, i);
    }
    std::vector<Status> taskResults(read_fd_list.size(), Status::OK());
    for (size_t i = 0; i < read_fd_list.size(); ++i) {
      taskResults[i] = tg.TaskResult(tids[i]);
    }

    for (size_t i = 0; i < taskResults.size(); i++) {
      if (taskResults[i].ok()) {
        lower_bound += 1;
      } else {
        // File exists for another token sequence
        break;
      }
    }
  }

  VINEYARD_DISCARD(BatchedClose(read_fd_list));

  std::vector<std::shared_ptr<FileDescriptor>> write_fd_list;
  std::vector<std::string> left_path(pathList.begin() + lower_bound,
                                     pathList.end());
  RETURN_ON_ERROR(
      BatchedOpen(left_path, write_fd_list, FileOperationType::WRITE));
  auto fn = [this, &write_fd_list, &tokenList, &kvCacheList,
             lower_bound](int i) -> Status {
    int tokenLength = (i + 1 + lower_bound) * chunkSize;

    RETURN_ON_ERROR(Write(write_fd_list[i], &tokenLength, sizeof(int)));
    RETURN_ON_ERROR(
        Write(write_fd_list[i], tokenList.data(), tokenLength * sizeof(int)));
    for (int currentTokenIndex = (i + lower_bound) * chunkSize;
         currentTokenIndex < (i + lower_bound + 1) * chunkSize;
         currentTokenIndex++) {
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        const LLMKV& k = kvCacheList[currentTokenIndex][currentLayer].first;
        const LLMKV& v = kvCacheList[currentTokenIndex][currentLayer].second;
        RETURN_ON_ERROR(Write(write_fd_list[i], k.data, k.length));
        RETURN_ON_ERROR(Write(write_fd_list[i], v.data, k.length));
      }
    }

    VINEYARD_DISCARD(Flush(write_fd_list[i]));
    return Status::OK();
  };

  if (write_fd_list.size() > 0) {
    parallel::ThreadGroup tg_write(
        std::min(write_fd_list.size(),
                 static_cast<size_t>(std::thread::hardware_concurrency())));
    std::vector<parallel::ThreadGroup::tid_t> tids_write(write_fd_list.size());
    for (size_t i = 0; i < write_fd_list.size(); ++i) {
      tids_write[i] = tg_write.AddTask(fn, i);
    }
    std::vector<Status> taskResults_write(write_fd_list.size(), Status::OK());
    for (size_t i = 0; i < write_fd_list.size(); ++i) {
      taskResults_write[i] = tg_write.TaskResult(tids_write[i]);
    }

    size_t upper_bound = 0;
    for (size_t i = 0; i < write_fd_list.size(); i++) {
      if (taskResults_write[i].ok()) {
        upper_bound += 1;
      } else {
        break;
      }
    }

    for (size_t i = upper_bound; i < write_fd_list.size(); i++) {
      VINEYARD_SUPPRESS(Delete(this->rootPath + pathList[i + lower_bound]));
    }
    updated = upper_bound * chunkSize;

    RETURN_ON_ERROR(BatchedClose(write_fd_list));
  }
  return Status::OK();
}

/**
 * @brief Query the kv state with the given token list in the file storage.
 *
 * @param tokenList The token list to be queried.
 * @param kvCacheList The kv state list of the token list to be fulfilled.
 *                    It's a 2D vector, the first dimension is the token index,
 *                    and the second dimension is the layer index. The kv state
 *                    is a pair of LLMKV, the first is the K tensor and the
 *                    second is the V tensor. It contains two fields: data and
 *                    length. The data is the pointer to the tensor, and
 *                    the length is the size of the tensor data.
 * @param matched It's a return value to indicate the number of tokens that have
 *                been matched successfully.
 *
 *           *****************************************************************
 *           * Important, the kv state is managed by the caller itself, the  *
 *           * caller need to malloc and free the memory of the kv state.    *
 *           *                                                               *
 *           * Assume the layer is 2, and the token list is [1,2] you should *
 *           * allocate the memory for the kv state like this:               *
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvCacheList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorNBytes);                     *
 *           *     value_state.data = malloc(tensorNBytes)                    *
 *           *     key_state.length = tensorNBytes;                           *
 *           *     value_state.length = tensorNBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvCacheList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, the key_state and value_state    *
 *           * will be fulfilled with the kv state of the token, then you    *
 *           * can copy the kv state to the LLM KV Cache. At last, you       *
 *           * must free the memory of the kv state manually.                *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The kvCacheList must be initialized before calling this function,
 * including the data and length of the kv tensor.
 *
 * @return Status
 */
Status FileStorage::Query(
    const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& matched) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }

  std::vector<std::string> paths;
  std::string dir = rootPath;
  RETURN_ON_ERROR(
      hasher->computePathForTokens(tokenList, chunkSize, hashChunkSize, paths));

  auto fn = [&](size_t i, size_t matched_start) -> Status {
    ghc::filesystem::path filePath(dir + paths[i]);
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();

    // If open failed, it means the kv state is not in the cache(file not exist)
    if (!Open(filePath.string(), fd, FileOperationType::READ).ok()) {
      return Status::ObjectNotExists("file doesn't exist");
    }
    size_t file_size = 0;
    auto s = GetFileSize(fd, file_size);
    if (!s.ok()) {
      VINEYARD_DISCARD(Close(fd));
      return Status::ObjectNotExists("cannot get file size");
    }
    if (file_size == 0) {
      VINEYARD_DISCARD(Close(fd));
      VINEYARD_DISCARD(Delete(filePath.string()));
      return Status::ObjectNotExists("file is empty");
    }

    int tokenLength = 0;
    RETURN_ON_ERROR(Read(fd, &tokenLength, sizeof(int)));
    std::vector<int> blockTokenList(tokenLength, 0);
    RETURN_ON_ERROR(Read(fd, blockTokenList.data(), tokenLength * sizeof(int)));

    if (!CompareTokenList(tokenList, blockTokenList, tokenLength)) {
      VINEYARD_DISCARD(Close(fd));
      return Status::ObjectNotExists("token mismatch");
    }
    for (int j = 0; j < chunkSize; j++) {
      if (matched_start + j >= tokenList.size() ||
          matched_start + j >= kvCacheList.size()) {
        break;
      }
      auto& kvState = kvCacheList[matched_start + j];
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        RETURN_ON_ASSERT(static_cast<int>(kvState.size()) == layer,
                         "The size of kvState is not equal to layer");
        LLMKV& k = kvState[currentLayer].first;
        LLMKV& v = kvState[currentLayer].second;
        RETURN_ON_ASSERT(
            k.length == tensorNBytes && v.length == tensorNBytes,
            "The size of kv tensor doesn't match with the tensorNBytes");
        RETURN_ON_ERROR(Read(fd, k.data, k.length));
        RETURN_ON_ERROR(Read(fd, v.data, v.length));
      }
    }
    VINEYARD_DISCARD(Close(fd));
    return Status::OK();
  };

  parallel::ThreadGroup tg(std::min(
      paths.size(), static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    tids[i] = tg.AddTask(fn, i, i * chunkSize);
  }
  std::vector<Status> taskResults(paths.size(), Status::OK());
  for (size_t i = 0; i < paths.size(); ++i) {
    taskResults[i] = tg.TaskResult(tids[i]);
  }

  matched = 0;
  for (size_t i = 0; i < paths.size(); i++) {
    if (taskResults[i].ok()) {
      matched += chunkSize;
    } else {
      break;
    }
  }
  return Status::OK();
}

Status FileStorage::Query(const std::vector<int>& prefix, int nextToken,
                          std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  // TBD
  return Status::NotImplemented();
}

Status FileStorage::Query(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& matched) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }
  if (prefix.size() % chunkSize != 0) {
    return Status::Invalid("Prefix size " + std::to_string(prefix.size()) +
                           " should be multiple of batch size " +
                           std::to_string(chunkSize) + "!");
  }

  size_t numPrefixChunks = prefix.size() / chunkSize;
  std::vector<int> totalTokenList(prefix.begin(), prefix.end());
  totalTokenList.insert(totalTokenList.end(), tokenList.begin(),
                        tokenList.end());

  std::vector<std::string> paths;
  std::string dir = rootPath;
  RETURN_ON_ERROR(hasher->computePathForTokens(totalTokenList, chunkSize,
                                               hashChunkSize, paths));

  auto fn = [&](size_t i, size_t matched_start) -> Status {
    ghc::filesystem::path filePath(dir + paths[i]);
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();

    // If open failed, it means the kv state is not in the cache(file not exist)
    if (!Open(filePath.string(), fd, FileOperationType::READ).ok()) {
      return Status::ObjectNotExists("Failed to open file '" +
                                     filePath.string() + "'");
    }
    size_t file_size = 0;
    auto s = GetFileSize(fd, file_size);
    if (!s.ok()) {
      VINEYARD_DISCARD(Close(fd));
      return Status::ObjectNotExists("Cannot get file size");
    }
    if (file_size == 0) {
      VINEYARD_DISCARD(Close(fd));
      VINEYARD_DISCARD(Delete(filePath.string()));
      return Status::ObjectNotExists("The target file is empty");
    }

    int tokenLength = 0;
    RETURN_ON_ERROR(Read(fd, &tokenLength, sizeof(int)));
    std::vector<int> blockTokenList(tokenLength, -1);
    RETURN_ON_ERROR(Read(fd, blockTokenList.data(), tokenLength * sizeof(int)));

    if (!CompareTokenList(totalTokenList, blockTokenList, tokenLength)) {
      VINEYARD_DISCARD(Close(fd));
      return Status::ObjectNotExists("Token mismatch");
    }
    for (int j = 0; j < chunkSize; j++) {
      if (matched_start + j >= totalTokenList.size() ||
          matched_start + j >= kvCacheList.size()) {
        break;
      }
      auto& kvState = kvCacheList[matched_start + j];
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        RETURN_ON_ASSERT(static_cast<int>(kvState.size()) == layer,
                         "The size of kvState is not equal to layer");
        LLMKV& k = kvState[currentLayer].first;
        LLMKV& v = kvState[currentLayer].second;
        RETURN_ON_ASSERT(
            k.length == tensorNBytes && v.length == tensorNBytes,
            "The size of kv tensor doesn't match with the tensorNBytes");
        RETURN_ON_ERROR(Read(fd, k.data, k.length));
        RETURN_ON_ERROR(Read(fd, v.data, v.length));
      }
    }

    VINEYARD_DISCARD(Close(fd));
    return Status::OK();
  };

  parallel::ThreadGroup tg(std::min(
      paths.size(), static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids(paths.size() -
                                                 numPrefixChunks);
  for (size_t i = numPrefixChunks; i < paths.size(); i++) {
    tids[i - numPrefixChunks] =
        tg.AddTask(fn, i, (i - numPrefixChunks) * chunkSize);
  }
  std::vector<Status> taskResults(paths.size() - numPrefixChunks, Status::OK());
  for (size_t i = numPrefixChunks; i < paths.size(); i++) {
    taskResults[i - numPrefixChunks] = tg.TaskResult(tids[i - numPrefixChunks]);
  }

  matched = 0;
  for (size_t i = numPrefixChunks; i < paths.size(); i++) {
    if (taskResults[i - numPrefixChunks].ok()) {
      matched += chunkSize;
    } else {
      break;
    }
  }
  return Status::OK();
}

Status FileStorage::BatchedQuery(
    const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& matched) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }

  std::vector<std::string> paths;
  RETURN_ON_ERROR(
      hasher->computePathForTokens(tokenList, chunkSize, hashChunkSize, paths));

  std::vector<std::shared_ptr<FileDescriptor>> read_fd_list;
  RETURN_ON_ERROR(BatchedOpen(paths, read_fd_list, FileOperationType::READ));
  auto read_fn = [this, &read_fd_list, &tokenList, &kvCacheList](
                     size_t i, size_t matched_start) -> Status {
    int tokenLength = 0;
    RETURN_ON_ERROR(Read(read_fd_list[i], &tokenLength, sizeof(int)));
    std::vector<int> blockTokenList(tokenLength, 0);
    RETURN_ON_ERROR(Read(read_fd_list[i], blockTokenList.data(),
                         tokenLength * sizeof(int)));

    if (!CompareTokenList(tokenList, blockTokenList, tokenLength)) {
      VINEYARD_DISCARD(Close(read_fd_list[i]));
      return Status::ObjectNotExists("Token mismatch");
    }

    for (int j = 0; j < chunkSize; j++) {
      if (matched_start + j >= tokenList.size() ||
          matched_start + j >= kvCacheList.size()) {
        break;
      }
      auto& kvState = kvCacheList[matched_start + j];
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        RETURN_ON_ASSERT(static_cast<int>(kvState.size()) == layer,
                         "The size of kvState is not equal to layer");
        LLMKV& k = kvState[currentLayer].first;
        LLMKV& v = kvState[currentLayer].second;
        RETURN_ON_ASSERT(
            k.length == tensorNBytes && v.length == tensorNBytes,
            "The size of kv tensor doesn't match with the tensorNBytes");
        RETURN_ON_ERROR(Read(read_fd_list[i], k.data, k.length));
        RETURN_ON_ERROR(Read(read_fd_list[i], v.data, v.length));
      }
    }
    VINEYARD_DISCARD(Close(read_fd_list[i]));
    return Status::OK();
  };

  parallel::ThreadGroup tg(
      std::min(read_fd_list.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids(read_fd_list.size());
  for (size_t i = 0; i < read_fd_list.size(); ++i) {
    tids[i] = tg.AddTask(read_fn, i, i * chunkSize);
  }
  std::vector<Status> taskResults(read_fd_list.size(), Status::OK());
  for (size_t i = 0; i < read_fd_list.size(); ++i) {
    taskResults[i] = tg.TaskResult(tids[i]);
  }

  matched = 0;
  for (size_t i = 0; i < read_fd_list.size(); i++) {
    if (taskResults[i].ok()) {
      matched += chunkSize;
    } else {
      break;
    }
  }
  return Status::OK();
}

bool FileStorage::CompareTokenList(const std::vector<int>& tokenList1,
                                   const std::vector<int>& tokenList2,
                                   size_t length) {
  if (tokenList1.size() < length || tokenList2.size() < length) {
    return false;
  }
  for (size_t i = 0; i < length; i++) {
    if (tokenList1[i] != tokenList2[i]) {
      return false;
    }
  }
  return true;
}

Status FileStorage::DefaultGCFunc() {
  auto now = std::chrono::high_resolution_clock::now();
  auto nanoseconds_since_epoch =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch());

  for (std::list<std::string>::iterator iter = gcList.begin();
       iter != gcList.end();) {
    std::string path = *iter;
    std::chrono::duration<int64_t, std::nano> accessTime(0);
    if (!GetFileAccessTime(path, accessTime).ok()) {
      if (IsFileExist(path)) {
        VLOG(100) << "Failed to get file access time: " << path;
        // skip this file, wait for next time.
        iter++;
      } else {
        VLOG(100) << "Default GC: " << path << " may be deleted by global GC!";
        // file may be deleted by global GC thread, remove it from gcList
        iter = gcList.erase(iter);
      }
    } else {
      VLOG(100) << "GC ttl:" << fileTTL.count();
      if ((accessTime + fileTTL).count() < nanoseconds_since_epoch.count()) {
        VLOG(100) << "GC: " << path << " is dead!";
        VLOG(100) << "Access time: " << GetTimestamp(accessTime);
        VLOG(100) << "Now: " << GetTimestamp(nanoseconds_since_epoch);
        RETURN_ON_ERROR(Delete(path));
        iter = gcList.erase(iter);
      } else {
        VLOG(100) << "GC: " << path << " is alive!";
        VLOG(100) << "Access time: " << GetTimestamp(accessTime);
        VLOG(100) << "Now: " << GetTimestamp(nanoseconds_since_epoch);
        iter++;
      }
    }
  }
  return Status::OK();
}

void FileStorage::DefaultGCThread(std::shared_ptr<FileStorage> fileStorage) {
  int64_t last_time =
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  while (1) {
    std::unique_lock<std::mutex> lock(fileStorage->gcMutex);
    if (fileStorage->cv.wait_for(
            lock, fileStorage->gcInterval, [&fileStorage, &last_time] {
              int64_t current_time =
                  std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::high_resolution_clock::now()
                          .time_since_epoch())
                      .count();
              return fileStorage->exitFlag ||
                     (current_time - last_time) >
                         fileStorage->gcInterval.count();
            })) {
      if (fileStorage->exitFlag) {
        VLOG(100) << "GC thread exit";
        return;
      }
      VLOG(100) << "GC thread timeout";
      Status status = fileStorage->DefaultGCFunc();
      if (!status.ok()) {
        LOG(ERROR) << "GC failed: " << status.ToString();
        // Not a fatal error and wait for next time.
      }
      last_time = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    }
  }
}

void FileStorage::PrintFileAccessTime(std::string path) {
  std::chrono::duration<int64_t, std::nano> accessTime;
  Status status = GetFileAccessTime(path, accessTime);
  if (!status.ok()) {
    VLOG(100) << "Failed to get file access time: " << status.ToString();
  } else {
    VLOG(100) << "File: " << path
              << " access time:" << GetTimestamp(accessTime);
  }
}

std::string FileStorage::GetTimestamp(
    std::chrono::duration<int64_t, std::nano> time) {
  auto duration_since_epoch =
      std::chrono::duration_cast<std::chrono::system_clock::duration>(time);
  std::chrono::time_point<std::chrono::system_clock> timestamp =
      std::chrono::system_clock::time_point(duration_since_epoch);
  time_t t = std::chrono::system_clock::to_time_t(timestamp);

  std::tm tm;
  localtime_r(&t, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

Status FileStorage::GlobalGCFunc() {
  auto now = std::chrono::high_resolution_clock::now();
  auto nanoseconds_since_epoch =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch());
  std::vector<std::string> fileList;
  RETURN_ON_ERROR(this->GetFileList(this->rootPath, fileList));
  VLOG(100) << "Global GC: " << fileList.size() << " files to check";
  for (std::vector<std::string>::iterator iter = fileList.begin();
       iter != fileList.end(); iter++) {
    std::string path = *iter;
    std::chrono::duration<int64_t, std::nano> accessTime(0);
    if (!GetFileAccessTime(path, accessTime).ok()) {
      continue;
    }
    VLOG(100) << "GC ttl:" << globalFileTTL.count();
    if ((accessTime + globalFileTTL).count() <
        nanoseconds_since_epoch.count()) {
      VLOG(100) << "Global GC: " << path << " is dead!";
      VLOG(100) << "Access time: " << GetTimestamp(accessTime);
      VLOG(100) << "Now: " << GetTimestamp(nanoseconds_since_epoch);
      VINEYARD_DISCARD(Delete(path));
    } else {
      VLOG(100) << "Global GC: " << path << " is alive!";
      VLOG(100) << "Access time: " << GetTimestamp(accessTime);
      VLOG(100) << "Now: " << GetTimestamp(nanoseconds_since_epoch);
    }
  }
  return Status::OK();
}

void FileStorage::GlobalGCThread(std::shared_ptr<FileStorage> fileStorage) {
  while (1) {
    sleep(fileStorage->globalGCInterval.count());
    if (fileStorage->enableGlobalGC) {
      VLOG(100) << "global GC thread wake";
      Status status = fileStorage->GlobalGCFunc();
      if (!status.ok()) {
        LOG(ERROR) << "GC failed: " << status.ToString();
        // Not a fatal error and wait for next time.
      }
    }
  }
}

void FileStorage::CloseGCThread() {
  std::lock_guard<std::mutex> gcLock(gcMutex);
  if (!exitFlag) {
    exitFlag = true;
    gcMutex.unlock();
    cv.notify_all();
    gcThread.join();
  }
}

void FileStorage::CloseCache() { CloseGCThread(); }

}  // namespace vineyard
