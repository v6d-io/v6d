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

#define RETURN_ON_ERROR_WITH_PATH_INDEX(index, status) \
  do {                                                 \
    auto _ret = (status);                              \
    if (!_ret.ok()) {                                  \
      return std::pair(index, _ret);                   \
    }                                                  \
  } while (0)

#define RETURN_ON_ASSERT_WITH_PATH_INDEX(index, condition, message)         \
  do {                                                                      \
    if (!(condition)) {                                                     \
      return std::pair(index, vineyard::Status::AssertionFailed(            \
                                  std::string(#condition ": ") + message)); \
    }                                                                       \
  } while (0)

namespace vineyard {

/**
 * @brief Update the kv state with the given token list in the file storage.
 *
 * @param tokenList The token list to be updated.
 * @param kvStateList The kv state list of the token list.
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
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvStateList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorBytes);                     *
 *           *     value_state.data = malloc(tensorBytes)                    *
 *           *     // Copy the k_state of LLM KV Cache to key_state.data     *
 *           *     // Copy the v_state of LLM KV Cache to value_state.data   *
 *           *     key_state.length = tensorBytes;                           *
 *           *     value_state.length = tensorBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvStateList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, you must release(free) the       *
 *           * kv buffer of the kvStateList manually                         *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The length of the token list should be as same as the length of the
 * kvStateList.
 *
 *
 * @example Suppose the token list is [1, 2, 3, 4], the layer is 2,
 *          then the kvStateList should be a 2D vector with size 4 * 2.
 *
 * @return Status
 */
Status FileStorage::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
    size_t& updated) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }
  std::vector<std::string> pathList;
  std::set<std::string> createFileSet;
  std::mutex createFileSetMutex;
  RETURN_ON_ERROR(hasher->computePathForTokens(tokenList, batchSize,
                                               splitNumber, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }

  std::vector<std::string> tempFilePaths(pathList.size());
  auto fn = [this, &tempFilePaths, &pathList, &tokenList, &kvStateList,
             &createFileSet,
             &createFileSetMutex](int i) -> std::pair<int, Status> {
    int tokenLength = (i + 1) * batchSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr = GetTmpFileDir() + "-" + std::to_string(i);
    tempFilePaths[i] = tmpPathStr;
    ghc::filesystem::path tmpPath(tmpPathStr);
    std::string pathStr = this->rootPath + pathList[i];
    ghc::filesystem::path path(pathStr);

    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Mkdir(path.parent_path().string()));

    if (Open(pathStr, fd, FileOperationType::READ).ok()) {
      int tokenLengthInFile;
      RETURN_ON_ERROR_WITH_PATH_INDEX(
          i, Read(fd, &tokenLengthInFile, sizeof(int)));
      std::vector<int> tokens;
      tokens.resize(tokenLengthInFile);
      RETURN_ON_ERROR_WITH_PATH_INDEX(
          i, Read(fd, tokens.data(), tokenLengthInFile * sizeof(int)));
      if (!CompareTokenList(tokenList, tokens, tokenLengthInFile)) {
        // Token list not match
        VINEYARD_DISCARD(Close(fd));
        return std::pair(
            i, Status::ObjectExists("File exists for another token sequence"));
      }
      // Skip this kv state
      VINEYARD_DISCARD(Close(fd));
      return std::pair(i, Status::OK());
    }

    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Mkdir(tmpPath.parent_path().string()));
    auto status = Open(tmpPathStr, fd, FileOperationType::WRITE);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to create temporary cache entry: "
                   << status.ToString();
      return std::pair(
          i, Status::Wrap(status, "Failed to create temporary cache entry"));
    }

    // Currently we do not consider delete.
    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Write(fd, &tokenLength, sizeof(int)));
    RETURN_ON_ERROR_WITH_PATH_INDEX(
        i, Write(fd, tokenList.data(), tokenLength * sizeof(int)));
    for (int currentTokenIndex = i * batchSize;
         currentTokenIndex < (i + 1) * batchSize; currentTokenIndex++) {
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        const LLMKV& k = kvStateList[currentTokenIndex][currentLayer].first;
        const LLMKV& v = kvStateList[currentTokenIndex][currentLayer].second;
        RETURN_ON_ERROR_WITH_PATH_INDEX(i, Write(fd, k.data, k.length));
        RETURN_ON_ERROR_WITH_PATH_INDEX(i, Write(fd, v.data, k.length));
      }
    }

    VINEYARD_DISCARD(Flush(fd));
    VINEYARD_DISCARD(Close(fd));
    status = MoveFileAtomic(tmpPathStr, pathStr);
    if (!status.ok()) {
      // Move failed. There exists a file with the same name.
      LOG(WARNING) << "Failed to move cache entry: " << status.ToString();
      VINEYARD_SUPPRESS(Delete(tmpPathStr));
      return std::pair(i, Status::Wrap(status, "Failed to move cache entry"));
    }
    std::lock_guard<std::mutex> lock(createFileSetMutex);
    createFileSet.insert(pathStr);
    return std::pair(i, Status::OK());
  };

  parallel::ThreadGroup tg(
      std::min(pathList.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  for (size_t i = 0; i < pathList.size(); i++) {
    tg.AddTask(fn, i);
  }

  std::vector<std::pair<int, Status>> ss = tg.TakeResults();
  std::map<int, bool> pathIndexMap;
  for (size_t i = 0; i < pathList.size(); i++) {
    if (ss[i].second.ok()) {
      pathIndexMap[ss[i].first] = true;
    }
  }

  int j = 0;
  {
    std::lock_guard<std::mutex> lock(gcMutex);
    for (size_t i = 0; i < pathList.size(); i++) {
      if (pathIndexMap.find(i) != pathIndexMap.end()) {
        j += 1;
        if (createFileSet.find(this->rootPath + pathList[i]) !=
            createFileSet.end()) {
          TouchFile(this->rootPath + pathList[i]);
          gcList.push_back(this->rootPath + pathList[i]);
        }
      } else {
        break;
      }
    }
  }
  updated = ((size_t) j) * batchSize;
  for (size_t i = j; i < pathList.size(); i++) {
    VINEYARD_SUPPRESS(Delete(this->rootPath + pathList[i]));
    VINEYARD_SUPPRESS(Delete(tempFilePaths[i]));
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
 * @param kvStateList The kv state list of the token list.
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
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvStateList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorBytes);                     *
 *           *     value_state.data = malloc(tensorBytes)                    *
 *           *     // Copy the k_state of LLM KV Cache to key_state.data     *
 *           *     // Copy the v_state of LLM KV Cache to value_state.data   *
 *           *     key_state.length = tensorBytes;                           *
 *           *     value_state.length = tensorBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvStateList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, you must release(free) the       *
 *           * kv buffer of the kvStateList manually                         *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The length of the token list should be as same as the length of the
 * kvStateList.
 *
 * @example Suppose the prefix is [1, 2], the token list is [3, 4], the layer is
 * 2, then the kvStateList should be a 2D vector with size 2 * 2.
 *
 * @return Status
 */
Status FileStorage::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
    size_t& updated) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }
  if (prefix.size() % batchSize != 0) {
    return Status::Invalid("Prefix size " + std::to_string(prefix.size()) +
                           " should be multiple of batch size " +
                           std::to_string(batchSize) + "!");
  }

  std::vector<std::string> pathList;
  std::set<std::string> createFileSet;
  std::mutex createFileSetMutex;
  std::vector<int> totalTokenList(prefix.begin(), prefix.end());
  totalTokenList.insert(totalTokenList.end(), tokenList.begin(),
                        tokenList.end());

  RETURN_ON_ERROR(hasher->computePathForTokens(totalTokenList, batchSize,
                                               splitNumber, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }

  std::vector<std::string> tempFilePaths(pathList.size());
  auto fn = [this, &tempFilePaths, &pathList, &prefix, &totalTokenList,
             &kvStateList, &createFileSet,
             &createFileSetMutex](size_t i) -> std::pair<int, Status> {
    int tokenLength = (i + 1) * batchSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr = GetTmpFileDir() + "-" + std::to_string(i);
    tempFilePaths[i] = tmpPathStr;
    ghc::filesystem::path tmpPath(tmpPathStr);
    std::string pathStr = this->rootPath + pathList[i];
    ghc::filesystem::path path(pathStr);

    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Mkdir(path.parent_path().string()));

    if (Open(pathStr, fd, FileOperationType::READ).ok()) {
      int tokenLength;
      RETURN_ON_ERROR_WITH_PATH_INDEX(i, Read(fd, &tokenLength, sizeof(int)));
      std::vector<int> tokens;
      tokens.resize(tokenLength);
      RETURN_ON_ERROR_WITH_PATH_INDEX(
          i, Read(fd, tokens.data(), tokenLength * sizeof(int)));
      if (!CompareTokenList(totalTokenList, tokens, tokenLength)) {
        // Token list not match
        VINEYARD_DISCARD(Close(fd));
        return std::pair(
            i, Status::ObjectExists("File exists for another token sequence"));
      }
      // Skip this kv state
      VINEYARD_DISCARD(Close(fd));
      return std::pair(i, Status::OK());
    }

    if ((i + 1) * batchSize <= prefix.size()) {
      return std::pair(
          i, Status::ObjectNotExists("The prefix is not in the file cache"));
    }

    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Mkdir(tmpPath.parent_path().string()));
    auto status = Open(tmpPathStr, fd, FileOperationType::WRITE);
    if (!status.ok()) {
      return std::pair(
          i, Status::Wrap(status, "Failed to create temporary cache entry"));
    }

    // Currently we do not consider delete.

    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Write(fd, &tokenLength, sizeof(int)));
    RETURN_ON_ERROR_WITH_PATH_INDEX(
        i, Write(fd, totalTokenList.data(), tokenLength * sizeof(int)));
    size_t kvStatePos =
        (i * batchSize) < prefix.size() ? 0 : (i * batchSize) - prefix.size();
    for (size_t currentTokenIndex = kvStatePos;
         currentTokenIndex < kvStatePos + batchSize; currentTokenIndex++) {
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        const LLMKV& k = kvStateList[currentTokenIndex][currentLayer].first;
        const LLMKV& v = kvStateList[currentTokenIndex][currentLayer].second;
        RETURN_ON_ERROR_WITH_PATH_INDEX(i, Write(fd, k.data, k.length));
        RETURN_ON_ERROR_WITH_PATH_INDEX(i, Write(fd, v.data, k.length));
      }
    }

    VINEYARD_DISCARD(Flush(fd));
    VINEYARD_DISCARD(Close(fd));
    if (!MoveFileAtomic(tmpPathStr, pathStr).ok()) {
      // Move failed. There exists a file with the same name.
      VINEYARD_SUPPRESS(Delete(tmpPathStr));
      return std::pair(i, Status::Wrap(status, "Failed to move cache entry"));
    }
    std::lock_guard<std::mutex> lock(createFileSetMutex);
    createFileSet.insert(pathStr);
    return std::pair(i, Status::OK());
  };

  parallel::ThreadGroup tg(
      std::min(pathList.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  for (size_t i = 0; i < pathList.size(); i++) {
    tg.AddTask(fn, i);
  }

  std::vector<std::pair<int, Status>> ss = tg.TakeResults();
  std::map<int, bool> pathIndexMap;
  for (size_t i = 0; i < pathList.size(); i++) {
    if (ss[i].second.ok()) {
      pathIndexMap[ss[i].first] = true;
    }
  }

  int j = 0;
  {
    std::lock_guard<std::mutex> lock(gcMutex);
    for (size_t i = 0; i < pathList.size(); i++) {
      if (pathIndexMap.find(i) != pathIndexMap.end()) {
        j += 1;
        if (((size_t) j) * batchSize > prefix.size() &&
            createFileSet.find(this->rootPath + pathList[i]) !=
                createFileSet.end()) {
          // Only this part is created.
          TouchFile(this->rootPath + pathList[i]);
          gcList.push_back(this->rootPath + pathList[i]);
        }
      } else {
        break;
      }
    }
  }
  updated =
      size_t(j * batchSize) < prefix.size() ? 0 : j * batchSize - prefix.size();
  for (size_t i = j; i < pathList.size(); i++) {
    VINEYARD_SUPPRESS(Delete(this->rootPath + pathList[i]));
    VINEYARD_SUPPRESS(Delete(tempFilePaths[i]));
  }

  return Status::OK();
}

Status FileStorage::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  // TBD
  return Status::NotImplemented();
}

/**
 * @brief Query the kv state with the given token list in the file storage.
 *
 * @param tokenList The token list to be queried.
 * @param kvStateList The kv state list of the token list to be fulfilled.
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
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvStateList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorBytes);                     *
 *           *     value_state.data = malloc(tensorBytes)                    *
 *           *     key_state.length = tensorBytes;                           *
 *           *     value_state.length = tensorBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvStateList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, the key_state and value_state    *
 *           * will be fulfilled with the kv state of the token, then you    *
 *           * can copy the kv state to the LLM KV Cache. At last, you       *
 *           * must free the memory of the kv state manually.                *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The kvStateList must be initialized before calling this function,
 * including the data and length of the kv tensor.
 *
 * @return Status
 */
Status FileStorage::Query(
    const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
    size_t& matched) {
  if (this->exitFlag) {
    return Status::Invalid("The file storage has been closed!");
  }
  std::vector<std::string> paths;
  std::string dir = rootPath;
  RETURN_ON_ERROR(
      hasher->computePathForTokens(tokenList, batchSize, splitNumber, paths));

  auto fn = [&](size_t i, size_t matched_start) -> std::pair<int, Status> {
    ghc::filesystem::path filePath(dir + paths[i]);
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();

    // If open failed, it means the kv state is not in the cache(file not exist)
    if (!Open(filePath.string(), fd, FileOperationType::READ).ok()) {
      return std::pair(i, Status::ObjectNotExists("file doesn't exist"));
    }
    size_t file_size = 0;
    auto s = GetFileSize(fd, file_size);
    if (!s.ok()) {
      VINEYARD_DISCARD(Close(fd));
      return std::pair(i, Status::ObjectNotExists("cannot get file size"));
    }
    if (file_size == 0) {
      VINEYARD_DISCARD(Close(fd));
      VINEYARD_DISCARD(Delete(filePath.string()));
      return std::pair(i, Status::ObjectNotExists("file is empty"));
    }

    int tokenLength;
    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Read(fd, &tokenLength, sizeof(int)));
    std::vector<int> prefix;
    prefix.resize(tokenLength);
    RETURN_ON_ERROR_WITH_PATH_INDEX(
        i, Read(fd, prefix.data(), tokenLength * sizeof(int)));

    if (!CompareTokenList(tokenList, prefix, prefix.size())) {
      VINEYARD_DISCARD(Close(fd));
      return std::pair(i, Status::ObjectNotExists("token mismatch"));
    } else {
      for (int j = 0; j < batchSize; j++) {
        if (matched_start + j >= tokenList.size() ||
            matched_start + j >= kvStateList.size()) {
          break;
        }
        auto& kvState = kvStateList[matched_start + j];
        for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
          RETURN_ON_ASSERT_WITH_PATH_INDEX(
              i, static_cast<int>(kvState.size()) == layer,
              "The size of kvState is not equal to layer");
          LLMKV& k = kvState[currentLayer].first;
          LLMKV& v = kvState[currentLayer].second;
          RETURN_ON_ASSERT_WITH_PATH_INDEX(
              i, k.length == tensorBytes && v.length == tensorBytes,
              "The size of kv tensor doesn't match with the tensorBytes");
          RETURN_ON_ERROR_WITH_PATH_INDEX(i, Read(fd, k.data, k.length));
          RETURN_ON_ERROR_WITH_PATH_INDEX(i, Read(fd, v.data, v.length));
        }
      }
    }

    VINEYARD_DISCARD(Close(fd));
    return std::pair(i, Status::OK());
  };

  parallel::ThreadGroup tg(std::min(
      paths.size(), static_cast<size_t>(std::thread::hardware_concurrency())));
  for (size_t i = 0; i < paths.size(); i++) {
    tg.AddTask(fn, i, i * batchSize);
  }

  matched = 0;
  std::vector<std::pair<int, Status>> ss = tg.TakeResults();
  std::map<int, bool> pathIndexMap;
  for (size_t i = 0; i < paths.size(); i++) {
    if (ss[i].second.ok()) {
      pathIndexMap[ss[i].first] = true;
    }
  }

  for (size_t i = 0; i < paths.size(); i++) {
    if (pathIndexMap.find(i) != pathIndexMap.end()) {
      matched += batchSize;
    } else {
      break;
    }
  }
  return Status::OK();
}

Status FileStorage::Query(const std::vector<int>& tokenList, int nextToken,
                          std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
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

Status FileStorage::DefaultGCFunc() {
  auto now = std::chrono::high_resolution_clock::now();
  auto nanoseconds_since_epoch =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch())
          .count();

  for (std::list<std::string>::iterator iter = gcList.begin();
       iter != gcList.end();) {
    std::string path = *iter;
    std::chrono::duration<int64_t, std::nano> accessTime(0);
    RETURN_ON_ERROR(GetFileAccessTime(path, accessTime));
    LOG(INFO) << "GC ttl:" << fileTTL.count();
    PrintFileAccessTime(path);
    if ((accessTime + fileTTL).count() < nanoseconds_since_epoch) {
      LOG(INFO) << "Dead!";
      RETURN_ON_ERROR(Delete(path));
      iter = gcList.erase(iter);
    } else {
      LOG(INFO) << "Alive!";
      iter++;
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
    // sleep(fileStorage->gcInterval.count());
    // LOG(INFO) << "GC thread wake";
    // Status status = fileStorage->DefaultGCFunc();
    // if (!status.ok()) {
    //   // TBD: process the error
    // }
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
        LOG(INFO) << "GC thread exit";
        return;
      }
      LOG(INFO) << "GC thread timeout";
      Status status = fileStorage->DefaultGCFunc();
      if (!status.ok()) {
        // TBD: process the error
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
    LOG(ERROR) << "Failed to get file access time: " << status.ToString();
  } else {
    LOG(INFO) << "File: " << path << " access time: " << accessTime.count();
  }
}

Status FileStorage::GlobalGCFunc() {
  auto now = std::chrono::high_resolution_clock::now();
  auto nanoseconds_since_epoch =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch())
          .count();
  std::vector<std::string> fileList;
  RETURN_ON_ERROR(this->GetFileList(this->rootPath, fileList));
  for (std::vector<std::string>::iterator iter = fileList.begin();
       iter != fileList.end(); iter++) {
    std::string path = *iter;
    std::chrono::duration<int64_t, std::nano> accessTime(0);
    if (!GetFileAccessTime(path, accessTime).ok()) {
      continue;
    }
    if ((accessTime + globalFileTTL).count() < nanoseconds_since_epoch) {
      LOG(INFO) << "Global GC: " << path << " is dead!";
      LOG(INFO) << "access time: " << accessTime.count()
                << " now: " << nanoseconds_since_epoch;
      Delete(path);
    } else {
      LOG(INFO) << "Global GC: " << path << " is alive!";
      LOG(INFO) << "access time: " << accessTime.count()
                << " now: " << nanoseconds_since_epoch;
    }
  }
  return Status::OK();
}

void FileStorage::GlobalGCThread(std::shared_ptr<FileStorage> fileStorage) {
  while (1) {
    sleep(fileStorage->globalGCInterval.count());
    if (fileStorage->enableGlobalGC) {
      LOG(INFO) << "global GC thread wake";
      Status status = fileStorage->GlobalGCFunc();
      if (!status.ok()) {
        // TBD: process the error
      }
    }
  }
}

void FileStorage::CloseCache() {
  std::lock_guard<std::mutex> gcLock(gcMutex);
  if (!exitFlag) {
    exitFlag = true;
    gcMutex.unlock();
    cv.notify_all();
    gcThread.join();
  }
  LOG(INFO) << "Close cache";
}

}  // namespace vineyard
