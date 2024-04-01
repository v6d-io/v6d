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
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/util/logging.h"
#include "common/util/status.h"
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
 * data, and the length is the size of the tensor data.
 * @param updated It's a return value to indicate the number of tokens that have
 * been updated successfully.
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
  std::vector<std::string> pathList;
  RETURN_ON_ERROR(hasher->computePathForTokens(tokenList, batchSize,
                                               splitNumber, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }

  std::vector<std::string> tempFilePaths(pathList.size());
  auto fn = [this, &tempFilePaths, &pathList, &tokenList,
             &kvStateList](int i) -> std::pair<int, Status> {
    int tokenLength = (i + 1) * batchSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr =
        GetTmpFileDir(pathList[i]) + "-" + std::to_string(i);
    tempFilePaths[i] = tmpPathStr;
    std::filesystem::path tmpPath(tmpPathStr);
    std::string pathStr = this->rootPath + pathList[i];
    std::filesystem::path path(pathStr);

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
  for (size_t i = 0; i < pathList.size(); i++) {
    if (pathIndexMap.find(i) != pathIndexMap.end()) {
      j += 1;
    }
  }
  updated = j * batchSize;
  for (size_t i = j; i < pathList.size(); i++) {
    VINEYARD_SUPPRESS(Delete(pathList[i]));
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
 * data, and the length is the size of the tensor data.
 * @param updated It's a return value to indicate the number of tokens that have
 * been updated successfully.
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
  if (prefix.size() % batchSize != 0) {
    return Status::Invalid("Prefix size " + std::to_string(prefix.size()) +
                           " should be multiple of batch size " +
                           std::to_string(batchSize) + "!");
  }

  std::vector<std::string> pathList;
  std::vector<int> totalTokenList(prefix.begin(), prefix.end());
  totalTokenList.insert(totalTokenList.end(), tokenList.begin(),
                        tokenList.end());

  RETURN_ON_ERROR(hasher->computePathForTokens(totalTokenList, batchSize,
                                               splitNumber, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }

  std::vector<std::string> tempFilePaths(pathList.size());
  auto fn = [&](size_t i) -> std::pair<int, Status> {
    int tokenLength = (i + 1) * batchSize;
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();
    std::string tmpPathStr =
        GetTmpFileDir(pathList[i]) + "-" + std::to_string(i);
    std::filesystem::path tmpPath(tmpPathStr);
    std::string pathStr = this->rootPath + pathList[i];
    std::filesystem::path path(pathStr);

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
  for (size_t i = 0; i < pathList.size(); i++) {
    if (pathIndexMap.find(i) != pathIndexMap.end()) {
      j += 1;
    }
  }
  updated =
      size_t(j * batchSize) < prefix.size() ? 0 : j * batchSize - prefix.size();
  for (size_t i = j; i < pathList.size(); i++) {
    VINEYARD_SUPPRESS(Delete(pathList[i]));
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
 *                    It must be initialized(allocated) before calling this
 * function. It's a 2D vector, the first dimension is the token index, and the
 * second dimension is the layer index. The kv state is a pair of LLMKV, the
 * first is the K tensor and the second is the V tensor. It contains two fields:
 *                    data and length. The data is the pointer to the tensor
 * data, and the length is the size of the tensor data.
 * @param matched It's a return value to indicate the number of tokens that have
 * been matched successfully.
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
  std::vector<std::string> paths;
  std::string dir = rootPath;
  RETURN_ON_ERROR(
      hasher->computePathForTokens(tokenList, batchSize, splitNumber, paths));

  auto fn = [&](size_t i, size_t matched_start) -> std::pair<int, Status> {
    std::filesystem::path filePath(dir + paths[i]);
    std::shared_ptr<FileDescriptor> fd = CreateFileDescriptor();

    // If open failed, it means the kv state is not in the cache(file not exist)
    if (!Open(filePath.string(), fd, FileOperationType::READ).ok()) {
      return std::pair(-1, Status::ObjectNotExists("file doesn't exist"));
    }
    size_t file_size = 0;
    auto s = GetFileSize(fd, file_size);
    if (!s.ok()) {
      VINEYARD_DISCARD(Close(fd));
      return std::pair(-1, Status::ObjectNotExists("cannot get file size"));
    }
    if (file_size == 0) {
      VINEYARD_DISCARD(Close(fd));
      VINEYARD_DISCARD(Delete(filePath.string()));
      return std::pair(-1, Status::ObjectNotExists("file is empty"));
    }

    int tokenLength;
    RETURN_ON_ERROR_WITH_PATH_INDEX(i, Read(fd, &tokenLength, sizeof(int)));
    std::vector<int> prefix;
    prefix.resize(tokenLength);
    RETURN_ON_ERROR_WITH_PATH_INDEX(
        i, Read(fd, prefix.data(), tokenLength * sizeof(int)));

    if (!CompareTokenList(tokenList, prefix, prefix.size())) {
      VINEYARD_DISCARD(Close(fd));
      return std::pair(-1, Status::ObjectNotExists("token mismatch"));
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
  // parallel::ThreadGroup tg(1);
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

}  // namespace vineyard
