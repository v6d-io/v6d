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
  std::string dir = rootPath;
  std::string path;
  int tokenSize;

  VINEYARD_ASSERT(batchSize > 0);
  // check if the root path ends with '/'
  if (dir.size() != 0 && dir.back() != '/') {
    dir += "/";
  }

  RETURN_ON_ERROR(hasher->computePathForTokens(tokenList, batchSize,
                                               splitNumber, pathList));
  if (pathList.size() == 0) {
    return Status::OK();
  }
  tokenSize = pathList.size() * batchSize;
  for (size_t i = 0; i < pathList.size(); i++) {
    std::shared_ptr<FileDescriptor> fd;
    std::filesystem::path filePath(dir + pathList[i]);
    RETURN_ON_ERROR(Mkdir(filePath.parent_path().string()));
    RETURN_ON_ERROR(Open(filePath.string(), fd, FileOperationType::WRITE));
    RETURN_ON_ERROR(Write(fd, &tokenSize, sizeof(int)));
    RETURN_ON_ERROR(Write(fd, tokenList.data(), tokenSize * sizeof(int)));
    for (size_t currentToken = i * batchSize;
         currentToken < (i + 1) * batchSize; currentToken++) {
      for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
        for (auto iter = kvStateList[currentToken].begin();
             iter != kvStateList[currentToken].end(); ++iter) {
          RETURN_ON_ERROR(
              Write(fd, reinterpret_cast<const char*>(&iter->second.first.data),
                    iter->second.first.length));
          RETURN_ON_ERROR(Write(
              fd, reinterpret_cast<const char*>(&iter->second.second.data),
              iter->second.second.length));
        }
      }
    }
    RETURN_ON_ERROR(Close(fd));
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
  LOG(INFO) << "Query";
  std::vector<std::string> paths;
  RETURN_ON_ERROR(
      hasher->computePathForTokens(tokenList, batchSize, splitNumber, paths));

  for (size_t i = 0; i < paths.size(); i++) {
    std::shared_ptr<FileDescriptor> fd;
    // If open failed, it means the kv state is not in the cache(file not exist)
    if (!Open(rootPath + paths[i], fd, FileOperationType::READ).ok()) {
      LOG(INFO) << "file not exist";
      return Status::OK();
    }

    int prefixSize;
    Read(fd, &prefixSize, sizeof(int));
    LOG(INFO) << "prefix length:" << prefixSize / sizeof(int);
    std::vector<int> prefix;
    prefix.resize(prefixSize / sizeof(int));
    Read(fd, prefix.data(), prefixSize);
    LOG(INFO) << "read token list:";
    std::string token_str = "";
    for (size_t j = 0; j < prefix.size(); j++) {
      token_str += std::to_string(prefix[j]) + " ";
    }
    LOG(INFO) << token_str;

    if (!CompareTokenList(tokenList, prefix, prefix.size())) {
      LOG(INFO) << "token list not match";
      RETURN_ON_ERROR(Close(fd));
      return Status::OK();
    } else {
      LOG(INFO) << "token list match";
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

          std::string data;
          for (int index = 0; index < tensorBytes; index++) {
            data +=
                std::to_string((reinterpret_cast<uint8_t*>(k.data))[index]) +
                " ";
          }
          LOG(INFO) << "layer " << currentLayer << ":";
          LOG(INFO) << "key_state: " << data;
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
