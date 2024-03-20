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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/util/logging.h"
#include "llm-cache/storage/file_storage.h"

namespace vineyard {

/**
 * prefix_size | layer | kv state len
 * prefix_len | prefix
 * kv state layer 1 | kv state layer 2 | kv state layer n
 * prefix_len | prefix
 * kv state layer 1 | kv state layer 2 | kv state layer n
 * prefix_len | prefix
 * kv state layer 1 | kv state layer 2 | kv state layer n
 * prefix_len | prefix
 * kv state layer 1 | kv state layer 2 | kv state layer n
 */

Status FileStorage::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  LOG(INFO) << "Update";
  // TBD: change batchSize to a fixed value
  batchSize = tokenList.size();
  int layer = kvStateList[0].size();
  int kvStateSize = kvStateList[0].begin()->second.first.length;

  std::vector<std::string> paths;
  RETURN_ON_ERROR(DefaultGetPathFromPrefix(tokenList, batchSize, paths));
  for (size_t i = 0; i < tokenList.size() / batchSize; i++) {
    std::shared_ptr<FileDescriptor> fd;

    // TODO: check if the kv state is already in the cache
    // TODO: hash conflict
    RETURN_ON_ERROR(Open(rootPath + paths[i], fd, FileOperationType::WRITE));
    FileHeader header;
    size_t fileSize;
    RETURN_ON_ERROR(GetFileSize(fd, fileSize));
    LOG(INFO) << "fileSize:" << fileSize;
    if (fileSize != 0) {
      Read(fd, &header, sizeof(FileHeader));
      if (header.layer != layer || header.kvStateSize != kvStateSize) {
        Close(fd);
        return Status::Invalid("Layer/KV state size mismatch!");
      }
      header.prefixNum += tokenList.size();
      Seek(fd, 0);
      Write(fd, &header, sizeof(FileHeader));
    }
    {
      header.prefixNum = tokenList.size();
      header.layer = layer;
      header.kvStateSize = kvStateSize;
      Write(fd, &header, sizeof(FileHeader));
    }
    LOG(INFO) << "Header: " << header.prefixNum << " " << header.layer << " "
              << header.kvStateSize;

    for (size_t j = 0; j < batchSize; j++) {
      // write
      int prefixSize = (i * batchSize + j + 1) * sizeof(int);
      LOG(INFO) << "Write prefix length:" << prefixSize / 4;
      LOG(INFO) << "write token list:";
      std::string token_str = "";
      for (size_t k = 0; k <= j + i * batchSize; k++) {
        token_str += std::to_string(tokenList[k]) + " ";
      }
      LOG(INFO) << token_str;
      Write(fd, &prefixSize, sizeof(int));
      Write(fd, tokenList.data(), prefixSize);
      for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
        LOG(INFO) << "write:";
        std::string data;
        for (int k = 0; k < kvStateSize; k++) {
          data += std::to_string((
                      reinterpret_cast<uint8_t*>(kvStateList[i * batchSize + j]
                                                     .find(currentLayer)
                                                     ->second.first.data))[k]) +
                  " ";
        }
        LOG(INFO) << "layer " << currentLayer << ":";
        LOG(INFO) << "key_state: " << data;
        Write(fd, (kvStateList[j].find(currentLayer))->second.first.data,
              kvStateSize);
        Write(fd, (kvStateList[j].find(currentLayer))->second.second.data,
              kvStateSize);
      }
    }

    RETURN_ON_ERROR(Close(fd));
  }

  LOG(INFO) << "update end";
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
  std::string token_str = " ";
  for (size_t i = 0; i < tokenList.size(); i++) {
    token_str += std::to_string(tokenList[i]) + " ";
  }
  LOG(INFO) << "Query token list: " << token_str;
  // TBD: change batchSize to a fixed value
  batchSize = tokenList.size();
  int layer = kvStateList[0].size();
  int kvStateSize = kvStateList[0].begin()->second.first.length;

  std::vector<std::string> paths;
  RETURN_ON_ERROR(DefaultGetPathFromPrefix(tokenList, batchSize, paths));
  for (size_t i = 0; i < tokenList.size() / batchSize; i++) {
    // per file
    std::shared_ptr<FileDescriptor> fd;

    // TODO: check if the kv state is already in the cache
    // TODO: hash conflict
    RETURN_ON_ERROR(Open(rootPath + paths[i], fd, FileOperationType::READ));
    FileHeader header;
    size_t fileSize;
    RETURN_ON_ERROR(GetFileSize(fd, fileSize));
    if (fileSize != 0) {
      Read(fd, &header, sizeof(FileHeader));
      if (header.layer != layer || header.kvStateSize != kvStateSize) {
        Close(fd);
        return Status::Invalid("Layer/KV state size mismatch!");
      }
    } else {
      RETURN_ON_ERROR(Close(fd));
      return Status::OK();
    }
    LOG(INFO) << "Header: " << header.prefixNum << " " << header.layer << " "
              << header.kvStateSize;

    // get the batchSize kv state
    // read
    std::vector<int> prefix(tokenList.begin(),
                            tokenList.begin() + (i + 1) * batchSize);
    std::vector<int> prefix_from_file;
    for (int k = 0; k < header.prefixNum; k++) {
      int prefixSize;
      Read(fd, &prefixSize, sizeof(int));
      LOG(INFO) << "Read prefix size:" << prefixSize;
      prefix_from_file.reserve(prefixSize / sizeof(int));
      prefix_from_file.resize(prefixSize / sizeof(int));
      Read(fd, prefix_from_file.data(), prefixSize);
      LOG(INFO) << "read token list:";
      std::string token_str = "";
      for (size_t k = 0; k < prefix_from_file.size(); k++) {
        token_str += std::to_string(prefix_from_file[k]) + " ";
      }
      LOG(INFO) << token_str;
      if (!CompareTokenList(prefix, prefix_from_file,
                            prefix_from_file.size())) {
        // Skip this batch in the file
        // It means that there exist hash conflict
        LOG(INFO) << "Do not match";
        return Status::OK();
        // prefix_from_file.clear();
        // size_t pos;
        // GetCurrentPos(fd, pos);
        // Seek(fd, pos + header.kvStateSize * 2 * header.layer);
        // continue;
      } else {
        LOG(INFO) << "Match";
        LOG(INFO) << "kv state size:" << kvStateSize;
        std::map<int, std::pair<LLMKV, LLMKV>> kvState;
        for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
          LLMKV k, v;
          k.data = new char[kvStateSize];
          v.data = new char[kvStateSize];
          Read(fd, k.data, kvStateSize);
          Read(fd, v.data, kvStateSize);
          std::string data;
          LOG(INFO) << "k state:";
          LOG(INFO) << "layer " << currentLayer << ":";
          for (int p = 0; p < kvStateSize; p++) {
            data +=
                std::to_string((reinterpret_cast<uint8_t*>(k.data))[p]) + " ";
          }
          LOG(INFO) << data;
          k.length = kvStateSize;
          v.length = kvStateSize;
          kvState.insert(std::make_pair(currentLayer, std::make_pair(k, v)));
        }
        kvStateList.push_back(kvState);
      }
    }

    // int currentToken = 0;
    // do {
    //   std::map<int, std::pair<LLMKV, LLMKV>> kvState;
    //   for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
    //     LLMKV k, v;
    //     k.data = new char[kvStateSize];
    //     v.data = new char[kvStateSize];
    //     Read(fd, k.data, kvStateSize);
    //     Read(fd, v.data, kvStateSize);
    //     k.length = kvStateSize;
    //     v.length = kvStateSize;
    //     kvState.insert(std::make_pair(currentLayer, std::make_pair(k, v)));
    //   }
    //   kvStateList.push_back(kvState);
    //   currentToken++;
    // } while (currentToken < batchSize);
    RETURN_ON_ERROR(Close(fd));
  }

  return Status::OK();
}

Status FileStorage::Query(const std::vector<int>& tokenList, int nextToken,
                          std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  // TBD
  return Status::NotImplemented();
}

Status FileStorage::DefaultGetPathFromPrefix(const std::vector<int>& tokenList,
                                             int batchSize,
                                             std::vector<std::string>& path) {
  std::string prefix;
  prefix += "/";
  for (size_t i = 0; i < tokenList.size(); i += batchSize) {
    for (int j = 0; j < batchSize; j++) {
      prefix += std::to_string(tokenList[i + j]);
    }
    LOG(INFO) << "path " << i << " : " << prefix;
    path.push_back(prefix);
  }

  // return the prefix size used to create the path
  return Status::OK();
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
