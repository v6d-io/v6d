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

#include <iostream>
#include <fstream>
#include <filesystem>

#include "common/util/logging.h"
#include "llm-cache/storage/file_storage.h"

namespace vineyard {

namespace fs = std::filesystem;

FileStorage::FileStorage(int chunkSize, int splitNumber,
                        int layer, int tensorBytes, int storedTokens, std::string path) {
    this->chunkSize = chunkSize;
    this->splitNumber = splitNumber;
    this->path = path;
    this->layer = layer;
    this->tensorBytes = tensorBytes;
    this->storedTokens = storedTokens;
}

Status FileStorage::Make(std::shared_ptr<FileStorage>& storage, int chunkSize, int splitNumber,
                        int layer, int tensorBytes, int storedTokens, std::string path) {
    storage = std::make_shared<FileStorage>(chunkSize, splitNumber, layer, tensorBytes, storedTokens, path);
    return Status::OK();
}

FileStorage::~FileStorage() {
  // TBD
}

Status FileStorage::Update(const std::vector<int>& tokenList, int nextToken,
                   const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
    // not implemented
    return Status::NotImplemented();
}

Status FileStorage::Update(const std::vector<int>& tokenList,
                   const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>&
                       kvStateList) {
    uint32_t hash;
    char hash_buffer[9];

    int token_size = tokenList.size() - tokenList.size() % chunkSize;
    // if the token list (upper_bound) is less than the chunk size, then return directly
    if (token_size < chunkSize) {
        return Status::OK();
    }

    // reserve the chunk size for the buffer and split by ' '
    std::vector<char> buffer(token_size * (sizeof(int) + 1), '\0');
    char *tokens_ptr;

    // split the token list into chunks
    for (int i = 0; i < token_size; i += chunkSize) {
        char* buffer_ptr = buffer.data();
        for (int j = i; j < i + chunkSize && j < token_size; j++) {
            buffer_ptr += std::sprintf(buffer_ptr, "%d ", tokenList[j]);
            if (j == storedTokens) {
                tokens_ptr = buffer_ptr;
            }
        }
        MurmurHash3_x86_32(buffer.data(), buffer_ptr - buffer.data(), 0, &hash);
        std::sprintf(hash_buffer, "%x", hash);
        int index = 0;
        std::string dir_str = path;
        while (index + splitNumber < 8) {
            // if the directory not ends with '/', then add '/'
            if (dir_str.back() != '/') {
                dir_str += "/";
            }
            dir_str += std::string(hash_buffer + index, splitNumber);
            index += splitNumber;
        }
        fs::path dir_path(dir_str);
        fs::path file_path = std::string(hash_buffer + index, 8 - index);
        fs::path full_path = dir_path / file_path;

        if (!fs::exists(dir_path)) {
            if (!fs::create_directories(dir_path)) {
                LOG(ERROR) << "Unable to create directory " << dir_path;
                return Status::IOError("Unable to create directory " + dir_path.string());
            }
        }

        std::ofstream file(full_path);
        if (!file.is_open()) {
            LOG(ERROR) << "Unable to create file " << full_path;
            return Status::IOError("Unable to create file " + full_path.string());
        }

        file.write(buffer.data(), tokens_ptr - buffer.data());
        file.write("\n", 1);
        // write the token 
        for (int currentToken = i; currentToken < i + chunkSize; currentToken++) {
            for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
                for (auto iter = kvStateList[currentToken].begin(); iter != kvStateList[currentToken].end(); ++iter) {
                    file.write(reinterpret_cast<const char*>(&iter->second.first.data), iter->second.first.length);
                    file.write(reinterpret_cast<const char*>(&iter->second.second.data), iter->second.second.length);
                }
            }
        }

        file.close();
    }

    return Status::OK();
}

Status FileStorage::Query(const std::vector<int>& tokenList, int token,
                  std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
    // not implemented
    return Status::NotImplemented();
}

Status FileStorage::Query(const std::vector<int>& tokenList,
                  std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
    uint32_t hash;
    char hash_buffer[9];

    int token_size = tokenList.size() - tokenList.size() % chunkSize;
    // if the token list (upper_bound) is less than the chunk size, then return directly
    if (token_size < chunkSize) {
        return Status::OK();
    }

    // reserve the chunk size for the buffer and split by ' '
    std::vector<char> buffer(token_size * (sizeof(int) + 1), '\0');

    // split the token list into chunks
    for (int i = 0; i < token_size; i += chunkSize) {
        char* buffer_ptr = buffer.data();
        for (int j = i; j < i + chunkSize && j < token_size; j++) {
            buffer_ptr += std::sprintf(buffer_ptr, "%d ", tokenList[j]);
        }
        MurmurHash3_x86_32(buffer.data(), buffer_ptr - buffer.data(), 0, &hash);
        std::sprintf(hash_buffer, "%x", hash);
        int index = 0;
        std::string dir_str = path;
        while (index + splitNumber < 8) {
            // if the directory not ends with '/', then add '/'
            if (dir_str.back() != '/') {
                dir_str += "/";
            }
            dir_str += std::string(hash_buffer + index, splitNumber);
            index += splitNumber;
        }
        fs::path dir_path(dir_str);
        fs::path file_path = std::string(hash_buffer + index, 8 - index);
        fs::path full_path = dir_path / file_path;
        
        // if the file not exists, return directly
        if (!fs::exists(full_path)) {
            return Status::OK();
        }

        std::ifstream file(full_path);
        if (!file.is_open()) {
            LOG(ERROR) << "Unable to open file " << full_path;
            return Status::IOError("Unable to open file " + full_path.string());
        }

        // read the first line that contains the token list
        std::string line;
        std::getline(file, line);
        std::vector<int> tokens;
        std::istringstream iss(line);
        int num;
        while (iss >> num) {
            tokens.push_back(num);
        }

        // if the token list is not equal to the current token list, return directly
        for (int t = 0; t < i + chunkSize && t < token_size; t++) {
            LOG(INFO) << "Token list: " << tokens[t] << " Current token list: " << tokenList[t];
            //if (tokens[t] != tokenList[t]) {
            //    LOG(INFO) << "########Token list not equal to the current token list";
            //    return Status::OK();
            //}
        }

        for (int currentToken = i; currentToken < i + chunkSize; currentToken++) {
            std::map<int, std::pair<LLMKV, LLMKV>> kv_state;
            for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
                LLMKV key_state;
                LLMKV value_state;
                key_state.data = malloc(tensorBytes);
                value_state.data = malloc(tensorBytes);
                key_state.length = tensorBytes;
                value_state.length = tensorBytes;
                file.read(reinterpret_cast<char*>(key_state.data), key_state.length);
                file.read(reinterpret_cast<char*>(value_state.data), value_state.length);
                kv_state[currentLayer] = std::make_pair(key_state, value_state);
            }
            kvStateList.push_back(kv_state);
        }

        file.close();
    }
    return Status::OK();
}

void FileStorage::Close() {
    
}

}  // namespace vineyard