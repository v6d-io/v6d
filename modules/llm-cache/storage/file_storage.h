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

#include "llm-cache/storage/storage.h"
#include "MurmurHash3/MurmurHash3.h"

namespace vineyard {

class FileStorage : public IStorage {
 public:
  FileStorage(int chunkSize, int splitNumber, 
     int layer, int tensorBytes, int storedTokens, std::string path = "/vineyard/llm");
  
  static Status Make(std::shared_ptr<FileStorage>& storage,int chunkSize, int splitNumber, 
     int layer, int tensorBytes, int storedTokens, std::string path = "/vineyard/llm");

  ~FileStorage();

  Status Update(const std::vector<int>& tokenList, int nextToken,
                const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) override;

  Status Update(const std::vector<int>& tokenList,
                const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>&
                    kvStateList) override;

  Status Query(const std::vector<int>& tokenList, int token,
               std::map<int, std::pair<LLMKV, LLMKV>>& kvState) override;

  Status Query(const std::vector<int>& tokenList,
               std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList)
      override;

  void Close() override;

  private:
    // Store the kv state in a file with chunkSize
    int chunkSize;
    // split the hash value into directory with splitNumber
    // e.g. splitNumber = 2, hash value = 1ae45b78, then the file path is 1a/e4/5b/78
    int splitNumber;
    // the path to store the kv state
    std::string path;
    int layer;
    int tensorBytes;
    // the number of tokens stored in the file to avoid the hash conflict
    int storedTokens;
};

}  // namespace vineyard
#endif  // MODULES_LLM_CACHE_STORAGE_FILE_STORAGE_H_
