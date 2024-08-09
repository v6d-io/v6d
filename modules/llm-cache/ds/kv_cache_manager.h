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

#include "llm-cache/ds/config.h"
#include "llm-cache/ds/kv_cache.h"
#include "llm-cache/storage/blob_storage.h"
#include "llm-cache/storage/file_storage.h"

#ifndef MODULES_LLM_CACHE_DS_KV_CACHE_MANAGER_H_
#define MODULES_LLM_CACHE_DS_KV_CACHE_MANAGER_H_

namespace vineyard {

class KVCacheManager {
 public:
  explicit KVCacheManager(std::shared_ptr<IStorage> storageImpl);

  ~KVCacheManager();

  static Status Make(Client& client, std::shared_ptr<KVCacheManager>& manager,
                     VineyardCacheConfig& config);

  static Status Make(std::shared_ptr<KVCacheManager>& manager,
                     FileCacheConfig& config);

  static Status Make(RPCClient& rpc_client, Client& ipc_client,
                     std::shared_ptr<KVCacheManager>& manager,
                     FileCacheConfig& config);

  Status Update(const std::vector<int>& tokenList, int nextToken,
                const std::vector<std::pair<LLMKV, LLMKV>>& kvState);

  Status Update(
      const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
      size_t& updated);

  Status Update(
      const std::vector<int>& prefix, const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
      size_t& updated);

  Status BatchedUpdate(
      const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
      size_t& updated);

  Status Query(const std::vector<int>& tokenList,
               std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
               size_t& matched);

  Status Query(const std::vector<int>& prefix, int nextToken,
               std::vector<std::pair<LLMKV, LLMKV>>& kvState);

  Status Query(const std::vector<int>& prefix,
               const std::vector<int>& tokenList,
               std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
               size_t& matched);

  Status BatchedQuery(
      const std::vector<int>& tokenList,
      std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
      size_t& matched);

  void Close();

  void StopGlobalGCThread();

  void StartGlobalGCThread();

  static Status ClearGlobalCache(Client& client, VineyardCacheConfig& config);

  std::shared_ptr<KVCacheConfig> Config() { return config; }

 private:
  std::shared_ptr<KVCacheConfig> config;
  std::shared_ptr<IStorage> storage;
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_KV_CACHE_MANAGER_H_
