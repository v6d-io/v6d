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

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "llm-cache/ds/kv_state_cache.h"

#ifndef MODULES_LLM_CACHE_DS_KV_STATE_CACHE_MANAGER_H_
#define MODULES_LLM_CACHE_DS_KV_STATE_CACHE_MANAGER_H_

namespace vineyard {

class KVStateCacheManager {
 private:
  Client& client;
  std::shared_ptr<KVStateCacheBuilder> kvStateCacheBuilder = nullptr;
  std::shared_ptr<KVStateCache> kvStateCache = nullptr;
  std::string llmCacheSyncLock;
  std::string llmCacheObjectName;
  std::thread syncThread;
  std::mutex syncMutex;
  int syncInterval;
  bool exitFlag = false;
  std::condition_variable cv;
  std::mutex exitMutex;

 public:
  KVStateCacheManager(Client& client,
                      std::shared_ptr<KVStateCacheBuilder>& cache,
                      int syncInterval, std::string& llmCacheSyncLock,
                      std::string& llmCacheObjectName);

  static Status Make(Client& client,
                     std::shared_ptr<KVStateCacheManager>& manager,
                     int dimension = 10, int cacheCapacity = 10, int layer = 1,
                     int blockSize = 5, int syncInterval = 3,
                     std::string llmCacheSyncLock = "llmCacheSyncLock",
                     std::string llmCacheObjectName = "llm_cache_object");

  Status Update(const std::vector<int>& tokenList, int nextToken,
                const std::map<int, std::pair<LLMKV, LLMKV>>& kvState);

  Status Update(
      const std::vector<int>& tokenList,
      const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvState);

  Status Query(const std::vector<int>& tokenList, int token,
               std::map<int, std::pair<LLMKV, LLMKV>>& kvState);

  Status Query(
      const std::vector<int>& tokenList,
      std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& listKVState);

  ~KVStateCacheManager();

 private:
  Status UpdateInternal(const std::vector<int>& tokenList, int nextToken,
                        const std::map<int, std::pair<LLMKV, LLMKV>>& kvState);

  Status QueryInternal(const std::vector<int>& tokenList, int token,
                       std::map<int, std::pair<LLMKV, LLMKV>>& kvState);

  void Delete(std::vector<int>& token);

  static void SyncThreadFunc(KVStateCacheManager* manager);

  Status Sync();

  Status AfterSyncFailed();

  static void AcquireServerLock(Client& client, std::string& lockKey,
                                std::string& actualKey);

  static void ReleaseServerLock(Client& client, std::string& actualKey);
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_KV_STATE_CACHE_MANAGER_H_
