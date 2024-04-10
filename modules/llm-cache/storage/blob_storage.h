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

#ifndef MODULES_LLM_CACHE_STORAGE_BLOB_STORAGE_H_
#define MODULES_LLM_CACHE_STORAGE_BLOB_STORAGE_H_

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "client/client.h"
#include "common/util/logging.h"

#include "llm-cache/ds/kv_state_cache.h"
#include "llm-cache/ds/refcnt_map.h"
#include "llm-cache/storage/storage.h"

namespace vineyard {

class BlobStorage : public IStorage {
 private:
  Client& client;
  std::shared_ptr<KVStateCacheBuilder> kvStateCacheBuilder = nullptr;
  std::shared_ptr<KVStateCache> kvStateCache = nullptr;
  std::shared_ptr<RefcntMapObjectBuilder> refcntMapObjectBuilder = nullptr;
  std::string llmCacheSyncLock;
  std::string llmCacheObjectName;
  std::string llmRefcntObjectName;
  std::thread syncThread;
  std::mutex cacheAccessMutex;
  int syncInterval;
  bool exitFlag = false;
  std::condition_variable cv;
  std::mutex exitMutex;
  bool isClosed = false;

 public:
  BlobStorage(Client& client, std::shared_ptr<KVStateCacheBuilder>& cache,
              int syncInterval, std::string& llmCacheSyncLock,
              std::string& llmCacheObjectName,
              std::string& llmRefcntObjectName);

  static Status Make(Client& client, std::shared_ptr<BlobStorage>& storage,
                     int tensorBytes = 10, int cacheCapacity = 10,
                     int layer = 1, int blockSize = 5, int syncInterval = 3,
                     std::string llmCacheSyncLock = "llmCacheSyncLock",
                     std::string llmCacheObjectName = "llm_cache_object",
                     std::string llmRefcntObjectName = "llm_refcnt_object");

  Status Update(const std::vector<int>& tokenList, int nextToken,
                const std::vector<std::pair<LLMKV, LLMKV>>& kvState) override;

  Status Update(
      const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
      size_t& updated) override;

  Status Update(
      const std::vector<int>& prefix, const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
      size_t& updated) override;

  Status Query(const std::vector<int>& tokenList, int token,
               std::vector<std::pair<LLMKV, LLMKV>>& kvState) override;

  Status Query(const std::vector<int>& tokenList,
               std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
               size_t& matched) override;

  void CloseCache() override;

  std::shared_ptr<KVStateCacheBuilder>& GetKVStateCacheBuilder() {
    return this->kvStateCacheBuilder;
  }

  std::shared_ptr<RefcntMapObjectBuilder>& GetRefcntMapObjectBuilder() {
    return this->refcntMapObjectBuilder;
  }

  void StopSync();

  static Status ClearGlobalCache(Client& client, std::string& llmCacheSyncLock,
                                 std::string& llmCacheObjectName,
                                 std::string& llmRefcntObjectName);

  ~BlobStorage();

 private:
  Status UpdateInternal(const std::vector<int>& tokenList, int nextToken,
                        const std::vector<std::pair<LLMKV, LLMKV>>& kvState);

  Status QueryInternal(const std::vector<int>& tokenList, int token,
                       std::vector<std::pair<LLMKV, LLMKV>>& kvState);

  void Delete(std::vector<int>& token);

  static void SyncThreadFunc(BlobStorage* storage);

  Status Sync();

  Status AfterSyncFailed();

  static void AcquireServerLock(Client& client, std::string& lockKey,
                                std::string& actualKey);

  static void ReleaseServerLock(Client& client, std::string& actualKey);

  Status SetRefcntMap(std::set<ObjectID>& blockIDSetToDelete,
                      std::set<ObjectID>& blockIDSetToAdd);

  void RefreshRefcnt();
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_STORAGE_BLOB_STORAGE_H_
