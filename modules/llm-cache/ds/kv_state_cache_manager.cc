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

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "client/client.h"
#include "common/util/logging.h"
#include "llm-cache/ds/kv_state_cache.h"
#include "llm-cache/ds/kv_state_cache_manager.h"

namespace vineyard {

KVStateCacheManager::KVStateCacheManager(
    Client& client, std::shared_ptr<KVStateCacheBuilder>& cache,
    int syncInterval, std::string& llmCacheSyncLock,
    std::string& llmCacheObjectName)
    : client(client) {
  this->syncInterval = syncInterval;
  this->kvStateCacheBuilder = cache;
  this->llmCacheSyncLock = llmCacheSyncLock;
  this->llmCacheObjectName = llmCacheObjectName;
  this->syncThread = std::thread(SyncThreadFunc, this);
}

Status KVStateCacheManager::Make(Client& client,
                                 std::shared_ptr<KVStateCacheManager>& manager,
                                 int dimension, int cacheCapacity, int layer,
                                 int blockSize, int syncInterval,
                                 std::string llmCacheSyncLock,
                                 std::string llmCacheObjectName) {
  RETURN_ON_ASSERT(client.Connected(), "The client is not connected.");
  // TBD
  // try to get cache object
  std::string actualKey;
  AcquireServerLock(client, llmCacheSyncLock, actualKey);

  // sync global cache object with vineyard
  ObjectID globalKVStateCacheID;
  Status status = client.GetName(llmCacheObjectName, globalKVStateCacheID);
  std::shared_ptr<KVStateCacheBuilder> kvStateCacheBuilder;
  if (status.ok()) {
    // if success, pull the cache object
    std::shared_ptr<KVStateCache> globalKVStateCache =
        std::dynamic_pointer_cast<KVStateCache>(
            client.FetchAndGetObject(globalKVStateCacheID));
    Status status = KVStateCacheBuilder::Make(client, kvStateCacheBuilder,
                                              globalKVStateCache);
    RETURN_ON_ASSERT(
        status.ok(),
        "Failed to make the cache object from global cache object.");
  } else {
    // if failed, create a new cache object
    VLOG(100) << "failed to get the cache object, create a new one.";
    Status status =
        KVStateCacheBuilder::Make(client, kvStateCacheBuilder, dimension,
                                  cacheCapacity, layer, blockSize);
    RETURN_ON_ASSERT(status.ok(), "Failed to make new cache object.");
  }

  // release the lock
  ReleaseServerLock(client, actualKey);

  // TBD
  // use lease to prevent the deadlock if the client is down
  manager = std::make_shared<KVStateCacheManager>(
      client, kvStateCacheBuilder, syncInterval, llmCacheSyncLock,
      llmCacheObjectName);
  return Status::OK();
}

Status KVStateCacheManager::UpdateInternal(
    const std::vector<int>& tokenList, int nextToken,
    const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  return kvStateCacheBuilder->Update(tokenList, nextToken, kvState);
}

Status KVStateCacheManager::QueryInternal(
    const std::vector<int>& tokenList, int token,
    std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  return kvStateCacheBuilder->Query(tokenList, token, kvState);
}

Status KVStateCacheManager::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  Status result =
      Status::Invalid("Query cache failed: can not gain the cache lock.");

  if (!syncMutex.try_lock()) {
    return result;
  }

  result = UpdateInternal(tokenList, nextToken, kvState);

  syncMutex.unlock();
  return result;
}

Status KVStateCacheManager::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvState) {
  Status result =
      Status::Invalid("Update cache failed: can not gain the cache lock.");
  if (!syncMutex.try_lock()) {
    return result;
  }

  std::vector<int> tokenListCopy;
  for (size_t i = 0; i < tokenList.size(); i++) {
    result = UpdateInternal(tokenListCopy, tokenList[i], kvState[i]);
    if (!result.ok()) {
      break;
    }
    tokenListCopy.push_back(tokenList[i]);
  }

  syncMutex.unlock();
  return result;
}

Status KVStateCacheManager::Query(
    const std::vector<int>& tokenList, int token,
    std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  Status result =
      Status::Invalid("Query cache failed: can not gain the cache lock.");

  if (!syncMutex.try_lock()) {
    return result;
  }

  result = QueryInternal(tokenList, token, kvState);
  syncMutex.unlock();

  return result;
}

Status KVStateCacheManager::Query(
    const std::vector<int>& tokenList,
    std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& listKVState) {
  Status result =
      Status::Invalid("Query cache failed: can not gain the cache lock.");
  if (!syncMutex.try_lock()) {
    return result;
  }

  std::vector<int> tokenListCopy;
  std::map<int, std::pair<LLMKV, LLMKV>> kvState;
  for (size_t i = 0; i < tokenList.size(); i++) {
    result = QueryInternal(tokenListCopy, tokenList[i], kvState);
    if (!result.ok()) {
      syncMutex.unlock();
      return result;
    }
    tokenListCopy.push_back(tokenList[i]);
    listKVState.push_back(kvState);
    kvState.clear();
  }

  syncMutex.unlock();
  return result;
}

KVStateCacheManager::~KVStateCacheManager() {
  LOG(INFO) << "Wait for sync thread to exit.";
  {
    std::lock_guard<std::mutex> lock(exitMutex);
    exitFlag = true;
  }
  cv.notify_one();
  syncThread.join();
  LOG(INFO) << "KVStateCacheManager exit.";
}

// This function is used for testing
void KVStateCacheManager::Delete(std::vector<int>& token) {
  std::shared_ptr<NodeData> evictedNode;
  kvStateCacheBuilder->GetRootTree()->Delete(token, evictedNode);
  kvStateCacheBuilder->Delete(evictedNode);
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(kvStateCacheBuilder->GetRootTree()->tree);
  }
}

Status KVStateCacheManager::Sync() {
  Status status;
  // 1. pull the cache object
  ObjectID globalKVStateCacheID;
  std::vector<ObjectID> deleteList;

  std::shared_ptr<KVStateCache> globalKVStateCache = nullptr;
  status = client.GetName(llmCacheObjectName, globalKVStateCacheID);
  if (status.ok()) {
    deleteList.push_back(globalKVStateCacheID);
    globalKVStateCache = std::dynamic_pointer_cast<KVStateCache>(
        client.FetchAndGetObject(globalKVStateCacheID));
  } else {
    // Not an error.
    VLOG(100) << "There is no cache object in the meta server.";
  }

  // 2. merge the cache object
  // only the global cache object with higher version will be merged
  VLOG(100) << "Current builder version:" << kvStateCacheBuilder->GetVersion()
            << " global version:"
            << (globalKVStateCache == nullptr
                    ? "null"
                    : std::to_string(globalKVStateCache->GetVersion()));
  if (globalKVStateCache != nullptr &&
      kvStateCacheBuilder->GetVersion() < globalKVStateCache->GetVersion()) {
    status = kvStateCacheBuilder->Merge(globalKVStateCache);
    RETURN_ON_ERROR(status);
  }
  kvStateCacheBuilder->UpdateVersion();

  // 3. push the cache object
  kvStateCache = std::dynamic_pointer_cast<KVStateCache>(
      kvStateCacheBuilder->_Seal(client));
  status = client.Persist(kvStateCache->id());
  RETURN_ON_ERROR(status);

  // 4. put the name of the new cache object to the meta server
  status = client.DropName(llmCacheObjectName);
  RETURN_ON_ERROR(status);
  status = client.PutName(kvStateCache->id(), llmCacheObjectName);
  RETURN_ON_ERROR(status);

  // 5. delete old cache object
  status = client.DelData(deleteList, false, true);
  if (!status.ok()) {
    LOG(ERROR) << "Delete old cache object failed: " << status.ToString()
               << " It may cause memory leak.";
  }

  // 6. create a global cache object replica
  kvStateCache->Resolve();
  status = KVStateCacheBuilder::Make(client, kvStateCacheBuilder, kvStateCache);
  RETURN_ON_ERROR(status);

  return Status::OK();

  // TBD
  // use lease to prevent the deadlock if the client is down
}

void KVStateCacheManager::SyncThreadFunc(KVStateCacheManager* manager) {
  uint64_t last_time = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  while (true) {
    std::unique_lock<std::mutex> lock(manager->exitMutex);
    if (manager->cv.wait_for(
            lock, std::chrono::seconds(manager->syncInterval),
            [&manager, &last_time] {
              uint64_t current_time =
                  std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
              return manager->exitFlag ||
                     static_cast<int>(current_time - last_time) >=
                         manager->syncInterval;
            })) {
      if (manager->exitFlag) {
        break;
      }
      manager->syncMutex.lock();
      std::string actualKey;

      AcquireServerLock(manager->client, manager->llmCacheSyncLock, actualKey);
      Status status = manager->Sync();
      if (!status.ok()) {
        while (!manager->AfterSyncFailed().ok()) {
          VLOG(100) << "Recover from sync failed failed. Retry later.";
          sleep(1);
        }
      }

      ReleaseServerLock(manager->client, actualKey);
      manager->syncMutex.unlock();

      last_time = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    }
  }
  LOG(INFO) << "Sync thread exit.";
}

Status KVStateCacheManager::AfterSyncFailed() {
  std::vector<ObjectID> deleteList;
  /**
   * If there is no global cache object, the local cache object will be
   * can be used directly. And Sync will be tried again later.
   * If there exists a global cache object, recover from the global object
   * and delete the cache object if the builder is sealed.
   */
  ObjectID globalKVStateCacheID;
  std::shared_ptr<KVStateCache> globalKVStateCache = nullptr;
  Status status = client.GetName(llmCacheObjectName, globalKVStateCacheID);
  if (status.ok()) {
    globalKVStateCache = std::dynamic_pointer_cast<KVStateCache>(
        client.FetchAndGetObject(globalKVStateCacheID));
  } else {
    VLOG(100) << "There is no cache object in the meta server.";
    return Status::OK();
  }

  status = KVStateCacheBuilder::Make(client, kvStateCacheBuilder,
                                     globalKVStateCache);
  RETURN_ON_ERROR(status);
  if (kvStateCache != nullptr && kvStateCache->id() != globalKVStateCacheID) {
    deleteList.push_back(kvStateCache->id());
  }
  status = client.DelData(deleteList, false, true);
  RETURN_ON_ERROR(status);
  kvStateCache = nullptr;

  return Status::OK();
}

void KVStateCacheManager::AcquireServerLock(Client& client,
                                            std::string& lockKey,
                                            std::string& actualKey) {
  bool result = false;
  while ((!(client.TryAcquireLock(lockKey, result, actualKey).ok())) ||
         !result) {
    VLOG(100) << "Failed to gain the lock, wait for next time.";
    sleep(1);
  }
}

void KVStateCacheManager::ReleaseServerLock(Client& client,
                                            std::string& actualKey) {
  bool result = false;
  while ((!(client.TryReleaseLock(actualKey, result).ok())) || !result) {
    VLOG(100) << "Failed to release the lock, wait for next time.";
    sleep(1);
  }
}

}  // namespace vineyard
