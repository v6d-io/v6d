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

KVStateCacheManager::KVStateCacheManager(int dimension, int cacheCapacity,
                                         int layer, int blockSize,
                                         int syncInterval, std::string socket) {
  this->syncInterval = syncInterval;
  VLOG(100) << "socket:" << socket;
  client.Connect(socket);

  // TBD
  // try to get cache object
  std::string actualKey;
  bool result;
  while (1) {
    client.TryAcquireLock(llmCacheSyncLock, result, actualKey);
    if (!result) {
      VLOG(100) << "failed to gain the lock, wait for next time.";
      sleep(1);
      continue;
    } else {
      break;
    }
  }

  // sync global cache object with vineyard
  ObjectID globalKVStateCacheID;
  Status status = client.GetName(llmCacheObjectName, globalKVStateCacheID);
  if (status.ok()) {
    // if success, pull the cache object
    std::shared_ptr<KVStateCache> globalKVStateCache =
        std::dynamic_pointer_cast<KVStateCache>(
            client.FetchAndGetObject(globalKVStateCacheID));
    kvStateCacheBuilder =
        std::make_shared<KVStateCacheBuilder>(client, globalKVStateCache);
  } else {
    // if failed, create a new cache object
    VLOG(100) << "failed to get the cache object, create a new one.";
    kvStateCacheBuilder = std::make_shared<KVStateCacheBuilder>(
        client, dimension, cacheCapacity, layer, blockSize);
  }

  // release the lock
  client.TryReleaseLock(actualKey, result);
  VINEYARD_ASSERT(result == true);

  // syncThread = new std::thread(threadFunc);
  syncThread = new std::thread(SyncThreadFunc, this);

  // TBD
  // use lease to prevent the deadlock if the client is down
}

void KVStateCacheManager::UpdateInternal(const std::vector<int>& tokenList,
                                         int nextToken,
                                         const KV_STATE_WITH_LAYER& kvState) {
  kvStateCacheBuilder->Update(client, tokenList, nextToken, kvState);
}

KV_STATE_WITH_LAYER KVStateCacheManager::QueryInternal(
    const std::vector<int>& tokenList, int token) {
  return kvStateCacheBuilder->Query(client, tokenList, token);
}

void KVStateCacheManager::Update(const std::vector<int>& tokenList,
                                 int nextToken,
                                 const KV_STATE_WITH_LAYER& kvState) {
  if (!syncMutex.try_lock()) {
    return;
  }

  UpdateInternal(tokenList, nextToken, kvState);

  syncMutex.unlock();
}

void KVStateCacheManager::Update(const std::vector<int>& tokenList,
                                 const LIST_KV_STATE_WITH_LAYER& kvState) {
  if (!syncMutex.try_lock()) {
    return;
  }

  std::vector<int> tokenListCopy;
  for (size_t i = 0; i < tokenList.size(); i++) {
    UpdateInternal(tokenListCopy, tokenList[i], kvState[i]);
    tokenListCopy.push_back(tokenList[i]);
  }

  syncMutex.unlock();
}

KV_STATE_WITH_LAYER KVStateCacheManager::Query(
    const std::vector<int>& tokenList, int token) {
  KV_STATE_WITH_LAYER result;

  if (!syncMutex.try_lock()) {
    return result;
  }

  result = QueryInternal(tokenList, token);
  syncMutex.unlock();

  return result;
}

LIST_KV_STATE_WITH_LAYER KVStateCacheManager::Query(
    const std::vector<int>& tokenList) {
  LIST_KV_STATE_WITH_LAYER listKVState;
  if (!syncMutex.try_lock()) {
    return listKVState;
  }

  std::vector<int> tokenListCopy;
  for (size_t i = 0; i < tokenList.size(); i++) {
    KV_STATE_WITH_LAYER kvState = QueryInternal(tokenListCopy, tokenList[i]);
    listKVState.push_back(kvState);
    tokenListCopy.push_back(tokenList[i]);
  }

  syncMutex.unlock();
  return listKVState;
}

KVStateCacheManager::~KVStateCacheManager() {
  LOG(INFO) << "Wait for sync thread to exit.";
  {
    std::lock_guard<std::mutex> lock(exitMutex);
    exitFlag = true;
  }
  cv.notify_one();
  syncThread->join();
  delete syncThread;
  LOG(INFO) << "KVStateCacheManager exit.";
}

// This function is used for testing
void KVStateCacheManager::Delete(std::vector<int> token) {
  std::shared_ptr<NodeData> evictedNode;
  kvStateCacheBuilder->GetRootTree()->Delete(token, evictedNode);
  kvStateCacheBuilder->Delete(evictedNode);
  raxShow(kvStateCacheBuilder->GetRootTree()->tree);
}

void KVStateCacheManager::Sync() {
  // 1. gain the lock
  std::string actualKey;
  bool result;
  client.TryAcquireLock(llmCacheSyncLock, result, actualKey);
  if (!result) {
    LOG(INFO) << "failed to gain the lock, wait for next time";
    return;
  }
  // 2. pull the cache object
  ObjectID globalKVStateCacheID;
  std::vector<ObjectID> deleteList;

  std::shared_ptr<KVStateCache> globalKVStateCache = nullptr;
  Status status = client.GetName(llmCacheObjectName, globalKVStateCacheID);
  if (status.ok()) {
    deleteList.push_back(globalKVStateCacheID);
    globalKVStateCache = std::dynamic_pointer_cast<KVStateCache>(
        client.FetchAndGetObject(globalKVStateCacheID));
  }

  // 3. merge the cache object
  // only the global cache object with higher version will be merged
  VLOG(100) << "Current builder version:" << kvStateCacheBuilder->GetVersion()
            << " global version:"
            << (globalKVStateCache == nullptr
                    ? "null"
                    : std::to_string(globalKVStateCache->GetVersion()));
  if (globalKVStateCache != nullptr &&
      kvStateCacheBuilder->GetVersion() < globalKVStateCache->GetVersion()) {
    kvStateCacheBuilder->Merge(client, globalKVStateCache);
  }
  kvStateCacheBuilder->UpdateVersion();

  // 4. push the cache object
  std::shared_ptr<Object> kvStateCache = kvStateCacheBuilder->_Seal(client);
  client.Persist(kvStateCache->id());

  // 5. put the name of the new cache object to the meta server
  client.DropName(llmCacheObjectName);
  status = client.PutName(kvStateCache->id(), llmCacheObjectName);
  if (!status.ok()) {
    throw std::runtime_error("Put cache object name failed.");
  }

  // 6. delete old cache object
  client.DelData(deleteList, true, true);

  // 7. create a global cache object replica
  std::dynamic_pointer_cast<KVStateCache>(kvStateCache)->Resolve();
  kvStateCacheBuilder = std::make_shared<KVStateCacheBuilder>(
      client, std::dynamic_pointer_cast<KVStateCache>(kvStateCache));

  // 8. release the lock
  while (1) {
    client.TryReleaseLock(actualKey, result);
    if (result == true) {
      break;
    }
    sleep(1);
  }

  // TBD
  // use lease to prevent the deadlock if the client is down
}

void KVStateCacheManager::SyncThreadFunc(KVStateCacheManager* manager) {
  uint64_t last_time = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  while (1) {
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
      manager->Sync();
      manager->syncMutex.unlock();
      last_time = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    }
  }
  LOG(INFO) << "Sync thread exit.";
}

}  // namespace vineyard
