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
#include "kv-state-cache/ds/kv_state_cache.h"
#include "kv-state-cache/utils/kv_state_cache_utils.h"

namespace vineyard {

static Client client;
static std::shared_ptr<KVStateCacheBuilder> kvStateCacheBuilder = nullptr;
static std::string llmCacheSyncLock = "llmCacheSyncLock";
static std::string llmCacheObjectName = "llm_cache_object";
static std::thread* syncThread;
static bool exitFlag = false;
static pthread_mutex_t syncMutex;

#ifndef SYNC_INTERVAL
#define SYNC_INTERVAL 3
#endif

// for test
void Delete(std::vector<int> token) {
  std::shared_ptr<NodeData> evictedNode;
  kvStateCacheBuilder->GetRootTree()->Delete(token, evictedNode);
  kvStateCacheBuilder->Delete(evictedNode);
  raxShow(kvStateCacheBuilder->GetRootTree()->tree);
}

void threadFunc();

void signalHandler(int signum) {
  /*
   * TBD
   * Avoid dead lock if the client is down when the lock is acquired.
   * Use lease to prevent dead lock in the future.
   */
  LOG(INFO) << "Interrupt signal (" << signum << ") received.\n";
  CloseKVStateCache();
  exit(signum);
}

void CloseKVStateCache() {
  exitFlag = true;
  syncThread->join();
}

void InitKVStateCache(int dimension, int cacheCapacity, int layer,
                      int blockSize, std::string socket) {
  if (kvStateCacheBuilder == nullptr) {
    VLOG(100) << "socket:" << socket;
    client.Connect(socket);

    pthread_mutex_init(&syncMutex, NULL);
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

    // // release the lock
    client.TryReleaseLock(actualKey, result);
    VINEYARD_ASSERT(result == true);

    syncThread = new std::thread(threadFunc);

    signal(SIGINT, signalHandler);
    // TBD
    // use lease to prevent the deadlock if the client is down
  }
}

void UpdateInternal(const std::vector<int>& tokenList, int nextToken,
                    const KV_STATE_WITH_LAYER& kvState) {
  kvStateCacheBuilder->Update(client, tokenList, nextToken, kvState);
}

void Update(const std::vector<int>& tokenList, int nextToken,
            const KV_STATE_WITH_LAYER& kvState) {
  if (pthread_mutex_trylock(&syncMutex)) {
    return;
  }

  UpdateInternal(tokenList, nextToken, kvState);

  pthread_mutex_unlock(&syncMutex);
}

void Update(const std::vector<int>& tokenList,
            const LIST_KV_STATE_WITH_LAYER& kvState) {
  if (pthread_mutex_trylock(&syncMutex)) {
    return;
  }
  std::vector<int> tokenListCopy;
  for (size_t i = 0; i < tokenList.size(); i++) {
    UpdateInternal(tokenListCopy, tokenList[i], kvState[i]);
    tokenListCopy.push_back(tokenList[i]);
  }
  pthread_mutex_unlock(&syncMutex);
}

KV_STATE_WITH_LAYER QueryInternal(const std::vector<int>& tokenList,
                                  int token) {
  return kvStateCacheBuilder->Query(client, tokenList, token);
}

KV_STATE_WITH_LAYER Query(const std::vector<int>& tokenList, int token) {
  KV_STATE_WITH_LAYER result;
  if (pthread_mutex_trylock(&syncMutex)) {
    return result;
  }

  result = QueryInternal(tokenList, token);
  pthread_mutex_unlock(&syncMutex);

  return result;
}

LIST_KV_STATE_WITH_LAYER Query(const std::vector<int>& tokenList) {
  LIST_KV_STATE_WITH_LAYER listKVState;
  if (pthread_mutex_trylock(&syncMutex)) {
    return listKVState;
  }

  std::vector<int> tokenListCopy;
  for (size_t i = 0; i < tokenList.size(); i++) {
    KV_STATE_WITH_LAYER kvState = QueryInternal(tokenListCopy, tokenList[i]);
    listKVState.push_back(kvState);
    tokenListCopy.push_back(tokenList[i]);
  }

  pthread_mutex_unlock(&syncMutex);
  return listKVState;
}

void sync() {
  LOG(INFO) << "Try sync.";

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
  client.DelData(deleteList);

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

void threadFunc() {
  while (1) {
    sleep(SYNC_INTERVAL);
    if (exitFlag) {
      break;
    }
    pthread_mutex_lock(&syncMutex);
    sync();
    pthread_mutex_unlock(&syncMutex);
  }
}

}  // namespace vineyard
