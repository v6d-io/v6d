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

#include "client/client.h"
#include "common/util/logging.h"
#include "kv-state-cache/ds/kv_state_cache.h"
#include "kv_state_cache_utils.h"

using namespace vineyard;

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
  //kvStateCacheBuilder->Delete(evictedNode);
  raxShow(kvStateCacheBuilder->GetRootTree()->tree);
}

void threadFunc();

void signalHandler(int signum) {
  /*
   * TBD
   * Avoid dead lock if the client is down when the lock is acquired.
   * Use lease to prevent dead lock in the future.
   */
  std::cout << "Interrupt signal (" << signum << ") received.\n";
  exitFlag = true;
  syncThread->join();
  exit(signum);
}

void InitKVStateCache(int dimension, int cacheCapacity, int layer) {
  if (kvStateCacheBuilder == nullptr) {
    std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
    LOG(INFO) << "socket:" << socket;
    client.Connect(socket);
    LOG(INFO) << "conneted";

    pthread_mutex_init(&syncMutex, NULL);
    // TBD
    // try to get cache object
    std::string acturalKey;
    bool result;
    while (1) {
      client.TryAcquireLock(llmCacheSyncLock, result, acturalKey);
      if (!result) {
        LOG(INFO) << "failed to gain the lock, wait for next time";
        sleep(1);
        continue;
      } else {
        break;
      }
    }

    // // sync global cache object with vineyard
    ObjectID globalKVStateCacheID;
    Status status = client.GetName(llmCacheObjectName, globalKVStateCacheID);
    if (status.ok()) {
      // if success, pull the cache object
      std::shared_ptr<KVStateCache> globalKVStateCache =
          std::dynamic_pointer_cast<KVStateCache>(
              client.GetObject(globalKVStateCacheID));
      // TBD cache stragety
      kvStateCacheBuilder =
          std::make_shared<KVStateCacheBuilder>(client, globalKVStateCache);
    } else {
      // if failed, create a new cache object
      LOG(INFO) << "failed to get the cache object, create a new one";
      kvStateCacheBuilder = std::make_shared<KVStateCacheBuilder>(
          client, dimension, cacheCapacity, layer);
    }

    // // release the lock
    client.TryReleaseLock(acturalKey, result);
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
  LOG(INFO) << "Update";
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
  LOG(INFO) << "Query";
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
  LOG(INFO) << "sync";

  // 1. gain the lock
  std::string acturalKey;
  bool result;
  client.TryAcquireLock(llmCacheSyncLock, result, acturalKey);
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
        client.GetObject(globalKVStateCacheID));
  }

  // 3. merge the cache object
  // only the global cache object with higher version will be merged
  LOG(INFO) << "Current builder version:" << kvStateCacheBuilder->GetVersion()
            << " global version:"
            << (globalKVStateCache == nullptr
                    ? "null"
                    : std::to_string(globalKVStateCache->GetVersion()));
  if (globalKVStateCache != nullptr &&
      kvStateCacheBuilder->GetVersion() < globalKVStateCache->GetVersion()) {
    LOG(INFO) << "kvStateCacheBuilder Merging...";
    kvStateCacheBuilder->Merge(client, globalKVStateCache);
  }
  kvStateCacheBuilder->UpdateVersion();

  // 4. push the cache object
  std::shared_ptr<Object> kvStateCache = kvStateCacheBuilder->_Seal(client);
  client.Persist(kvStateCache->id());
  LOG(INFO) << "before seal builderObjectID is" << kvStateCache->id();
  // 5. put the name of the new cache object to the meta server
  LOG(INFO) << "stage 5" << llmCacheObjectName;
  client.DropName(llmCacheObjectName);
  status = client.PutName(kvStateCache->id(), llmCacheObjectName);
  if (status.ok()) {
    LOG(INFO) << "put name success";
  } else {
    LOG(INFO) << "put name failed with status:" + status.ToString();
  }

  LOG(INFO) << "stage 6";
  // 6. delete old cache object
  client.DelData(deleteList);

  LOG(INFO) << "stage 7";
  // 7. create a global cache object replica
  std::dynamic_pointer_cast<KVStateCache>(kvStateCache)->Resolve();
  kvStateCacheBuilder = std::make_shared<KVStateCacheBuilder>(
      client, std::dynamic_pointer_cast<KVStateCache>(kvStateCache));

  LOG(INFO) << "stage 8";
  // 8. release the lock
  client.TryReleaseLock(acturalKey, result);
  VINEYARD_ASSERT(result == true);

  // TBD
  // use lease to prevent the deadlock if the client is down
}

void threadFunc() {
  while (1) {
    sleep(SYNC_INTERVAL);
    if (exitFlag) {
      break;
    }
    LOG(INFO) << "Try sync";
    pthread_mutex_lock(&syncMutex);
    sync();
    pthread_mutex_unlock(&syncMutex);
  }
}
