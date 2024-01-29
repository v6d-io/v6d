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

using namespace vineyard;

static Client client;
static std::shared_ptr<KVStateCacheBuilder> kv_state_cache_builder = nullptr;
static std::string llm_cache_sync_lock = "llm_cache_sync_lock";
static std::string llm_cache_object_name = "llm_cache_object";
static std::thread* sync_thread;
static bool exit_flag = false;
static pthread_mutex_t sync_mutex;

#ifndef SYNC_INTERVAL
#define SYNC_INTERVAL 3
#endif

void threadFunc();

void signalHandler(int signum) {
  /*
   * TBD
   * Avoid dead lock if the client is down when the lock is acquired.
   * Use lease to prevent dead lock in the future.
   */
  std::cout << "Interrupt signal (" << signum << ") received.\n";
  exit_flag = true;
  sync_thread->join();
  exit(signum);
}

void initKVStateCache(int dimension = 10, int cache_capacity = 10) {
  if (kv_state_cache_builder == nullptr) {
    std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
    LOG(INFO) << "socket:" << socket;
    client.Connect(socket);
    LOG(INFO) << "conneted";

    pthread_mutex_init(&sync_mutex, NULL);
    // TBD
    // try to get cache object
    std::string actural_key;
    bool result;
    while (1) {
      client.TryAcquireLock(llm_cache_sync_lock, result, actural_key);
      if (!result) {
        LOG(INFO) << "failed to gain the lock, wait for next time";
        sleep(1);
        continue;
      } else {
        break;
      }
    }

    // // sync global cache object with vineyard
    ObjectID global_kv_state_cache_id;
    Status status =
        client.GetName(llm_cache_object_name, global_kv_state_cache_id);
    if (status.ok()) {
      // if success, pull the cache object
      std::shared_ptr<KVStateCache> global_kv_state_cache =
          std::dynamic_pointer_cast<KVStateCache>(
              client.GetObject(global_kv_state_cache_id));
      // TBD cache stragety
      kv_state_cache_builder =
          std::make_shared<KVStateCacheBuilder>(client, global_kv_state_cache);
    } else {
      // if failed, create a new cache object
      LOG(INFO) << "failed to get the cache object, create a new one";
      kv_state_cache_builder = std::make_shared<KVStateCacheBuilder>(
          client, dimension, cache_capacity);
    }

    // // release the lock
    client.TryReleaseLock(actural_key, result);
    VINEYARD_ASSERT(result == true);

    sync_thread = new std::thread(threadFunc);

    signal(SIGINT, signalHandler);
    // TBD
    // use lease to prevent the deadlock if the client is down
  }
}

void updateInternal(const std::vector<int>& token_list, int next_token,
                    const KV_STATE_WITH_LAYER& kv_state) {
  kv_state_cache_builder->Update(client, token_list, next_token, kv_state);
}

void update(const std::vector<int>& token_list, int next_token,
            const KV_STATE_WITH_LAYER& kv_state) {
  LOG(INFO) << "update";
  if (pthread_mutex_trylock(&sync_mutex)) {
    return;
  }

  updateInternal(token_list, next_token, kv_state);

  pthread_mutex_unlock(&sync_mutex);
}

void update(const std::vector<int>& token_list,
            const LIST_KV_STATE_WITH_LAYER& kv_state) {
  if (pthread_mutex_trylock(&sync_mutex)) {
    return;
  }
  std::vector<int> token_list_copy;
  for (size_t i = 0; i < token_list.size(); i++) {
    updateInternal(token_list_copy, token_list[i], kv_state[i]);
    token_list_copy.push_back(token_list[i]);
  }
  pthread_mutex_unlock(&sync_mutex);
}

KV_STATE_WITH_LAYER queryInternal(const std::vector<int>& token_list,
                                  int token) {
  return kv_state_cache_builder->Query(client, token_list, token);
}

KV_STATE_WITH_LAYER query(const std::vector<int>& token_list, int token) {
  LOG(INFO) << "query";
  KV_STATE_WITH_LAYER result;
  if (pthread_mutex_trylock(&sync_mutex)) {
    return result;
  }

  result = queryInternal(token_list, token);
  pthread_mutex_unlock(&sync_mutex);

  return result;
}

LIST_KV_STATE_WITH_LAYER query(const std::vector<int>& token_list) {
  LIST_KV_STATE_WITH_LAYER list_kv_state;
  if (pthread_mutex_trylock(&sync_mutex)) {
    return list_kv_state;
  }

  std::vector<int> token_list_copy;
  for (size_t i = 0; i < token_list.size(); i++) {
    KV_STATE_WITH_LAYER kv_state =
        queryInternal(token_list_copy, token_list[i]);
    list_kv_state.push_back(kv_state);
    token_list_copy.push_back(token_list[i]);
  }

  pthread_mutex_unlock(&sync_mutex);
  return list_kv_state;
}

void sync() {
  LOG(INFO) << "sync";

  // 1. gain the lock
  std::string actural_key;
  bool result;
  client.TryAcquireLock(llm_cache_sync_lock, result, actural_key);
  if (!result) {
    LOG(INFO) << "failed to gain the lock, wait for next time";
    return;
  }
  // 2. pull the cache object
  ObjectID global_kv_state_cache_id;
  std::vector<ObjectID> delete_list;

  std::shared_ptr<KVStateCache> global_kv_state_cache = nullptr;
  Status status =
      client.GetName(llm_cache_object_name, global_kv_state_cache_id);
  if (status.ok()) {
    delete_list.push_back(global_kv_state_cache_id);
    global_kv_state_cache = std::dynamic_pointer_cast<KVStateCache>(
        client.GetObject(global_kv_state_cache_id));
  }

  // 3. merge the cache object
  kv_state_cache_builder->Merge(client, global_kv_state_cache);

  // 4. push the cache object
  std::shared_ptr<Object> kv_state_cache =
      kv_state_cache_builder->_Seal(client);
  client.Persist(kv_state_cache->id());

  // 5. put the name of the new cache object to the meta server
  LOG(INFO) << "stage 5";
  client.DropName(llm_cache_object_name);
  status = client.PutName(kv_state_cache->id(), llm_cache_object_name);
  if (status.ok()) {
    LOG(INFO) << "put name success";
  } else {
    LOG(INFO) << "put name failed with status:" + status.ToString();
  }

  LOG(INFO) << "stage 6";
  // 6. delete old cache object
  client.DelData(delete_list);

  LOG(INFO) << "stage 7";
  // 7. create a global cache object replica
  // TBD cache stragety
  std::dynamic_pointer_cast<KVStateCache>(kv_state_cache)->Resolve();
  kv_state_cache_builder = std::make_shared<KVStateCacheBuilder>(
      client, std::dynamic_pointer_cast<KVStateCache>(kv_state_cache));

  LOG(INFO) << "stage 8";
  // 8. release the lock
  client.TryReleaseLock(actural_key, result);
  VINEYARD_ASSERT(result == true);

  // TBD
  // use lease to prevent the deadlock if the client is down
}

void threadFunc() {
  while (1) {
    sleep(SYNC_INTERVAL);
    if (exit_flag) {
      break;
    }
    LOG(INFO) << "Try sync";
    pthread_mutex_lock(&sync_mutex);
    sync();
    pthread_mutex_unlock(&sync_mutex);
    // break;
  }
}

/*
  a. vineyardd with global cache object | sealed
  b. client get the object replica
  c. client update replica
  d. client seal the local object and try to push object to server (modified
  sealed object and global cache version) ⅰ. if success
      1. vineyardd modify global object meta
      2. client reconstruct the local object replica
      3. goto c
    ⅱ. if failed
      1. client pull the global object
      2. merge the object with local cache (e.g. create a new child_cache_object
  and merge)
      3. goto d
*/
