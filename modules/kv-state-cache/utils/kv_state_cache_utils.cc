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

void threadFunc();

void init_kv_state_cache(int dimension = 10, int cache_capacity = 10) {
  if (kv_state_cache_builder == nullptr) {
    std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
    LOG(INFO) << "socket:" << socket;
    client.Connect(socket);
    LOG(INFO) << "conneted";

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
    // TBD
    // use lease to prevent the deadlock if the client is down
  }
}

void update(const std::vector<int>& token_list, int next_token,
            const KV_STATE_WITH_LAYER& kv_state) {
  LOG(INFO) << "update";
  kv_state_cache_builder->Update(client, token_list, next_token, kv_state);
}

void update(const std::vector<int>& token_list,
            const LIST_KV_STATE_WITH_LAYER& kv_state) {
  std::vector<int> token_list_copy;
  for (size_t i = 0; i < token_list.size(); i++) {
    update(token_list_copy, token_list[i], kv_state[i]);
    token_list_copy.push_back(token_list[i]);
  }
}

KV_STATE_WITH_LAYER query(const std::vector<int>& token_list, int token) {
  LOG(INFO) << "query";
  return kv_state_cache_builder->Query(client, token_list, token);
}

LIST_KV_STATE_WITH_LAYER query(const std::vector<int>& token_list) {
  LIST_KV_STATE_WITH_LAYER list_kv_state;
  std::vector<int> token_list_copy;
  for (size_t i = 0; i < token_list.size(); i++) {
    KV_STATE_WITH_LAYER kv_state = query(token_list_copy, token_list[i]);
    list_kv_state.push_back(kv_state);
    token_list_copy.push_back(token_list[i]);
  }
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

  client.GetName(llm_cache_object_name, global_kv_state_cache_id);
  std::shared_ptr<KVStateCache> global_kv_state_cache = nullptr;
  if (global_kv_state_cache_id != InvalidObjectID()) {
    delete_list.push_back(global_kv_state_cache_id);
    global_kv_state_cache = std::dynamic_pointer_cast<KVStateCache>(
        client.GetObject(global_kv_state_cache_id));
  }

  // 3. merge the cache object
  std::shared_ptr<KVStateCacheBuilder> merged_kv_state_cache_builder =
      kv_state_cache_builder->Merge(client, global_kv_state_cache);
  if (global_kv_state_cache != nullptr) {
    if (global_kv_state_cache->GetVersion() ==
        kv_state_cache_builder->GetVersion() - 1) {
      merged_kv_state_cache_builder = kv_state_cache_builder;
    } else {
      // TBD
    }
  }

  // 4. push the cache object
  std::shared_ptr<Object> kv_state_cache =
      merged_kv_state_cache_builder->_Seal(client);
  // 5. put the name of the new cache object to the meta server
  client.DropName(llm_cache_object_name);
  client.PutName(kv_state_cache->id(), llm_cache_object_name);
  // 6. delete old cache object
  client.DelData(delete_list);
  // 7. create a global cache object replica
  // TBD cache stragety
  kv_state_cache_builder = std::make_shared<KVStateCacheBuilder>(
      client, std::dynamic_pointer_cast<KVStateCache>(kv_state_cache));
  // 8. release the lock
  client.TryReleaseLock(actural_key, result);
  VINEYARD_ASSERT(result == true);

  // TBD
  // use lease to prevent the deadlock if the client is down
}

void threadFunc() {
  while (1) {
    sleep(5);
    LOG(INFO) << "Try sync";
    // sync();
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
