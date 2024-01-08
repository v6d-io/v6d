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
#include "common/util/status.h"
#include "kv-state-cache/ds/kv_state_cache.h"

static struct KVStateCacheStruct {
  vineyard::Client client;
  vineyard::KVStateCacheBuilder* kv_state_cache_builder;
  RadixTree* root_tree;

  KVStateCacheStruct() {
    LOG(INFO) << "init kv cache";
    if (kv_state_cache_builder == nullptr) {
      kv_state_cache_builder = new vineyard::KVStateCacheBuilder();
      kv_state_cache_builder->GetTree(root_tree);
      std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
      LOG(INFO) << "socket:" << socket;
      client.Connect(socket);
      LOG(INFO) << "conneted";
    }
  }
} kv_state_cache_struct;

void update(const std::vector<int>& token_list, int next_token,
            const KV_STATE_WITH_LAYER& kv_state) {
  vineyard::Status status =
      kv_state_cache_struct.kv_state_cache_builder->Update(
          kv_state_cache_struct.client, token_list, next_token, kv_state);
  if (!status.ok()) {
    // TBD. Check the status
  }
}

void update(const std::vector<int>& token_list,
            const LIST_KV_STATE_WITH_LAYER& kv_state) {
  vineyard::Status status =
      kv_state_cache_struct.kv_state_cache_builder->Update(
          kv_state_cache_struct.client, token_list, kv_state);
  if (status.ok()) {
    // TBD. Check the status
  }
}

LIST_KV_STATE_WITH_LAYER query(const std::vector<int>& token_list) {
  std::vector<
      std::map<int, std::pair<std::vector<double>, std::vector<double>>>>
      kv_state;
  kv_state_cache_struct.kv_state_cache_builder->Query(
      kv_state_cache_struct.client, token_list, kv_state);
  return kv_state;
}

KV_STATE_WITH_LAYER query(const std::vector<int>& token_list, int token) {
  std::map<int, std::pair<std::vector<double>, std::vector<double>>> kv_state;
  kv_state_cache_struct.kv_state_cache_builder->Query(
      kv_state_cache_struct.client, token_list, token, kv_state);
  return kv_state;
}