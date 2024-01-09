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
      root_tree = new RadixTree();
      kv_state_cache_builder = new vineyard::KVStateCacheBuilder(root_tree);
      std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
      LOG(INFO) << "socket:" << socket;
      client.Connect(socket);
      LOG(INFO) << "conneted";
    }
  }
} kv_state_cache_struct;

void update(const std::vector<int>& token_list, int next_token,
            const KV_STATE_WITH_LAYER& kv_state) {
  NodeWithCustomData *node_with_custom_data = kv_state_cache_struct.root_tree->insert(token_list, next_token);
  vineyard::KVStateCacheBuilder *kv_state_cache_builder = (vineyard::KVStateCacheBuilder *)node_with_custom_data->get_custom_data();
  kv_state_cache_builder->Lock();
  if (kv_state_cache_builder->isFull()) {
    kv_state_cache_struct.root_tree->Delete(token_list, next_token);
    kv_state_cache_builder->Split();
    update(token_list, next_token, kv_state);
  } else {
    offset_data *data = kv_state_cache_builder->Update(kv_state_cache_struct.client, kv_state);
    Node *node = node_with_custom_data->get_node();
    node->set_data(data, sizeof(offset_data));
  }
  kv_state_cache_builder->UnLock();
}

void update(const std::vector<int>& token_list,
            const LIST_KV_STATE_WITH_LAYER& kv_state) {
  std::vector<int> token_list_copy;
  for (int i = 0; i < token_list.size(); i++) {
    update(token_list_copy, token_list[i], kv_state[i]);
    token_list_copy.push_back(token_list[i]);
  }
}

LIST_KV_STATE_WITH_LAYER query(const std::vector<int>& token_list) {
  LIST_KV_STATE_WITH_LAYER list_kv_state;
  std::vector<int> token_list_copy;
  for (int i = 0; i < token_list.size(); i++) {
    KV_STATE_WITH_LAYER kv_state = query(token_list_copy, token_list[i]);
    list_kv_state.push_back(kv_state);
    token_list_copy.push_back(token_list[i]);
  }
  return list_kv_state;
}

KV_STATE_WITH_LAYER query(const std::vector<int>& token_list, int token) {
  KV_STATE_WITH_LAYER kv_state;
  RadixTree *tree;
  NodeWithCustomData *node_with_custom_data = kv_state_cache_struct.root_tree->get(token_list, token);
  if (node_with_custom_data != nullptr) {
    offset_data *data = (offset_data *) node_with_custom_data->get_node()->get_data();
    int offset_k = data->offset_k;
    int offset_v = data->offset_v;
    
    vineyard::KVStateCacheBuilder *kv_state_cache_builder = (vineyard::KVStateCacheBuilder *)node_with_custom_data->get_custom_data();
    kv_state_cache_builder->Lock();
    kv_state_cache_builder->Query(kv_state_cache_struct.client, offset_k, offset_v, kv_state);
    kv_state_cache_builder->UnLock();
  }
  return kv_state;
}