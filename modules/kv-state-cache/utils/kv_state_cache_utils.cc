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
#include "kv-state-cache/radix-tree/radix-tree.h"
#include "kv-state-cache/strategy/LRU_strategy.h"
#include "kv-state-cache/utils/kv_state_cache_utils.h"

using namespace vineyard;
struct KVStateCacheStruct {
  Client client;
  KVStateCacheBuilder* kv_state_cache_builder;
  RadixTree* root_tree;
  CacheStrategy* cache_strategy;
  int dimension;

  KVStateCacheStruct() {
    std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
    LOG(INFO) << "socket:" << socket;
    client.Connect(socket);
    LOG(INFO) << "conneted";
  }

  void SetDimension(int dimension, int cache_capacity = 10) {
    this->dimension = dimension;
    kv_state_cache_builder = new KVStateCacheBuilder(client, dimension);
    root_tree =
        new RadixTree(kv_state_cache_builder, sizeof(KVStateCacheBuilder));
    cache_strategy = new LRUStrategy(cache_capacity);
  }
} kv_state_cache_struct;

void init_kv_state_cache(int dimension) {
  kv_state_cache_struct.SetDimension(dimension);
}

KVStateCacheBuilder* split(
    KVStateCacheBuilder* kv_state_cache_builder,
    std::vector<std::shared_ptr<NodeWithTreeAttri>> node_with_tree_attri_list) {
  // Split the tree if the list of kv_state is full.
  assert(node_with_tree_attri_list.size() > 0);
  KVStateCacheBuilder* child_kv_state_cache_builder = new KVStateCacheBuilder(
      kv_state_cache_struct.client, kv_state_cache_struct.dimension);
  for (size_t i = 0; i < node_with_tree_attri_list.size(); i++) {
    std::shared_ptr<offset_data> data = std::static_pointer_cast<offset_data>(
        node_with_tree_attri_list[i]->get_node()->get_data());
    int index = data->offset;

    // Transfer the data from this builder to the child builder.
    const TensorBuilder<double>* k_builder =
        kv_state_cache_builder->getKBuilder();
    const TensorBuilder<double>* v_builder =
        kv_state_cache_builder->getVBuilder();
    std::shared_ptr<offset_data> new_offset_data =
        child_kv_state_cache_builder->Update(
            k_builder->data() + index * kv_state_cache_struct.dimension,
            v_builder->data() + index * kv_state_cache_struct.dimension,
            kv_state_cache_struct.dimension * sizeof(double));
    node_with_tree_attri_list[i]->get_node()->set_data(new_offset_data,
                                                       sizeof(offset_data));
    // Clear the bitmap.
    kv_state_cache_builder->DeleteKVCache(index);
  }
  kv_state_cache_builder->SetChildKVStateCacheBuilder(
      child_kv_state_cache_builder);
  return child_kv_state_cache_builder;
}

void update(const std::vector<int>& token_list, int next_token,
            const KV_STATE_WITH_LAYER& kv_state) {
  LOG(INFO) << "update";
  std::vector<int> token_list_copy = token_list;
  token_list_copy.push_back(next_token);

  // Create a empty node of tokens from radix tree.
  std::shared_ptr<NodeWithTreeAttri> node_with_tree_attri =
      kv_state_cache_struct.root_tree->insert(token_list_copy);
  RadixTree* sub_tree = node_with_tree_attri->get_tree();
  KVStateCacheBuilder* kv_state_cache_builder =
      (KVStateCacheBuilder*) sub_tree->get_custom_data();

  // TBD
  // Use lock to protect the kv_state_cache_builder
  // kv_state_cache_builder->Lock();

  if (kv_state_cache_builder->isFull()) {
    /**
     * If the kv-state cache of the tree is full, triggle split. Delete the empty node from
     * the radix tree and split the tree. Then, kv-state cache split according to the new
     * tree.
     */
    kv_state_cache_struct.root_tree->Delete(token_list_copy);
    RadixTree* new_tree = sub_tree->split();

    std::vector<std::shared_ptr<NodeWithTreeAttri>> node_with_tree_attri_list =
        new_tree->traverse();
    KVStateCacheBuilder* new_kv_state_cache_builder =
        split(kv_state_cache_builder, node_with_tree_attri_list);
    new_tree->set_custom_data(new_kv_state_cache_builder,
                              sizeof(KVStateCacheBuilder));

    // kv_state_cache_builder->UnLock();
    update(token_list, next_token, kv_state);
  } else {
    // Update the kv-state cache.
    std::shared_ptr<offset_data> data =
        kv_state_cache_builder->Update(kv_state);
    std::shared_ptr<Node> node = node_with_tree_attri->get_node();
    node->set_data(data, sizeof(offset_data));

    std::vector<int> evicted_tokens;
    kv_state_cache_struct.cache_strategy->put(token_list, next_token,
                                              evicted_tokens);

    // Delete the evicted kv-state cache.
    if (evicted_tokens.size() > 0) {
      std::string evicted_tokens_str = "";
      for (size_t i = 0; i < evicted_tokens.size(); i++) {
        evicted_tokens_str += std::to_string(evicted_tokens[i]);
      }
      LOG(INFO) << "evicted tokens: " << evicted_tokens_str;

      std::shared_ptr<NodeWithTreeAttri> evicted_node_with_tree_attri =
          kv_state_cache_struct.root_tree->get(evicted_tokens);
      KVStateCacheBuilder* evicted_kv_state_cache_builder =
          (KVStateCacheBuilder*) evicted_node_with_tree_attri->get_tree()
              ->get_custom_data();

      LOG(INFO) << "evicted index: "
                << std::static_pointer_cast<offset_data>(
                       evicted_node_with_tree_attri->get_node()->get_data())
                       ->offset;

      evicted_kv_state_cache_builder->DeleteKVCache(
          std::static_pointer_cast<offset_data>(
              evicted_node_with_tree_attri->get_node()->get_data())
              ->offset);
      kv_state_cache_struct.root_tree->Delete(evicted_tokens);
    }
    // kv_state_cache_builder->UnLock();
  }
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

  std::vector<int> token_list_copy = token_list;
  token_list_copy.push_back(token);

  KV_STATE_WITH_LAYER kv_state;
  std::shared_ptr<NodeWithTreeAttri> node_with_tree_attri =
      kv_state_cache_struct.root_tree->get(token_list_copy);
  if (node_with_tree_attri != nullptr) {
    std::shared_ptr<offset_data> data = std::static_pointer_cast<offset_data>(
        node_with_tree_attri->get_node()->get_data());
    int offset = data->offset;

    KVStateCacheBuilder* kv_state_cache_builder =
        (KVStateCacheBuilder*) node_with_tree_attri->get_tree()
            ->get_custom_data();
    // kv_state_cache_builder->Lock();
    kv_state_cache_builder->Query(kv_state_cache_struct.client, offset,
                                  kv_state);
    // kv_state_cache_builder->UnLock();
    std::vector<int> evicted_tokens;
    kv_state_cache_struct.cache_strategy->put(token_list, token,
                                              evicted_tokens);
    assert(evicted_tokens.size() == 0);
  }
  return kv_state;
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