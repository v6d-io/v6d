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

#include <map>
#include <vector>

#include "client/client.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "kv-state-cache/radix-tree/radix-tree.h"
#include "kv-state-cache/strategy/LRU_strategy.h"
#include "kv_state_cache_block.h"

#ifndef MODULES_KV_STATE_CACHE_H_
#define MODULES_KV_STATE_CACHE_H_

namespace vineyard {

class KVStateCache : public vineyard::Registered<KVStateCache> {
 private:
  KVStateCacheBlockBuilder* kv_state_cache_builder;
  RadixTree* root_tree;
  CacheStrategy* cache_strategy;
  int dimension;
  ObjectID id;

 public:
  void Construct(const ObjectMeta& meta) override;

  friend class KVStateCacheBlockBuilder;
};

class KVStateCacheBuilder : public vineyard::ObjectBuilder {
  KVStateCacheBlockBuilder* kv_state_cache_builder;
  RadixTree* root_tree;
  CacheStrategy* cache_strategy;
  int dimension;

 public:
  KVStateCacheBuilder(Client& client, int dimension, int cache_capacity);

  KVStateCacheBlockBuilder* split(
      Client& client, KVStateCacheBlockBuilder* kv_state_cache_builder,
      std::vector<std::shared_ptr<NodeWithTreeAttri>>
          node_with_tree_attri_list);

  void update(Client& client, const std::vector<int>& token_list,
              int next_token, const KV_STATE_WITH_LAYER& kv_state);

  KV_STATE_WITH_LAYER query(Client& client, const std::vector<int>& token_list,
                            int token);

  Status Build(Client& client) override;

  std::shared_ptr<Object> _Seal(Client& client) override;
};

}  // namespace vineyard

#endif