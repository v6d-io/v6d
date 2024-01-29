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

struct BlockStruct {
  union {
    void* kv_state_cache_block;
    uint64_t objectID;
  };
  bool is_pointer;
};

class KVStateCache : public vineyard::Registered<KVStateCache> {
 private:
  std::shared_ptr<KVStateCacheBlock> kv_state_cache_block;
  std::shared_ptr<RadixTree> root_tree;
  int dimension;
  int cache_capacity;
  uint64_t version;

 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<KVStateCache>{new KVStateCache()});
  }

  void Construct(const ObjectMeta& meta) override;

  void Resolve();

  // for test
  std::shared_ptr<KVStateCacheBlock> GetKVStateCacheBlock() {
    return this->kv_state_cache_block;
  }

  int GetDemension() { return this->dimension; }

  int GetCacheCapacity() { return this->cache_capacity; }

  uint64_t GetVersion() { return this->version; }

  std::shared_ptr<RadixTree> GetRootTree() { return this->root_tree; }

  ~KVStateCache();

  friend class KVStateCacheBuilder;
};

class KVStateCacheBuilder : public vineyard::ObjectBuilder {
  std::shared_ptr<KVStateCacheBlockBuilder> kv_state_cache_block_builder;
  std::shared_ptr<RadixTree> root_tree;
  int dimension;
  uint64_t version;

 public:
  KVStateCacheBuilder(Client& client, int dimension, int cache_capacity);

  KVStateCacheBuilder(Client& client, std::shared_ptr<KVStateCache> cache);

  KVStateCacheBlockBuilder* Split(
      Client& client, KVStateCacheBlockBuilder* kv_state_cache_block_builder,
      std::vector<std::shared_ptr<NodeWithTreeAttri>>
          node_with_tree_attri_list);

  void Update(Client& client, const std::vector<int>& token_list,
              int next_token, const KV_STATE_WITH_LAYER& kv_state);

  KV_STATE_WITH_LAYER Query(Client& client, const std::vector<int>& token_list,
                            int token);

  void Delete(std::shared_ptr<NodeWithTreeAttri> evicted_node);

  void Merge(Client& client, std::shared_ptr<KVStateCache> kv_state_cache);

  uint64_t GetVersion() { return this->version; }

  Status Build(Client& client) override;

  std::shared_ptr<Object> _Seal(Client& client) override;

  std::shared_ptr<KVStateCacheBlockBuilder> GetKVStateCacheBlockBuilder() {
    return this->kv_state_cache_block_builder;
  }

  uint64_t GetDemension() { return this->dimension; }

  std::shared_ptr<RadixTree> GetRootTree() { return this->root_tree; }

  ~KVStateCacheBuilder();
};

}  // namespace vineyard

#endif