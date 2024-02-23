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

struct TreeData {
  union {
    void* kvStateCacheBlockBuilder;
    uint64_t builderObjectID;
  };
  bool isPtr = true;
};

class KVStateCache : public vineyard::Registered<KVStateCache> {
 private:
  std::vector<std::shared_ptr<KVStateCacheBlock>> kvStateCacheBlockList;
  std::map<uint64_t, std::shared_ptr<KVStateCacheBlock>> kvStateCacheBlockMap;
  std::shared_ptr<RadixTree> rootTree;
  int dimension;
  int cacheCapacity;
  int layer;
  uint64_t version;

 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<KVStateCache>{new KVStateCache()});
  }

  void Construct(const ObjectMeta& meta) override;

  void Resolve();

  // for test
  std::vector<std::shared_ptr<KVStateCacheBlock>> GetKVStateCacheBlockList() {
    return this->kvStateCacheBlockList;
  }

  int GetDemension() { return this->dimension; }

  int GetCacheCapacity() { return this->cacheCapacity; }

  uint64_t GetVersion() { return this->version; }

  std::shared_ptr<RadixTree> GetRootTree() { return this->rootTree; }

  int GetLayer() { return this->layer; }

  ~KVStateCache();

  friend class KVStateCacheBuilder;
};

class KVStateCacheBuilder : public vineyard::ObjectBuilder {
  std::shared_ptr<RadixTree> rootTree;
  int dimension;
  int layer = 1;
  uint64_t version;

 public:
  KVStateCacheBuilder(Client& client, int dimension, int cacheCapacity,
                      int layer);

  KVStateCacheBuilder(Client& client, std::shared_ptr<KVStateCache> cache);

  KVStateCacheBlockBuilder* Split(
      Client& client, KVStateCacheBlockBuilder* kvStateCacheBlockBuilder,
      std::vector<std::shared_ptr<NodeData>> nodeDataList);

  void Update(Client& client, const std::vector<int>& token_list,
              int next_token, const KV_STATE_WITH_LAYER& kv_state);

  KV_STATE_WITH_LAYER Query(Client& client, const std::vector<int>& token_list,
                            int token);

  void Delete(Client& client, std::shared_ptr<NodeData> evicted_node);

  void Merge(Client& client, std::shared_ptr<KVStateCache> kv_state_cache);

  uint64_t GetVersion() { return this->version; }

  void UpdateVersion() { this->version++; }

  Status Build(Client& client) override;

  std::shared_ptr<Object> _Seal(Client& client) override;

  uint64_t GetDemension() { return this->dimension; }

  std::shared_ptr<RadixTree> GetRootTree() { return this->rootTree; }

  int GetLayer() { return this->layer; }

  ~KVStateCacheBuilder();
};

}  // namespace vineyard

#endif