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
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "client/client.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "llm-cache/ds/kv_state_cache_block.h"
#include "llm-cache/radix-tree/radix-tree.h"

#ifndef MODULES_LLM_CACHE_DS_KV_STATE_CACHE_H_
#define MODULES_LLM_CACHE_DS_KV_STATE_CACHE_H_

namespace vineyard {

class KVStateCache : public vineyard::Registered<KVStateCache> {
 private:
  std::vector<std::shared_ptr<KVStateCacheBlock>> kvStateCacheBlockList;
  std::shared_ptr<RadixTree> rootTree;
  int tensorBytes;
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
  std::vector<std::shared_ptr<KVStateCacheBlock>>& GetKVStateCacheBlockList() {
    return this->kvStateCacheBlockList;
  }

  int GetTensorBytes() { return this->tensorBytes; }

  int GetCacheCapacity() { return this->cacheCapacity; }

  uint64_t GetVersion() { return this->version; }

  std::shared_ptr<RadixTree> GetRootTree() { return this->rootTree; }

  int GetLayer() { return this->layer; }

  void GetCurrentBlockIDSet(std::set<ObjectID>& objectIDSet);

  ~KVStateCache();

  friend class KVStateCacheBuilder;
};

class KVStateCacheBuilder : public vineyard::ObjectBuilder {
  Client& client;
  std::shared_ptr<RadixTree> rootTree;
  std::set<ObjectID> blockIDSetToDelete;
  int tensorBytes;
  int layer;
  uint64_t version;
  int blockSize;
  int cacheCapacity;

 public:
  KVStateCacheBuilder(Client& client, int tensorBytes, int layer,
                      std::shared_ptr<RadixTree>& rootTree);

  static Status Make(Client& client,
                     std::shared_ptr<KVStateCacheBuilder>& kvStateCacheBuilder,
                     int tensorBytes = 10, int cacheCapacity = 10,
                     int layer = 1, int blockSize = DEFAULT_BLOCK_SIZE);

  static Status Make(Client& client,
                     std::shared_ptr<KVStateCacheBuilder>& kvStateCacheBuilder,
                     std::shared_ptr<KVStateCache>& cache);

  Status Split(KVStateCacheBlockBuilder* kvStateCacheBlockBuilder,
               std::vector<std::shared_ptr<NodeData>> nodeDataList,
               KVStateCacheBlockBuilder*& childKVStateCacheBlockBuilder);

  Status Update(const std::vector<int>& token_list, int next_token,
                const std::vector<std::pair<LLMKV, LLMKV>>& kv_state);

  Status Query(const std::vector<int>& token_list, int token,
               std::vector<std::pair<LLMKV, LLMKV>>& kv_state);

  void Delete(std::shared_ptr<NodeData> evicted_node);

  Status Merge(std::shared_ptr<KVStateCache> kv_state_cache);

  uint64_t GetVersion() { return this->version; }

  void UpdateVersion() { this->version++; }

  void RollbackVersion() { this->version--; }

  Status Build(Client& client) override;

  std::shared_ptr<Object> _Seal(Client& client) override;

  uint64_t GetTensorBytes() { return this->tensorBytes; }

  std::shared_ptr<RadixTree> GetRootTree() { return this->rootTree; }

  int GetLayer() { return this->layer; }

  void Close();

  std::set<ObjectID>& GetBlockIDSetToDelete() {
    return this->blockIDSetToDelete;
  }

  void GetCurrentBlockIDSet(std::set<ObjectID>& objectIDSet);

  void ClearBlockIDSetToDelete() { this->blockIDSetToDelete.clear(); }

  ~KVStateCacheBuilder();
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_KV_STATE_CACHE_H_
