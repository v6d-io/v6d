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

#include <unordered_map>
#include <vector>

#include "cache_strategy.h"

#ifndef MODULES_LRU_STRATEGY_H_
#define MODULES_LRU_STRATEGY_H_

namespace vineyard {

struct LRUCacheNode {
  std::shared_ptr<LRUCacheNode> next;
  std::shared_ptr<LRUCacheNode> prev;
  std::vector<int> tokens;
};

class LRUStrategy : public CacheStrategy {
 private:
  int current_size;

  std::shared_ptr<LRUCacheNode> header;

  std::shared_ptr<LRUCacheNode> tail;

  LRUStrategy();

  std::shared_ptr<LRUCacheNode> Remove();

  ~LRUStrategy();

 public:
  LRUStrategy(int capacity);

  LRUStrategy(const std::vector<std::vector<int>>& cache_list, int capacity);

  void MoveToHead(std::shared_ptr<LRUCacheNode> cache_node);

  std::shared_ptr<LRUCacheNode> InsertToHeader(
      const std::vector<int>& tokens, std::vector<int>& evicted_tokens);

  void Remove(std::shared_ptr<LRUCacheNode> cache_node);

  std::shared_ptr<LRUCacheNode> GetHeader();

  int GetCapacity() { return capacity; }

  void PrintLRUList();
};

}  // namespace vineyard

#endif