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

#include "LRU_strategy.h"

namespace vineyard {

LRUStrategy::LRUStrategy(int capacity) {
  this->capacity = capacity;
  this->header = this->tail = nullptr;
  this->current_size = 0;
  radix_tree = new RadixTree();
}

LRUStrategy::LRUStrategy(const std::vector<std::vector<int>>& cache_list,
                         int capacity) {
  // TBD
}

void LRUStrategy::put(const std::vector<int>& prefix, int token,
                      std::vector<int>& evicted_tokens) {
  LOG(INFO) << "put";

  std::vector<int> tokens = prefix;
  tokens.push_back(token);

  std::shared_ptr<NodeWithTreeAttri> node_with_tree_attri =
      radix_tree->Query(tokens);
  if (node_with_tree_attri != nullptr) {
    std::shared_ptr<LRUCacheNode> cache_node =
        std::static_pointer_cast<LRUCacheNode>(
            node_with_tree_attri->get_node()->get_data());
    MoveToHead(cache_node);
    return;
  }

  if (current_size == capacity) {
    std::shared_ptr<LRUCacheNode> cache_node = Remove();
    evicted_tokens = cache_node->tokens;
    radix_tree->Delete(cache_node->tokens);
  }

  node_with_tree_attri = radix_tree->Insert(tokens);
  std::shared_ptr<Node> rax_node = node_with_tree_attri->get_node();
  std::shared_ptr<LRUCacheNode> cache_node = std::make_shared<LRUCacheNode>();

  rax_node->set_data(std::static_pointer_cast<void>(cache_node),
                     sizeof(LRUCacheNode));
  cache_node->tokens = tokens;

  if (header == nullptr) {
    header = cache_node;
    tail = cache_node;
  } else {
    cache_node->next = header;
    header->prev = cache_node;
    header = cache_node;
  }

  current_size++;
}

void LRUStrategy::MoveToHead(std::shared_ptr<LRUCacheNode> cache_node) {
  if (cache_node == header) {
    return;
  }

  if (cache_node == tail) {
    tail = tail->prev;
    tail->next = nullptr;
  } else {
    cache_node->prev->next = cache_node->next;
    cache_node->next->prev = cache_node->prev;
  }

  cache_node->next = header;
  header->prev = cache_node;
  header = cache_node;
  cache_node->prev = nullptr;
}

std::shared_ptr<LRUCacheNode> LRUStrategy::Remove() {
  std::shared_ptr<LRUCacheNode> cache_node = tail;
  if (tail->prev != nullptr) {
    tail->prev->next = nullptr;
    tail = tail->prev;
  } else {
    header = nullptr;
    tail = nullptr;
  }
  current_size--;
  return cache_node;
}

void LRUStrategy::Remove(const std::vector<int>& prefix, int token) {
  std::vector<int> tokens = prefix;
  tokens.push_back(token);

  std::shared_ptr<NodeWithTreeAttri> node_with_tree_attri =
      radix_tree->Query(tokens);
  if (node_with_tree_attri == nullptr) {
    return;
  }

  std::shared_ptr<LRUCacheNode> cache_node =
      std::static_pointer_cast<LRUCacheNode>(
          node_with_tree_attri->get_node()->get_data());
  if (cache_node == header) {
    header = header->next;
    header->prev = nullptr;
  } else if (cache_node == tail) {
    tail = tail->prev;
    tail->next = nullptr;
  } else {
    cache_node->prev->next = cache_node->next;
    cache_node->next->prev = cache_node->prev;
  }
  current_size--;
  radix_tree->Delete(tokens);
}

LRUStrategy::~LRUStrategy() { delete radix_tree; }

}  // namespace vineyard
