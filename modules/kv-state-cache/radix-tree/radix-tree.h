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

#ifndef RADIX_TREE_H
#define RADIX_TREE_H

#include "radix.h"

#include "common/util/logging.h"
#include "kv-state-cache/strategy/LRU_strategy.h"

#include <map>
#include <memory>
#include <vector>

using namespace vineyard;

typedef struct nodeData {
  std::shared_ptr<void> data;
  int data_length;
  std::shared_ptr<LRUCacheNode> cache_node;
} nodeData;

class Node {
 private:
  nodeData* data;
  raxNode* node;

 public:
  Node(raxNode* node) {
    this->data = (nodeData*) raxGetData(node);
    this->node = node;
  }

  Node(nodeData* data) {
    this->data = data;
    this->node = NULL;
  }

  void set_data(std::shared_ptr<void> data, int data_length) {
    if (this->node == NULL) {
      LOG(INFO) << "set data failed, node is null";
      return;
    }
    this->data->data = data;
    this->data->data_length = data_length;
    raxSetData(this->node, this->data);
  }

  void set_cache_node(std::shared_ptr<LRUCacheNode> cache_node) {
    if (this->node == NULL) {
      LOG(INFO) << "set data failed, node is null";
      return;
    }
    this->data->cache_node = cache_node;
    raxSetData(this->node, this->data);
  }

  std::shared_ptr<void> get_data() { return this->data->data; }

  int get_data_length() { return this->data->data_length; }

  std::shared_ptr<LRUCacheNode> get_cache_node() {
    return this->data->cache_node;
  }
};

class RadixTree;

class NodeWithTreeAttri {
 private:
  std::shared_ptr<Node> node;
  RadixTree* belong_to;

 public:
  NodeWithTreeAttri(std::shared_ptr<Node> node, void* belong_to) {
    this->node = node;
    this->belong_to = (RadixTree*) belong_to;
  }

  std::shared_ptr<Node> get_node() { return node; }

  RadixTree* get_tree() { return belong_to; }
};

class RadixTree {
 private:
  void* custom_data;
  int custom_data_length;
  // the whole radix tree for prefix match
  rax* tree;
  // the sub tree for mapping a vineyard object
  rax* sub_tree;
  LRUStrategy* lru_strategy;

 public:
  RadixTree(int cache_capacity = 10) {
    LOG(INFO) << "init radix tree";
    this->tree = raxNew();
    this->sub_tree = this->tree;
    this->custom_data = NULL;
    this->custom_data_length = 0;
    lru_strategy = new LRUStrategy(cache_capacity);
  }

  RadixTree(void* custom_data, int custom_data_length,
            int cache_capacity = 10) {
    LOG(INFO) << "init radix tree with custom data";
    this->tree = raxNew();
    this->sub_tree = this->tree;
    this->custom_data = custom_data;
    this->custom_data_length = custom_data_length;
    this->lru_strategy = new LRUStrategy(cache_capacity);
  }

  std::shared_ptr<NodeWithTreeAttri> Insert(
      std::vector<int> tokens,
      std::shared_ptr<NodeWithTreeAttri> evicted_node) {
    // insert the token vector to the radix tree
    int* insert_tokens_array = tokens.data();
    size_t insert_tokens_array_len = tokens.size();
    nodeData* dummy_data = new nodeData();
    nodeData* old_data;
    raxNode* dataNode = NULL;
    int retval = raxInsertAndReturnDataNode(this->tree, insert_tokens_array,
                              insert_tokens_array_len, dummy_data, (void**) &dataNode, (void**) &old_data);
    if (dataNode == NULL) {
      LOG(INFO) << "insert failed";
      return NULL;
    }
    LOG(INFO) << "insert success";

  
    if (retval == 0) {
      // (retval == 0 ) means the token vector already exists in the radix tree
      // remove the token vector from the lru cache as it will be inserted again
      std::shared_ptr<Node> node = std::make_shared<Node>(old_data);
      std::shared_ptr<LRUCacheNode> cache_node = node->get_cache_node();
      lru_strategy->Remove(cache_node);
      delete old_data;
    }

    // refresh the lru cache
    std::vector<int> evicted_tokens;
    std::shared_ptr<LRUCacheNode> cache_node =
        lru_strategy->InsertToHeader(tokens, evicted_tokens);
    if (cache_node == nullptr) {
      LOG(INFO) << "WTF?";
    }
    dummy_data->cache_node = cache_node;
    raxSetData(dataNode, dummy_data);
    if (evicted_tokens.size() > 0) {
      this->Delete(evicted_tokens, evicted_node);
    }

    return std::make_shared<NodeWithTreeAttri>(std::make_shared<Node>(dataNode),
                                               this);
  }

  void Delete(std::vector<int> tokens,
              std::shared_ptr<NodeWithTreeAttri>& evicted_node) {
    // remove the token vector from the radix tree
    int* delete_tokens_array = tokens.data();
    size_t delete_tokens_array_len = tokens.size();

    nodeData* old_data;
    int retval = raxRemove(this->tree, delete_tokens_array,
                           delete_tokens_array_len, (void**) &old_data);
    if (retval == 1) {
      LOG(INFO) << "remove success";
      std::shared_ptr<Node> node = std::make_shared<Node>(old_data);
      evicted_node = std::make_shared<NodeWithTreeAttri>(node, this);
      delete old_data;
    } else {
      LOG(INFO) << "remove failed";
    }
  }

  std::shared_ptr<NodeWithTreeAttri> Query(std::vector<int> key) {
    int* tokens = key.data();
    size_t tokens_len = key.size();

    raxNode* dataNode =
        raxFindAndReturnDataNode(this->tree, tokens, tokens_len);
    if (dataNode == NULL) {
      LOG(INFO) << "get failed";
      return NULL;
    }
    LOG(INFO) << "get success";

    // refresh the lru cache
    std::shared_ptr<Node> node = std::make_shared<Node>(dataNode);
    std::shared_ptr<LRUCacheNode> cache_node = node->get_cache_node();
    lru_strategy->MoveToHead(cache_node);

    return std::make_shared<NodeWithTreeAttri>(node, this);
  }

  std::string Serialize() { return std::string("this is a serialized string"); }

  static RadixTree* Deserialize(std::string data) {
    LOG(INFO) << "deserialize with data:" + data;
    return new RadixTree();
  }

  RadixTree* Split(std::vector<int> tokens) {
    nodeData* dummy_data = new nodeData();
    raxNode* sub_tree_root_node = raxSplit(this->tree, tokens.data(),
                                           tokens.size(), dummy_data);
    RadixTree* sub_tree = new RadixTree();
    sub_tree->tree = this->tree;
    this->sub_tree = raxNew();
    this->sub_tree->head = sub_tree_root_node;
    return sub_tree;
  }

  // Get child node list from this tree.
  std::vector<std::shared_ptr<NodeWithTreeAttri>> TraverseSubTree() {
    if (this->sub_tree == NULL) {
      LOG(INFO) << "traverse failed";
      return std::vector<std::shared_ptr<NodeWithTreeAttri>>();
    }
    std::vector<std::shared_ptr<NodeWithTreeAttri>> nodes;

    std::vector<std::shared_ptr<raxNode>> dataNodeList;
    raxNode* headNode = this->sub_tree->head;
    raxTraverseSubTree(headNode, dataNodeList);
    for (int i = 0; i < dataNodeList.size(); i++) {
      nodes.push_back(std::make_shared<NodeWithTreeAttri>(
          std::make_shared<Node>(dataNodeList[i].get()), this));
    }
    return nodes;
  }

  void* GetCustomData() { return custom_data; }

  void SetCustomData(void* custom_data, int custom_data_length) {
    this->custom_data = custom_data;
    this->custom_data_length = custom_data_length;
  }
};

#endif
