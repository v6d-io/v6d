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

extern "C" {
#include "radix.h"
}
#include "common/util/logging.h"

#include <map>
#include <memory>
#include <vector>

typedef struct nodeData {
  std::shared_ptr<void> data;
  int data_length;
} nodeData;

class Node {
 private:
  raxNode* node;
  std::shared_ptr<void> data;
  int data_length;

 public:
  Node(raxNode* node) {
    this->node = node;
    this->data = NULL;
    this->data_length = 0;
  }
  void set_data(std::shared_ptr<void> data, int data_length) {
    nodeData* node_data = new nodeData();
    node_data->data = data;
    node_data->data_length = data_length;
    raxSetData(this->node, node_data);
  }

  std::shared_ptr<void> get_data() {
    nodeData* nodedata = (nodeData*) raxGetData(this->node);
    return nodedata->data;
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
  rax* tree;

 public:
  RadixTree() {
    LOG(INFO) << "init radix tree";
    this->tree = raxNew();
    this->custom_data = NULL;
    this->custom_data_length = 0;
  }

  RadixTree(void* custom_data, int custom_data_length) {
    LOG(INFO) << "init radix tree with custom data";
    this->tree = raxNew();
    this->custom_data = custom_data;
    this->custom_data_length = custom_data_length;
  }

  // void insert(const std::vector<int> key, void** data, int data_length) {
  //   const int* tokens = key.data();
  //   size_t tokens_len = key.size();

  //   nodeData *insert_data = new nodeData();
  //   ((nodeData*)data)->data = data;
  //   ((nodeData*)data)->data_length = data_length;
  //   int retval = raxInsert(this->tree, tokens, tokens_len, insert_data,
  //   NULL); if (retval == 0) {
  //     if (errno == 0) {
  //       LOG(INFO) << "overwrite an existing token list";
  //     } else {
  //       LOG(INFO) << "insert failed with errno:" + std::to_string(errno);
  //     }
  //   } else {
  //       LOG(INFO) << "insert success";
  //   }
  // }

  std::shared_ptr<NodeWithTreeAttri> Insert(std::vector<int> tokens) {
    // insert the token vector to the radix tree
    int* insert_tokens_array = tokens.data();
    size_t insert_tokens_array_len = tokens.size();
    nodeData* dummy_data = new nodeData();
    raxNode* dataNode =
        raxInsertAndReturnDataNode(this->tree, insert_tokens_array,
                                   insert_tokens_array_len, dummy_data, NULL);
    if (dataNode == NULL) {
      LOG(INFO) << "insert failed";
      return NULL;
    }
    LOG(INFO) << "insert success";
    // return new NodeWithTreeAttri(new Node(dataNode), this);
    return std::make_shared<NodeWithTreeAttri>(std::make_shared<Node>(dataNode),
                                               this);
  }

  void Delete(std::vector<int> tokens) {
    // remove the token vector from the radix tree
    int* delete_tokens_array = tokens.data();
    size_t delete_tokens_array_len = tokens.size();
    int retval = raxRemove(this->tree, delete_tokens_array,
                           delete_tokens_array_len, NULL);
    if (retval == 1) {
      LOG(INFO) << "remove success";
    } else {
      LOG(INFO) << "remove failed";
    }
  }

  // void insert(const std::vector<int>& prefix, int key, void** data,
  //             int data_length) {
  //   std::vector<int> key_vec = prefix;
  //   key_vec.push_back(key);

  //   const int* tokens = key_vec.data();
  //   size_t tokens_len = key_vec.size();

  //   nodeData *insert_data = new nodeData();
  //   ((nodeData*)data)->data = data;
  //   ((nodeData*)data)->data_length = data_length;
  //   int retval = raxInsert(this->tree, tokens, tokens_len, insert_data,
  //   NULL); if (retval == 0) {
  //     if (errno == 0) {
  //       LOG(INFO) << "overwrite an existing token list";
  //     } else {
  //       LOG(INFO) << "insert failed with errno:" + std::to_string(errno);
  //     }
  //   } else {
  //       LOG(INFO) << "insert success";
  //   }
  // }

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
    // return new NodeWithTreeAttri(new Node(dataNode), this);
    return std::make_shared<NodeWithTreeAttri>(std::make_shared<Node>(dataNode),
                                               this);
  }

  // std::shared_ptr<NodeWithTreeAttri> get(std::vector<int> prefix, int key) {
  //   std::vector<int> key_vec = prefix;
  //   key_vec.push_back(key);
  //   return get(key_vec);
  // }

  std::string serialize() { return std::string("this is a serialized string"); }

  static RadixTree* Deserialize(std::string data) {
    LOG(INFO) << "deserialize with data:" + data;
    return new RadixTree();
  }

  RadixTree* Split() {
    LOG(INFO) << "splits is not implemented";
    return this;
  }

  // Get child node list from this tree.
  std::vector<std::shared_ptr<NodeWithTreeAttri>> Travel() {
    if (this->tree == NULL) {
      LOG(INFO) << "traverse failed";
      return std::vector<std::shared_ptr<NodeWithTreeAttri>>();
    }
    // std::vector<NodeWithTreeAttri *> nodes;
    std::vector<std::shared_ptr<NodeWithTreeAttri>> nodes;

    int numele = this->tree->numele;
    raxNode** dataNodeList = (raxNode**) malloc(sizeof(raxNode*) * (numele));
    raxNode** current = dataNodeList;
    raxNode* headNode = this->tree->head;
    raxTraverse(headNode, &dataNodeList);
    for (int i = 0; i < numele; i++, current++) {
      // nodes.push_back(new NodeWithTreeAttri(new Node(*current), this));
      nodes.push_back(std::make_shared<NodeWithTreeAttri>(
          std::make_shared<Node>(*current), this));
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
