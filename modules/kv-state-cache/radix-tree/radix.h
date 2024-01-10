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
#include "common/util/logging.h"

class Node {
 private:
  std::vector<int> key;
  std::shared_ptr<void> data;
  int data_length;
  std::vector<Node*> children;

 public:
  void set_data(std::shared_ptr<void> data, int data_length) {
    this->data = data;
    this->data_length = data_length;
  }

  std::shared_ptr<void> get_data() { return this->data; }
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

static std::map<std::string, std::shared_ptr<Node>> storage;

class RadixTree {
 private:
  void* custom_data;
  int custom_data_length;

 public:
  RadixTree() {
    LOG(INFO) << "init radix tree";
    custom_data = NULL;
    custom_data_length = 0;
  }

  RadixTree(void* custom_data, int custom_data_length) {
    LOG(INFO) << "init radix tree with custom data";
    this->custom_data = custom_data;
    this->custom_data_length = custom_data_length;
  }

  void* GetCustomData() { return custom_data; }

  void insert(const std::vector<int> key, std::shared_ptr<void> data,
              int data_length) {
    std::shared_ptr<Node> node = std::make_shared<Node>();
    node->set_data(data, data_length);
    std::string key_str = "";
    for (size_t i = 0; i < key.size(); ++i) {
      key_str += std::to_string(key[i]);
    }
    storage.insert(std::make_pair(key_str, node));
  }

  std::shared_ptr<NodeWithTreeAttri> insert(const std::vector<int> tokens,
                                            int next_token) {
    std::shared_ptr<Node> node = std::make_shared<Node>();
    std::string key_str = "";
    for (size_t i = 0; i < tokens.size(); ++i) {
      key_str += std::to_string(tokens[i]);
    }
    key_str += std::to_string(next_token);
    storage.insert(std::make_pair(key_str, node));
    return std::make_shared<NodeWithTreeAttri>(node, this);
  }

  void Delete(const std::vector<int> tokens, int next_token) {
    std::string key_str = "";
    for (size_t i = 0; i < tokens.size(); ++i) {
      key_str += std::to_string(tokens[i]);
    }
    key_str += std::to_string(next_token);
    auto iter = storage.find(key_str);
    if (iter != storage.end()) {
      storage.erase(iter);
    }
  }

  void insert(const std::vector<int>& prefix, int key,
              std::shared_ptr<void> data, int data_length) {
    std::vector<int> key_vec = prefix;
    key_vec.push_back(key);
    insert(key_vec, data, data_length);
  }

  std::shared_ptr<NodeWithTreeAttri> get(std::vector<int> key) {
    std::string key_str = "";
    for (size_t i = 0; i < key.size(); ++i) {
      key_str += std::to_string(key[i]);
    }
    auto iter = storage.find(key_str);
    if (iter != storage.end()) {
      LOG(INFO) << "find key of :" + key_str;
      return std::make_shared<NodeWithTreeAttri>(iter->second, this);
    }
    LOG(INFO) << "cannot find key of :" + key_str;
    return nullptr;
  }

  std::shared_ptr<NodeWithTreeAttri> get(std::vector<int> prefix, int key) {
    std::vector<int> key_vec = prefix;
    key_vec.push_back(key);
    return get(key_vec);
  }

  std::string serialize() { return std::string("this is a serialized string"); }

  static RadixTree* deserialize(std::string data) {
    LOG(INFO) << "deserialize with data:" + data;
    return new RadixTree();
  }

  RadixTree* split() {
    LOG(INFO) << "splits is not implemented";
    return nullptr;
  }

  // Get child node list from this tree.
  std::vector<std::shared_ptr<NodeWithTreeAttri>> traverse() {
    std::vector<std::shared_ptr<NodeWithTreeAttri>> nodes;
    for (auto iter = storage.begin(); iter != storage.end(); ++iter) {
      nodes.push_back(std::make_shared<NodeWithTreeAttri>(iter->second, this));
    }
    return nodes;
  }

  void* get_custom_data() { return custom_data; }

  void set_custom_data(void* custom_data, int custom_data_length) {
    this->custom_data = custom_data;
    this->custom_data_length = custom_data_length;
  }
};