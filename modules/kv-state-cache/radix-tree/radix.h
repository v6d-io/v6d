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
  void* data;
  int data_length;
  std::vector<Node*> children;

 public:
  void set_data(void* data, int data_length) {
    this->data = data;
    this->data_length = data_length;
  }

  void* get_data() { return this->data; }
};

static std::map<std::string, struct Node*> storage;

class RadixTree {
 public:
  void insert(const std::vector<int> key, void* data, int data_length) {
    Node* node = new Node();
    node->set_data(data, data_length);
    std::string key_str = "";
    for (int i = 0; i < key.size(); ++i) {
      key_str += std::to_string(key[i]);
    }
    storage.insert(std::make_pair(key_str, node));
  }

  void insert(const std::vector<int>& prefix, int key, void* data,
              int data_length) {
    std::vector<int> key_vec = prefix;
    key_vec.push_back(key);
    insert(key_vec, data, data_length);
  }

  struct Node* get(std::vector<int> key) {
    std::string key_str = "";
    for (int i = 0; i < key.size(); ++i) {
      key_str += std::to_string(key[i]);
    }
    auto iter = storage.find(key_str);
    if (iter != storage.end()) {
      LOG(INFO) << "find key of :" + key_str;
      return iter->second;
    }
    LOG(INFO) << "cannot find key of :" + key_str;
    return nullptr;
  }

  struct Node* get(std::vector<int> prefix, int key) {
    std::vector<int> key_vec = prefix;
    key_vec.push_back(key);
    return get(key_vec);
  }

  std::string serialize() { return std::string("this is a serialized string"); }

  static RadixTree *deserialize(std::string data) {
    LOG(INFO) << "deserialize with data:" + data;
    return new RadixTree();
  }

  void drop(std::vector<int> key) {
    std::string key_str = "";
    for (int i = 0; i < key.size(); ++i) {
      key_str += std::to_string(key[i]);
    }
    auto iter = storage.find(key_str);
    if (iter != storage.end()) {
      storage.erase(iter);
    }
  }

  void* getNodeData(void* node) { return NULL; }

  void setNodeData(void* node, void* data, int data_length) {
    ((Node*) node)->set_data(data, data_length);
  }

  RadixTree* split() {
    LOG(INFO) << "splits is not implemented";
    return nullptr;
  }

  std::vector<Node *> traverse() {
    std::vector<Node *> nodes;
    for (auto iter = storage.begin(); iter != storage.end(); ++iter) {
      nodes.push_back(iter->second);
    }
    return nodes;
  }

  bool find_insert_position(std::vector<int> token_list, int next_token, RadixTree *&node) {
    // if the position is in the subtree, return false
    // else return true
  }
};