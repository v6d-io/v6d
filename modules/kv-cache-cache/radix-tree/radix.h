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

#include <vector>
#include <types.h>

class Node {
 private:
  std::vector<int> key;
  void *data;
  int data_length;
  std::vector<Node *> children;

 public:
  void set_data(void *data, int data_length) {
    this->data = data;
    this->data_length = data_length;
  }

  void *get_data() {
    return this->data;
  }
};

class RadixTree {
 public:

	void insert(const std::vector<int> key, void *data, int data_length) {

  }

	void insert(const std::vector<int> &prefix, int key, void *data, int data_length) {

  }

  struct Node *get(std::vector<int> key) {
    return nullptr;
  }

  struct Node *get(std::vector<int> prefix, int key) {
    return nullptr;
  }

  std::string serialize() {
    return NULL;
  }

  RadixTree deserialize(std::string data) {

  }

  void drop(std::vector<int> key) {

  }

  void *getNodeData(void *node) {
    return NULL;
  }

  void setNodeData(void *node, void *data, int data_length) {

  }
    
  RadixTree *splits() {

  }
};