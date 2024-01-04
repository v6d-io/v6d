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

namespace vineyard {

class node {

};

class RadixTree {
 public:

	void insert(std::vector<int> key, void *data, int data_length) {

  }

  struct node *get(std::vector<int> key) {

  }

  void *serialize() {
    return NULL;
  }

  void deserialize(void *data) {

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

}  // namespace vineyard