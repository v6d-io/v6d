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

#include <unistd.h>
#include <iostream>
#include <random>
#include <vector>

#include "rax/radix.h"

#include "common/util/logging.h"
#include "llm-cache/ds/kv_state_cache_manager.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void print_tokens(const std::vector<int>& tokens) {
  std::string tokens_str = "";
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens_str += std::to_string(tokens[i]);
  }
  LOG(INFO) << "Current tokens: " + tokens_str;
}

void radix_tree_insert_test() {
  std::shared_ptr<RadixTree> radix_tree = std::make_shared<RadixTree>(10);

  /* insert a token list*/
  std::vector<int> tokens;
  std::shared_ptr<NodeData> node_data;
  for (int i = 0; i < 10; i++) {
    tokens.push_back(i);
    VINEYARD_ASSERT(radix_tree->Insert(tokens, node_data) != NULL);
    VINEYARD_ASSERT(node_data == NULL);
  }
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(radix_tree->tree);
  }
  /* insert new token and check whether the old token is evicted */
  tokens.clear();
  for (int i = 1; i < 10; i++) {
    tokens.push_back(i);
    std::vector<int> tokens_copy = tokens;
    VINEYARD_ASSERT(radix_tree->Insert(tokens_copy, node_data) != NULL);
    VINEYARD_ASSERT(node_data != NULL);
  }

  /* insert a token that prefix is not in the radix tree */
  tokens.clear();
  for (int i = 10; i > 0; i--) {
    tokens.push_back(i);
  }
  VINEYARD_ASSERT(radix_tree->Insert(tokens, node_data) == NULL);
}

void radix_tree_delete_test() {
  std::shared_ptr<RadixTree> radix_tree = std::make_shared<RadixTree>(10);

  /* insert a token list*/
  std::vector<int> tokens;
  std::shared_ptr<NodeData> node_data;
  for (int i = 0; i < 10; i++) {
    tokens.push_back(i);
    std::vector<int> tokens_copy = tokens;
    VINEYARD_ASSERT(radix_tree->Insert(tokens_copy, node_data) != NULL);
    VINEYARD_ASSERT(node_data == NULL);
  }

  /* delete a token list*/

  tokens.clear();
  node_data = NULL;
  for (int i = 0; i < 5; i++) {
    tokens.push_back(i);
  }
  radix_tree->Delete(tokens, node_data);
  VINEYARD_ASSERT(node_data != NULL);

  /* delete a token list that is not in the radix tree */
  tokens.clear();
  node_data = NULL;
  for (int i = 10; i > 0; i--) {
    tokens.push_back(i);
  }
  radix_tree->Delete(tokens, node_data);
  VINEYARD_ASSERT(node_data == NULL);
}

void radix_tree_query_test() {
  std::shared_ptr<RadixTree> radix_tree = std::make_shared<RadixTree>(10);

  /* insert a token list*/
  std::vector<int> tokens;
  std::shared_ptr<NodeData> node_data;
  for (int i = 0; i < 10; i++) {
    tokens.push_back(i);
    std::vector<int> tokens_copy = tokens;
    VINEYARD_ASSERT(radix_tree->Insert(tokens_copy, node_data) != NULL);
    VINEYARD_ASSERT(node_data == NULL);
  }

  /* query a token list*/
  tokens.clear();
  for (int i = 0; i < 5; i++) {
    tokens.push_back(i);
  }
  VINEYARD_ASSERT(radix_tree->Query(tokens) != NULL);

  /* query a token list that is not in the radix tree */
  tokens.clear();
  for (int i = 10; i > 0; i--) {
    tokens.push_back(i);
  }
  VINEYARD_ASSERT(radix_tree->Query(tokens) == NULL);
}

void radix_tree_serialize_and_deserialize() {
  std::shared_ptr<RadixTree> radix_tree = std::make_shared<RadixTree>(10);

  /* insert a token list*/
  std::vector<int> tokens;
  std::shared_ptr<NodeData> node_data;
  for (int i = 0; i < 10; i++) {
    tokens.push_back(i);
    std::vector<int> tokens_copy = tokens;
    VINEYARD_ASSERT(radix_tree->Insert(tokens_copy, node_data) != NULL);
    VINEYARD_ASSERT(node_data == NULL);
  }

  /* serialize radix tree */
  std::string serialized_radix_tree = radix_tree->Serialize();

  /* deserialize radix tree */
  std::shared_ptr<RadixTree> deserialized_radix_tree =
      radix_tree->Deserialize(serialized_radix_tree);

  /* query to check whether all token list exist */
  tokens.clear();
  for (int i = 0; i < 10; i++) {
    tokens.push_back(i);
    std::vector<int> tokens_copy = tokens;
    print_tokens(tokens);
    VINEYARD_ASSERT(deserialized_radix_tree->Query(tokens_copy) != NULL);
  }
}

void radix_tree_split() {
  std::shared_ptr<RadixTree> radix_tree = std::make_shared<RadixTree>(20);

  /* insert a token list*/
  std::vector<int> tokens;
  std::shared_ptr<NodeData> node_data;
  for (int i = 0; i < 10; i++) {
    tokens.push_back(i);
    std::vector<int> tokens_copy = tokens;
    VINEYARD_ASSERT(radix_tree->Insert(tokens_copy, node_data) != NULL);
    VINEYARD_ASSERT(node_data == NULL);
  }

  /* split a token list*/
  tokens.clear();
  for (int i = 0; i < 5; i++) {
    tokens.push_back(i);
  }
  print_tokens(tokens);
  std::shared_ptr<NodeData> subTreeHeader;
  std::vector<std::shared_ptr<NodeData>> node_data_list =
      radix_tree->Split(tokens, subTreeHeader);
  VINEYARD_ASSERT(node_data_list.size() == 7);
}

int main() {
  LOG(INFO) << "Start to test radix tree insert...";
  radix_tree_insert_test();
  LOG(INFO) << "Finish radix tree insert test!";
  LOG(INFO) << "Start to test radix tree delete...";
  radix_tree_delete_test();
  LOG(INFO) << "Finish radix tree delete test!";
  LOG(INFO) << "Start to test radix tree query...";
  radix_tree_query_test();
  LOG(INFO) << "Finish radix tree query test!";
  LOG(INFO) << "Start to test radix tree serialize and deserialize...";
  radix_tree_serialize_and_deserialize();
  LOG(INFO) << "Finish radix tree serialize and deserialize test!";
  LOG(INFO) << "Start to test radix tree split...";
  radix_tree_split();
  LOG(INFO) << "Finish radix tree split test!";
}
