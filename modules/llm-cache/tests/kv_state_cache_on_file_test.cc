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

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "common/util/logging.h"

#include "llm-cache/ds/config.h"
#include "llm-cache/ds/kv_state_cache_manager.h"

using namespace vineyard;  //  NOLINT(build/namespaces)

constexpr int BATCHSIZE = 16;
constexpr int SPLITNUMBER = 3;
constexpr int LAYER = 64;
constexpr int TENSORBYTES = 800;
const std::string PATH = "/tmp/vineyard/llm";

std::shared_ptr<KVStateCacheManager> manager;

void init() {
  FileCacheConfig config;
  config.batchSize = BATCHSIZE;
  config.root = PATH;
  config.splitNumber = SPLITNUMBER;
  config.layer = LAYER;
  VINEYARD_CHECK_OK(KVStateCacheManager::Make(manager, config));
}

std::vector<int> generate_random_tokens(size_t max_length) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 10000);

  size_t length = dist(gen) % max_length + 1;
  std::vector<int> tokens(length);
  for (size_t i = 0; i < length; ++i) {
    tokens[i] = dist(gen);
  }
  return tokens;
}

std::map<int, std::pair<LLMKV, LLMKV>> generate_kv_state(int token) {
  std::map<int, std::pair<LLMKV, LLMKV>> kv_state;
  for (int currentLayer = 0; currentLayer < LAYER; currentLayer++) {
    LLMKV key_state;
    LLMKV value_state;
    key_state.data = malloc(TENSORBYTES);
    key_state.length = TENSORBYTES;
    value_state.data = malloc(TENSORBYTES);
    value_state.length = TENSORBYTES;

    for (int i = 0; i < TENSORBYTES; i++) {
      (reinterpret_cast<char*>(key_state.data))[i] = static_cast<char>(token);
      (reinterpret_cast<char*>(value_state.data))[i] = static_cast<char>(token);
    }
    kv_state.insert(
        std::make_pair(currentLayer, std::make_pair(key_state, value_state)));
  }
  return kv_state;
}

void inference(std::vector<std::vector<int>>& tokens) {
  LOG(INFO) << "inference...";
  std::map<int, std::pair<LLMKV, LLMKV>> kv_state;

  std::vector<std::map<int, std::pair<LLMKV, LLMKV>>> kv_state_list;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::vector<int> inference_tokens;
    kv_state_list.clear();
    for (size_t j = 0; j < tokens[i].size(); ++j) {
      kv_state = generate_kv_state(tokens[i][j]);
      kv_state_list.push_back(kv_state);
      inference_tokens.push_back(tokens[i][j]);
    }
    Status status = manager->Update(inference_tokens, kv_state_list);
  }

  /*
 for (size_t i = 0; i < tokens.size(); ++i) {
  std::vector<int> inference_tokens;
  for (size_t j = 0; j < tokens[i].size(); ++j) {
    kv_state_list.push_back(kv_state);
    inference_tokens.push_back(tokens[i][j]);
  }
  Status status = manager->Query(inference_tokens, kv_state_list);
 }
 */
}

int main(int argc, char** argv) {
  init();

  const size_t num_lists = 10;
  std::vector<std::vector<int>> all_token_lists;
  for (size_t i = 0; i < num_lists; ++i) {
    all_token_lists.push_back(generate_random_tokens(200));
  }

  inference(all_token_lists);

  LOG(INFO) << "Passed the kv cache on file test.";
  return 0;
}
