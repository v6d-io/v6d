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

#include <iostream>
#include <random>
#include <vector>

#include "common/util/logging.h"
#include "kv-state-cache/utils/kv_state_cache_utils.h"

using namespace vineyard;

#define DEMENSION 10

void init() { init_kv_state_cache(DEMENSION); }

void print_current_tokens(const std::vector<int>& prefix, int next_token) {
  std::string tokens_str = "";
  for (size_t i = 0; i < prefix.size(); ++i) {
    tokens_str += std::to_string(prefix[i]);
  }
  tokens_str += std::to_string(next_token);
  LOG(INFO) << "Current tokens: " + tokens_str;
  LOG(INFO) << tokens_str;
}

void print_kv_state(
    const std::map<int, std::pair<std::vector<double>, std::vector<double>>>&
        kv_state) {
  LOG(INFO) << "kv_state: ";
  for (auto iter = kv_state.begin(); iter != kv_state.end(); ++iter) {
    std::string key_state_str = "";
    std::string value_state_str = "";
    for (int i = 0; i < DEMENSION; ++i) {
      key_state_str += std::to_string(iter->second.first[i]) + " ";
      value_state_str += std::to_string(iter->second.second[i]) + " ";
    }
    LOG(INFO) << "key_state: " << key_state_str;
    LOG(INFO) << "value_state: " << value_state_str;
  }
}

// we do not consider the layer.
std::map<int, std::pair<std::vector<double>, std::vector<double>>>
generate_kv_state(int token) {
  std::vector<double> key_state;
  std::vector<double> value_state;
  for (int i = 0; i < DEMENSION; ++i) {
    key_state.push_back(((double) token) / DEMENSION * (i + 1));
    value_state.push_back(((double) token) / DEMENSION * (i + 1) * 2);
  }

  std::map<int, std::pair<std::vector<double>, std::vector<double>>> kv_state;
  kv_state.insert(std::make_pair(1, std::make_pair(key_state, value_state)));
  return kv_state;
}

void inference(std::vector<int> tokens) {
  LOG(INFO) << "inference";
  std::vector<int> inference_tokens;
  std::map<int, std::pair<std::vector<double>, std::vector<double>>> kv_state;

  for (size_t i = 0; i < tokens.size(); ++i) {
    kv_state = query(inference_tokens, tokens[i]);
    if (kv_state.size() == 0) {
      LOG(INFO) << "======================================";
      LOG(INFO) << "Can not find the kv_state from cache:";
      print_current_tokens(inference_tokens, tokens[i]);
      LOG(INFO) << "Generate the kv_state and update the cache.";
      kv_state = generate_kv_state(tokens[i]);
      update(inference_tokens, tokens[i], kv_state);
      print_kv_state(kv_state);
      LOG(INFO) << "======================================";
    } else {
      LOG(INFO) << "--------------------------------------";
      LOG(INFO) << "Find the kv_state from cache:";
      print_current_tokens(inference_tokens, tokens[i]);
      print_kv_state(kv_state);
      LOG(INFO) << "--------------------------------------";
    }
    inference_tokens.push_back(tokens[i]);
  }
}

int main() {
  init();
  std::vector<int> round_1_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> round_2_tokens = {1, 2, 3, 4, 5, 7, 8, 9, 10};
  inference(round_1_tokens);
  inference(round_2_tokens);
  inference(round_2_tokens);
  inference(round_1_tokens);
  return 0;
}