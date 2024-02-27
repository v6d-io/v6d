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
#include "kv-state-cache/radix-tree/radix.h"

#include "common/util/logging.h"
#include "kv-state-cache/utils/kv_state_cache_utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int dimension = 10;
int capacity = 20;
int layer = 3;
int block_size = 5;

std::vector<int> round_1_tokens = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69};
std::vector<int> round_2_tokens = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14};
std::vector<int> round_3_tokens = {1, 2, 3, 9, 10, 11, 12, 13, 14};
std::vector<int> round_4_tokens = {1, 2, 3, 4, 5, 6};

std::vector<std::vector<int>> tokens_list;

void init(int dimension, int capacity, int layer, int block_size,
          std::string socket) {
  InitKVStateCache(dimension, capacity, layer, block_size, socket);
}

void print_current_tokens(const std::vector<int>& prefix, int next_token) {
  std::string tokens_str = "";
  for (size_t i = 0; i < prefix.size(); ++i) {
    tokens_str += std::to_string(prefix[i]) + " ";
  }
  tokens_str += std::to_string(next_token);
  LOG(INFO) << "Current tokens: " + tokens_str;
}

void print_kv_state(
    const std::map<int, std::pair<std::vector<double>, std::vector<double>>>&
        kv_state) {
  LOG(INFO) << "kv_state: ";
  for (auto iter = kv_state.begin(); iter != kv_state.end(); ++iter) {
    std::string key_state_str = "";
    std::string value_state_str = "";
    for (int i = 0; i < dimension; ++i) {
      key_state_str += std::to_string(iter->second.first[i]) + " ";
      value_state_str += std::to_string(iter->second.second[i]) + " ";
    }
    LOG(INFO) << "layer " << iter->first << ":";
    LOG(INFO) << "key_state: " << key_state_str;
    LOG(INFO) << "value_state: " << value_state_str;
    LOG(INFO) << "---------------------";
  }
}

// we do not consider the layer.
std::map<int, std::pair<std::vector<double>, std::vector<double>>>
generate_kv_state(int token) {
  std::map<int, std::pair<std::vector<double>, std::vector<double>>> kv_state;
  for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
    std::vector<double> key_state;
    std::vector<double> value_state;
    for (int i = 0; i < dimension; ++i) {
      key_state.push_back((static_cast<double>(token)) / dimension * (i + 1) +
                          currentLayer * 10);
      value_state.push_back((static_cast<double>(token)) / dimension * (i + 1) *
                                2 +
                            currentLayer * 10);
    }

    kv_state.insert(
        std::make_pair(currentLayer, std::make_pair(key_state, value_state)));
  }
  return kv_state;
}

void check_kv_state(
    const std::map<int, std::pair<std::vector<double>, std::vector<double>>>&
        kv_state,
    int& token) {
  VINEYARD_ASSERT(kv_state.size() == (size_t) layer);
  for (auto iter = kv_state.begin(); iter != kv_state.end(); ++iter) {
    VINEYARD_ASSERT(iter->second.first.size() == (size_t) dimension);
    VINEYARD_ASSERT(iter->second.second.size() == (size_t) dimension);
    for (int i = 0; i < dimension; ++i) {
      if (iter->second.first[i] !=
          (static_cast<double>(token)) / dimension * (i + 1) +
              iter->first * 10) {
        LOG(INFO) << "token:" << token << " dimension" << dimension
                  << " layer:" << iter->first;
        LOG(INFO) << "key_state[" << i << "]: " << iter->second.first[i]
                  << ". But is should be "
                  << (static_cast<double>(token)) / dimension * (i + 1) +
                         iter->first * 10;
        throw std::runtime_error("key_state error!");
      }
      if (iter->second.second[i] !=
          (static_cast<double>(token)) / dimension * (i + 1) * 2 +
              iter->first * 10) {
        LOG(INFO) << "token:" << token << " dimension" << dimension
                  << " layer:" << iter->first;
        LOG(INFO) << "value_state[" << i << "]: " << iter->second.second[i]
                  << ". But is should be "
                  << (static_cast<double>(token)) / dimension * (i + 1) * 2 +
                         iter->first * 10;
        throw std::runtime_error("value_state error!");
      }
    }
  }
}

void inference(std::vector<int> tokens, bool block = false) {
  std::vector<int> inference_tokens;
  std::map<int, std::pair<std::vector<double>, std::vector<double>>> kv_state;

  for (size_t i = 0; i < tokens.size(); ++i) {
    kv_state = Query(inference_tokens, tokens[i]);
    if (kv_state.size() == 0) {
      LOG(INFO) << "Can not find the kv_state from cache:";
      print_current_tokens(inference_tokens, tokens[i]);
      LOG(INFO) << "Generate the kv_state and update the cache.";
      kv_state = generate_kv_state(tokens[i]);
      print_kv_state(kv_state);
      Update(inference_tokens, tokens[i], kv_state);
    } else {
      LOG(INFO) << "Find the kv_state from cache:";
      print_current_tokens(inference_tokens, tokens[i]);
      check_kv_state(kv_state, tokens[i]);
    }
    LOG(INFO) << "--------------------------------------";
    inference_tokens.push_back(tokens[i]);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./kv_state_cache_test <ipc_socket_name>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "-d") == 0) {
      dimension = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-c") == 0) {
      capacity = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-l") == 0) {
      layer = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-b") == 0) {
      block_size = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-s") == 0) {
      for (int j = i + 1; j < argc; j++) {
        if (strcmp(argv[j], "1") == 0) {
          tokens_list.push_back(round_1_tokens);
        } else if (strcmp(argv[j], "2") == 0) {
          tokens_list.push_back(round_2_tokens);
        } else if (strcmp(argv[j], "3") == 0) {
          tokens_list.push_back(round_3_tokens);
        } else if (strcmp(argv[j], "4") == 0) {
          tokens_list.push_back(round_4_tokens);
        } else {
          break;
        }
      }
    }
  }

  LOG(INFO) << "Test KVStateCache with dimension: " << dimension
            << ", capacity: " << capacity << ", layer: " << layer
            << ", block_size: " << block_size << ".";

  init(dimension, capacity, layer, block_size, ipc_socket);

  for (size_t i = 0; i < tokens_list.size(); i++) {
    inference(tokens_list[i]);
  }

  sleep(5);

  for (size_t i = 0; i < tokens_list.size(); i++) {
    inference(tokens_list[i]);
  }

  LOG(INFO) << "inference end";
  CloseKVStateCache();
  LOG(INFO) << "Passed KVStateCache tests...";
  return 0;
}