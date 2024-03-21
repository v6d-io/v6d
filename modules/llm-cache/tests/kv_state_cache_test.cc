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
#include "llm-cache/ds/config.h"
#include "llm-cache/ds/kv_state_cache_manager.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int tensorBytes = 80;
int capacity = 20;
int layer = 3;
int block_size = 5;

VineyardCacheConfig config;
std::string llmCacheObjectName = "cache_test_cache_object";
std::string llmCacheSyncLock = "cache_test_cache_lock";
std::string llmRefcntObjectName = "cache_test_refcnt_object";

std::vector<int> round_1_tokens = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69};
std::vector<int> round_2_tokens = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14};
std::vector<int> round_3_tokens = {1, 2, 3, 9, 10, 11, 12, 13, 14};
std::vector<int> round_4_tokens = {1, 2, 3, 4, 5, 6};

std::vector<std::vector<int>> tokens_list = {round_1_tokens, round_2_tokens,
                                             round_3_tokens, round_4_tokens};

std::shared_ptr<KVStateCacheManager> init(Client& client) {
  std::shared_ptr<KVStateCacheManager> kv_state_cache_manager;
  VINEYARD_CHECK_OK(
      KVStateCacheManager::Make(client, kv_state_cache_manager, config));
  return kv_state_cache_manager;
}

void print_current_tokens(const std::vector<int>& prefix, int next_token) {
  std::string tokens_str = "";
  for (size_t i = 0; i < prefix.size(); ++i) {
    tokens_str += std::to_string(prefix[i]) + " ";
  }
  tokens_str += std::to_string(next_token);
  LOG(INFO) << "Current tokens: " + tokens_str;
}

void print_kv_state(const std::map<int, std::pair<LLMKV, LLMKV>>& kv_state) {
  LOG(INFO) << "kv_state: ";
  for (auto iter = kv_state.begin(); iter != kv_state.end(); ++iter) {
    uint8_t* key_state_data =
        reinterpret_cast<uint8_t*>(iter->second.first.data);
    uint8_t* value_state_data =
        reinterpret_cast<uint8_t*>(iter->second.second.data);
    // print the first tensorBytes bytes
    std::string key_state_str = "";
    std::string value_state_str = "";
    for (int j = 0; j < tensorBytes; j++) {
      key_state_str += std::to_string(key_state_data[j]) + " ";
      value_state_str += std::to_string(value_state_data[j]) + " ";
    }
    LOG(INFO) << "layer " << iter->first << ":";
    LOG(INFO) << "key_state: " << key_state_str;
    LOG(INFO) << "value_state: " << value_state_str;
    LOG(INFO) << "---------------------";
  }
}

// we do not consider the layer.
std::map<int, std::pair<LLMKV, LLMKV>> generate_kv_state(int token) {
  std::map<int, std::pair<LLMKV, LLMKV>> kv_state;
  for (int currentLayer = 0; currentLayer < layer; currentLayer++) {
    LLMKV key_state;
    LLMKV value_state;
    key_state.data = malloc(tensorBytes);
    value_state.data = malloc(tensorBytes);

    key_state.length = tensorBytes;
    value_state.length = tensorBytes;

    for (int i = 0; i < tensorBytes; ++i) {
      (reinterpret_cast<uint8_t*>(key_state.data))[i] =
          (static_cast<uint8_t>(token)) + i + currentLayer;
      (reinterpret_cast<uint8_t*>(value_state.data))[i] =
          (static_cast<uint8_t>(token)) + i + currentLayer;
    }
    kv_state.insert(
        std::make_pair(currentLayer, std::make_pair(key_state, value_state)));
  }
  return kv_state;
}

void check_kv_state(const std::map<int, std::pair<LLMKV, LLMKV>>& kv_state,
                    int& token) {
  VINEYARD_ASSERT(kv_state.size() == (size_t) layer);
  for (auto iter = kv_state.begin(); iter != kv_state.end(); ++iter) {
    VINEYARD_ASSERT(iter->second.first.length == (size_t) tensorBytes);
    VINEYARD_ASSERT(iter->second.second.length == (size_t) tensorBytes);
    for (int i = 0; i < tensorBytes; ++i) {
      if ((reinterpret_cast<uint8_t*>(iter->second.first.data))[i] !=
          (static_cast<uint8_t>(token)) + i + iter->first) {
        LOG(INFO) << "token:" << token << " tensorBytes" << tensorBytes
                  << " layer:" << iter->first;
        LOG(INFO) << "key_state[" << i << "]: "
                  << (reinterpret_cast<uint8_t*>(iter->second.first.data))[i]
                  << ". But is should be "
                  << (static_cast<uint8_t>(token)) + i + iter->first;
        throw std::runtime_error("key_state error!");
      }
      if (reinterpret_cast<uint8_t*>(iter->second.second.data)[i] !=
          (static_cast<uint8_t>(token)) + i + iter->first) {
        LOG(INFO) << "token:" << token << " tensorBytes" << tensorBytes
                  << " layer:" << iter->first;
        LOG(INFO) << "value_state[" << i << "]: "
                  << (reinterpret_cast<uint8_t*>(iter->second.second.data))[i]
                  << ". But is should be "
                  << (static_cast<uint8_t>(token)) + i + iter->first * 10;
        throw std::runtime_error("value_state error!");
      }
    }
  }
}

void inference(std::shared_ptr<KVStateCacheManager>& kv_state_cache_manager,
               std::vector<int> tokens, bool block = false) {
  std::vector<int> inference_tokens;
  std::map<int, std::pair<LLMKV, LLMKV>> kv_state;
  for (size_t i = 0; i < tokens.size(); ++i) {
    kv_state.clear();
    Status result =
        kv_state_cache_manager->Query(inference_tokens, tokens[i], kv_state);
    if (!result.ok() || kv_state.empty()) {
      LOG(INFO) << "Can not find the kv_state from cache:";
      print_current_tokens(inference_tokens, tokens[i]);
      LOG(INFO) << "Generate the kv_state and update the cache.";
      kv_state = generate_kv_state(tokens[i]);
      print_kv_state(kv_state);
      Status status =
          kv_state_cache_manager->Update(inference_tokens, tokens[i], kv_state);
      if (!status.ok()) {
        // Not a error. May be the cache is full.
        VLOG(100) << "Put kv state into cache failed.";
      }
    } else {
      LOG(INFO) << "Find the kv_state from cache:";
      print_current_tokens(inference_tokens, tokens[i]);
      check_kv_state(kv_state, tokens[i]);
    }
    LOG(INFO) << "--------------------------------------";
    inference_tokens.push_back(tokens[i]);
  }
}

void threadFunc(std::string socket) {
  Client client;
  VINEYARD_CHECK_OK(client.Connect(socket));
  std::shared_ptr<KVStateCacheManager> manager = init(client);

  for (size_t i = 0; i < tokens_list.size(); i++) {
    inference(manager, tokens_list[i]);
  }

  sleep(5);

  for (size_t i = 0; i < tokens_list.size(); i++) {
    inference(manager, tokens_list[i]);
  }

  LOG(INFO) << "inference end";

  manager->Close();
  client.Disconnect();
}

void clearGlobalObject(std::vector<std::string>& sockets) {
  Client client;
  VINEYARD_CHECK_OK(client.Connect(sockets[0]));

  VINEYARD_CHECK_OK(KVStateCacheManager::ClearGlobalCache(client, config));
  client.Disconnect();

  for (size_t i = 0; i < sockets.size(); i++) {
    Client client;
    VINEYARD_CHECK_OK(client.Connect(sockets[i]));
    std::shared_ptr<InstanceStatus> status;
    VINEYARD_CHECK_OK(client.InstanceStatus(status));

    if (status->memory_usage != 0) {
      std::vector<ObjectMeta> objects;
      std::vector<ObjectMeta> metas = client.ListObjectMeta(".*", true);
      LOG(INFO) << "Object:";
      for (size_t i = 0; i < metas.size(); i++) {
        LOG(INFO) << metas[i].ToString();
      }
      // VINEYARD_ASSERT(false);
    }
    client.Disconnect();
  }
}

int main(int argc, char** argv) {
  std::vector<std::string> sockets;
  if (argc < 2) {
    printf(
        "usage ./kv_state_cache_test --client-num <client_num> "
        "--vineyard-ipc-sockets <ipc_socket_1> ... <ipc_socket_n> -d "
        "<tensorBytes> -c <capacity> -l <layer> -b <blockSize>\n");
    return 1;
  }

  if (strcmp(argv[1], "--client-num") != 0) {
    return 1;
  }

  int client_num = std::stoi(argv[2]);

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "-d") == 0) {
      tensorBytes = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-c") == 0) {
      capacity = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-l") == 0) {
      layer = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-b") == 0) {
      block_size = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-s") == 0) {
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
    } else if (strcmp(argv[i], "--vineyard-ipc-sockets") == 0) {
      for (int j = 0; j < client_num; j++) {
        sockets.push_back(std::string(argv[i + j + 1]));
      }
    }
  }

  LOG(INFO) << "Test KVStateCache with tensorBytes: " << tensorBytes
            << ", capacity: " << capacity << ", layer: " << layer
            << ", block_size: " << block_size << " and use " << client_num
            << " client.";

  config = VineyardCacheConfig(tensorBytes, capacity, layer, block_size, 3,
                               llmCacheSyncLock, llmCacheObjectName,
                               llmRefcntObjectName);

  std::vector<std::thread> threads;
  for (int i = 0; i < client_num; i++) {
    threads.push_back(std::thread(threadFunc, sockets[i]));
  }

  for (int i = 0; i < client_num; i++) {
    threads[i].join();
    LOG(INFO) << "Thread:" << i << " exit.";
  }

  sleep(5);
  clearGlobalObject(sockets);

  size_t total_memory_usage = 0;
  for (size_t i = 0; i < sockets.size(); i++) {
    Client client;
    VINEYARD_CHECK_OK(client.Connect(sockets[i]));
    std::shared_ptr<InstanceStatus> status;
    VINEYARD_CHECK_OK(client.InstanceStatus(status));
    LOG(INFO) << "Client " << client.instance_id()
              << " memory usage:" << status->memory_usage;
    total_memory_usage += status->memory_usage;
    client.Disconnect();
  }
  LOG(INFO) << "Total memory usage:" << total_memory_usage;

  LOG(INFO) << "Passed KVStateCache tests...";
  return 0;
}
