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

#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

#include "llm-cache/ds/kv_state_cache_manager.h"

using namespace vineyard;  //  NOLINT(build/namespaces)

constexpr int TENSORBYTES = 800;
constexpr int CAPACITY = 1000;
constexpr int LAYER = 64;
constexpr int BLOCK_SIZE = 100;

std::shared_ptr<KVStateCacheManager> manager;
VineyardCacheConfig config(TENSORBYTES, CAPACITY, LAYER, BLOCK_SIZE, 3);
Client client;

void init(std::string socket) {
  VINEYARD_CHECK_OK(client.Connect(socket));
  VINEYARD_CHECK_OK(KVStateCacheManager::Make(client, manager, config));
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

    kv_state.insert(
        std::make_pair(currentLayer, std::make_pair(key_state, value_state)));
  }
  return kv_state;
}

// test the performance of Query and Update function
void benchmark_inference(std::vector<std::vector<int>>& tokens) {
  LOG(INFO) << "inference for benchmark";
  std::map<int, std::pair<LLMKV, LLMKV>> kv_state;

  std::chrono::steady_clock::time_point start, end;
  double token_list_size = 0;
  std::chrono::duration<double> update_duration(0);
  std::chrono::duration<double> query_duration(0);
  double total_update_duration = 0;
  double total_query_duration = 0;

  for (size_t i = 0; i < tokens.size(); ++i) {
    std::vector<int> inference_tokens;
    for (size_t j = 0; j < tokens[i].size(); ++j) {
      start = std::chrono::steady_clock::now();
      kv_state = generate_kv_state(tokens[i][j]);
      Status status = manager->Query(inference_tokens, tokens[i][j], kv_state);
      if (!status.ok()) {
        VLOG(100) << "KV state is not in the cache.";
      }
      end = std::chrono::steady_clock::now();
      query_duration += end - start;

      if (kv_state.size() == 0) {
        start = std::chrono::steady_clock::now();
        Status status =
            manager->Update(inference_tokens, tokens[i][j], kv_state);
        if (!status.ok()) {
          // Not a error. May be the cache is full.
          VLOG(100) << "Put kv state into cache failed.";
        }
        end = std::chrono::steady_clock::now();
        update_duration += end - start;
      }
      inference_tokens.push_back(tokens[i][j]);
      token_list_size++;
    }
    total_update_duration += update_duration.count();
    total_query_duration += query_duration.count();
  }

  LOG(INFO) << "Token list size is " << token_list_size
            << "Total Update time is " << total_update_duration << "s "
            << "Total Query time is " << total_query_duration << "s "
            << "Average update time is "
            << token_list_size / total_update_duration << "token/s "
            << "Average query time is "
            << token_list_size / total_query_duration << "token/s ";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./kv_state_cache_benchmark <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  init(ipc_socket);

  std::atomic<bool> inference_done(false);

  std::thread memory_monitor([&]() {
    Client client;
    size_t max_memory_usage = 0;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    while (!inference_done) {
      sleep(1);
      std::shared_ptr<InstanceStatus> status;
      VINEYARD_CHECK_OK(client.InstanceStatus(status));
      LOG(INFO) << "status->memory_usage is:" << status->memory_usage;
      if (status->memory_usage > max_memory_usage) {
        max_memory_usage = status->memory_usage;
      }
    }
    LOG(INFO) << "Max memory usage is " << max_memory_usage;
  });

  std::thread inference([&]() {
    const size_t num_lists = 10;
    std::vector<std::vector<int>> all_token_lists;
    for (size_t i = 0; i < num_lists; ++i) {
      all_token_lists.push_back(generate_random_tokens(2000));
    }

    benchmark_inference(all_token_lists);
    sleep(5);
    inference_done = true;
    manager->Close();
  });

  memory_monitor.join();
  inference.join();
  return 0;
}
