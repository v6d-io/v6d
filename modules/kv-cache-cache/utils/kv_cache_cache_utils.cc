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

#include <cstdlib>
#include "modules/kv-cache-cache/ds/kv_state_cache.h"
#include "common/util/status.h"

static vineyard::KVStateCacheBuilder *kv_state_cache_builder;
static vineyard::Client client;

__attribute((constructor)) void init() {
    if (kv_state_cache_builder == nullptr) {
        kv_state_cache_builder = new vineyard::KVStateCacheBuilder();
        // client.Connect()
    }
}

void Update(const std::vector<int> &token_list, int next_token, const std::map<int, std::vector<std::vector<double>, std::vector<double>>> &kv_state) {
    vineyard::Status status = kv_state_cache_builder->Update(client, token_list, next_token, kv_state);
    if (!status.ok()) {
        // TBD. Check if the cache is full.
    }
}

void Update(const std::vector<int> &token_list, const std::vector<std::map<int, std::pair<std::vector<double>, std::vector<double>>>> &kv_state) {
    vineyard::Status status = kv_state_cache_builder->Update(client, token_list, kv_state);
    if (status.ok()) {
        // TBD. Check if the cache is full.
    }
}

std::vector<std::map<int, std::pair<std::vector<double>, std::vector<double>>>> Query(const std::vector<int> &token_list) {
    std::vector<std::map<int, std::pair<std::vector<double>, std::vector<double>>>> kv_state;
    kv_state_cache_builder->Query(client, token_list, kv_state);
    return kv_state;
}

std::map<int, std::pair<std::vector<double>, std::vector<double>>> Query(const std::vector<int> &token_list, int token) {
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> kv_state;
    kv_state_cache_builder->Query(client, token_list, token, kv_state);
    return kv_state;
}