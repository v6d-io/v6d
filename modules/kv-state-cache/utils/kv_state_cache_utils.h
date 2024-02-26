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

#include "kv-state-cache/ds/kv_state_cache.h"

#ifndef MODULES_KV_STATE_CACHE_UTILS_H_
#define MODULES_KV_STATE_CACHE_UTILS_H_

void InitKVStateCache(
    int dimension = 10, int cacheCapacity = 10, int layer = 1,
    int blockSize = 5,
    std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET")));

void Update(const std::vector<int>& tokenList, int nextToken,
            const KV_STATE_WITH_LAYER& kvState);

void Update(const std::vector<int>& tokenList,
            const LIST_KV_STATE_WITH_LAYER& kvState);

KV_STATE_WITH_LAYER Query(const std::vector<int>& tokenList, int token);

LIST_KV_STATE_WITH_LAYER Query(const std::vector<int>& tokenList);

void Delete(std::vector<int> token);

void CloseKVStateCache();

#endif