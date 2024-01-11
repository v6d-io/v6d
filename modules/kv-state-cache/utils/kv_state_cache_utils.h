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

void init_kv_state_cache(int dimension = 10, int cache_capacity = 10);

void update(const std::vector<int>& token_list, int next_token,
            const KV_STATE_WITH_LAYER& kv_state);

void update(const std::vector<int>& token_list,
            const LIST_KV_STATE_WITH_LAYER& kv_state);

KV_STATE_WITH_LAYER query(const std::vector<int>& token_list, int token);

LIST_KV_STATE_WITH_LAYER query(const std::vector<int>& token_list);

#endif