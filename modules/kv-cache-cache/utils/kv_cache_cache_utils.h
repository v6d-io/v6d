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
#include <map>

#ifndef MODULES_KV_STATE_CACHE_UTILS_H_
#define MODULES_KV_STATE_CACHE_UTILS_H_

void Update(const std::vector<int> &token_list, int next_token, const std::map<int, std::vector<std::vector<double>, std::vector<double>>> &kv_state);

void Update(const std::vector<int> &token_list, const std::vector<std::map<int, std::pair<std::vector<double>, std::vector<double>>>> &kv_state);

std::vector<std::map<double, std::pair<std::vector<double>, std::vector<int>>>> Query(const std::vector<int> &token_list);

std::map<int, std::pair<std::vector<double>, std::vector<double>>> Query(const std::vector<int> &token_list, int token);

#endif