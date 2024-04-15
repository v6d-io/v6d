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

#ifndef MODULES_LLM_CACHE_STORAGE_STORAGE_H_
#define MODULES_LLM_CACHE_STORAGE_STORAGE_H_

#include <map>
#include <utility>
#include <vector>

#include "common/util/status.h"
#include "llm-cache/ds/kv_state_cache_block.h"

namespace vineyard {

class IStorage {
 public:
  virtual ~IStorage() {}

  virtual Status Update(
      const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
      size_t& updated) = 0;

  virtual Status Update(
      const std::vector<int>& tokenList, int nextToken,
      const std::vector<std::pair<LLMKV, LLMKV>>& kvState) = 0;

  virtual Status Update(
      const std::vector<int>& prefix, const std::vector<int>& tokenList,
      const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
      size_t& updated) = 0;

  virtual Status Query(
      const std::vector<int>& tokenList,
      std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvStateList,
      size_t& matched) = 0;

  virtual Status Query(const std::vector<int>& tokenList, int nextToken,
                       std::vector<std::pair<LLMKV, LLMKV>>& kvState) = 0;

  virtual void CloseCache() = 0;

  virtual void StartGlobalGCThread() {}

  virtual void StopGlobalGCThread() {}
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_STORAGE_STORAGE_H_
