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

#ifndef MODULES_KV_CACHE_CACHE_H_
#define MODULES_KV_CACHE_CACHE_H_

#include <iostream>
#include <vector>
#include <map>
#include <array>

#include "modules/basic/ds/tensor.h"
#include "client/ds/i_object.h"
#include "modules/kv-cache-cache/radix-tree/radix.h"

namespace vineyard {

struct offset_data {
  short offset_k;
  short offset_v;
};

#define LIST_SIZE 50

/**
 * @brief KVCacheCache is a cache for kv-cache of LLM. When a new prompt comes, LLM can
 * query KVCacheCache to get the state of the kv-cache to avoid caclulating the kv-cache
 * again if the new prompt is similar to the previous one.
 * 
 * KVCacheCache is stored in vineyard as a vineyard object which contains a radix tree.
 * The token sequence is the key of the radix tree and the value point out the offset
 * of the kv-cache in the tensor list.
 * 
 * KVCacheCache can be shared by multiple machines.
 */

class KVCacheCache : public vineyard::Registered<KVCacheCache> {
 private:
  RadixTree tree;
  TensorBuilder<int> *key_state_builder;
  TensorBuilder<int> *value_state_builder;

 public:
  void Construct(const ObjectMeta& meta) override;
};

class KVCacheCacheBuilder : public vineyard::ObjectBuilder {
 private:
  RadixTree tree;
  std::array<TensorBuilder<int> *, LIST_SIZE> key_state_builder_array;
  std::array<TensorBuilder<int> *, LIST_SIZE> value_state_builder_array;

  /**
   * @brief Splits the radix-tree into several radix-trees if the number of kv-state
   * is larger than LIST_SIZE.
   */
  Status Splits();

  /**
   * @brief Travel the radix-tree and update the kv-state when splitting the radix-tree.
   */
  Status TravelAndUpdateNode();

 public:
  KVCacheCacheBuilder();

  /**
   * @brief Update the kv-state using next token.
   * 
   * @param token_list The token list of the prompt.
   * @param next_token The next token of the prompt.
   * @param kv_state The kv-state of the prompt. A LLM inference can contain multiple kv-states
   * for each layer.
   */
  Status Update(const std::vector<int> &token_list, int next_token, const std::map<int, std::vector<std::vector<int>, std::vector<int>>> &kv_state);

  /**
   * @brief Update the kv-state using the whole token list.
   * 
   * @param token_list The token list of the prompt.
   * @param kv_state The kv-state of the prompt. A LLM inference can contain multiple kv-states
   * for each layer.
   */
  Status Update(const std::vector<int> &token_list, const std::vector<std::map<int, std::pair<std::vector<int>, std::vector<int>>>> &kv_state);

  /**
   * @brief Query the kv-state using the whole token list.
   * 
   * @param token_list The token list of the prompt.
   * @param kv_state The kv-state of the prompt returned by radix-tree. If the kv-state is not
   * found, the kv-state will be empty.
   */
  Status Query(const std::vector<int> &token_list, std::vector<std::map<int, std::pair<std::vector<int>, std::vector<int>>>> &kv_state);

  /**
   * @brief Query the kv-state using the whole token list.
   * 
   * @param token_list The token list of the prompt.
   * @param token The token of the prompt.
   * @param kv_state The kv-state of the prompt returned by radix-tree. If the kv-state is not
   * found, the kv-state will be empty.
   */
  Status Query(const std::vector<int> &token_list, int token, std::map<int, std::pair<std::vector<int>, std::vector<int>>> &kv_state);

  Status Build(Client &client) override;

  std::shared_ptr<Object> _Seal(Client& client) override;
};

}   // namespace vineyard

#endif