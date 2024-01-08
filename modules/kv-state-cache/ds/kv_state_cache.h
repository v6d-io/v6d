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

#ifndef MODULES_KV_STATE_CACHE_H_
#define MODULES_KV_STATE_CACHE_H_

#include <array>
#include <iostream>
#include <map>
#include <vector>

#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "kv-state-cache/radix-tree/radix.h"

typedef std::map<int, std::pair<std::vector<double>, std::vector<double>>>
    KV_STATE_WITH_LAYER;
typedef std::vector<
    std::map<int, std::pair<std::vector<double>, std::vector<double>>>>
    LIST_KV_STATE_WITH_LAYER;

namespace vineyard {

struct offset_data {
  short offset_k;
  short offset_v;
};

#define LIST_SIZE 2

/**
 * @brief KVStateCache is a cache for kv-cache of LLM. When a new prompt comes,
 * LLM can query KVStateCache to get the state of the kv-cache to avoid
 * caclulating the kv-cache again if the new prompt is similar to the previous
 * one.
 *
 * KVStateCache is stored in vineyard as a vineyard object which contains a
 * radix tree. The token sequence is the key of the radix tree and the value
 * point out the offset of the kv-cache in the tensor list.
 *
 * KVStateCache can be shared by multiple machines.
 */

class KVStateCache : public vineyard::Registered<KVStateCache> {
 private:
  RadixTree *tree;
  std::array<std::unique_ptr<BlobWriter>, LIST_SIZE> key_state_writer_array;
  std::array<std::unique_ptr<BlobWriter>, LIST_SIZE> value_state_writer_array;
  uint64_t bitmap;
  pthread_spinlock_t spin_lock;
  ObjectID id;

 public:
  void Construct(const ObjectMeta& meta) override;

  friend class KVStateCacheBuilder;
};

class KVStateCacheBuilder : public vineyard::ObjectBuilder {
 private:
  RadixTree *tree;
  std::array<std::unique_ptr<BlobWriter>, LIST_SIZE> key_state_writer_array;
  std::array<std::unique_ptr<BlobWriter>, LIST_SIZE> value_state_writer_array;
  std::map<RadixTree *, KVStateCacheBuilder *> kv_state_cache_builder_map;
  uint64_t bitmap;
  pthread_spinlock_t spin_lock;
  ObjectID id;

  /**
   * @brief Splits the radix-tree into several radix-trees if the number of
   * kv-state is larger than LIST_SIZE.
   */
  Status Split();

  Status UpdateInternal(Client& client, const std::vector<int>& token_list,
                        int next_token, const KV_STATE_WITH_LAYER& kv_state);

  Status QueryInternal(Client& client, const std::vector<int>& token_list,
                       int token, KV_STATE_WITH_LAYER& kv_state);

  /**
   * @brief Travel the radix-tree and update the kv-state when splitting the
   * radix-tree.
   */
  Status TravelAndUpdateNode();

 public:
  KVStateCacheBuilder();

  KVStateCacheBuilder(KVStateCache& kv_state_cache);

  KVStateCacheBuilder(RadixTree* tree);

  /**
   * @brief Update the kv-state using next token.
   *
   * @param token_list The token list of the prompt.
   * @param next_token The next token of the prompt.
   * @param kv_state The kv-state of the prompt. A LLM inference can contain
   * multiple kv-states for each layer.
   */
  Status Update(Client& client, const std::vector<int>& token_list,
                int next_token, const KV_STATE_WITH_LAYER& kv_state);

  /**
   * @brief Update the kv-state using the whole token list.
   *
   * @param token_list The token list of the prompt.
   * @param kv_state The kv-state of the prompt. A LLM inference can contain
   * multiple kv-states for each layer.
   */
  Status Update(Client& client, const std::vector<int>& token_list,
                const LIST_KV_STATE_WITH_LAYER& kv_state);

  /**
   * @brief Query the kv-state using the whole token list.
   *
   * @param token_list The token list of the prompt.
   * @param token The token of the prompt.
   * @param kv_state The kv-state of the prompt returned by radix-tree. If the
   * kv-state is not found, the data of kv-state is invalid.
   */
  Status Query(Client& client, const std::vector<int>& token_list, int token,
               KV_STATE_WITH_LAYER& kv_state);

  /**
   * @brief Query the kv-state using the whole token list.
   *
   * @param token_list The token list of the prompt.
   * @param kv_state The kv-state of the prompt returned by radix-tree. If the
   * kv-state is not found, the data of kv-state is invalid.
   */
  Status Query(Client& client, const std::vector<int>& token_list,
               LIST_KV_STATE_WITH_LAYER& kv_state);

  Status Build(Client& client) override;

  std::shared_ptr<Object> _Seal(Client& client) override;

  Status GetTree(RadixTree*& tree);
};

}  // namespace vineyard

#endif