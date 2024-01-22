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

#include "basic/ds/tensor.h"
#include "common/util/logging.h"
#include "kv-state-cache/ds/kv_state_cache.h"

using namespace vineyard;

std::vector<int> tokens;
RadixTree* radix_tree;
std::vector<std::vector<double>> k_state_list;
std::vector<std::vector<double>> v_state_list;
std::vector<std::shared_ptr<NodeWithTreeAttri>> nodes_with_tree_attri_list;

#define DIMENSION 10
#define TOKEN_NUM 10
#define CACHE_CAPACITY 10

void prepareData(KVStateCacheBuilder* kv_state_cache_builder) {
  radix_tree = new RadixTree(10);
  radix_tree->SetCustomData(kv_state_cache_builder,
                            sizeof(KVStateCacheBuilder));

  for (int i = 0; i < TOKEN_NUM; i++) {
    tokens.push_back(i);
  }

  LOG(INFO) << "stage 1";
  for (int i = 0; i < TOKEN_NUM; i++) {
    std::vector<double> key_state;
    for (int j = 0; j < DIMENSION; ++j) {
      key_state.push_back(((double) (j)) * 0.1 + (double) i);
    }
    k_state_list.push_back(key_state);
  }

  LOG(INFO) << "stage 2";
  for (int i = 0; i < TOKEN_NUM; i++) {
    std::vector<double> value_state;
    for (int j = 0; j < DIMENSION; ++j) {
      value_state.push_back(((double) (j)) * 0.1 + (double) i);
    }
    v_state_list.push_back(value_state);
  }
}

void updateTest(Client& client, KVStateCacheBuilder* builder) {
  std::vector<int> prefix;

  for (size_t i = 0; i < tokens.size(); ++i) {
    KV_STATE_WITH_LAYER kv_state;
    kv_state.insert(
        std::make_pair(1, std::make_pair(k_state_list[i], v_state_list[i])));
    LOG(INFO) << "update test";
    builder->Update(client, prefix, tokens[i], kv_state);
    prefix.push_back(tokens[i]);
  }
}

void queryTest(Client& client, KVStateCacheBuilder* builder) {
  std::vector<int> prefix;
  KV_STATE_WITH_LAYER kv_state;

  for (int i = 0; i < TOKEN_NUM; i++) {
    kv_state = builder->Query(client, prefix, tokens[i]);
    std::vector<double> key_state = kv_state[1].first;
    std::vector<double> value_state = kv_state[1].second;

    VINEYARD_ASSERT(
        key_state.size() == (size_t) DIMENSION,
        "Expected key_state.size() == " + std::to_string(DIMENSION) +
            ", but got + key_state.size() == " +
            std::to_string(key_state.size()));
    VINEYARD_ASSERT(
        value_state.size() == (size_t) DIMENSION,
        "Expected value_state.size() == " + std::to_string(DIMENSION) +
            ", but got + value_state.size() == " +
            std::to_string(value_state.size()));
    for (int j = 0; j < DIMENSION; ++j) {
      VINEYARD_ASSERT(key_state[j] == k_state_list[i][j],
                      "Expected key_state[" + std::to_string(j) +
                          "] == " + std::to_string(k_state_list[i][j]) +
                          ", but got + key_state[" + std::to_string(j) +
                          "] == " + std::to_string(key_state[j]));
      VINEYARD_ASSERT(value_state[j] == v_state_list[i][j],
                      "Expected value_state[" + std::to_string(j) +
                          "] == " + std::to_string(v_state_list[i][j]) +
                          ", but got + value_state[" + std::to_string(j) +
                          "] == " + std::to_string(value_state[j]));
    }
    prefix.push_back(tokens[i]);
  }
}

void sealAndConstructTest(Client& client, KVStateCacheBuilder* builder) {
  ObjectID id = builder->_Seal(client)->id();
  std::shared_ptr<KVStateCache> kv_state_cache =
      std::dynamic_pointer_cast<KVStateCache>(client.GetObject(id));
  std::shared_ptr<KVStateCacheBlock> kv_state_cache_block =
      kv_state_cache->GetKVStateCacheBlock();
  std::shared_ptr<KVStateCacheBlockBuilder> kv_state_cache_block_builder =
      builder->GetKVStateCacheBlockBuilder();

  // compare kv_state_cache_block and kv_state_cache_block_builder
  VINEYARD_ASSERT(kv_state_cache_block->GetDimension() ==
                  kv_state_cache_block_builder->GetDimension());

  VINEYARD_ASSERT(kv_state_cache_block->GetBitmap() ==
                  kv_state_cache_block_builder->GetBitmap());

  LOG(INFO) << "Bitmap:";
  LOG(INFO) << kv_state_cache_block_builder->GetBitmapStr();
  LOG(INFO) << kv_state_cache_block->GetBitmapStr();

  const std::shared_ptr<TensorBuilder<double>> k_tensor_builder =
      kv_state_cache_block_builder->getKBuilder();
  const std::shared_ptr<TensorBuilder<double>> v_tensor_builder =
      kv_state_cache_block_builder->getVBuilder();

  std::shared_ptr<const Tensor<double>> k_tensor =
      kv_state_cache_block->GetKTensor();
  std::shared_ptr<const Tensor<double>> v_tensor =
      kv_state_cache_block->GetVTensor();

  for (int i = 0; i < TOKEN_NUM; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      VINEYARD_ASSERT(k_tensor->data()[i * DIMENSION + j] ==
                      k_tensor_builder->data()[i * DIMENSION + j]);
      VINEYARD_ASSERT(v_tensor->data()[i * DIMENSION + j] ==
                      v_tensor_builder->data()[i * DIMENSION + j]);
    }
  }
}

void splitTest(Client& client, KVStateCacheBuilder* builder) {}

int main() {
  std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
  Client client;
  client.Connect(socket);

  LOG(INFO) << "Build kv state cache";
  KVStateCacheBuilder* kv_state_cache_builder =
      new KVStateCacheBuilder(client, DIMENSION, CACHE_CAPACITY);

  LOG(INFO) << "Prepare data";
  prepareData(kv_state_cache_builder);

  LOG(INFO) << "Test update";
  updateTest(client, kv_state_cache_builder);

  LOG(INFO) << "Test query";
  queryTest(client, kv_state_cache_builder);

  LOG(INFO) << "Test seal and construct";
  sealAndConstructTest(client, kv_state_cache_builder);

  return 0;
}