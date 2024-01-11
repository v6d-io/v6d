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

#include "kv_state_cache_block.h"
#include "client/client.h"
#include "common/util/logging.h"

namespace vineyard {

// this function will be removed in the future
std::string toBinaryString(unsigned long long num) {
  std::string result;
  const int bits = 8 * sizeof(unsigned long long);
  for (int i = bits - 1; i >= 0; --i) {
    result += ((num >> i) & 1) ? '1' : '0';
  }
  return result;
}

void KVStateCacheBlock::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  // TBD
  std::string tree_data;
  meta.GetKeyValue("bitmap", this->bitmap);
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(Client& client,
                                                   int dimension) {
  pthread_spin_init(&(this->spin_lock), 0);
  this->bitmap = UINT64_MAX;
  this->k_builder = new TensorBuilder<double>(client, {LIST_SIZE, dimension});
  this->v_builder = new TensorBuilder<double>(client, {LIST_SIZE, dimension});
  this->dimension = dimension;
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(
    Client& client, KVStateCacheBlock& kv_state_cache) {
  pthread_spin_init(&(this->spin_lock), 0);
  this->bitmap = kv_state_cache.bitmap;
  this->k_builder =
      new TensorBuilder<double>(client, {LIST_SIZE, kv_state_cache.dimension});
  this->v_builder =
      new TensorBuilder<double>(client, {LIST_SIZE, kv_state_cache.dimension});
  this->dimension = kv_state_cache.dimension;
  // TBD:
  // transfer the data from kv_state_cache to this builder
}

// current we do not consider the layer.
Status KVStateCacheBlockBuilder::Query(Client& client, int index,
                                       KV_STATE_WITH_LAYER& kv_state) {
  std::vector<double> k_state;
  std::vector<double> v_state;

  for (int i = 0; i < this->dimension; ++i) {
    k_state.push_back(((double*) k_builder->data())[index * dimension + i]);
  }

  for (int i = 0; i < this->dimension; ++i) {
    v_state.push_back(((double*) v_builder->data())[index * dimension + i]);
  }

  kv_state.insert(std::make_pair(1, std::make_pair(k_state, v_state)));
  return Status::OK();
}

bool KVStateCacheBlockBuilder::isFull() {
  int index = ffsll(this->bitmap) - 1;
  return index < 0 || index > LIST_SIZE;
}

Status KVStateCacheBlockBuilder::Build(Client& client) {
  // TBD craete vineyard object
  pthread_spin_lock(&(this->spin_lock));
  ObjectMeta meta;
  meta.SetTypeName(type_name<KVStateCacheBlock>());
  meta.AddKeyValue("bitmap", this->bitmap);
  for (int i = 0; i < LIST_SIZE; ++i) {
    // TBD
    // create tensor meta
  }
  // TBD check the status
  client.CreateMetaData(meta, id);
  pthread_spin_unlock(&(this->spin_lock));
  return Status::OK();
}

std::shared_ptr<Object> KVStateCacheBlockBuilder::_Seal(Client& client) {
  pthread_spin_lock(&(this->spin_lock));
  // TBD
  // Sync with vineyard server preriodically
  pthread_spin_unlock(&(this->spin_lock));
  return nullptr;
}

std::shared_ptr<offset_data> KVStateCacheBlockBuilder::Update(
    const KV_STATE_WITH_LAYER& kv_state) {
  int index = ffsll(this->bitmap) - 1;
  assert(index >= 0 && index < LIST_SIZE);
  std::vector<double> k_state = (kv_state.find(1)->second).first;
  std::vector<double> v_state = (kv_state.find(1)->second).second;
  assert(k_state.size() == this->dimension);
  assert(v_state.size() == this->dimension);

  double* key_data = (double*) k_builder->data();
  double* value_data = (double*) v_builder->data();
  for (int i = 0; i < this->dimension; ++i) {
    key_data[index * this->dimension + i] = k_state[i];
  }
  for (int i = 0; i < this->dimension; ++i) {
    value_data[index * this->dimension + i] = v_state[i];
  }
  std::shared_ptr<offset_data> data = std::make_shared<offset_data>();
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap, index);
  return data;
}

std::shared_ptr<offset_data> KVStateCacheBlockBuilder::Update(
    double* k_data, double* v_data, unsigned long data_length) {
  int index = ffsll(this->bitmap) - 1;
  assert(index >= 0 && index < LIST_SIZE);
  double* key_data = (double*) k_builder->data();
  double* value_data = (double*) v_builder->data();
  assert(this->dimension == data_length);
  for (unsigned long i = 0; i < data_length; ++i) {
    key_data[index * this->dimension + i] = k_data[i];
  }
  for (unsigned long i = 0; i < data_length; ++i) {
    value_data[index * this->dimension + i] = v_data[i];
  }
  std::shared_ptr<offset_data> data = std::make_shared<offset_data>();
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap, index);
  return data;
}

void KVStateCacheBlockBuilder::SetChildKVStateCacheBlockBuilder(
    KVStateCacheBlockBuilder* child_kv_state_cache_builder) {
  this->child_kv_state_cache_builder_list.push_back(
      child_kv_state_cache_builder);
}

}  // namespace vineyard
