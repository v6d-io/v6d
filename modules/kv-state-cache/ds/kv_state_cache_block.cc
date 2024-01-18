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
std::string KVStateCacheBlock::GetBitmapStr() {
  std::string result;
  const int bits = 8 * sizeof(unsigned long long);
  for (int i = bits - 1; i >= 0; --i) {
    result += ((this->bitmap >> i) & 1) ? '1' : '0';
  }
  return result;
}

std::string KVStateCacheBlockBuilder::GetBitmapStr() {
  std::string result;
  const int bits = 8 * sizeof(unsigned long long);
  for (int i = bits - 1; i >= 0; --i) {
    result += ((this->bitmap >> i) & 1) ? '1' : '0';
  }
  return result;
}

void KVStateCacheBlock::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);

  std::string typeName = type_name<KVStateCacheBlock>();

  VINEYARD_ASSERT(meta.GetTypeName() == typeName,
                  "Expect typename '" + typeName + "', but got '" +
                      meta.GetTypeName() + "'");

  // TBD
  // 1. construct the k_builder and v_builder
  this->k_tensor = std::dynamic_pointer_cast<Tensor<double>>(
      this->meta_.GetMember("k_builder"));
  this->v_tensor = std::dynamic_pointer_cast<Tensor<double>>(
      this->meta_.GetMember("v_builder"));
  // 2. construct the child kv_state_cache_block_builder
  int child_num = this->meta_.GetKeyValue<int>("child_num");
  for (int i = 0; i < child_num; ++i) {
    std::shared_ptr<KVStateCacheBlock> child_kv_state_cache_block_builder =
        std::dynamic_pointer_cast<KVStateCacheBlock>(this->meta_.GetMember(
            "child_kv_state_cache_block_" + std::to_string(i)));
    this->child_kv_state_cache_block_list.push_back(
        child_kv_state_cache_block_builder);
  }
  // 3. construct the member field
  this->bitmap = this->meta_.GetKeyValue<unsigned long long>("bitmap");
  this->dimension = this->meta_.GetKeyValue<int>("dimension");
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(Client& client,
                                                   int dimension) {
  pthread_spin_init(&(this->spin_lock), 0);
  this->bitmap = UINT64_MAX;
  std::vector<int64_t> shape = {LIST_SIZE, dimension};
  this->k_builder = std::make_shared<TensorBuilder<double>>(client, shape);
  this->v_builder = std::make_shared<TensorBuilder<double>>(client, shape);
  this->dimension = dimension;
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(
    Client& client, std::shared_ptr<KVStateCacheBlock> kv_state_cache_block) {
  pthread_spin_init(&(this->spin_lock), 0);
  this->bitmap = kv_state_cache_block->bitmap;
  this->dimension = kv_state_cache_block->dimension;
  std::vector<int64_t> shape = {LIST_SIZE, dimension};
  this->k_builder = std::make_shared<TensorBuilder<double>>(client, shape);
  this->v_builder = std::make_shared<TensorBuilder<double>>(client, shape);

  // transfer the data from kv_state_cache to this builder
  memcpy(this->k_builder->data(), kv_state_cache_block->k_tensor->data(),
         LIST_SIZE * this->dimension * sizeof(double));
  memcpy(this->v_builder->data(), kv_state_cache_block->v_tensor->data(),
         LIST_SIZE * this->dimension * sizeof(double));
  for (size_t i = 0;
       i < kv_state_cache_block->child_kv_state_cache_block_list.size(); ++i) {
    this->child_kv_state_cache_builder_list.push_back(
        new KVStateCacheBlockBuilder(
            client, kv_state_cache_block->child_kv_state_cache_block_list[i]));
  }
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

int KVStateCacheBlockBuilder::FindEmptySlot() {
  int index = ffsll(this->bitmap) - 1;
  VINEYARD_ASSERT(index >= 0 && index < LIST_SIZE);
  return index;
}

bool KVStateCacheBlockBuilder::IsFull() {
  int index = ffsll(this->bitmap) - 1;
  return index < 0 || index >= LIST_SIZE;
}

void KVStateCacheBlockBuilder::Update(const KV_STATE_WITH_LAYER& kv_state,
                                      offset_data* data) {
  int index = this->FindEmptySlot();
  LOG(INFO) << "index:" << index;
  std::vector<double> k_state = (kv_state.find(1)->second).first;
  std::vector<double> v_state = (kv_state.find(1)->second).second;
  VINEYARD_ASSERT(k_state.size() == (size_t) this->dimension);
  VINEYARD_ASSERT(v_state.size() == (size_t) this->dimension);

  double* key_data = (double*) k_builder->data();
  double* value_data = (double*) v_builder->data();
  for (int i = 0; i < this->dimension; ++i) {
    key_data[index * this->dimension + i] = k_state[i];
  }
  for (int i = 0; i < this->dimension; ++i) {
    value_data[index * this->dimension + i] = v_state[i];
  }
  data->offset = index;

  LOG(INFO) << "before:" << this->bitmap;
  ACQUIRE_BIT_RESOURCE(this->bitmap, index);
  LOG(INFO) << "after:" << this->bitmap;
}

void KVStateCacheBlockBuilder::Update(double* k_data, double* v_data,
                                      unsigned long data_length,
                                      offset_data* data) {
  int index = FindEmptySlot();
  double* key_data = (double*) k_builder->data();
  double* value_data = (double*) v_builder->data();
  VINEYARD_ASSERT((unsigned long) this->dimension == data_length);
  for (unsigned long i = 0; i < data_length; ++i) {
    key_data[index * this->dimension + i] = k_data[i];
  }
  for (unsigned long i = 0; i < data_length; ++i) {
    value_data[index * this->dimension + i] = v_data[i];
  }
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap, index);
  LOG(INFO) << "bitmap:" << this->GetBitmapStr();
}

void KVStateCacheBlockBuilder::SetChildKVStateCacheBlockBuilder(
    KVStateCacheBlockBuilder* child_kv_state_cache_builder) {
  this->child_kv_state_cache_builder_list.push_back(
      child_kv_state_cache_builder);
}

Status KVStateCacheBlockBuilder::Build(Client& client) {
  // TBD craete vineyard object
  // pthread_spin_lock(&(this->spin_lock));
  // ObjectMeta meta;
  // meta.SetTypeName(type_name<KVStateCacheBlock>());
  // meta.AddKeyValue("bitmap", this->bitmap);
  // for (int i = 0; i < LIST_SIZE; ++i) {
  //   // TBD
  //   // create tensor meta
  // }
  // // TBD check the status
  // client.CreateMetaData(meta, id);
  // pthread_spin_unlock(&(this->spin_lock));
  return Status::OK();
}

std::shared_ptr<Object> KVStateCacheBlockBuilder::_Seal(Client& client) {
  this->Build(client);
  // pthread_spin_lock(&(this->spin_lock));
  // pthread_spin_unlock(&(this->spin_lock));

  std::shared_ptr<KVStateCacheBlock> kv_state_cache_block =
      std::make_shared<KVStateCacheBlock>();

  // TBD
  // 1. seal k_builder and v_builder
  kv_state_cache_block->meta_.AddMember("k_builder", k_builder->Seal(client));
  kv_state_cache_block->meta_.AddMember("v_builder", v_builder->Seal(client));
  // 2. seal child kv_state_cache_block_builder
  for (size_t i = 0; i < this->child_kv_state_cache_builder_list.size(); ++i) {
    kv_state_cache_block->meta_.AddMember(
        "child_kv_state_cache_block_" + std::to_string(i),
        this->child_kv_state_cache_builder_list[i]->_Seal(client));
  }
  kv_state_cache_block->meta_.AddKeyValue(
      "child_num", this->child_kv_state_cache_builder_list.size());
  // 3. store the member field to meta
  kv_state_cache_block->meta_.AddKeyValue("bitmap", this->bitmap);
  kv_state_cache_block->meta_.AddKeyValue("dimension", this->dimension);
  // 4. set the object type to meta
  kv_state_cache_block->meta_.SetTypeName(type_name<KVStateCacheBlock>());

  VINEYARD_CHECK_OK(client.CreateMetaData(kv_state_cache_block->meta_,
                                          kv_state_cache_block->id_));
  return kv_state_cache_block;
}

}  // namespace vineyard
