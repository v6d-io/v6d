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
  // 1. construct the keyStateTensorBuilder and valueStateTensorBuilder
  this->keyStateTensor = std::dynamic_pointer_cast<Tensor<double>>(
      this->meta_.GetMember("keyStateTensorBuilder"));
  this->valueStateTensor = std::dynamic_pointer_cast<Tensor<double>>(
      this->meta_.GetMember("valueStateTensorBuilder"));
  // 2. construct the member field
  this->bitmap = this->meta_.GetKeyValue<unsigned long long>("bitmap");
  this->dimension = this->meta_.GetKeyValue<int>("dimension");
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(Client& client,
                                                   int dimension) {
  this->bitmap = UINT64_MAX;
  std::vector<int64_t> shape = {LIST_SIZE, dimension};
  this->keyStateTensorBuilder =
      std::make_shared<TensorBuilder<double>>(client, shape);
  this->valueStateTensorBuilder =
      std::make_shared<TensorBuilder<double>>(client, shape);
  this->dimension = dimension;
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(
    Client& client, std::shared_ptr<KVStateCacheBlock> kvStateCacheBlock) {
  this->bitmap = kvStateCacheBlock->bitmap;
  this->dimension = kvStateCacheBlock->dimension;
  std::vector<int64_t> shape = {LIST_SIZE, dimension};
  this->keyStateTensorBuilder =
      std::make_shared<TensorBuilder<double>>(client, shape);
  this->valueStateTensorBuilder =
      std::make_shared<TensorBuilder<double>>(client, shape);

  // transfer the data from kv_state_cache to this builder
  memcpy(this->keyStateTensorBuilder->data(),
         kvStateCacheBlock->keyStateTensor->data(),
         LIST_SIZE * this->dimension * sizeof(double));
  memcpy(this->valueStateTensorBuilder->data(),
         kvStateCacheBlock->valueStateTensor->data(),
         LIST_SIZE * this->dimension * sizeof(double));
}

// current we do not consider the layer.
Status KVStateCacheBlockBuilder::Query(Client& client, int index,
                                       KV_STATE_WITH_LAYER& kvState) {
  std::vector<double> keyStateVector;
  std::vector<double> valueStateVector;

  for (int i = 0; i < this->dimension; ++i) {
    keyStateVector.push_back(
        ((double*) keyStateTensorBuilder->data())[index * dimension + i]);
  }

  for (int i = 0; i < this->dimension; ++i) {
    valueStateVector.push_back(
        ((double*) valueStateTensorBuilder->data())[index * dimension + i]);
  }

  kvState.insert(
      std::make_pair(1, std::make_pair(keyStateVector, valueStateVector)));
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

void KVStateCacheBlockBuilder::Update(const KV_STATE_WITH_LAYER& kvState,
                                      OffsetData* data) {
  int index = this->FindEmptySlot();
  LOG(INFO) << "index:" << index;
  std::vector<double> keyStateVector = (kvState.find(1)->second).first;
  std::vector<double> valueStateVector = (kvState.find(1)->second).second;
  VINEYARD_ASSERT(keyStateVector.size() == (size_t) this->dimension);
  VINEYARD_ASSERT(valueStateVector.size() == (size_t) this->dimension);

  double* keyData = (double*) keyStateTensorBuilder->data();
  double* valueData = (double*) valueStateTensorBuilder->data();
  for (int i = 0; i < this->dimension; ++i) {
    keyData[index * this->dimension + i] = keyStateVector[i];
  }
  for (int i = 0; i < this->dimension; ++i) {
    valueData[index * this->dimension + i] = valueStateVector[i];
  }
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap, index);
}

void KVStateCacheBlockBuilder::Update(double* keyState, double* valueState,
                                      unsigned long dataLength,
                                      OffsetData* data) {
  int index = FindEmptySlot();
  double* keyData = (double*) keyStateTensorBuilder->data();
  double* valueData = (double*) valueStateTensorBuilder->data();
  VINEYARD_ASSERT((unsigned long) this->dimension == dataLength);
  for (unsigned long i = 0; i < dataLength; ++i) {
    keyData[index * this->dimension + i] = keyState[i];
  }
  for (unsigned long i = 0; i < dataLength; ++i) {
    valueData[index * this->dimension + i] = valueState[i];
  }
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap, index);
}

Status KVStateCacheBlockBuilder::Build(Client& client) { return Status::OK(); }

std::shared_ptr<Object> KVStateCacheBlockBuilder::_Seal(Client& client) {
  LOG(INFO) << "block seal:" << this;
  this->Build(client);

  std::shared_ptr<KVStateCacheBlock> kvStateCacheBlock =
      std::make_shared<KVStateCacheBlock>();

  // 1. seal keyStateTensorBuilder and valueStateTensorBuilder
  kvStateCacheBlock->meta_.AddMember("keyStateTensorBuilder",
                                     keyStateTensorBuilder->Seal(client));
  kvStateCacheBlock->meta_.AddMember("valueStateTensorBuilder",
                                     valueStateTensorBuilder->Seal(client));

  // 2. store the member field to meta
  kvStateCacheBlock->meta_.AddKeyValue("bitmap", this->bitmap);
  kvStateCacheBlock->meta_.AddKeyValue("dimension", this->dimension);
  // 3. set the object type to meta
  kvStateCacheBlock->meta_.SetTypeName(type_name<KVStateCacheBlock>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(kvStateCacheBlock->meta_, kvStateCacheBlock->id_));
  return kvStateCacheBlock;
}

}  // namespace vineyard
