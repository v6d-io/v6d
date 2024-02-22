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
  this->layer = this->meta_.GetKeyValue<int>("layer");
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    this->keyStateTensorList.push_back(
        std::dynamic_pointer_cast<Tensor<double>>(this->meta_.GetMember(
            "keyStateTensorBuilder_" + std::to_string(currentLayer))));
    this->valueStateTensorList.push_back(
        std::dynamic_pointer_cast<Tensor<double>>(this->meta_.GetMember(
            "valueStateTensorBuilder_" + std::to_string(currentLayer))));
  }
  // 2. construct the member field
  this->bitmap = this->meta_.GetKeyValue<unsigned long long>("bitmap");
  this->dimension = this->meta_.GetKeyValue<int>("dimension");
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(Client& client,
                                                   int dimension, int layer) {
  this->bitmap = UINT64_MAX;
  std::vector<int64_t> shape = {LIST_SIZE, dimension};
  for (int i = 0; i < layer; i++) {
    this->keyStateTensorBuilderList.push_back(
        std::make_shared<TensorBuilder<double>>(client, shape));
    this->valueStateTensorBuilderList.push_back(
        std::make_shared<TensorBuilder<double>>(client, shape));
  }
  this->dimension = dimension;
  this->layer = layer;
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(
    Client& client, std::shared_ptr<KVStateCacheBlock> kvStateCacheBlock) {
  this->bitmap = kvStateCacheBlock->bitmap;
  this->dimension = kvStateCacheBlock->dimension;
  this->layer = kvStateCacheBlock->layer;
  std::vector<int64_t> shape = {LIST_SIZE, dimension};
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    this->keyStateTensorBuilderList.push_back(
        std::make_shared<TensorBuilder<double>>(client, shape));
    this->valueStateTensorBuilderList.push_back(
        std::make_shared<TensorBuilder<double>>(client, shape));
  }

  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    memcpy(this->keyStateTensorBuilderList[currentLayer]->data(),
           kvStateCacheBlock->keyStateTensorList[currentLayer]->data(),
           LIST_SIZE * this->dimension * sizeof(double));
    memcpy(this->valueStateTensorBuilderList[currentLayer]->data(),
           kvStateCacheBlock->valueStateTensorList[currentLayer]->data(),
           LIST_SIZE * this->dimension * sizeof(double));
  }
}

// current we do not consider the layer.
Status KVStateCacheBlockBuilder::Query(Client& client, int index,
                                       KV_STATE_WITH_LAYER& kvState) {
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    std::vector<double> keyStateVector;
    std::vector<double> valueStateVector;

    for (int i = 0; i < this->dimension; ++i) {
      keyStateVector.push_back(
          ((double*) keyStateTensorBuilderList[currentLayer]
               ->data())[index * dimension + i]);
    }

    for (int i = 0; i < this->dimension; ++i) {
      valueStateVector.push_back(
          ((double*) valueStateTensorBuilderList[currentLayer]
               ->data())[index * dimension + i]);
    }

    kvState.insert(std::make_pair(
        currentLayer, std::make_pair(keyStateVector, valueStateVector)));
  }
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
  LOG(INFO) << "layer:" << layer;
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    std::vector<double> keyStateVector =
        (kvState.find(currentLayer)->second).first;
    std::vector<double> valueStateVector =
        (kvState.find(currentLayer)->second).second;
    LOG(INFO) << "vector size:" << keyStateVector.size() << " "
              << valueStateVector.size() << " demension" << this->dimension;
    VINEYARD_ASSERT(keyStateVector.size() == (size_t) this->dimension);
    VINEYARD_ASSERT(valueStateVector.size() == (size_t) this->dimension);

    double* keyData = (double*) keyStateTensorBuilderList[currentLayer]->data();
    double* valueData =
        (double*) valueStateTensorBuilderList[currentLayer]->data();
    memcpy(keyData + index * this->dimension, keyStateVector.data(),
           this->dimension * sizeof(double));
    memcpy(valueData + index * this->dimension, valueStateVector.data(),
           this->dimension * sizeof(double));
  }
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap, index);
}

short KVStateCacheBlockBuilder::Split(KVStateCacheBlockBuilder* child,
                                      int index) {
  // TBD
  VINEYARD_ASSERT(this->layer == child->layer);
  int childIndex = child->FindEmptySlot();
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    std::shared_ptr<TensorBuilder<double>> keyStateTensorBuilder =
        keyStateTensorBuilderList[currentLayer];
    std::shared_ptr<TensorBuilder<double>> valueStateTensorBuilder =
        valueStateTensorBuilderList[currentLayer];
    std::shared_ptr<TensorBuilder<double>> childKeyStateTensorBuilder =
        child->keyStateTensorBuilderList[currentLayer];
    std::shared_ptr<TensorBuilder<double>> childValueStateTensorBuilder =
        child->valueStateTensorBuilderList[currentLayer];

    double* keyState =
        (double*) keyStateTensorBuilder->data() + index * this->dimension;
    double* valueState =
        (double*) valueStateTensorBuilder->data() + index * this->dimension;
    double* childKeyState = (double*) childKeyStateTensorBuilder->data() +
                            childIndex * this->dimension;
    double* childValueState = (double*) childValueStateTensorBuilder->data() +
                              childIndex * this->dimension;

    memcpy(childKeyState, keyState, this->dimension * sizeof(double));
    memcpy(childValueState, valueState, this->dimension * sizeof(double));
  }
  ACQUIRE_BIT_RESOURCE(child->bitmap, childIndex);
  FREE_BIT_RESOURCE(this->bitmap, index);
  return childIndex;
}

Status KVStateCacheBlockBuilder::Build(Client& client) { return Status::OK(); }

std::shared_ptr<Object> KVStateCacheBlockBuilder::_Seal(Client& client) {
  LOG(INFO) << "block seal:" << this;
  this->Build(client);

  std::shared_ptr<KVStateCacheBlock> kvStateCacheBlock =
      std::make_shared<KVStateCacheBlock>();

  // 1. seal keyStateTensorBuilder and valueStateTensorBuilder
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    kvStateCacheBlock->meta_.AddMember(
        "keyStateTensorBuilder_" + std::to_string(currentLayer),
        keyStateTensorBuilderList[currentLayer]->Seal(client));
    kvStateCacheBlock->meta_.AddMember(
        "valueStateTensorBuilder_" + std::to_string(currentLayer),
        valueStateTensorBuilderList[currentLayer]->Seal(client));
  }

  // 2. store the member field to meta
  kvStateCacheBlock->meta_.AddKeyValue("bitmap", this->bitmap);
  kvStateCacheBlock->meta_.AddKeyValue("dimension", this->dimension);
  kvStateCacheBlock->meta_.AddKeyValue("layer", this->layer);
  // 3. set the object type to meta
  kvStateCacheBlock->meta_.SetTypeName(type_name<KVStateCacheBlock>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(kvStateCacheBlock->meta_, kvStateCacheBlock->id_));
  return kvStateCacheBlock;
}

}  // namespace vineyard
