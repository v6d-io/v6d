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

#include <memory>
#include <string>
#include <utility>

#include "client/client.h"
#include "common/util/logging.h"
#include "llm-cache/ds/kv_state_cache_block.h"

namespace vineyard {

// this function will be removed in the future
std::string KVStateCacheBlock::GetBitmapStr() {
  std::string result;
  const int bits = 8 * sizeof(uint64_t);
  for (int i = 0; i < this->bitmapSize; i++) {
    for (int j = bits - 1; j >= 0; --j) {
      result += (((this->bitmap[i]) >> j) & 1) ? '1' : '0';
    }
  }
  return result;
}

std::string KVStateCacheBlockBuilder::GetBitmapStr() {
  std::string result;
  const int bits = 8 * sizeof(uint64_t);
  for (int i = 0; i < this->bitmapSize; i++) {
    for (int j = bits - 1; j >= 0; --j) {
      result += (((this->bitmap[i]) >> j) & 1) ? '1' : '0';
    }
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
  this->bitmapSize = this->meta_.GetKeyValue<int>("bitmap_size");
  VLOG(100) << "construct bitmap size:" << this->bitmapSize;
  this->bitmap = new uint64_t[this->bitmapSize];
  for (int i = 0; i < this->bitmapSize; i++) {
    this->bitmap[i] =
        this->meta_.GetKeyValue<uint64_t>("bitmap_" + std::to_string(i));
  }
  this->dimension = this->meta_.GetKeyValue<int>("dimension");
  this->blockSize = this->meta_.GetKeyValue<int>("block_size");
}

KVStateCacheBlock::~KVStateCacheBlock() { delete this->bitmap; }

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(Client& client,
                                                   int dimension, int layer,
                                                   int blockSize) {
  this->blockSize = blockSize;
  this->bitmapSize = (blockSize + 63) / 64;
  this->bitmap = new uint64_t[this->bitmapSize];
  memset(this->bitmap, UINT8_MAX, this->bitmapSize * sizeof(uint64_t));
  std::vector<int64_t> shape = {(int64_t)(blockSize), dimension};
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
  this->bitmapSize = kvStateCacheBlock->bitmapSize;
  this->blockSize = kvStateCacheBlock->blockSize;
  VLOG(100) << "create builder from block object, bitmap size:"
            << this->bitmapSize << " block size:" << blockSize;
  this->bitmap = new uint64_t[this->bitmapSize];
  for (int i = 0; i < this->bitmapSize; i++) {
    this->bitmap[i] = kvStateCacheBlock->bitmap[i];
  }
  this->dimension = kvStateCacheBlock->dimension;
  this->layer = kvStateCacheBlock->layer;
  std::vector<int64_t> shape = {(int64_t)(blockSize), dimension};
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    this->keyStateTensorBuilderList.push_back(
        std::make_shared<TensorBuilder<double>>(client, shape));
    this->valueStateTensorBuilderList.push_back(
        std::make_shared<TensorBuilder<double>>(client, shape));
  }

  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    memcpy(this->keyStateTensorBuilderList[currentLayer]->data(),
           kvStateCacheBlock->keyStateTensorList[currentLayer]->data(),
           (int64_t)(blockSize) * this->dimension * sizeof(double));
    memcpy(this->valueStateTensorBuilderList[currentLayer]->data(),
           kvStateCacheBlock->valueStateTensorList[currentLayer]->data(),
           (int64_t)(blockSize) * this->dimension * sizeof(double));
  }
}

// current we do not consider the layer.
int KVStateCacheBlockBuilder::Query(Client& client, int index,
                                    KV_STATE_WITH_LAYER& kvState) {
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    memcpy((kvState.find(currentLayer)->second).first.data,
           keyStateTensorBuilderList[currentLayer]->data() + index * dimension,
           dimension * sizeof(double));
    memcpy(
        (kvState.find(currentLayer)->second).second.data,
        valueStateTensorBuilderList[currentLayer]->data() + index * dimension,
        dimension * sizeof(double));
  }
  return 0;
}

int KVStateCacheBlockBuilder::FindEmptySlot() {
  for (int i = 0; i < this->bitmapSize; i++) {
    if (this->bitmap[i] != 0) {
      int index = ffsll(this->bitmap[i]) - 1;
      return index + i * 64;
    }
  }
  return -1;
}

bool KVStateCacheBlockBuilder::IsFull() {
  int left = this->blockSize;
  for (int i = 0; i < this->bitmapSize; i++) {
    if (this->bitmap[i] != 0 && ffsll(this->bitmap[i]) - 1 < left) {
      return false;
    }
    left -= sizeof(uint64_t) * 8;
  }
  return true;
}

void KVStateCacheBlockBuilder::Update(const KV_STATE_WITH_LAYER& kvState,
                                      OffsetData* data) {
  int index = this->FindEmptySlot();
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    K_STATE keyState = (kvState.find(currentLayer)->second).first;
    V_STATE valueState = (kvState.find(currentLayer)->second).second;
    VINEYARD_ASSERT(keyState.length ==
                    (size_t) this->dimension * sizeof(double));
    VINEYARD_ASSERT(valueState.length ==
                    (size_t) this->dimension * sizeof(double));

    double* keyData = keyStateTensorBuilderList[currentLayer]->data();
    double* valueData = valueStateTensorBuilderList[currentLayer]->data();
    memcpy(keyData + index * this->dimension, keyState.data,
           this->dimension * sizeof(double));
    memcpy(valueData + index * this->dimension, valueState.data,
           this->dimension * sizeof(double));
  }
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap[index / 64], index % 64);
}

int16_t KVStateCacheBlockBuilder::Split(KVStateCacheBlockBuilder* child,
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

    double* keyState = keyStateTensorBuilder->data() + index * this->dimension;
    double* valueState =
        valueStateTensorBuilder->data() + index * this->dimension;
    double* childKeyState =
        childKeyStateTensorBuilder->data() + childIndex * this->dimension;
    double* childValueState =
        childValueStateTensorBuilder->data() + childIndex * this->dimension;

    memcpy(childKeyState, keyState, this->dimension * sizeof(double));
    memcpy(childValueState, valueState, this->dimension * sizeof(double));
  }
  ACQUIRE_BIT_RESOURCE(child->bitmap[childIndex / 64], childIndex % 64);
  FREE_BIT_RESOURCE(this->bitmap[index / 64], index % 64);
  return childIndex;
}

Status KVStateCacheBlockBuilder::Build(Client& client) { return Status::OK(); }

std::shared_ptr<Object> KVStateCacheBlockBuilder::_Seal(Client& client) {
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
  kvStateCacheBlock->meta_.AddKeyValue("bitmap_size", this->bitmapSize);
  for (int i = 0; i < this->bitmapSize; i++) {
    kvStateCacheBlock->meta_.AddKeyValue("bitmap_" + std::to_string(i),
                                         this->bitmap[i]);
  }

  kvStateCacheBlock->meta_.AddKeyValue("block_size", this->blockSize);
  kvStateCacheBlock->meta_.AddKeyValue("dimension", this->dimension);
  kvStateCacheBlock->meta_.AddKeyValue("layer", this->layer);
  // 3. set the object type to meta
  kvStateCacheBlock->meta_.SetTypeName(type_name<KVStateCacheBlock>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(kvStateCacheBlock->meta_, kvStateCacheBlock->id_));
  return kvStateCacheBlock;
}

void KVStateCacheBlockBuilder::PrintKVStateCacheBlock() {
  LOG(INFO) << "builder:" << this;
  for (int i = 0; i < this->blockSize; i++) {
    LOG(INFO) << "index:" << i << " bitmap:" << this->GetBitmapStr();
  }

  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    LOG(INFO) << "layer:" << currentLayer;
    for (int i = 0; i < this->blockSize; i++) {
      LOG(INFO) << "index:" << i;
      std::string keyState = "";
      std::string valueState = "";
      for (int j = 0; j < this->dimension; j++) {
        keyState += std::to_string((keyStateTensorBuilderList[currentLayer]
                                        ->data())[i * dimension + j]) +
                    " ";
        valueState += std::to_string((valueStateTensorBuilderList[currentLayer]
                                          ->data())[i * dimension + j]) +
                      " ";
      }
      LOG(INFO) << "keyState:" << keyState;
      LOG(INFO) << "valueState:" << valueState;
    }
  }

  LOG(INFO) << "==========================";
}

KVStateCacheBlockBuilder::~KVStateCacheBlockBuilder() { delete this->bitmap; }

}  // namespace vineyard
