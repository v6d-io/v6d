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
        std::dynamic_pointer_cast<KVTensor>(this->meta_.GetMember(
            "keyStateTensorBuilder_" + std::to_string(currentLayer))));
    this->valueStateTensorList.push_back(
        std::dynamic_pointer_cast<KVTensor>(this->meta_.GetMember(
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
  this->tensorBytes = this->meta_.GetKeyValue<int>("tensorBytes");
  this->blockSize = this->meta_.GetKeyValue<int>("block_size");
}

KVStateCacheBlock::~KVStateCacheBlock() { delete this->bitmap; }

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(Client& client,
                                                   int tensorBytes, int layer,
                                                   int blockSize)
    : client(client) {
  this->blockSize = blockSize;
  this->bitmapSize = (blockSize + 63) / 64;
  this->bitmap = new uint64_t[this->bitmapSize];
  memset(this->bitmap, UINT8_MAX, this->bitmapSize * sizeof(uint64_t));
  std::vector<int64_t> shape = {(int64_t)(blockSize), tensorBytes};
  for (int i = 0; i < layer; i++) {
    this->keyStateTensorBuilderList.push_back(
        std::make_shared<KVTensorBuilder>(client, shape));
    this->valueStateTensorBuilderList.push_back(
        std::make_shared<KVTensorBuilder>(client, shape));
  }
  this->tensorBytes = tensorBytes;
  this->layer = layer;
}

KVStateCacheBlockBuilder::KVStateCacheBlockBuilder(
    Client& client, std::shared_ptr<KVStateCacheBlock> kvStateCacheBlock)
    : client(client) {
  this->bitmapSize = kvStateCacheBlock->bitmapSize;
  this->blockSize = kvStateCacheBlock->blockSize;
  VLOG(100) << "create builder from block object, bitmap size:"
            << this->bitmapSize << " block size:" << blockSize;
  this->bitmap = new uint64_t[this->bitmapSize];
  for (int i = 0; i < this->bitmapSize; i++) {
    this->bitmap[i] = kvStateCacheBlock->bitmap[i];
  }
  this->tensorBytes = kvStateCacheBlock->tensorBytes;
  this->layer = kvStateCacheBlock->layer;
  std::vector<int64_t> shape = {(int64_t)(blockSize), this->tensorBytes};
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    this->keyStateTensorBuilderList.push_back(
        std::make_shared<KVTensorBuilder>(client, shape));
    this->valueStateTensorBuilderList.push_back(
        std::make_shared<KVTensorBuilder>(client, shape));
  }

  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    memcpy(this->keyStateTensorBuilderList[currentLayer]->data(),
           kvStateCacheBlock->keyStateTensorList[currentLayer]->data(),
           (int64_t)(blockSize) * this->tensorBytes);
    memcpy(this->valueStateTensorBuilderList[currentLayer]->data(),
           kvStateCacheBlock->valueStateTensorList[currentLayer]->data(),
           (int64_t)(blockSize) * this->tensorBytes);
  }
}

Status KVStateCacheBlockBuilder::Make(
    Client& client, TreeData* treeData,
    KVStateCacheBlockBuilder*& kvStateCacheBlockBuilder) {
  RETURN_ON_ASSERT(treeData != nullptr && treeData->isPtr == false);
  ObjectID blockObjectID = treeData->builderObjectID;

  std::shared_ptr<KVStateCacheBlock> blockObject;
  RETURN_ON_ERROR(client.FetchAndGetObject(blockObjectID, blockObject));
  kvStateCacheBlockBuilder = new KVStateCacheBlockBuilder(client, blockObject);
  if (blockObjectID != blockObject->id()) {
    // If the object is migrated, we should delete the copied object.
    Status status = client.DelData(blockObject->id());
    if (!status.ok()) {
      LOG(ERROR) << "Delete object failed: " << status.ToString()
                 << " It may cause memory leak.";
    }
  }
  return Status::OK();
}

Status KVStateCacheBlockBuilder::Query(
    int index, std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  RETURN_ON_ASSERT((index >= 0 && index < this->blockSize),
                   "Index out of range: " + std::to_string(index));
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    LLMKV keyState = (kvState.find(currentLayer)->second).first;
    LLMKV valueState = (kvState.find(currentLayer)->second).second;
    keyState.data =
        keyStateTensorBuilderList[currentLayer]->data() + index * tensorBytes;
    keyState.length = tensorBytes;
    valueState.data =
        valueStateTensorBuilderList[currentLayer]->data() + index * tensorBytes;
    valueState.length = tensorBytes;
    kvState.emplace(currentLayer, std::make_pair(keyState, valueState));
  }
  return Status::OK();
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

Status KVStateCacheBlockBuilder::Update(
    const std::map<int, std::pair<LLMKV, LLMKV>>& kvState, OffsetData* data) {
  int index = this->FindEmptySlot();
  RETURN_ON_ASSERT((index >= 0 && index < this->blockSize),
                   "Index out of range: " + std::to_string(index));

  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    LLMKV keyState = (kvState.find(currentLayer)->second).first;
    LLMKV valueState = (kvState.find(currentLayer)->second).second;
    RETURN_ON_ASSERT((keyState.length == (size_t) this->tensorBytes &&
                      valueState.length == (size_t) this->tensorBytes));

    uint8_t* keyData = keyStateTensorBuilderList[currentLayer]->data();
    uint8_t* valueData = valueStateTensorBuilderList[currentLayer]->data();
    memcpy(keyData + index * this->tensorBytes, keyState.data,
           this->tensorBytes);
    memcpy(valueData + index * this->tensorBytes, valueState.data,
           this->tensorBytes);
  }
  data->offset = index;

  ACQUIRE_BIT_RESOURCE(this->bitmap[index / 64], index % 64);
  return Status::OK();
}

int16_t KVStateCacheBlockBuilder::Split(KVStateCacheBlockBuilder* child,
                                        int index) {
  // Child builder must be empty.
  int childIndex = child->FindEmptySlot();
  for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
    std::shared_ptr<KVTensorBuilder> keyStateTensorBuilder =
        keyStateTensorBuilderList[currentLayer];
    std::shared_ptr<KVTensorBuilder> valueStateTensorBuilder =
        valueStateTensorBuilderList[currentLayer];
    std::shared_ptr<KVTensorBuilder> childKeyStateTensorBuilder =
        child->keyStateTensorBuilderList[currentLayer];
    std::shared_ptr<KVTensorBuilder> childValueStateTensorBuilder =
        child->valueStateTensorBuilderList[currentLayer];

    uint8_t* keyState =
        keyStateTensorBuilder->data() + index * this->tensorBytes;
    uint8_t* valueState =
        valueStateTensorBuilder->data() + index * this->tensorBytes;
    uint8_t* childKeyState =
        childKeyStateTensorBuilder->data() + childIndex * this->tensorBytes;
    uint8_t* childValueState =
        childValueStateTensorBuilder->data() + childIndex * this->tensorBytes;

    memcpy(childKeyState, keyState, this->tensorBytes);
    memcpy(childValueState, valueState, this->tensorBytes);
  }
  ACQUIRE_BIT_RESOURCE(child->bitmap[childIndex / 64], childIndex % 64);
  FREE_BIT_RESOURCE(this->bitmap[index / 64], index % 64);
  return childIndex;
}

Status KVStateCacheBlockBuilder::Build(Client& client) { return Status::OK(); }

std::shared_ptr<Object> KVStateCacheBlockBuilder::_Seal(Client& client) {
  VINEYARD_CHECK_OK(this->Build(client));

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
  kvStateCacheBlock->meta_.AddKeyValue("tensorBytes", this->tensorBytes);
  kvStateCacheBlock->meta_.AddKeyValue("layer", this->layer);
  // 3. set the object type to meta
  kvStateCacheBlock->meta_.SetTypeName(type_name<KVStateCacheBlock>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(kvStateCacheBlock->meta_, kvStateCacheBlock->id_));
  this->set_sealed(true);
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
      uint8_t* key_state_data = keyStateTensorBuilderList[currentLayer]->data();
      uint8_t* value_state_data =
          valueStateTensorBuilderList[currentLayer]->data();
      // print the first tensorBytes bytes
      std::string keyState = "";
      std::string valueState = "";
      for (int j = 0; j < this->tensorBytes; j++) {
        keyState += std::to_string(key_state_data[i * tensorBytes + j]) + " ";
        valueState +=
            std::to_string(value_state_data[i * tensorBytes + j]) + " ";
      }
      LOG(INFO) << "keyState:" << keyState;
      LOG(INFO) << "valueState:" << valueState;
    }
  }

  LOG(INFO) << "==========================";
}

KVStateCacheBlockBuilder::~KVStateCacheBlockBuilder() { delete this->bitmap; }

}  // namespace vineyard
