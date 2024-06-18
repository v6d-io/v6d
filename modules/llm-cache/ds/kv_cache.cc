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

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "rax/radix.h"

#include "client/client.h"
#include "common/util/base64.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "llm-cache/ds/kv_cache.h"
#include "llm-cache/radix-tree/radix-tree.h"

namespace vineyard {

void KVCache::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  Resolve();
}

void KVCache::Resolve() {
  std::string typeName = type_name<KVCache>();

  VINEYARD_ASSERT(this->meta_.GetTypeName() == typeName,
                  "Expect typename '" + typeName + "', but got '" +
                      this->meta_.GetTypeName() + "'");

  // 1. construct the radix tree
  this->rootTree = RadixTree::Deserialize(
      base64_decode(this->meta_.GetKeyValue<std::string>("radix_tree")));
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(this->rootTree->GetRootTree());
  }

  // 2. construct the member field
  this->tensorNBytes = this->meta_.GetKeyValue<int>("tensorNBytes");
  this->version = this->meta_.GetKeyValue<uint64_t>("version");
  this->layer = this->meta_.GetKeyValue<int>("layer");
  VLOG(100) << "construct the member field success, with tensorNBytes:"
            << this->tensorNBytes << " version:" << this->version
            << " layer:" << this->layer;
}

void KVCache::GetCurrentBlockIDSet(std::set<ObjectID>& objectIDSet) {
  std::set<void*> subTreeData = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeData.begin(); iter != subTreeData.end(); ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (!treeData->isPtr) {
      objectIDSet.insert(treeData->builderObjectID);
    }
  }
}

KVCache::~KVCache() {}

KVCacheBuilder::KVCacheBuilder(Client& client, int tensorNBytes, int layer,
                               std::shared_ptr<RadixTree>& rootTree)
    : client(client) {
  this->tensorNBytes = tensorNBytes;
  this->version = 0;
  this->layer = layer;
  this->rootTree = rootTree;
}

Status KVCacheBuilder::Make(Client& client,
                            std::shared_ptr<KVCacheBuilder>& kvCacheBuilder,
                            int tensorNBytes, int cacheCapacity, int layer,
                            int blockSize) {
  KVCacheBlockBuilder* builder =
      new KVCacheBlockBuilder(client, tensorNBytes, layer, blockSize);

  std::shared_ptr<RadixTree> rootTree =
      std::make_shared<RadixTree>(cacheCapacity);

  TreeData* treeData = new TreeData();
  treeData->kvCacheBlockBuilder = builder;
  treeData->isPtr = true;

  std::shared_ptr<NodeData> rootTreeHeader = rootTree->GetRootNode();
  rootTreeHeader->treeData->data = treeData;
  rootTreeHeader->treeData->dataLength = sizeof(TreeData);
  rootTree->SetSubtreeData(treeData);

  kvCacheBuilder = std::shared_ptr<KVCacheBuilder>(
      new KVCacheBuilder(client, tensorNBytes, layer, rootTree));
  return Status::OK();
}

Status KVCacheBuilder::Make(Client& client,
                            std::shared_ptr<KVCacheBuilder>& kvCacheBuilder,
                            std::shared_ptr<KVCache>& cache) {
  kvCacheBuilder = std::make_shared<KVCacheBuilder>(
      client, cache->GetTensorNBytes(), cache->GetLayer(), cache->rootTree);
  return Status::OK();
}

Status KVCacheBuilder::Split(
    KVCacheBlockBuilder* kvCacheBlockBuilder,
    std::vector<std::shared_ptr<NodeData>> nodeDataList,
    KVCacheBlockBuilder*& childKVCacheBlockBuilder) {
  // Split the tree if the list of kvState is full.
  childKVCacheBlockBuilder =
      new KVCacheBlockBuilder(client, this->tensorNBytes, this->layer,
                              kvCacheBlockBuilder->GetBlockSize());
  VINEYARD_ASSERT(childKVCacheBlockBuilder != nullptr,
                  "Not enough memory for new block builder.");

  for (size_t i = 0; i < nodeDataList.size(); i++) {
    OffsetData* data =
        reinterpret_cast<OffsetData*>(nodeDataList[i]->nodeData->data);
    if (data == nullptr)
      continue;
    int index = data->offset;

    // Transfer the data from this builder to the child builder.
    data->offset = kvCacheBlockBuilder->Split(childKVCacheBlockBuilder, index);
  }
  VLOG(100) << "builder:" << kvCacheBlockBuilder
            << " bitmap:" << kvCacheBlockBuilder->GetBitmapStr();
  VLOG(100) << "child_builder:" << childKVCacheBlockBuilder
            << " bitmap:" << childKVCacheBlockBuilder->GetBitmapStr();
  return Status::OK();
}

Status KVCacheBuilder::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  std::vector<int> tokenListCopy = tokenList;
  tokenListCopy.push_back(nextToken);

  // Create a empty node of tokens from radix tree.
  std::shared_ptr<NodeData> evictedNodeData = nullptr;
  std::shared_ptr<NodeData> nodeData =
      this->rootTree->Insert(tokenListCopy, evictedNodeData);
  RETURN_ON_ASSERT(nodeData != nullptr, "Update llm cache failed.");

  KVCacheBlockBuilder* kvCacheBlockBuilder;
  TreeData* treeData = reinterpret_cast<TreeData*>(nodeData->treeData->data);
  if (treeData->isPtr) {
    kvCacheBlockBuilder =
        reinterpret_cast<KVCacheBlockBuilder*>(treeData->kvCacheBlockBuilder);
  } else {
    ObjectID blockObjectID = treeData->builderObjectID;
    RETURN_ON_ERROR(
        KVCacheBlockBuilder::Make(client, treeData, kvCacheBlockBuilder));
    treeData->kvCacheBlockBuilder = kvCacheBlockBuilder;
    treeData->isPtr = true;
    blockIDSetToDelete.insert(blockObjectID);
  }

  if (evictedNodeData != nullptr) {
    Delete(evictedNodeData);
  }

  if (kvCacheBlockBuilder->IsFull()) {
    /**
     * If the kv-state cache of the tree is full, trigger split. Delete the
     * empty node from the radix tree and split the tree. Then, kv-state cache
     * split according to the new tree.
     */
    VLOG(100) << "trigger splits";
    std::shared_ptr<NodeData> evictedNodeData = nullptr;
    this->rootTree->Delete(tokenListCopy, evictedNodeData);

    std::shared_ptr<NodeData> subTreeHeader;
    std::vector<std::shared_ptr<NodeData>> nodeDataList =
        rootTree->Split(tokenListCopy, subTreeHeader);
    RETURN_ON_ASSERT(nodeDataList.size() != 0, "Split llm cache failed.");
    KVCacheBlockBuilder* newKVCacheBlockBuilder;
    Status status =
        Split(kvCacheBlockBuilder, nodeDataList, newKVCacheBlockBuilder);
    RETURN_ON_ERROR(status);

    TreeData* newTreeData = new TreeData();
    RETURN_ON_ASSERT(newTreeData != nullptr, "Split llm cache failed.");
    newTreeData->kvCacheBlockBuilder = newKVCacheBlockBuilder;
    newTreeData->isPtr = true;

    subTreeHeader->treeData->data = newTreeData;
    subTreeHeader->treeData->dataLength = sizeof(TreeData);
    rootTree->SetSubtreeData(newTreeData);
    VLOG(100) << "block split success";

    // kv_cache_builder->UnLock();
    status = Update(tokenList, nextToken, kvState);
    RETURN_ON_ERROR(status);
  } else {
    // Update the kv-state cache.
    OffsetData* data = new OffsetData();
    RETURN_ON_ASSERT(data != nullptr, "Not enough memory for new offset data.");

    RETURN_ON_ERROR(kvCacheBlockBuilder->Update(kvState, data));
    nodeData->nodeData->data = data;
    nodeData->nodeData->dataLength = sizeof(OffsetData);
  }

  VLOG(100) << "builder:" << kvCacheBlockBuilder
            << " bitmap:" << kvCacheBlockBuilder->GetBitmapStr();
  return Status::OK();
}

Status KVCacheBuilder::Query(const std::vector<int>& tokenList, int token,
                             std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  std::vector<int> tokenListCopy = tokenList;
  tokenListCopy.push_back(token);

  std::shared_ptr<NodeData> nodeData = this->rootTree->Query(tokenListCopy);

  RETURN_ON_ASSERT(nodeData != nullptr, "Query llm cache failed.");

  OffsetData* data = reinterpret_cast<OffsetData*>(nodeData->nodeData->data);
  int offset = data->offset;

  TreeData* treeData = reinterpret_cast<TreeData*>(nodeData->treeData->data);
  KVCacheBlockBuilder* kvCacheBlockBuilder;
  if (treeData->isPtr) {
    kvCacheBlockBuilder =
        reinterpret_cast<KVCacheBlockBuilder*>(treeData->kvCacheBlockBuilder);
  } else {
    ObjectID blockObjectID = treeData->builderObjectID;
    RETURN_ON_ERROR(
        KVCacheBlockBuilder::Make(client, treeData, kvCacheBlockBuilder));
    treeData->kvCacheBlockBuilder = kvCacheBlockBuilder;
    treeData->isPtr = true;
    blockIDSetToDelete.insert(blockObjectID);
  }

  return kvCacheBlockBuilder->Query(offset, kvState);
}

void KVCacheBuilder::Delete(std::shared_ptr<NodeData> evictedNodeData) {
  TreeData* treeData =
      reinterpret_cast<TreeData*>(evictedNodeData->treeData->data);
  KVCacheBlockBuilder* kvCacheBlockBuilder;
  if (treeData->isPtr) {
    kvCacheBlockBuilder =
        reinterpret_cast<KVCacheBlockBuilder*>(treeData->kvCacheBlockBuilder);
  } else {
    ObjectID blockObjectID = treeData->builderObjectID;
    Status status =
        KVCacheBlockBuilder::Make(client, treeData, kvCacheBlockBuilder);
    if (!status.ok()) {
      // Not a deadly error, just log it and return.
      LOG(FATAL) << "Failed to make kvCacheBlockBuilder. It may cause "
                    "memory leak.";
      return;
    }
    treeData->kvCacheBlockBuilder = kvCacheBlockBuilder;
    treeData->isPtr = true;

    blockIDSetToDelete.insert(blockObjectID);
  }

  OffsetData* data =
      reinterpret_cast<OffsetData*>(evictedNodeData->nodeData->data);
  kvCacheBlockBuilder->DeleteKVCache(data->offset);
  delete data;
  // TBD
  // Refactor this code. The data should be deleted by the RadixTree
  // delete (DataWrapper*) evictedNodeData->nodeData;
  if (evictedNodeData->cleanTreeData) {
    this->rootTree->ClearSubtreeData(treeData);
    std::shared_ptr<Object> blockObject = kvCacheBlockBuilder->_Seal(client);
    Status status = client.DelData(blockObject->id());
    if (!status.ok()) {
      LOG(ERROR) << "Delete object failed: " << status.ToString()
                 << " It may cause memory leak.";
    }
    delete kvCacheBlockBuilder;
  }
  evictedNodeData->RecycleSource();
}

Status KVCacheBuilder::Merge(std::shared_ptr<KVCache> kvCache) {
  if (kvCache == nullptr) {
    return Status::OK();
  }

  std::shared_ptr<KVCacheBuilder> globalCacheBuilder;
  Status status = KVCacheBuilder::Make(client, globalCacheBuilder, kvCache);
  RETURN_ON_ERROR(status);

  std::shared_ptr<RadixTree> globalCacheTree = kvCache->GetRootTree();

  std::set<std::vector<int>> insertTokenList;
  std::vector<std::vector<int>> evicted_token_list;
  RadixTree::MergeTree(this->rootTree, globalCacheTree, evicted_token_list,
                       insertTokenList);

  VLOG(100) << "insert token list size:" << insertTokenList.size()
            << " evicted token list size:" << evicted_token_list.size();
  for (size_t i = 0; i < evicted_token_list.size(); i++) {
    std::vector<int> tokenList =
        evicted_token_list[evicted_token_list.size() - i - 1];
    std::shared_ptr<NodeData> evictedNodeData;
    this->rootTree->Delete(tokenList, evictedNodeData);
    Delete(evictedNodeData);
  }

  /**
   * Set use lexicographical order to insert the token list, so the insert token
   * list is sorted and will not cause insert failed.(Radix tree will reject a
   * insert operation if the prefix of the insert token list is not in the
   * tree.)
   */
  for (auto it = insertTokenList.begin(); it != insertTokenList.end(); ++it) {
    std::vector<int> tokenList =
        std::vector<int>((*it).begin(), (*it).end() - 1);
    std::vector<std::pair<LLMKV, LLMKV>> kvState;
    for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
      LLMKV key_state;
      LLMKV value_state;
      key_state.data = nullptr;
      key_state.length = 0;
      value_state.data = nullptr;
      value_state.length = 0;

      kvState.emplace_back(key_state, value_state);
    }
    Status status = globalCacheBuilder->Query(tokenList, (*it).back(), kvState);
    if (status.ok()) {
      status = this->Update(tokenList, (*it).back(), kvState);
    }
    RETURN_ON_ERROR(status);
  }

  this->version = globalCacheBuilder->GetVersion();
  globalCacheBuilder->Close();
  return Status::OK();
}

void KVCacheBuilder::GetCurrentBlockIDSet(std::set<ObjectID>& objectIDSet) {
  std::set<void*> subTreeData = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeData.begin(); iter != subTreeData.end(); ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (!treeData->isPtr) {
      objectIDSet.insert(treeData->builderObjectID);
    }
  }
}

Status KVCacheBuilder::Build(Client& client) { return Status::OK(); }

std::shared_ptr<Object> KVCacheBuilder::_Seal(Client& client) {
  VINEYARD_CHECK_OK(this->Build(client));

  std::shared_ptr<KVCache> kvCache = std::make_shared<KVCache>();

  // 1. store the member variables to cache object meta
  kvCache->meta_.AddKeyValue("tensorNBytes", this->tensorNBytes);
  kvCache->meta_.AddKeyValue("version", this->version);
  kvCache->meta_.AddKeyValue("layer", this->layer);

  // 2. seal all the block and put object id to cache object and
  // change the tree data from pointer to object id

  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (!treeData->isPtr) {
      continue;
    }

    KVCacheBlockBuilder* kvCacheBlockBuilder =
        reinterpret_cast<KVCacheBlockBuilder*>(treeData->kvCacheBlockBuilder);
    std::shared_ptr<Object> kvCacheBlock = kvCacheBlockBuilder->_Seal(client);
    VINEYARD_CHECK_OK(client.Persist(kvCacheBlock->id()));
    treeData->builderObjectID = kvCacheBlock->id();
    treeData->isPtr = false;
  }

  // 3. put the serialized sequence radix tree to cache object meta
  kvCache->meta_.AddKeyValue("radix_tree",
                             base64_encode(this->rootTree->Serialize()));

  // 4. put the object type to the meta
  kvCache->meta_.SetTypeName(type_name<KVCache>());

  VINEYARD_CHECK_OK(client.CreateMetaData(kvCache->meta_, kvCache->id_));
  VLOG(100) << "KVCacheBuilder::_Seal: " << kvCache->id_;
  this->set_sealed(true);
  return kvCache;
}

KVCacheBuilder::~KVCacheBuilder() {
  // get all subtree data and node data
  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  std::set<void*> nodeDataSet = rootTree->GetAllNodeData();
  // delete all subtree data and node data
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (treeData->isPtr == true && treeData->kvCacheBlockBuilder != nullptr) {
      delete reinterpret_cast<KVCacheBlockBuilder*>(
          treeData->kvCacheBlockBuilder);
      delete treeData;
    }
  }
  for (auto iter = nodeDataSet.begin(); iter != nodeDataSet.end(); ++iter) {
    OffsetData* data = reinterpret_cast<OffsetData*>(*iter);
    if (data != nullptr) {
      delete data;
    }
  }
}

void KVCacheBuilder::Close() {
  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (treeData->isPtr && treeData->kvCacheBlockBuilder != nullptr) {
      std::shared_ptr<Object> object =
          reinterpret_cast<KVCacheBlockBuilder*>(treeData->kvCacheBlockBuilder)
              ->_Seal(client);
      Status status = client.DelData(object->id());
      if (!status.ok()) {
        LOG(ERROR) << "Delete object failed: " << status.ToString()
                   << " It may cause memory leak.";
      }
    } else if (!treeData->isPtr) {
      blockIDSetToDelete.insert(treeData->builderObjectID);
    }
  }
}

}  // namespace vineyard
