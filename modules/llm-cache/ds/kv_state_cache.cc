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
#include "llm-cache/ds/kv_state_cache.h"
#include "llm-cache/radix-tree/radix-tree.h"

namespace vineyard {

void KVStateCache::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  Resolve();
}

void KVStateCache::Resolve() {
  std::string typeName = type_name<KVStateCache>();

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
  this->tensorBytes = this->meta_.GetKeyValue<int>("tensorBytes");
  this->version = this->meta_.GetKeyValue<uint64_t>("version");
  this->layer = this->meta_.GetKeyValue<int>("layer");
  VLOG(100) << "construct the member field success, with tensorBytes:"
            << this->tensorBytes << " version:" << this->version
            << " layer:" << this->layer;
}

void KVStateCache::GetCurrentBlockIDSet(std::set<ObjectID>& objectIDSet) {
  std::set<void*> subTreeData = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeData.begin(); iter != subTreeData.end(); ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (!treeData->isPtr) {
      objectIDSet.insert(treeData->builderObjectID);
    }
  }
}

KVStateCache::~KVStateCache() {}

KVStateCacheBuilder::KVStateCacheBuilder(Client& client, int tensorBytes,
                                         int layer,
                                         std::shared_ptr<RadixTree>& rootTree)
    : client(client) {
  this->tensorBytes = tensorBytes;
  this->version = 0;
  this->layer = layer;
  this->rootTree = rootTree;
}

Status KVStateCacheBuilder::Make(
    Client& client, std::shared_ptr<KVStateCacheBuilder>& kvStateCacheBuilder,
    int tensorBytes, int cacheCapacity, int layer, int blockSize) {
  KVStateCacheBlockBuilder* builder =
      new KVStateCacheBlockBuilder(client, tensorBytes, layer, blockSize);

  std::shared_ptr<RadixTree> rootTree =
      std::make_shared<RadixTree>(cacheCapacity);

  TreeData* treeData = new TreeData();
  treeData->kvStateCacheBlockBuilder = builder;
  treeData->isPtr = true;

  std::shared_ptr<NodeData> rootTreeHeader = rootTree->GetRootNode();
  rootTreeHeader->treeData->data = treeData;
  rootTreeHeader->treeData->dataLength = sizeof(TreeData);
  rootTree->SetSubtreeData(treeData);

  kvStateCacheBuilder = std::shared_ptr<KVStateCacheBuilder>(
      new KVStateCacheBuilder(client, tensorBytes, layer, rootTree));
  return Status::OK();
}

Status KVStateCacheBuilder::Make(
    Client& client, std::shared_ptr<KVStateCacheBuilder>& kvStateCacheBuilder,
    std::shared_ptr<KVStateCache>& cache) {
  kvStateCacheBuilder = std::make_shared<KVStateCacheBuilder>(
      client, cache->GetTensorBytes(), cache->GetLayer(), cache->rootTree);
  return Status::OK();
}

Status KVStateCacheBuilder::Split(
    KVStateCacheBlockBuilder* kvStateCacheBlockBuilder,
    std::vector<std::shared_ptr<NodeData>> nodeDataList,
    KVStateCacheBlockBuilder*& childKVStateCacheBlockBuilder) {
  // Split the tree if the list of kvState is full.
  childKVStateCacheBlockBuilder =
      new KVStateCacheBlockBuilder(client, this->tensorBytes, this->layer,
                                   kvStateCacheBlockBuilder->GetBlockSize());
  VINEYARD_ASSERT(childKVStateCacheBlockBuilder != nullptr,
                  "Not enough memory for new block builder.");

  for (size_t i = 0; i < nodeDataList.size(); i++) {
    OffsetData* data =
        reinterpret_cast<OffsetData*>(nodeDataList[i]->nodeData->data);
    if (data == nullptr)
      continue;
    int index = data->offset;

    // Transfer the data from this builder to the child builder.
    data->offset =
        kvStateCacheBlockBuilder->Split(childKVStateCacheBlockBuilder, index);
  }
  VLOG(100) << "builder:" << kvStateCacheBlockBuilder
            << " bitmap:" << kvStateCacheBlockBuilder->GetBitmapStr();
  VLOG(100) << "child_builder:" << childKVStateCacheBlockBuilder
            << " bitmap:" << childKVStateCacheBlockBuilder->GetBitmapStr();
  return Status::OK();
}

Status KVStateCacheBuilder::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  std::vector<int> tokenListCopy = tokenList;
  tokenListCopy.push_back(nextToken);

  // Create a empty node of tokens from radix tree.
  std::shared_ptr<NodeData> evictedNodeData = nullptr;
  std::shared_ptr<NodeData> nodeData =
      this->rootTree->Insert(tokenListCopy, evictedNodeData);
  RETURN_ON_ASSERT(nodeData != nullptr, "Update llm cache failed.");

  KVStateCacheBlockBuilder* kvStateCacheBlockBuilder;
  TreeData* treeData = reinterpret_cast<TreeData*>(nodeData->treeData->data);
  if (treeData->isPtr) {
    kvStateCacheBlockBuilder = reinterpret_cast<KVStateCacheBlockBuilder*>(
        treeData->kvStateCacheBlockBuilder);
  } else {
    ObjectID blockObjectID = treeData->builderObjectID;
    RETURN_ON_ERROR(KVStateCacheBlockBuilder::Make(client, treeData,
                                                   kvStateCacheBlockBuilder));
    treeData->kvStateCacheBlockBuilder = kvStateCacheBlockBuilder;
    treeData->isPtr = true;
    blockIDSetToDelete.insert(blockObjectID);
  }

  if (evictedNodeData != nullptr) {
    Delete(evictedNodeData);
  }

  if (kvStateCacheBlockBuilder->IsFull()) {
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
    KVStateCacheBlockBuilder* newKVStateCacheBlockBuilder;
    Status status = Split(kvStateCacheBlockBuilder, nodeDataList,
                          newKVStateCacheBlockBuilder);
    RETURN_ON_ERROR(status);

    TreeData* newTreeData = new TreeData();
    RETURN_ON_ASSERT(newTreeData != nullptr, "Split llm cache failed.");
    newTreeData->kvStateCacheBlockBuilder = newKVStateCacheBlockBuilder;
    newTreeData->isPtr = true;

    subTreeHeader->treeData->data = newTreeData;
    subTreeHeader->treeData->dataLength = sizeof(TreeData);
    rootTree->SetSubtreeData(newTreeData);
    VLOG(100) << "block split success";

    // kv_state_cache_builder->UnLock();
    status = Update(tokenList, nextToken, kvState);
    RETURN_ON_ERROR(status);
  } else {
    // Update the kv-state cache.
    OffsetData* data = new OffsetData();
    RETURN_ON_ASSERT(data != nullptr, "Not enough memory for new offset data.");

    RETURN_ON_ERROR(kvStateCacheBlockBuilder->Update(kvState, data));
    nodeData->nodeData->data = data;
    nodeData->nodeData->dataLength = sizeof(OffsetData);
  }

  VLOG(100) << "builder:" << kvStateCacheBlockBuilder
            << " bitmap:" << kvStateCacheBlockBuilder->GetBitmapStr();
  return Status::OK();
}

Status KVStateCacheBuilder::Query(
    const std::vector<int>& tokenList, int token,
    std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  std::vector<int> tokenListCopy = tokenList;
  tokenListCopy.push_back(token);

  std::shared_ptr<NodeData> nodeData = this->rootTree->Query(tokenListCopy);

  RETURN_ON_ASSERT(nodeData != nullptr, "Query llm cache failed.");

  OffsetData* data = reinterpret_cast<OffsetData*>(nodeData->nodeData->data);
  int offset = data->offset;

  TreeData* treeData = reinterpret_cast<TreeData*>(nodeData->treeData->data);
  KVStateCacheBlockBuilder* kvStateCacheBlockBuilder;
  if (treeData->isPtr) {
    kvStateCacheBlockBuilder = reinterpret_cast<KVStateCacheBlockBuilder*>(
        treeData->kvStateCacheBlockBuilder);
  } else {
    ObjectID blockObjectID = treeData->builderObjectID;
    RETURN_ON_ERROR(KVStateCacheBlockBuilder::Make(client, treeData,
                                                   kvStateCacheBlockBuilder));
    treeData->kvStateCacheBlockBuilder = kvStateCacheBlockBuilder;
    treeData->isPtr = true;
    blockIDSetToDelete.insert(blockObjectID);
  }

  return kvStateCacheBlockBuilder->Query(offset, kvState);
}

void KVStateCacheBuilder::Delete(std::shared_ptr<NodeData> evictedNodeData) {
  TreeData* treeData =
      reinterpret_cast<TreeData*>(evictedNodeData->treeData->data);
  KVStateCacheBlockBuilder* kvStateCacheBlockBuilder;
  if (treeData->isPtr) {
    kvStateCacheBlockBuilder = reinterpret_cast<KVStateCacheBlockBuilder*>(
        treeData->kvStateCacheBlockBuilder);
  } else {
    ObjectID blockObjectID = treeData->builderObjectID;
    Status status = KVStateCacheBlockBuilder::Make(client, treeData,
                                                   kvStateCacheBlockBuilder);
    if (!status.ok()) {
      // Not a deadly error, just log it and return.
      LOG(FATAL) << "Failed to make kvStateCacheBlockBuilder. It may cause "
                    "memory leak.";
      return;
    }
    treeData->kvStateCacheBlockBuilder = kvStateCacheBlockBuilder;
    treeData->isPtr = true;

    blockIDSetToDelete.insert(blockObjectID);
  }

  OffsetData* data =
      reinterpret_cast<OffsetData*>(evictedNodeData->nodeData->data);
  kvStateCacheBlockBuilder->DeleteKVCache(data->offset);
  delete data;
  // TBD
  // Refactor this code. The data should be deleted by the RadixTree
  // delete (DataWrapper*) evictedNodeData->nodeData;
  if (evictedNodeData->cleanTreeData) {
    this->rootTree->ClearSubtreeData(treeData);
    std::shared_ptr<Object> blockObject =
        kvStateCacheBlockBuilder->_Seal(client);
    Status status = client.DelData(blockObject->id());
    if (!status.ok()) {
      LOG(ERROR) << "Delete object failed: " << status.ToString()
                 << " It may cause memory leak.";
    }
    delete kvStateCacheBlockBuilder;
  }
  evictedNodeData->RecycleSource();
}

Status KVStateCacheBuilder::Merge(std::shared_ptr<KVStateCache> kvStateCache) {
  if (kvStateCache == nullptr) {
    return Status::OK();
  }

  std::shared_ptr<KVStateCacheBuilder> globalCacheBuilder;
  Status status =
      KVStateCacheBuilder::Make(client, globalCacheBuilder, kvStateCache);
  RETURN_ON_ERROR(status);

  std::shared_ptr<RadixTree> globalCacheTree = kvStateCache->GetRootTree();

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
    std::map<int, std::pair<LLMKV, LLMKV>> kvState;
    for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
      LLMKV key_state;
      LLMKV value_state;
      key_state.data = malloc(this->tensorBytes);
      key_state.length = this->tensorBytes;
      value_state.data = malloc(this->tensorBytes);
      value_state.length = this->tensorBytes;

      kvState.insert(
          std::make_pair(currentLayer, std::make_pair(key_state, value_state)));
    }
    Status status = globalCacheBuilder->Query(tokenList, (*it).back(), kvState);
    if (status.ok()) {
      status = this->Update(tokenList, (*it).back(), kvState);
    }
    for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
      LLMKV key_state = kvState[currentLayer].first;
      LLMKV value_state = kvState[currentLayer].second;
      free(key_state.data);
      free(value_state.data);
    }
    RETURN_ON_ERROR(status);
  }

  this->version = globalCacheBuilder->GetVersion();
  globalCacheBuilder->Close();
  return Status::OK();
}

void KVStateCacheBuilder::GetCurrentBlockIDSet(
    std::set<ObjectID>& objectIDSet) {
  std::set<void*> subTreeData = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeData.begin(); iter != subTreeData.end(); ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (!treeData->isPtr) {
      objectIDSet.insert(treeData->builderObjectID);
    }
  }
}

Status KVStateCacheBuilder::Build(Client& client) { return Status::OK(); }

std::shared_ptr<Object> KVStateCacheBuilder::_Seal(Client& client) {
  VINEYARD_CHECK_OK(this->Build(client));

  std::shared_ptr<KVStateCache> kvStateCache = std::make_shared<KVStateCache>();

  // 1. store the member variables to cache object meta
  kvStateCache->meta_.AddKeyValue("tensorBytes", this->tensorBytes);
  kvStateCache->meta_.AddKeyValue("version", this->version);
  kvStateCache->meta_.AddKeyValue("layer", this->layer);

  // 2. seal all the block and put object id to cache object and
  // change the tree data from pointer to object id

  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (!treeData->isPtr) {
      continue;
    }

    KVStateCacheBlockBuilder* kvStateCacheBlockBuilder =
        reinterpret_cast<KVStateCacheBlockBuilder*>(
            treeData->kvStateCacheBlockBuilder);
    std::shared_ptr<Object> kvStateCacheBlock =
        kvStateCacheBlockBuilder->_Seal(client);
    VINEYARD_CHECK_OK(client.Persist(kvStateCacheBlock->id()));
    treeData->builderObjectID = kvStateCacheBlock->id();
    treeData->isPtr = false;
  }

  // 3. put the serialized sequence radix tree to cache object meta
  kvStateCache->meta_.AddKeyValue("radix_tree",
                                  base64_encode(this->rootTree->Serialize()));

  // 4. put the object type to the meta
  kvStateCache->meta_.SetTypeName(type_name<KVStateCache>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(kvStateCache->meta_, kvStateCache->id_));
  VLOG(100) << "KVStateCacheBuilder::_Seal: " << kvStateCache->id_;
  this->set_sealed(true);
  return kvStateCache;
}

KVStateCacheBuilder::~KVStateCacheBuilder() {
  // get all subtree data and node data
  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  std::set<void*> nodeDataSet = rootTree->GetAllNodeData();
  // delete all subtree data and node data
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (treeData->isPtr == true &&
        treeData->kvStateCacheBlockBuilder != nullptr) {
      delete reinterpret_cast<KVStateCacheBlockBuilder*>(
          treeData->kvStateCacheBlockBuilder);
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

void KVStateCacheBuilder::Close() {
  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (treeData->isPtr && treeData->kvStateCacheBlockBuilder != nullptr) {
      std::shared_ptr<Object> object =
          reinterpret_cast<KVStateCacheBlockBuilder*>(
              treeData->kvStateCacheBlockBuilder)
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
