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

#include "client/client.h"
#include "common/util/base64.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "llm-cache/ds/kv_state_cache.h"
#include "llm-cache/radix-tree/radix-tree.h"

#include "rax/radix.h"

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
  // raxShow(this->rootTree->GetRootTree());

  // 2. construct the kvStateCacheBlockBuilder list
  size_t numBlocks = this->meta_.GetKeyValue<size_t>("numBlocks");
  for (size_t i = 0; i < numBlocks; i++) {
    std::shared_ptr<Object> kvStateCacheBlockObject = this->meta_.GetMember(
        "kv_state_cache_block_builder_" + std::to_string(i));
    this->kvStateCacheBlockList.push_back(
        std::dynamic_pointer_cast<KVStateCacheBlock>(kvStateCacheBlockObject));
  }

  // 3. construct the member field
  this->dimension = this->meta_.GetKeyValue<int>("dimension");
  this->version = this->meta_.GetKeyValue<uint64_t>("version");
  this->layer = this->meta_.GetKeyValue<int>("layer");
  VLOG(100) << "construct the member field success, with dimension:"
            << this->dimension << " version:" << this->version
            << " layer:" << this->layer;
}

KVStateCache::~KVStateCache() {}

KVStateCacheBuilder::KVStateCacheBuilder(Client& client, int dimension,
                                         int cacheCapacity, int layer,
                                         int blockSize) {
  this->dimension = dimension;
  this->version = 0;
  this->layer = layer;
  KVStateCacheBlockBuilder* builder =
      new KVStateCacheBlockBuilder(client, this->dimension, layer, blockSize);

  this->rootTree = std::make_shared<RadixTree>(cacheCapacity);

  TreeData* treeData = new TreeData();
  treeData->kvStateCacheBlockBuilder = builder;
  treeData->isPtr = true;

  std::shared_ptr<NodeData> rootTreeHeader = this->rootTree->GetRootNode();
  rootTreeHeader->treeData->data = treeData;
  rootTreeHeader->treeData->dataLength = sizeof(TreeData);
  this->rootTree->SetSubtreeData(treeData);
}

KVStateCacheBuilder::KVStateCacheBuilder(Client& client,
                                         std::shared_ptr<KVStateCache> cache) {
  this->dimension = cache->GetDimension();
  this->version = cache->GetVersion();
  this->layer = cache->GetLayer();
  // 1. create block builder from block
  std::vector<std::shared_ptr<KVStateCacheBlock>> kvStateCacheBlockList =
      cache->GetKVStateCacheBlockList();
  this->rootTree = cache->GetRootTree();
  std::set<void*> subTreeData = cache->rootTree->GetSubTreeDataSet();

  for (auto iter = subTreeData.begin(); iter != subTreeData.end(); ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    VINEYARD_ASSERT(treeData->isPtr == false);
    std::shared_ptr<KVStateCacheBlock> kvStateCacheBlock =
        kvStateCacheBlockList[treeData->builderObjectID];
    KVStateCacheBlockBuilder* kvStateCacheBlockBuilder =
        new KVStateCacheBlockBuilder(client, kvStateCacheBlock);

    treeData->kvStateCacheBlockBuilder = kvStateCacheBlockBuilder;
    treeData->isPtr = true;
  }
}

KVStateCacheBlockBuilder* KVStateCacheBuilder::Split(
    Client& client, KVStateCacheBlockBuilder* kvStateCacheBlockBuilder,
    std::vector<std::shared_ptr<NodeData>> nodeDataList) {
  // Split the tree if the list of kvState is full.
  VINEYARD_ASSERT(nodeDataList.size() > 0);
  KVStateCacheBlockBuilder* childKVStateCacheBlockBuilder =
      new KVStateCacheBlockBuilder(client, this->dimension, this->layer,
                                   kvStateCacheBlockBuilder->GetBlockSize());
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
  return childKVStateCacheBlockBuilder;
}

void KVStateCacheBuilder::Update(Client& client,
                                 const std::vector<int>& tokenList,
                                 int nextToken,
                                 const KV_STATE_WITH_LAYER& kvState) {
  std::vector<int> tokenListCopy = tokenList;
  tokenListCopy.push_back(nextToken);

  // Create a empty node of tokens from radix tree.
  std::shared_ptr<NodeData> evictedNodeData = nullptr;
  std::shared_ptr<NodeData> nodeData =
      this->rootTree->Insert(tokenListCopy, evictedNodeData);
  if (nodeData == nullptr) {
    return;
  }
  KVStateCacheBlockBuilder* kvStateCacheBlockBuilder =
      reinterpret_cast<KVStateCacheBlockBuilder*>(
          (reinterpret_cast<TreeData*>(nodeData->treeData->data))
              ->kvStateCacheBlockBuilder);
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
    KVStateCacheBlockBuilder* newKVStateCacheBlockBuilder =
        Split(client, kvStateCacheBlockBuilder, nodeDataList);

    TreeData* newTreeData = new TreeData();
    newTreeData->kvStateCacheBlockBuilder = newKVStateCacheBlockBuilder;
    newTreeData->isPtr = true;

    subTreeHeader->treeData->data = newTreeData;
    subTreeHeader->treeData->dataLength = sizeof(TreeData);
    rootTree->SetSubtreeData(newTreeData);
    VLOG(100) << "block split success";

    // kv_state_cache_builder->UnLock();
    Update(client, tokenList, nextToken, kvState);
  } else {
    // Update the kv-state cache.
    OffsetData* data = new OffsetData();
    kvStateCacheBlockBuilder->Update(kvState, data);
    nodeData->nodeData->data = data;
    nodeData->nodeData->dataLength = sizeof(OffsetData);
  }

  VLOG(100) << "builder:" << kvStateCacheBlockBuilder
            << " bitmap:" << kvStateCacheBlockBuilder->GetBitmapStr();
}

int KVStateCacheBuilder::Query(Client& client,
                               const std::vector<int>& tokenList, int token,
                               KV_STATE_WITH_LAYER& kvState) {
  std::vector<int> tokenListCopy = tokenList;
  tokenListCopy.push_back(token);

  std::shared_ptr<NodeData> nodeData = this->rootTree->Query(tokenListCopy);

  if (nodeData != nullptr) {
    OffsetData* data = reinterpret_cast<OffsetData*>(nodeData->nodeData->data);
    int offset = data->offset;

    KVStateCacheBlockBuilder* kvStateCacheBlockBuilder =
        reinterpret_cast<KVStateCacheBlockBuilder*>(
            (reinterpret_cast<TreeData*>(nodeData->treeData->data))
                ->kvStateCacheBlockBuilder);

    return kvStateCacheBlockBuilder->Query(client, offset, kvState);
  }
  return -1;
}

void KVStateCacheBuilder::Delete(std::shared_ptr<NodeData> evictedNodeData) {
  TreeData* treeData =
      reinterpret_cast<TreeData*>(evictedNodeData->treeData->data);
  KVStateCacheBlockBuilder* kvStateCacheBlockBuilder =
      reinterpret_cast<KVStateCacheBlockBuilder*>(
          treeData->kvStateCacheBlockBuilder);
  OffsetData* data =
      reinterpret_cast<OffsetData*>(evictedNodeData->nodeData->data);
  kvStateCacheBlockBuilder->DeleteKVCache(data->offset);
  delete data;
  // TBD
  // Refactor this code. The data should be deleted by the RadixTree
  // delete (DataWrapper*) evictedNodeData->nodeData;
  if (evictedNodeData->cleanTreeData) {
    this->rootTree->ClearSubtreeData(treeData);
    delete kvStateCacheBlockBuilder;
  }
  evictedNodeData->RecycleSource();
}

void KVStateCacheBuilder::Merge(Client& client,
                                std::shared_ptr<KVStateCache> kvStateCache) {
  if (kvStateCache == nullptr) {
    return;
  }

  std::shared_ptr<KVStateCacheBuilder> globalCacheBuilder =
      std::make_shared<KVStateCacheBuilder>(client, kvStateCache);
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
    KV_STATE_WITH_LAYER kvState;
    for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
      K_STATE key_state;
      V_STATE value_state;
      key_state.data = malloc(this->dimension * sizeof(double));
      key_state.length = this->dimension * sizeof(double);
      value_state.data = malloc(this->dimension * sizeof(double));
      value_state.length = this->dimension * sizeof(double);

      kvState.insert(
          std::make_pair(currentLayer, std::make_pair(key_state, value_state)));
    }
    globalCacheBuilder->Query(client, tokenList, (*it).back(), kvState);
    this->Update(client, tokenList, (*it).back(), kvState);
    for (int currentLayer = 0; currentLayer < this->layer; currentLayer++) {
      K_STATE key_state = kvState[currentLayer].first;
      V_STATE value_state = kvState[currentLayer].second;
      free(key_state.data);
      free(value_state.data);
    }
  }

  this->version = globalCacheBuilder->GetVersion();
  return;
}

Status KVStateCacheBuilder::Build(Client& client) { return Status::OK(); }

std::shared_ptr<Object> KVStateCacheBuilder::_Seal(Client& client) {
  this->Build(client);

  std::shared_ptr<KVStateCache> kvStateCache = std::make_shared<KVStateCache>();

  // 1. store the member variables to cache object meta
  kvStateCache->meta_.AddKeyValue("dimension", this->dimension);
  kvStateCache->meta_.AddKeyValue("version", this->version);
  kvStateCache->meta_.AddKeyValue("layer", this->layer);

  // 2. seal all the block and put object id to cache object and
  // change the tree data from pointer to object id

  int count = 0;
  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    VINEYARD_ASSERT(treeData != nullptr);
    VINEYARD_ASSERT(treeData->isPtr == true);

    KVStateCacheBlockBuilder* kvStateCacheBlockBuilder =
        reinterpret_cast<KVStateCacheBlockBuilder*>(
            treeData->kvStateCacheBlockBuilder);
    std::shared_ptr<Object> kvStateCacheBlock =
        kvStateCacheBlockBuilder->_Seal(client);
    kvStateCache->meta_.AddMember(
        "kv_state_cache_block_builder_" + std::to_string(count),
        kvStateCacheBlock);
    treeData->builderObjectID = count;
    treeData->isPtr = false;
    count++;
  }

  kvStateCache->meta_.AddKeyValue("numBlocks", count);

  // 3. put the serialized sequence radix tree to cache object meta
  kvStateCache->meta_.AddKeyValue("radix_tree",
                                  base64_encode(this->rootTree->Serialize()));

  // 4. put the object type to the meta
  kvStateCache->meta_.SetTypeName(type_name<KVStateCache>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(kvStateCache->meta_, kvStateCache->id_));
  VLOG(100) << "KVStateCacheBuilder::_Seal: " << kvStateCache->id_;
  return kvStateCache;
}

KVStateCacheBuilder::~KVStateCacheBuilder() {
  // get all subtree data and node data
  std::set<void*> subTreeDataSet = rootTree->GetSubTreeDataSet();
  std::set<void*> nodeDataSet = rootTree->GetAllNodeData();
  // 2. delete all subtree data and node data
  for (auto iter = subTreeDataSet.begin(); iter != subTreeDataSet.end();
       ++iter) {
    TreeData* treeData = reinterpret_cast<TreeData*>(*iter);
    if (treeData->isPtr == true) {
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

}  // namespace vineyard
