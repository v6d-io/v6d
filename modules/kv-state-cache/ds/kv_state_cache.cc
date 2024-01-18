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

#include "client/client.h"
#include "common/util/base64.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "kv-state-cache/radix-tree/radix-tree.h"
#include "kv_state_cache.h"

namespace vineyard {

void KVStateCache::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  Resolve();
}

void KVStateCache::Resolve() {
  LOG(INFO) << "Resolve";
  std::string typeName = type_name<KVStateCache>();

  VINEYARD_ASSERT(this->meta_.GetTypeName() == typeName,
                  "Expect typename '" + typeName + "', but got '" +
                      this->meta_.GetTypeName() + "'");

  // 1. construct the kv_state_cache_block_builder
  this->kv_state_cache_block = std::dynamic_pointer_cast<KVStateCacheBlock>(
      this->meta_.GetMember("root_kv_state_cache_block"));
  // 2. construct the radix tree
  this->root_tree = RadixTree::Deserialize(
      base64_decode(this->meta_.GetKeyValue<std::string>("radix_tree")));
  // 3. construct the member field
  this->dimension = this->meta_.GetKeyValue<int>("dimension");
}

KVStateCache::~KVStateCache() {
  // TBD
}

KVStateCacheBuilder::KVStateCacheBuilder(Client& client, int dimension,
                                         int cache_capacity) {
  this->dimension = dimension;
  this->version = 0;
  this->kv_state_cache_block_builder =
      std::make_shared<KVStateCacheBlockBuilder>(client, this->dimension);

  this->root_tree = std::make_shared<RadixTree>(cache_capacity);
  this->root_tree->SetCustomData(this->kv_state_cache_block_builder.get(),
                                 sizeof(KVStateCacheBlockBuilder));
}

KVStateCacheBuilder::KVStateCacheBuilder(Client& client,
                                         std::shared_ptr<KVStateCache> cache) {
  // TBD
  this->dimension = cache->GetDemension();
  this->version = cache->GetVersion();
  this->kv_state_cache_block_builder =
      std::make_shared<KVStateCacheBlockBuilder>(client,
                                                 cache->GetKVStateCacheBlock());

  this->root_tree = cache->GetRootTree();
  this->root_tree->SetCustomData(this->kv_state_cache_block_builder.get(),
                                 sizeof(KVStateCacheBlockBuilder));
}

KVStateCacheBlockBuilder* KVStateCacheBuilder::Split(
    Client& client, KVStateCacheBlockBuilder* kv_state_cache_block_builder,
    std::vector<std::shared_ptr<NodeWithTreeAttri>> node_with_tree_attri_list) {
  // Split the tree if the list of kv_state is full.
  VINEYARD_ASSERT(node_with_tree_attri_list.size() > 0);
  KVStateCacheBlockBuilder* child_kv_state_cache_block_builder =
      new KVStateCacheBlockBuilder(client, this->dimension);
  for (size_t i = 0; i < node_with_tree_attri_list.size(); i++) {
    offset_data* data =
        (offset_data*) node_with_tree_attri_list[i]->get_node()->get_data();
    int index = data->offset;

    // Transfer the data from this builder to the child builder.
    const std::shared_ptr<TensorBuilder<double>> k_builder =
        kv_state_cache_block_builder->getKBuilder();
    const std::shared_ptr<TensorBuilder<double>> v_builder =
        kv_state_cache_block_builder->getVBuilder();
    offset_data* new_offset_data = new offset_data();
    child_kv_state_cache_block_builder->Update(
        k_builder->data() + index * this->dimension,
        v_builder->data() + index * this->dimension,
        this->dimension * sizeof(double), new_offset_data);
    node_with_tree_attri_list[i]->get_node()->set_data(new_offset_data,
                                                       sizeof(offset_data));
    // Clear the bitmap.
    kv_state_cache_block_builder->DeleteKVCache(index);
  }
  kv_state_cache_block_builder->SetChildKVStateCacheBlockBuilder(
      child_kv_state_cache_block_builder);
  return child_kv_state_cache_block_builder;
}

void KVStateCacheBuilder::Update(Client& client,
                                 const std::vector<int>& token_list,
                                 int next_token,
                                 const KV_STATE_WITH_LAYER& kv_state) {
  LOG(INFO) << "update";
  std::vector<int> token_list_copy = token_list;
  token_list_copy.push_back(next_token);

  // Create a empty node of tokens from radix tree.
  std::shared_ptr<NodeWithTreeAttri> evicted_node = nullptr;
  std::shared_ptr<NodeWithTreeAttri> node_with_tree_attri =
      this->root_tree->Insert(token_list_copy, evicted_node);
  if (node_with_tree_attri == nullptr) {
    LOG(INFO) << "insert failed";
    return;
  }
  std::shared_ptr<RadixTree> sub_tree = node_with_tree_attri->get_tree();
  KVStateCacheBlockBuilder* kv_state_cache_block_builder =
      (KVStateCacheBlockBuilder*) sub_tree->GetCustomData();
  if (evicted_node != nullptr) {
    offset_data* data = (offset_data*) evicted_node->get_node()->get_data();
    KVStateCacheBlockBuilder* builder =
        (KVStateCacheBlockBuilder*) evicted_node->get_tree()->GetCustomData();
    builder->DeleteKVCache(data->offset);

    delete (offset_data*) evicted_node->get_node()->get_data();
  }

  // TBD
  // Use lock to protect the kv_state_cache_builder
  // kv_state_cache_builder->Lock();

  if (kv_state_cache_block_builder->IsFull()) {
    /**
     * If the kv-state cache of the tree is full, triggle split. Delete the
     * empty node from the radix tree and split the tree. Then, kv-state cache
     * split according to the new tree.
     */
    std::shared_ptr<NodeWithTreeAttri> evicted_node = nullptr;
    this->root_tree->Delete(token_list_copy, evicted_node);
    std::shared_ptr<RadixTree> new_tree = sub_tree->Split(token_list_copy);

    std::vector<std::shared_ptr<NodeWithTreeAttri>> node_with_tree_attri_list =
        RadixTree::TraverseTreeWithoutSubTree(new_tree);
    KVStateCacheBlockBuilder* new_kv_state_cache_block_builder =
        Split(client, kv_state_cache_block_builder, node_with_tree_attri_list);
    new_tree->SetCustomData(new_kv_state_cache_block_builder,
                            sizeof(KVStateCacheBlockBuilder));

    // kv_state_cache_builder->UnLock();
    Update(client, token_list, next_token, kv_state);
  } else {
    // Update the kv-state cache.
    offset_data* data = new offset_data();
    kv_state_cache_block_builder->Update(kv_state, data);
    std::shared_ptr<Node> node = node_with_tree_attri->get_node();
    node->set_data(data, sizeof(offset_data));
  }

  LOG(INFO) << "bitmap:" << kv_state_cache_block_builder->GetBitmapStr();
}

static std::shared_ptr<Node> node;

KV_STATE_WITH_LAYER KVStateCacheBuilder::Query(
    Client& client, const std::vector<int>& token_list, int token) {
  std::vector<int> token_list_copy = token_list;
  token_list_copy.push_back(token);

  KV_STATE_WITH_LAYER kv_state;
  std::shared_ptr<NodeWithTreeAttri> node_with_tree_attri =
      this->root_tree->Query(token_list_copy);
  /**/
  if (node_with_tree_attri != nullptr) {
    offset_data* data =
        (offset_data*) node_with_tree_attri->get_node()->get_data();
    int offset = data->offset;

    KVStateCacheBlockBuilder* kv_state_cache_block_builder =
        (KVStateCacheBlockBuilder*) node_with_tree_attri->get_tree()
            ->GetCustomData();
    // kv_state_cache_builder->Lock();

    kv_state_cache_block_builder->Query(client, offset, kv_state);
    // kv_state_cache_builder->UnLock();
    node = node_with_tree_attri->get_node();
  }
  LOG(INFO) << "query success";
  return kv_state;
}

std::shared_ptr<KVStateCacheBuilder> KVStateCacheBuilder::Merge(
    Client& client, std::shared_ptr<KVStateCache> kv_state_cache) {
  // TBD
  if (kv_state_cache == nullptr) {
    return nullptr;
  }
  // VINEYARD_ASSERT(false);
  return nullptr;
}

Status KVStateCacheBuilder::Build(Client& client) {
  // TBD
  return Status::OK();
}

std::shared_ptr<Object> KVStateCacheBuilder::_Seal(Client& client) {
  this->Build(client);

  std::shared_ptr<KVStateCache> kv_state_cache =
      std::make_shared<KVStateCache>();

  // 1. store the member variables to cache object meta
  kv_state_cache->meta_.AddKeyValue("dimension", this->dimension);

  // 2. seal all the kv_state_cache_block
  // 3. put cache_block_object_id to cache object meta
  kv_state_cache->meta_.AddMember(
      "root_kv_state_cache_block",
      this->kv_state_cache_block_builder->_Seal(client));

  // 4. put the serialized sequence radix tree to cache object meta
  kv_state_cache->meta_.AddKeyValue(
      "radix_tree", base64_encode(this->root_tree->Serialize()));

  // 5. put the object type to the meta
  kv_state_cache->meta_.SetTypeName(type_name<KVStateCache>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(kv_state_cache->meta_, kv_state_cache->id_));
  LOG(INFO) << "KVStateCacheBuilder::_Seal: " << kv_state_cache->id_;
  return kv_state_cache;
}

KVStateCacheBuilder::~KVStateCacheBuilder() {
  // TBD
  std::vector<std::shared_ptr<NodeWithTreeAttri>> node_with_tree_attri_list =
      RadixTree::TraverseTreeWithoutSubTree(this->root_tree);
  for (size_t i = 0; i < node_with_tree_attri_list.size(); i++) {
    delete (offset_data*) node_with_tree_attri_list[i]->get_node()->get_data();
  }
}

}  // namespace vineyard
