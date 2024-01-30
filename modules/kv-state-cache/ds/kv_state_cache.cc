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
#include "kv-state-cache/radix-tree/radix.h"
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

  // 1. construct the radix tree
  this->root_tree = RadixTree::Deserialize(
      base64_decode(this->meta_.GetKeyValue<std::string>("radix_tree")));
  LOG(INFO) << "Resolve RadixTree success" << std::endl;
  raxShow(this->root_tree->GetTree());

  // 2. construct the kv_state_cache_block_builder list
  size_t num_blocks = this->meta_.GetKeyValue<size_t>("num_blocks");
  LOG(INFO) << "num blocks:" << num_blocks;
  for (size_t i = 0; i < num_blocks; i++) {
    std::shared_ptr<Object> kv_state_cache_block_object =
        this->meta_.GetMember("kv_state_cache_block_builder_" + std::to_string(i));
    this->kv_state_cache_block_list.push_back(
        std::dynamic_pointer_cast<KVStateCacheBlock>(kv_state_cache_block_object));
    this->kv_state_cache_block_map[kv_state_cache_block_object->id()] =
        std::dynamic_pointer_cast<KVStateCacheBlock>(kv_state_cache_block_object);
    LOG(INFO) << "kv_state_cache_block_object:" << kv_state_cache_block_object->id();
  }

  // 3. construct the member field
  this->dimension = this->meta_.GetKeyValue<int>("dimension");
  LOG(INFO) << "construct the member field success" << std::endl;
}

KVStateCache::~KVStateCache() {
  // TBD
}

KVStateCacheBuilder::KVStateCacheBuilder(Client& client, int dimension,
                                         int cache_capacity) {
  this->dimension = dimension;
  this->version = 0;
  KVStateCacheBlockBuilder* builder = new KVStateCacheBlockBuilder(client, this->dimension);

  this->root_tree = std::make_shared<RadixTree>(cache_capacity);

  TreeData  *tree_data = new TreeData();
  tree_data->kv_state_cache_block_builder = builder;
  tree_data->is_ptr = true;

  this->root_tree->SetCustomData(tree_data, sizeof(TreeData));
  LOG(INFO) << "set builder:" << builder << " to tree:" << this->root_tree->GetTree()->head;
  // this->AddKVStateCacheBlockBuilder(builder, this->root_tree);
}

KVStateCacheBuilder::KVStateCacheBuilder(Client& client,
                                         std::shared_ptr<KVStateCache> cache) {
  // TBD
  this->dimension = cache->GetDemension();
  this->version = cache->GetVersion();
  // 1. create block builder from block
  std::map<uint64_t, std::shared_ptr<KVStateCacheBlock>> kv_state_cache_block_map =
      cache->kv_state_cache_block_map;
  this->root_tree = cache->GetRootTree();
  std::set<std::shared_ptr<RadixTree>> radix_tree_list = cache->root_tree->GetSubTreeSet();

  for (auto iter = radix_tree_list.begin(); iter != radix_tree_list.end(); ++iter) {
    std::shared_ptr<RadixTree> radix_tree = *iter;
    TreeData *tree_data = (TreeData *)radix_tree->GetCustomData();
    VINEYARD_ASSERT(tree_data->is_ptr == false);
    std::shared_ptr<KVStateCacheBlock> kv_state_cache_block =
        kv_state_cache_block_map[tree_data->builder_object_id];
    KVStateCacheBlockBuilder* kv_state_cache_block_builder =
        new KVStateCacheBlockBuilder(client, kv_state_cache_block);

    tree_data->kv_state_cache_block_builder = kv_state_cache_block_builder;
    tree_data->is_ptr = true;

    // this->AddKVStateCacheBlockBuilder(kv_state_cache_block_builder, radix_tree);
  }
}

KVStateCacheBlockBuilder* KVStateCacheBuilder::Split(
    Client& client, KVStateCacheBlockBuilder* kv_state_cache_block_builder,
    std::vector<std::shared_ptr<NodeWithTreeAttri>> node_with_tree_attri_list) {
  LOG(INFO) << "split";
  // Split the tree if the list of kv_state is full.
  VINEYARD_ASSERT(node_with_tree_attri_list.size() > 0);
  KVStateCacheBlockBuilder* child_kv_state_cache_block_builder =
      new KVStateCacheBlockBuilder(client, this->dimension);
  for (size_t i = 0; i < node_with_tree_attri_list.size(); i++) {
    offset_data* data =
        (offset_data*) node_with_tree_attri_list[i]->get_node()->get_data();
    if (data == nullptr)
      continue;
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
        this->dimension, new_offset_data);
    node_with_tree_attri_list[i]->get_node()->set_data(new_offset_data,
                                                       sizeof(offset_data));
    // Clear the bitmap.
    kv_state_cache_block_builder->DeleteKVCache(index);
  }
  LOG(INFO) << "builder:" << kv_state_cache_block_builder << " bitmap:" << kv_state_cache_block_builder->GetBitmapStr();
  LOG(INFO) << "child_builder:" << child_kv_state_cache_block_builder << " bitmap:" << child_kv_state_cache_block_builder->GetBitmapStr();
  // kv_state_cache_block_builder->SetChildKVStateCacheBlockBuilder(
  //     child_kv_state_cache_block_builder,
  //     node_with_tree_attri_list[0]->get_tree());
  // AddKVStateCacheBlockBuilder(child_kv_state_cache_block_builder,
  //                             node_with_tree_attri_list[0]->get_tree());
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
  std::shared_ptr<RadixTree> tree = node_with_tree_attri->get_tree();
  KVStateCacheBlockBuilder* kv_state_cache_block_builder = (KVStateCacheBlockBuilder* )
      ((TreeData*) tree->GetCustomData())->kv_state_cache_block_builder;
  if (evicted_node != nullptr) {
    Delete(evicted_node);
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
    LOG(INFO) << "triggle splits";
    LOG(INFO) << "split tree:" << tree->GetTree();
    std::shared_ptr<NodeWithTreeAttri> evicted_node = nullptr;
    this->root_tree->Delete(token_list_copy, evicted_node);
    std::shared_ptr<RadixTree> new_tree = root_tree->Split(token_list_copy);
    LOG(INFO) << "root tree:" << root_tree << " new tree:" << new_tree;

    std::vector<std::shared_ptr<NodeWithTreeAttri>> node_with_tree_attri_list =
        RadixTree::TraverseTreeWithoutSubTree(new_tree);
    KVStateCacheBlockBuilder* new_kv_state_cache_block_builder =
        Split(client, kv_state_cache_block_builder, node_with_tree_attri_list);

    TreeData* new_tree_data = new TreeData();
    new_tree_data->kv_state_cache_block_builder =
        new_kv_state_cache_block_builder;
    new_tree_data->is_ptr = true;

    new_tree->SetCustomData(new_tree_data, sizeof(TreeData));
    LOG(INFO) << "set builder:" << new_kv_state_cache_block_builder << " to tree:" << new_tree->GetTree()->head << " tree data:" << new_tree_data;
    LOG(INFO) << "block split success";

    // kv_state_cache_builder->UnLock();
    Update(client, token_list, next_token, kv_state);
  } else {
    // Update the kv-state cache.
    offset_data* data = new offset_data();
    kv_state_cache_block_builder->Update(kv_state, data);
    std::shared_ptr<Node> node = node_with_tree_attri->get_node();
    node->set_data(data, sizeof(offset_data));
  }

  LOG(INFO) << "builder:" << kv_state_cache_block_builder << " bitmap:" << kv_state_cache_block_builder->GetBitmapStr();
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

    KVStateCacheBlockBuilder* kv_state_cache_block_builder = (KVStateCacheBlockBuilder* )
        ((TreeData*) node_with_tree_attri->get_tree()
            ->GetCustomData())->kv_state_cache_block_builder;
    // kv_state_cache_builder->Lock();

    LOG(INFO) << "offset:" << offset;
    LOG(INFO) << "kv_state_cache_block_builder:" << kv_state_cache_block_builder;
    kv_state_cache_block_builder->Query(client, offset, kv_state);
    // kv_state_cache_builder->UnLock();
    node = node_with_tree_attri->get_node();
  }
  return kv_state;
}

void KVStateCacheBuilder::Delete(std::shared_ptr<NodeWithTreeAttri> evicted_node) {
  LOG(INFO) << "stage1";
  KVStateCacheBlockBuilder* kv_state_cache_block_builder = (KVStateCacheBlockBuilder* )
      ((TreeData*) evicted_node->get_tree()->GetCustomData())->kv_state_cache_block_builder;
  LOG(INFO) << "stage2, builder:" << kv_state_cache_block_builder;
  offset_data* data = (offset_data*) evicted_node->get_node()->get_data();
  LOG(INFO) << "stage3";
  kv_state_cache_block_builder->DeleteKVCache(data->offset);
  LOG(INFO) << "stage4";
  delete data;
}

void KVStateCacheBuilder::Merge(
    Client& client, std::shared_ptr<KVStateCache> kv_state_cache) {
  // TBD
  if (kv_state_cache == nullptr) {
    return;
  }
  std::shared_ptr<KVStateCacheBuilder> global_cache_builder =
      std::make_shared<KVStateCacheBuilder>(client, kv_state_cache);
  std::shared_ptr<RadixTree> global_cache_tree = kv_state_cache->GetRootTree();

  std::set<std::vector<int>> insert_token_list;
  std::vector<std::vector<int>> evicted_token_list;
  mergeTree(this->root_tree->GetRootTree(),
            global_cache_tree->GetRootTree(),
            evicted_token_list, insert_token_list,
            this->root_tree->GetCacheCapacity());

  for (size_t i = 0; i < evicted_token_list.size(); i++) {
    std::vector<int> token_list = evicted_token_list[i];
    std::shared_ptr<NodeWithTreeAttri> evicted_node;
    this->root_tree->Delete(token_list, evicted_node);
    Delete(evicted_node);
  }

  for (auto it = insert_token_list.begin(); it != insert_token_list.end();
       ++it) {
    std::vector<int> token_list = *it;
    KV_STATE_WITH_LAYER kv_state =
        global_cache_builder->Query(client, std::vector<int>(token_list.begin(), token_list.end() - 1), token_list.back());
    this->Update(client, token_list, token_list[token_list.size() - 1],
                 kv_state);
  }
  return;
}

Status KVStateCacheBuilder::Build(Client& client) {
  // TBD
  return Status::OK();
}

std::shared_ptr<Object> KVStateCacheBuilder::_Seal(Client& client) {
  LOG(INFO) << "cache seal";
  this->Build(client);

  std::shared_ptr<KVStateCache> kv_state_cache =
      std::make_shared<KVStateCache>();

  // 1. store the member variables to cache object meta
  kv_state_cache->meta_.AddKeyValue("dimension", this->dimension);

  // 2. seal all the block and put object id to cache object and
  // change the tree data from pointer to object id
  // LOG(INFO) << "block size:" << this->kv_state_cache_block_builder_list.size();
  // for (int i = 0; i < this->kv_state_cache_block_builder_list.size(); i++) {
  //   LOG(INFO) << "block builder:" << this->kv_state_cache_block_builder_list[i];
  // }

  // refactor related code.
  TreeData* tree_data = (TreeData*)((customData*)this->root_tree->GetTree()->head->custom_data)->data;
  KVStateCacheBlockBuilder* kv_state_cache_block_builder = (KVStateCacheBlockBuilder*)
      tree_data->kv_state_cache_block_builder;
  LOG(INFO) << "builder:" << kv_state_cache_block_builder;
  std::shared_ptr<Object> kv_state_cache_block =
      kv_state_cache_block_builder->_Seal(client);
  LOG(INFO) << "block:" << kv_state_cache_block;
  kv_state_cache->meta_.AddMember("kv_state_cache_block_builder_0",
                                  kv_state_cache_block);
  tree_data->builder_object_id = kv_state_cache_block->id();
  tree_data->is_ptr = false;

  raxIterator iter;
  raxStart(&iter, this->root_tree->GetTree());
  raxSeek(&iter, "^", nullptr, 0);
  raxShow(this->root_tree->GetTree());
  int count = 1;
  while (raxNext(&iter)) {
    std::string str = "";
    for (int i = 0; i < iter.key_len; i++) {
      str += std::to_string(iter.key[i]) + " ";
    }
    LOG(INFO) << "key:" << str;
    if (iter.node->issubtree) {
      LOG(INFO) << "node:" << iter.node;
      TreeData* tree_data = (TreeData*)((customData*) iter.node->custom_data)->data;
      LOG(INFO) << "stage1";
      VINEYARD_ASSERT(tree_data != nullptr);
      LOG(INFO) << "stage2";
      VINEYARD_ASSERT(tree_data->is_ptr == true);

      LOG(INFO) << "stage3";
      KVStateCacheBlockBuilder* kv_state_cache_block_builder = (KVStateCacheBlockBuilder*)
          tree_data->kv_state_cache_block_builder;
      LOG(INFO) << "builder:" << kv_state_cache_block_builder;
      LOG(INFO) << "stage4";
      std::shared_ptr<Object> kv_state_cache_block =
          kv_state_cache_block_builder->_Seal(client);
      LOG(INFO) << "stage5";
      kv_state_cache->meta_.AddMember(
          "kv_state_cache_block_builder_" + std::to_string(count),
          kv_state_cache_block);
      tree_data->builder_object_id = kv_state_cache_block->id();
      tree_data->is_ptr = false;
      count++;
    }
  }
  kv_state_cache->meta_.AddKeyValue("num_blocks",
                                    count);

  // 3. put the serialized sequence radix tree to cache object meta
  kv_state_cache->meta_.AddKeyValue(
      "radix_tree", base64_encode(this->root_tree->Serialize()));

  // 4. put the object type to the meta
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
