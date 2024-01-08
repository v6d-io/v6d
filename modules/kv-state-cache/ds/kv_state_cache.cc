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

#include "kv_state_cache.h"
#include "client/client.h"

// Set the bit to 0, which means the resource is being used
#define CLR_UINT64_BIT(value, bit) ((value) |= (((int64_t) 1) << (bit)))
// Set the bit to 1, which means the resource is not being used
#define SET_UINT64_BIT(value, bit) ((value) &= ~(((int64_t) 1) << (bit)))

namespace vineyard {

// this function will be removed in the future
std::string toBinaryString(unsigned long long num) {
  std::string result;
  const int bits = 8 * sizeof(unsigned long long);
  for (int i = bits - 1; i >= 0; --i) {
    result += ((num >> i) & 1) ? '1' : '0';
  }
  return result;
}

void KVStateCache::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  // TBD
  std::string tree_data;
  meta.GetKeyValue("tree", tree_data);
  meta.GetKeyValue("bitmap", this->bitmap);
  tree = RadixTree::deserialize(tree_data);
}

KVStateCacheBuilder::KVStateCacheBuilder() {
  pthread_spin_init(&(this->spin_lock), 0);
  this->bitmap = UINT64_MAX;
  this->tree = new RadixTree();
}

KVStateCacheBuilder::KVStateCacheBuilder(KVStateCache& kv_state_cache) {
  pthread_spin_init(&(this->spin_lock), 0);
  this->bitmap = kv_state_cache.bitmap;
  this->tree = kv_state_cache.tree;
  // TBD:
  // transfer the data from kv_state_cache to this builder
}

KVStateCacheBuilder::KVStateCacheBuilder(RadixTree *tree) {
  pthread_spin_init(&(this->spin_lock), 0);
  this->bitmap = UINT64_MAX;
  this->tree = tree;
}

Status KVStateCacheBuilder::Split() {
  // Split the tree if the list of kv_state is full
  RadixTree *sub_tree = this->tree->split();
  KVStateCacheBuilder *child_kv_state_cache_builder = new KVStateCacheBuilder(sub_tree);
  std::vector<Node *> sub_tree_node_list = sub_tree->traverse();
  for (int i = 0; i < sub_tree_node_list.size(); i++) {
    offset_data *data = (offset_data *) sub_tree_node_list[i]->get_data();
    int index_k = data->offset_k;
    int index_v = data->offset_v;

    // transfer the data from this builder to the child builder
    child_kv_state_cache_builder->key_state_writer_array[i] = std::move(this->key_state_writer_array[index_k]);
    child_kv_state_cache_builder->value_state_writer_array[i] = std::move(this->value_state_writer_array[index_v]);

    // clear the bitmap
    SET_UINT64_BIT(this->bitmap, index_k);
    SET_UINT64_BIT(this->bitmap, index_v);
    CLR_UINT64_BIT(child_kv_state_cache_builder->bitmap, i);
    CLR_UINT64_BIT(child_kv_state_cache_builder->bitmap, i);
  }
  this->kv_state_cache_builder_map.insert(std::make_pair(sub_tree, child_kv_state_cache_builder));
  return Status::OK();
}

Status KVStateCacheBuilder::UpdateInternal(
    Client& client, const std::vector<int>& token_list, int next_token,
    const KV_STATE_WITH_LAYER& kv_state) {
  const std::vector<double>& key_state = (kv_state.find(1)->second).first;
  const std::vector<double>& value_state = (kv_state.find(1)->second).second;

  // set the key and value state
  int index = ffsll(this->bitmap) - 1;
  if (index < 0 || index > LIST_SIZE) {
    // split the tree
    Split();
    // retry.
    return Status::IOError();
    // TBD
    // check the status
  }

  // prepare blob writer
  // TBD
  // Here need to check the insert position. If the position is the subtree, we must
  // find the subtree and insert the data into the subtree.

  client.CreateBlob(key_state.size() * sizeof(double),
                    this->key_state_writer_array[index]);
  client.CreateBlob(value_state.size() * sizeof(double),
                    this->value_state_writer_array[index]);
  double* key_data = (double*) this->key_state_writer_array[index]->data();
  double* value_data = (double*) this->value_state_writer_array[index]->data();

  for (int i = 0; i < key_state.size(); ++i) {
    key_data[i] = key_state[i];
  }
  for (int i = 0; i < value_state.size(); ++i) {
    value_data[i] = value_state[i];
  }

  // construct the tree data and insert the key-value into the tree
  offset_data* data = new offset_data();
  data->offset_k = index;
  data->offset_v = index;

  tree->insert(token_list, next_token, data, (int) sizeof(offset_data));

  SET_UINT64_BIT(this->bitmap, index);
  LOG(INFO) << toBinaryString(this->bitmap);
  return Status::OK();
}

// current we do not consider the layer.
Status KVStateCacheBuilder::Update(Client& client,
                                   const std::vector<int>& token_list,
                                   int next_token,
                                   const KV_STATE_WITH_LAYER& kv_state) {
  pthread_spin_lock(&(this->spin_lock));
  Status status =
      this->UpdateInternal(client, token_list, next_token, kv_state);
  pthread_spin_unlock(&(this->spin_lock));
  return status;
}

Status KVStateCacheBuilder::Update(Client& client,
                                   const std::vector<int>& token_list,
                                   const LIST_KV_STATE_WITH_LAYER& kv_state) {
  pthread_spin_lock(&(this->spin_lock));
  std::vector<int> token_list_copy;
  for (int i = 0; i < token_list.size(); ++i) {
    int next_token = token_list[i];
    const KV_STATE_WITH_LAYER& kv_state_map = kv_state[i];
    this->UpdateInternal(client, token_list_copy, next_token, kv_state_map);
    token_list_copy.push_back(token_list[i]);
  }
  pthread_spin_unlock(&(this->spin_lock));
  return Status::OK();
}

// current we do not consider the layer.
Status KVStateCacheBuilder::QueryInternal(Client& client,
                                          const std::vector<int>& token_list,
                                          int token,
                                          KV_STATE_WITH_LAYER& kv_state) {
  Node* node = this->tree->get(token_list, token);
  if (node == nullptr) {
    return Status::ObjectNotExists();
  }

  // TBD
  // Here need to check the insert position. If the position is the subtree, we must
  // find the subtree and insert the data into the subtree.
  offset_data* data = (offset_data*) node->get_data();
  int index_k = data->offset_k;
  int index_v = data->offset_v;
  std::vector<double> k_state;
  std::vector<double> v_state;

  for (int i = 0; i < key_state_writer_array[index_k]->size() / sizeof(double);
       ++i) {
    k_state.push_back(((double*) key_state_writer_array[index_k]->data())[i]);
  }

  for (int i = 0;
       i < value_state_writer_array[index_v]->size() / sizeof(double); ++i) {
    v_state.push_back(((double*) value_state_writer_array[index_v]->data())[i]);
  }

  kv_state.insert(std::make_pair(1, std::make_pair(k_state, v_state)));
  return Status::OK();
}

Status KVStateCacheBuilder::Query(Client& client,
                                  const std::vector<int>& token_list, int token,
                                  KV_STATE_WITH_LAYER& kv_state) {
  pthread_spin_lock(&(this->spin_lock));
  Status query_status = QueryInternal(client, token_list, token, kv_state);
  pthread_spin_unlock(&(this->spin_lock));
  return query_status;
}

// current we do not consider the layer.
Status KVStateCacheBuilder::Query(Client& client,
                                  const std::vector<int>& token_list,
                                  LIST_KV_STATE_WITH_LAYER& kv_state) {
  pthread_spin_lock(&(this->spin_lock));
  std::vector<int> token_list_copy;
  for (int i = 0; i < token_list.size(); i++) {
    KV_STATE_WITH_LAYER kv_state_map;
    int next_token = token_list[i];
    Status query_status =
        QueryInternal(client, token_list_copy, next_token, kv_state_map);
    // TBD check the query status
    kv_state.push_back(kv_state_map);
    token_list_copy.push_back(token_list[i]);
  }
  pthread_spin_unlock(&(this->spin_lock));
  return Status::OK();
}

Status KVStateCacheBuilder::Build(Client& client) {
  // TBD craete vineyard object
  ObjectMeta meta;
  meta.SetTypeName(type_name<KVStateCache>());
  meta.AddKeyValue("tree", this->tree->serialize());
  meta.AddKeyValue("bitmap", this->bitmap);
  for (int i = 0; i < LIST_SIZE; ++i) {
    if (this->bitmap & (((int64_t) 1) << i)) {
      meta.AddMember("key_state_builder_array_" + std::to_string(i),
                     this->key_state_writer_array[i]->id());
      meta.AddMember("value_state_builder_array_" + std::to_string(i),
                     this->value_state_writer_array[i]->id());
    }
  }
  // TBD check the status
  client.CreateMetaData(meta, id);
  return Status::OK();
}

std::shared_ptr<Object> KVStateCacheBuilder::_Seal(Client& client) {
  pthread_spin_lock(&(this->spin_lock));
  // TBD
  // Sync with vineyard server preriodically
  pthread_spin_unlock(&(this->spin_lock));
  return nullptr;
}

Status KVStateCacheBuilder::GetTree(RadixTree *&tree) {
  tree = this->tree;
  return Status::OK();
}

}  // namespace vineyard