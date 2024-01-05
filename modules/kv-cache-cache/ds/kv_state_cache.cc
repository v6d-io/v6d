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

#define SET_UINT64_BIT(value, bit) ((value) |= (((int64_t)1) << (bit)))
#define CLR_UINT64_BIT(value, bit) ((value) &= ~(((int64_t)1) << (bit)))
namespace vineyard {

void KVStateCache::Construct(const ObjectMeta& meta) {
    Object::Construct(meta);
    // TBD
    std::string tree_data;
    meta.GetKeyValue("tree", tree_data);
    meta.GetKeyValue("bitmap", this->bitmap);
    tree = this->tree.deserialize(tree_data);
}

KVStateCacheBuilder::KVStateCacheBuilder() {
    this->spin_lock = 0;
    this->bitmap = UINT64_MAX;
}

KVStateCacheBuilder::KVStateCacheBuilder(KVStateCache &kv_state_cache) {
    this->spin_lock = 0;
    this->bitmap = kv_state_cache.bitmap;
    this->tree = kv_state_cache.tree;
    // TBD transfer the data from kv_state_cache to this builder
}

Status KVStateCacheBuilder::UpdateInternal(Client &client, const std::vector<int> &token_list, int next_token, const std::map<int, std::pair<std::vector<double>, std::vector<double>>> &kv_state) {
    const std::vector<double> &key_state = (kv_state.find(1)->second).first;
    const std::vector<double> &value_state = (kv_state.find(1)->second).second;

    // set the key and value state
    int index = ffsll(this->bitmap) - 1;
    if (index < 0) {
        // TBD: split the tree
    }

    // prepare tensor builder
    client.CreateBlob(key_state.size() * sizeof(double), this->key_state_writer_array[index]);
    client.CreateBlob(value_state.size() * sizeof(double), this->value_state_writer_array[index]);
    double* key_data = (double *)this->key_state_writer_array[index]->data();
    double* value_data = (double *)this->value_state_writer_array[index]->data();

    for (int i = 0; i < key_state.size(); ++i) {
        key_data[i] = key_state[i];
    }
    for (int i = 0; i < value_state.size(); ++i) {
        value_data[i] = value_state[i];
    }

    // construct the tree data and insert the key-value into the tree
    offset_data data;
    data.offset_k = index;
    data.offset_v = index;

    tree.insert(token_list, next_token, &data, (int)sizeof(offset_data));
    return Status::OK();
}

// current we do not consider the layer.
Status KVStateCacheBuilder::Update(Client &client, const std::vector<int> &token_list, int next_token, const std::map<int, std::pair<std::vector<double>, std::vector<double>>> &kv_state) {
    pthread_spin_lock(&(this->spin_lock));
    Status status = this->UpdateInternal(client, token_list, next_token, kv_state);
    pthread_spin_unlock(&(this->spin_lock));
    return status;
}

Status KVStateCacheBuilder::Update(Client &client, const std::vector<int> &token_list, const std::vector<std::map<int, std::pair<std::vector<double>, std::vector<double>>>> &kv_state) {
    pthread_spin_lock(&(this->spin_lock));
    std::vector<int> token_list_copy;
    for (int i = 0; i < token_list.size(); ++i) {
        int next_token = token_list[i];
        const std::map<int, std::pair<std::vector<double>, std::vector<double>>> &kv_state_map = kv_state[i];
        this->UpdateInternal(client, token_list_copy, next_token, kv_state_map);
        token_list_copy.push_back(token_list[i]);
    }
    pthread_spin_unlock(&(this->spin_lock));
    return Status::OK();
}

// current we do not consider the layer.
Status KVStateCacheBuilder::QueryInternal(Client &client, const std::vector<int> &token_list, int token, std::map<int, std::pair<std::vector<double>, std::vector<double>>> &kv_state) {
    Node *node = this->tree.get(token_list, token);
    if (node == nullptr) {
        return Status::ObjectNotExists();
    }

    offset_data *data = (offset_data *)node->get_data();
    int index_k = data->offset_k;
    int index_v = data->offset_v;
    std::vector<double> k_state;
    std::vector<double> v_state;

    kv_state.insert(std::make_pair(1, std::make_pair(k_state, v_state)));
    return Status::OK();
}

Status KVStateCacheBuilder::Query(Client &client, const std::vector<int> &token_list, int token, std::map<int, std::pair<std::vector<double>, std::vector<double>>> &kv_state) {
    pthread_spin_lock(&(this->spin_lock));
    Status query_status = QueryInternal(client, token_list, token, kv_state);
    pthread_spin_unlock(&(this->spin_lock));
    return query_status;
}

// current we do not consider the layer.
Status KVStateCacheBuilder::Query(Client &client, const std::vector<int> &token_list, std::vector<std::map<int, std::pair<std::vector<double>, std::vector<double>>>> &kv_state) {
    pthread_spin_lock(&(this->spin_lock));
    std::vector<int> token_list_copy;
    for (int i = 0; i < token_list.size(); i++) {
        std::map<int, std::pair<std::vector<double>, std::vector<double>>> kv_state_map;
        int next_token = token_list[i];
        Status query_status = QueryInternal(client, token_list_copy, next_token, kv_state_map);
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
    meta.AddKeyValue("tree", this->tree.serialize());
    meta.AddKeyValue("bitmap", this->bitmap);
    for (int i = 0; i < LIST_SIZE; ++i) {
        if (this->bitmap & (((int64_t)1) << i)) {
            meta.AddMember("key_state_builder_array_" + std::to_string(i), this->key_state_writer_array[i]->id());
            meta.AddMember("value_state_builder_array_" + std::to_string(i), this->value_state_writer_array[i]->id());
        }
    }
    // TBD check the status
    client.CreateMetaData(meta, id);
    return Status::OK();
}

std::shared_ptr<Object> KVStateCacheBuilder::_Seal(Client& client) {
    pthread_spin_lock(&(this->spin_lock));
    pthread_spin_unlock(&(this->spin_lock));
    return nullptr;
}

} // namespace vineyard