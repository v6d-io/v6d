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

#include "kv_cache_cache.h"

namespace vineyard {

void KVCacheCache::Construct(const ObjectMeta& meta) {
    Object::Construct(meta);
    // TBD
}

KVCacheCacheBuilder::KVCacheCacheBuilder() {
    // TBD
}

Status KVCacheCacheBuilder::Update(const std::vector<int> &token_list, int next_token, const std::map<int, std::vector<std::vector<int>, std::vector<int>>> &kv_state) {
    // TBD
    return Status::OK();
}

Status KVCacheCacheBuilder::Update(const std::vector<int> &token_list, const std::vector<std::map<int, std::pair<std::vector<int>, std::vector<int>>>> &kv_state) {
    // TBD
    return Status::OK();
}

Status KVCacheCacheBuilder::Query(const std::vector<int> &token_list, std::vector<std::map<int, std::pair<std::vector<int>, std::vector<int>>>> &kv_state) {
    // TBD
}

Status KVCacheCacheBuilder::Query(const std::vector<int> &token_list, int token, std::map<int, std::pair<std::vector<int>, std::vector<int>>> &kv_state) {
    // TBD
}

Status KVCacheCacheBuilder::Build(Client& client) {
    // TBD
    return Status::OK();
}

std::shared_ptr<Object> KVCacheCacheBuilder::_Seal(Client& client) {

}

} // namespace vineyard