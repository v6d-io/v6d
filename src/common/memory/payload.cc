/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include "common/memory/payload.h"

#include <cstdint>

namespace vineyard {

json Payload::ToJSON() const {
  json payload;
  this->ToJSON(payload);
  return payload;
}

void Payload::ToJSON(json& tree) const {
  tree["object_id"] = object_id;
  tree["store_fd"] = store_fd;
  tree["data_offset"] = data_offset;
  tree["data_size"] = data_size;
  tree["map_size"] = map_size;
  tree["pointer"] = reinterpret_cast<uintptr_t>(pointer);
  tree["is_sealed"] = is_sealed;
  tree["is_owner"] = is_owner;
}

void Payload::FromJSON(const json& tree) {
  object_id = tree["object_id"].get<ObjectID>();
  store_fd = tree["store_fd"].get<int>();
  data_offset = tree["data_offset"].get<ptrdiff_t>();
  data_size = tree["data_size"].get<int64_t>();
  map_size = tree["map_size"].get<int64_t>();
  pointer = reinterpret_cast<uint8_t*>(tree["pointer"].get<uintptr_t>());
  is_sealed = tree["is_sealed"].get<bool>();
  is_owner = tree["is_owner"].get<bool>();
}

Payload Payload::FromJSON1(const json& tree) {
  Payload payload;
  payload.FromJSON(tree);
  return payload;
}

json PlasmaPayload::ToJSON() const {
  json payload;
  this->ToJSON(payload);
  return payload;
}

void PlasmaPayload::ToJSON(json& tree) const {
  tree["plasma_id"] = plasma_id;
  tree["object_id"] = object_id;
  tree["plasma_size"] = plasma_size;
  tree["store_fd"] = store_fd;
  tree["data_offset"] = data_offset;
  tree["data_size"] = data_size;
  tree["map_size"] = map_size;
  tree["pointer"] = reinterpret_cast<uintptr_t>(pointer);
  tree["ref_cnt"] = ref_cnt;
  tree["is_sealed"] = is_sealed;
  tree["is_owner"] = is_owner;
}

void PlasmaPayload::FromJSON(const json& tree) {
  plasma_id = tree["plasma_id"].get<PlasmaID>();
  object_id = tree["object_id"].get<ObjectID>();
  plasma_size = tree["plasma_size"].get<int64_t>();
  store_fd = tree["store_fd"].get<int>();
  data_offset = tree["data_offset"].get<ptrdiff_t>();
  data_size = tree["data_size"].get<int64_t>();
  map_size = tree["map_size"].get<int64_t>();
  pointer = reinterpret_cast<uint8_t*>(tree["pointer"].get<uintptr_t>());
  ref_cnt = tree["ref_cnt"].get<int64_t>();
  is_sealed = tree["is_sealed"].get<bool>();
  is_owner = tree["is_owner"].get<bool>();
  pointer = nullptr;
}

PlasmaPayload PlasmaPayload::FromJSON1(const json& tree) {
  PlasmaPayload plasma_payload;
  plasma_payload.FromJSON(tree);
  return plasma_payload;
}

}  // namespace vineyard
