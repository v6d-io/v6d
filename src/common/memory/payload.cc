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
}

void Payload::FromJSON(const json& tree) {
  object_id = tree["object_id"].get<ObjectID>();
  store_fd = tree["store_fd"].get<int>();
  data_offset = tree["data_offset"].get<ptrdiff_t>();
  data_size = tree["data_size"].get<int64_t>();
  map_size = tree["map_size"].get<int64_t>();
  pointer = nullptr;
}

Payload Payload::FromJSON1(const json& tree) {
  Payload payload;
  payload.FromJSON(tree);
  return payload;
}

json ExternalPayload::ToJSON() const {
  json payload;
  this->ToJSON(payload);
  return payload;
}

void ExternalPayload::ToJSON(json& tree) const {
  tree["external_id"] = object_id;
  tree["external_size"] = external_size;
  tree["store_fd"] = store_fd;
  tree["data_offset"] = data_offset;
  tree["data_size"] = data_size;
  tree["map_size"] = map_size;
}

void ExternalPayload::FromJSON(const json& tree) {
  external_id = tree["external_id"].get<ObjectID>();
  external_size = tree["external_size"].get<int64_t>();
  store_fd = tree["store_fd"].get<int>();
  data_offset = tree["data_offset"].get<ptrdiff_t>();
  data_size = tree["data_size"].get<int64_t>();
  map_size = tree["map_size"].get<int64_t>();
  pointer = nullptr;
}

ExternalPayload ExternalPayload::FromJSON1(const json& tree) {
  ExternalPayload external_payload;
  external_payload.FromJSON(tree);
  return external_payload;
}

}  // namespace vineyard
