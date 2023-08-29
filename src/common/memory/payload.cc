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

#include "common/memory/payload.h"

#include <cstdint>

namespace vineyard {

Payload::Payload()
    : object_id(EmptyBlobID()),
      store_fd(-1),
      arena_fd(-1),
      data_offset(0),
      data_size(0),
      map_size(0),
      ref_cnt(0),
      pointer(nullptr),
      is_sealed(false),
      is_owner(true),
      is_spilled(false),
      is_gpu(false) {
  pinned.store(0);
}

Payload::Payload(ObjectID object_id, int64_t size, uint8_t* ptr, int fd,
                 int64_t msize, ptrdiff_t offset)
    : object_id(object_id),
      store_fd(fd),
      arena_fd(-1),
      data_offset(offset),
      data_size(size),
      map_size(msize),
      ref_cnt(0),
      pointer(ptr),
      is_sealed(false),
      is_owner(true),
      is_spilled(false),
      is_gpu(false) {
  pinned.store(0);
}

Payload::Payload(ObjectID object_id, int64_t size, uint8_t* ptr, int fd,
                 int arena_fd, int64_t msize, ptrdiff_t offset)
    : object_id(object_id),
      store_fd(fd),
      arena_fd(arena_fd),
      data_offset(offset),
      data_size(size),
      map_size(msize),
      ref_cnt(0),
      pointer(ptr),
      is_sealed(false),
      is_owner(true),
      is_spilled(false),
      is_gpu(false) {
  pinned.store(0);
}

Payload::Payload(const Payload& payload) {
  object_id = payload.object_id;
  store_fd = payload.store_fd;
  arena_fd = payload.arena_fd;
  data_offset = payload.data_offset;
  data_size = payload.data_size;
  map_size = payload.map_size;
  ref_cnt = payload.ref_cnt;
  pointer = payload.pointer;
  is_sealed = payload.is_sealed;
  is_owner = payload.is_owner;
  is_spilled = payload.is_spilled;
  is_gpu = payload.is_gpu;
  pinned.store(payload.pinned.load());
}

Payload& Payload::operator=(const Payload& payload) {
  object_id = payload.object_id;
  store_fd = payload.store_fd;
  arena_fd = payload.arena_fd;
  data_offset = payload.data_offset;
  data_size = payload.data_size;
  map_size = payload.map_size;
  ref_cnt = payload.ref_cnt;
  pointer = payload.pointer;
  is_sealed = payload.is_sealed;
  is_owner = payload.is_owner;
  is_spilled = payload.is_spilled;
  is_gpu = payload.is_gpu;
  pinned.store(payload.pinned.load());
  return *this;
}

std::shared_ptr<Payload> Payload::MakeEmpty() {
  static std::shared_ptr<Payload> payload = std::make_shared<Payload>();
  return payload;
}

bool Payload::operator==(const Payload& other) const {
  return ((object_id == other.object_id) && (store_fd == other.store_fd) &&
          (data_offset == other.data_offset) && (data_size == other.data_size));
}

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
  tree["is_gpu"] = is_gpu;
}

void Payload::FromJSON(const json& tree) {
  object_id = tree["object_id"].get<ObjectID>();
  store_fd = tree["store_fd"].get<int>();
  data_offset = tree["data_offset"].get<ptrdiff_t>();
  data_size = tree["data_size"].get<int64_t>();
  map_size = tree["map_size"].get<int64_t>();
  pointer = reinterpret_cast<uint8_t*>(tree["pointer"].get<uintptr_t>());
  is_sealed = tree.value("is_sealed", false);
  is_owner = tree.value("is_owner", true);
  is_gpu = tree.value("is_gpu", false);
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
  tree["ref_cnt"] = ref_cnt;
  tree["pointer"] = reinterpret_cast<uintptr_t>(pointer);
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
  ref_cnt = tree["ref_cnt"].get<int64_t>();
  pointer = reinterpret_cast<uint8_t*>(tree["pointer"].get<uintptr_t>());
  is_sealed = tree.value("is_sealed", false);
  is_owner = tree.value("is_owner", true);
  pointer = nullptr;
}

PlasmaPayload PlasmaPayload::FromJSON1(const json& tree) {
  PlasmaPayload plasma_payload;
  plasma_payload.FromJSON(tree);
  return plasma_payload;
}

}  // namespace vineyard
