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

#ifndef MODULES_MSGPACK_PACKED_OBJECT_H_
#define MODULES_MSGPACK_PACKED_OBJECT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"
#include "msgpack/vbuffer.h"

namespace vineyard {

class PackedObject {
 public:
  explicit PackedObject(Object const& object) : PackedObject(object.meta()) {}

  explicit PackedObject(Object const* object) : PackedObject(object->meta()) {}

  explicit PackedObject(std::shared_ptr<Object> const& object)
      : PackedObject(object->meta()) {}

  explicit PackedObject(std::unique_ptr<Object> const& object)
      : PackedObject(object->meta()) {}

  explicit PackedObject(ObjectMeta const& meta) : meta_(meta) {
    buffer.Append(Buffer(meta.MetaData().dump()));
    std::vector<uint64_t> elements;
    auto const& buffers = meta.GetBufferSet()->AllBuffers();
    elements.emplace_back(buffers.size());
    for (auto const& item : buffers) {
      elements.emplace_back(item.first);
      elements.emplace_back(item.second->size());
    }
    buffer.Append(Buffer(elements));
    for (auto const& item : buffers) {
      buffer.Append(Buffer(item.second->data(), item.second->size()));
    }
  }

 private:
  const ObjectMeta meta_;
  VBuffer buffer;
};

}  // namespace vineyard

#endif  // MODULES_MSGPACK_PACKED_OBJECT_H_
