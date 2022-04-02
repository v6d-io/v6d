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

#ifndef MODULES_MSGPACK_PICKLE_H_
#define MODULES_MSGPACK_PICKLE_H_

#include <iostream>
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

class Pickler {
 public:
  explicit Pickler(Object const& object) : Pickler(object.meta()) {}

  explicit Pickler(Object const* object) : Pickler(object->meta()) {}

  explicit Pickler(std::shared_ptr<Object> const& object)
      : Pickler(object->meta()) {}

  explicit Pickler(std::unique_ptr<Object> const& object)
      : Pickler(object->meta()) {}

  explicit Pickler(ObjectMeta const& meta) : meta_(meta) {
    this->buildPickleBuffers();
  }

 private:
  void buildPickleBuffers() {
    BufferBuilder builder;
    // Framer framer;

    // header
    builder.PutByte('\x80');
    builder.PutByte(5);

    auto const& buffers = meta_.GetBufferSet()->AllBuffers();

    // start frame
    {
      BufferBuilder inner;
      inner.PutChars(
          "\x8c\x0bvineyard._C\x94\x8c\x1c_object_from_pickled_"
          "buffers\x94");
      inner.PutChars("\x93\x94");  // start the tuple
      for (auto const& item : buffers) {
        // put object id
        inner.PutByte('\x8a');  // LONG1
        inner.PutByte(8);
        inner.PutUInt64(item.first);
        // put PickleBuffer
        inner.PutChars("\x97\x98");  // NEXT_BUFFER + READONLY_BUFFER
      }
      inner.PutByte('.');

      auto buffer = inner.Finish();
      builder.PutByte('\x95');
      builder.PutUInt64(buffer.size());
      builder.Append(buffer);
    }

    auto buffer = builder.Finish();
    std::cout << "buffer: " << buffer.ToString() << std::endl;

    // buffer.Append(Buffer(meta.MetaData().dump()));
    // std::vector<uint64_t> elements;
    // for (auto const& item : buffers) {
    //   elements.emplace_back(item.first);
    //   elements.emplace_back(item.second->size());
    // }
    // buffer.Append(Buffer(elements));
    // for (auto const& item : buffers) {
    //   buffer.Append(Buffer(item.second->data(), item.second->size()));
    // }
    // builder.Append();
  }

  const ObjectMeta meta_;
  VBuffer buffer;

  static constexpr const uint8_t PROTO = '\x80';
  static constexpr const uint8_t STOP = '.';
};

}  // namespace vineyard

#endif  // MODULES_MSGPACK_PICKLE_H_
