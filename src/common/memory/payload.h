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

#ifndef SRC_COMMON_MEMORY_PAYLOAD_H_
#define SRC_COMMON_MEMORY_PAYLOAD_H_

#include <memory>

#include "common/util/json.h"
#include "common/util/uuid.h"

namespace vineyard {

struct Payload {
  ObjectID object_id;
  int store_fd;
  int arena_fd;
  ptrdiff_t data_offset;
  int64_t data_size;
  int64_t map_size;
  uint8_t* pointer;

  Payload()
      : object_id(EmptyBlobID()),
        store_fd(-1),
        arena_fd(-1),
        data_offset(0),
        data_size(0),
        map_size(0),
        pointer(nullptr) {}

  Payload(ObjectID object_id, int64_t size, uint8_t* ptr, int fd, int64_t msize,
          ptrdiff_t offset)
      : object_id(object_id),
        store_fd(fd),
        arena_fd(-1),
        data_offset(offset),
        data_size(size),
        map_size(msize),
        pointer(ptr) {}

  Payload(ObjectID object_id, int64_t size, uint8_t* ptr, int fd, int arena_fd,
          int64_t msize, ptrdiff_t offset)
      : object_id(object_id),
        store_fd(fd),
        arena_fd(arena_fd),
        data_offset(offset),
        data_size(size),
        map_size(msize),
        pointer(ptr) {}

  static std::shared_ptr<Payload> MakeEmpty() {
    static std::shared_ptr<Payload> payload = std::make_shared<Payload>();
    return payload;
  }

  bool operator==(const Payload& other) const {
    return ((object_id == other.object_id) && (store_fd == other.store_fd) &&
            (data_offset == other.data_offset) &&
            (data_size == other.data_size));
  }

  json ToJSON() const;

  void ToJSON(json& tree) const;

  void FromJSON(const json& tree);

  /**
   * @brief A static variant for `FromJSON`.
   */
  static Payload FromJSON1(const json& tree);
};

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_PAYLOAD_H_
