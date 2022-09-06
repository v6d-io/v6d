/** Copyright 2020-2022 Alibaba Group Holding Limited.

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
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

struct PlasmaPayload;

class BulkStore;

struct Payload {
  ObjectID object_id;
  int store_fd;
  int arena_fd;
  ptrdiff_t data_offset;
  int64_t data_size;
  int64_t map_size;
  int64_t ref_cnt;
  uint8_t* pointer;  // the direct pointer for this blob on the server side
  bool is_sealed;
  bool is_owner;
  bool is_spilled;
  bool is_gpu;  // indicate if the Object is on the GPU

  enum class Kind {
    kMalloc = 0,
    kAllocator = 1,
    kDiskMMap = 2,
  };
  Kind kind = Kind::kMalloc;

  Payload()
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
        is_gpu(false) {}

  Payload(ObjectID object_id, int64_t size, uint8_t* ptr, int fd, int64_t msize,
          ptrdiff_t offset)
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
        is_gpu(false) {}

  Payload(ObjectID object_id, int64_t size, uint8_t* ptr, int fd, int arena_fd,
          int64_t msize, ptrdiff_t offset)
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
        is_gpu(false) {}

  static std::shared_ptr<Payload> MakeEmpty() {
    static std::shared_ptr<Payload> payload = std::make_shared<Payload>();
    return payload;
  }

  bool operator==(const Payload& other) const {
    return ((object_id == other.object_id) && (store_fd == other.store_fd) &&
            (data_offset == other.data_offset) &&
            (data_size == other.data_size));
  }

  inline void Reset() { is_sealed = false, is_owner = true; }

  inline void MarkAsSealed() { is_sealed = true; }

  inline bool IsSealed() { return is_sealed; }

  inline void RemoveOwner() { is_owner = false; }

  inline bool IsOwner() { return is_owner; }

  bool IsSpilled() { return is_spilled; }

  inline bool IsGPU() { return is_gpu; }

  json ToJSON() const;

  void ToJSON(json& tree) const;

  void FromJSON(const json& tree);

  /**
   * @brief A static variant for `FromJSON`.
   */
  static Payload FromJSON1(const json& tree);
};

struct PlasmaPayload : public Payload {
  PlasmaID plasma_id;
  int64_t plasma_size;

  PlasmaPayload() : Payload(), plasma_id(), plasma_size(0) {}

  PlasmaPayload(PlasmaID plasma_id, ObjectID object_id, int64_t plasma_size,
                int64_t size, uint8_t* ptr, int fd, int64_t msize,
                ptrdiff_t offset)
      : Payload(object_id, size, ptr, fd, msize, offset),
        plasma_id(plasma_id),
        plasma_size(plasma_size) {}

  PlasmaPayload(PlasmaID plasma_id, ObjectID object_id, int64_t plasma_size,
                int64_t size, uint8_t* ptr, int fd, int arena_fd, int64_t msize,
                ptrdiff_t offset)
      : Payload(object_id, size, ptr, fd, arena_fd, msize, offset),
        plasma_id(plasma_id),
        plasma_size(plasma_size) {}

  PlasmaPayload(PlasmaID plasma_id, int64_t size, uint8_t* ptr, int fd,
                int64_t msize, ptrdiff_t offset)
      : Payload(EmptyBlobID(), size, ptr, fd, msize, offset),
        plasma_id(plasma_id),
        plasma_size(0) {}

  explicit PlasmaPayload(Payload _p)
      : Payload(_p),
        plasma_id(PlasmaIDFromString(ObjectIDToString(_p.object_id))),
        plasma_size(0) {}

  static std::shared_ptr<PlasmaPayload> MakeEmpty() {
    static std::shared_ptr<PlasmaPayload> payload =
        std::make_shared<PlasmaPayload>();
    return payload;
  }

  bool operator==(const PlasmaPayload& other) const {
    return ((object_id == other.object_id) && (store_fd == other.store_fd) &&
            (data_offset == other.data_offset) &&
            (plasma_size == other.plasma_size) &&
            (plasma_id == other.plasma_id) && (data_size == other.data_size));
  }

  Payload ToNormalPayload() const {
    return Payload(object_id, data_size, pointer, store_fd, arena_fd, map_size,
                   data_offset);
  }

  json ToJSON() const;

  void ToJSON(json& tree) const;

  void FromJSON(const json& tree);

  /**
   * @brief A static variant for `FromJSON`.
   */
  static PlasmaPayload FromJSON1(const json& tree);
};

template <typename T>
struct ID_traits {};

template <>
struct ID_traits<ObjectID> {
  typedef Payload P;
};

template <>
struct ID_traits<PlasmaID> {
  typedef PlasmaPayload P;
};

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_PAYLOAD_H_
