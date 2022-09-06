/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/memory.h is referred and derived from project
 * apache-arrow,
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/plasma/memory.h
 *
 * which has the following license:
 *
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
 */

#ifndef SRC_SERVER_MEMORY_MEMORY_H_
#define SRC_SERVER_MEMORY_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "oneapi/tbb/concurrent_hash_map.h"

#include "common/memory/gpu/unified_memory.h"
#include "common/memory/payload.h"
#include "common/util/logging.h"
#include "common/util/macros.h"
#include "common/util/status.h"
#include "server/memory/gpu/gpuallocator.h"
#include "server/memory/usage.h"

namespace vineyard {

template <typename ID, typename P>
class BulkStoreBase {
 public:
  using object_map_t = tbb::concurrent_hash_map<ID, std::shared_ptr<P>>;

  virtual ~BulkStoreBase();

  Status Get(ID const& id, std::shared_ptr<P>& object);

  /**
   * Like `Get()`, but optionally bypass the "sealed" check.
   */
  Status GetUnsafe(ID const& id, const bool unsafe, std::shared_ptr<P>& object);

  /**
   * This methods only return available objects, and doesn't fail when object
   * does not exists.
   */
  Status Get(std::vector<ID> const& ids,
             std::vector<std::shared_ptr<P>>& objects);

  Status GetUnsafe(std::vector<ID> const& ids, const bool unsafe,
                   std::vector<std::shared_ptr<P>>& objects);

  bool Exists(ID const& object_id);

  Status Seal(ID const& object_id);

  Status Delete(ID const& object_id);

  Status DeleteGPU(ID const& object_id);

  object_map_t const& List() const { return objects_; }

  size_t Footprint() const;
  size_t FootprintLimit() const;
  size_t FootprintGPU() const;
  size_t FootprintLimitGPU() const;

  Status MakeArena(size_t const size, int& fd, uintptr_t& base);

  Status PreAllocate(
      size_t const size,
#if defined(DEFAULT_ALLOCATOR)
      std::string const& allocator = VINEYARD_TO_STRING(DEFAULT_ALLOCATOR));
#else
      std::string const& allocator = "mimalloc");
#endif

  Status FinalizeArena(int const fd, std::vector<size_t> const& offsets,
                       std::vector<size_t> const& sizes);

  Status MoveOwnership(std::map<ID, P> const& to_process_ids);

  Status RemoveOwnership(std::set<ID> const& ids,
                         std::map<ID, P>& successed_ids);

  void SetMemSpillUpBound(size_t mem_spill_upper_bound) {
    mem_spill_upper_bound_ = mem_spill_upper_bound;
  }
  void SetMemSpillLowBound(size_t mem_spill_lower_bound) {
    mem_spill_lower_bound_ = mem_spill_lower_bound;
  }

 protected:
  uint8_t* AllocateMemory(size_t size, int* fd, int64_t* map_size,
                          ptrdiff_t* offset);
  /**
   * @brief Allocate memory on GPU
   *
   * @param size the size of memory
   * @return uint8_t*
   */
  uint8_t* AllocateMemoryGPU(size_t size);

  struct Arena {
    int fd;
    size_t size;
    uintptr_t base;
    static std::set<ID> spans;
  };

  std::unordered_map<int /* fd */, Arena> arenas_;

  object_map_t objects_;

  size_t mem_spill_upper_bound_;

  size_t mem_spill_lower_bound_;
};

class BulkStore
    : public BulkStoreBase<ObjectID, Payload>,
      public detail::ColdObjectTracker<ObjectID, Payload, BulkStore>,
      public std::enable_shared_from_this<BulkStore> {
 public:
  /*
   * @brief Allocate space for a new blob.
   */
  Status Create(const size_t size, ObjectID& object_id,
                std::shared_ptr<Payload>& object);

  /*
   * @brief Decrease the reference count of a blob, when its reference count
   * reaches zero. It will trigger `OnRelease` behavior. See ColdObjectTracker
   */
  Status Release(ObjectID const& id, int conn);

  /*
   * @brief Allocate space for a new blob on gpu.
   */
  Status CreateGPU(const size_t size, ObjectID& object_id,
                   std::shared_ptr<Payload>& object);

  /*
   * @brief Allocate space for a new blob on disk.
   */
  Status CreateDisk(const size_t size, const std::string& path,
                    ObjectID& object_id, std::shared_ptr<Payload>& object);

  /*
   * @brief Decrease the reference count of a blob, when its reference count
   * reaches zero. It will trigger `OnRelease` behavior. See ColdObjectTracker
   * Not support on GPU for now
   */
  Status Release_GPU(ObjectID const& id, int conn);

 protected:
  /**
   * @brief change the reference count of the object on the client-side cache.
   */
  Status FetchAndModify(ObjectID const& id, int64_t& ref_cnt, int64_t changes);

  /**
   * @brief Required by `ColdObjectTracker`. When reference count reaches zero,
   * mark the blob as cold blob
   */
  Status OnRelease(ObjectID const& id);

  /**
   * @brief Required by `ColdObjectTracker`. Currently, the deletion does not
   * respect the reference count.
   */
  Status OnDelete(ObjectID const& id);

 private:
  inline std::shared_ptr<BulkStore> shared_from_self() override {
    return shared_from_this();
  }

  friend class detail::ColdObjectTracker<ObjectID, Payload, BulkStore>;
  friend class SocketConnection;
  friend class VineyardServer;
};

class PlasmaBulkStore
    : public BulkStoreBase<PlasmaID, PlasmaPayload>,
      public std::enable_shared_from_this<PlasmaBulkStore>,
      protected detail::DependencyTracker<PlasmaID, PlasmaPayload,
                                          PlasmaBulkStore> {
 public:
  /*
   * @brief Allocate space for a new blob.
   */
  Status Create(size_t const data_size, size_t const plasma_size,
                PlasmaID const& plasma_id, ObjectID& object_id,
                std::shared_ptr<PlasmaPayload>& object);

  /*
   * @brief Decrease the reference count of a blob, when its reference count
   * reaches zero. It will trigger `OnRelease` behavior. See DependencyTracker
   */
  Status Release(PlasmaID const& id, int conn);

  /**
   * @brief delete a object lazily, this will add the object to a delete queue
   * and do the actual deletion when their reference count reaches zero.
   */
  Status Delete(PlasmaID const& id);

 protected:
  Status FetchAndModify(PlasmaID const& id, int64_t& ref_cnt, int64_t changes);

  /**
   * @brief Required by `DependencyTracker`. When reference count reaches zero,
   * evict the blob for more space eagerly.
   */
  Status OnRelease(PlasmaID const& id);

  /**
   * @brief Required by `DependencyTracker`. Delete the blob when its reference
   * count reaches zero.
   */
  Status OnDelete(PlasmaID const& id);

 private:
  friend class detail::DependencyTracker<PlasmaID, PlasmaPayload,
                                         PlasmaBulkStore>;
  friend class SocketConnection;
};

/**
 * @brief A wrapper of `BulkStore` that provides a simple interface to act like
 * a Plasma mock.
 */
namespace detail {

template <typename ObjectIDType>
struct bulk_store_t {};

template <>
struct bulk_store_t<ObjectID> {
  using type = BulkStore;
};

template <>
struct bulk_store_t<PlasmaID> {
  using type = PlasmaBulkStore;
};

}  // namespace detail

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_MEMORY_H_
