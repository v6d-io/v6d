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

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "oneapi/tbb/concurrent_hash_map.h"

#include "common/memory/payload.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "server/memory/usage.h"

namespace vineyard {

template <typename ID, typename P>
class BulkStoreBase {
 public:
  using object_map_t = tbb::concurrent_hash_map<ID, std::shared_ptr<P>>;

  ~BulkStoreBase();

  Status Get(ID const& id, std::shared_ptr<P>& object);

  Status GetUnchecked(ID const& id, std::shared_ptr<P>& object);

  /**
   * This methods only return available objects, and doesn't fail when object
   * does not exists.
   */
  Status Get(std::vector<ID> const& ids,
             std::vector<std::shared_ptr<P>>& objects);

  bool Exists(ID const& object_id);

  Status Seal(ID const& object_id);

  Status Delete(ID const& object_id);

  object_map_t const& List() const { return objects_; }

  size_t Footprint() const;
  size_t FootprintLimit() const;

  Status MakeArena(size_t const size, int& fd, uintptr_t& base);

  Status PreAllocate(size_t const size);

  Status FinalizeArena(int const fd, std::vector<size_t> const& offsets,
                       std::vector<size_t> const& sizes);

  Status MoveOwnership(std::map<ID, P> const& to_process_ids);

  Status RemoveOwnership(std::set<ID> const& ids,
                         std::map<ID, P>& successed_ids);

 protected:
  uint8_t* AllocateMemory(size_t size, int* fd, int64_t* map_size,
                          ptrdiff_t* offset);
  struct Arena {
    int fd;
    size_t size;
    uintptr_t base;
    static std::set<ID> spans;
  };

  std::unordered_map<int /* fd */, Arena> arenas_;

  object_map_t objects_;
};

class BulkStore
    : public BulkStoreBase<ObjectID, Payload>,
      protected detail::ColdObjectTracker<ObjectID, Payload, BulkStore> {
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

  friend class detail::ColdObjectTracker<ObjectID, Payload, BulkStore>;
  friend class SocketConnection;
};

/**
 * @brief A wrapper of `BulkStore` that provides a simple interface to act like
 * a Plasma mock.
 */
class PlasmaBulkStore
    : public BulkStoreBase<PlasmaID, PlasmaPayload>,
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

  friend class detail::DependencyTracker<PlasmaID, PlasmaPayload,
                                         PlasmaBulkStore>;
  friend class SocketConnection;
};

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_MEMORY_H_
