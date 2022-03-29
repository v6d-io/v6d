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
#include "common/util/lifecycle.h"
#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

//  Track Dependency lest dangling objects and double deletion.
template <typename ID, typename P, typename Der>
class DependencyTracker
    : public LifeCycleTracker<ID, P, DependencyTracker<ID, P, Der>> {
 public:
  using dependency_map_t = tbb::concurrent_hash_map</*socket_connection*/ int,
                                                    std::unordered_set<ID>>;

  DependencyTracker() {}

  Status AddDependency(std::unordered_set<ID> const& ids, int conn) {
    for (auto const& id : ids) {
      RETURN_ON_ERROR(AddDependency(id, conn));
    }
    return Status::OK();
  }

  Status AddDependency(ID const& id, int conn) {
    RETURN_ON_ERROR(this->IncreaseReferenceCount(id));

    typename dependency_map_t::accessor accessor;
    if (!dependency_.find(accessor, conn)) {
      dependency_.emplace(conn, std::unordered_set<ID>());
    }

    if (dependency_.find(accessor, conn)) {
      auto& objects = accessor->second;
      if (objects.find(id) == objects.end()) {
        objects.emplace(id);
      }
    }
    return Status::OK();
  }

  Status RemoveDependency(std::unordered_set<ID> const& ids, int conn) {
    for (auto const& id : ids) {
      RETURN_ON_ERROR(RemoveDependency(id, conn));
    }
    return Status::OK();
  }

  Status RemoveDependency(ID const& id, int conn) {
    typename dependency_map_t::accessor accessor;
    if (!dependency_.find(accessor, conn)) {
      return Status::Invalid("connection not exist.");
    } else {
      auto& objects = accessor->second;
      if (objects.find(id) == objects.end()) {
        return Status::ObjectNotExists();
      } else {
        objects.erase(id);
      }
    }

    RETURN_ON_ERROR(this->DecreaseReferenceCount(id));

    return Status::OK();
  }

  /**
   * Note:
   * If object_id exists in dependency_, then its reference count should
   * determinately greater than 0, thus it will determinately not be deleted.
   * Delete will not remove dependency.
   */
  Status PreDelete(ID const& id) {
    return LifeCycleTracker<ID, P, DependencyTracker<ID, P, Der>>::PreDelete(
        id);
  }

  /// remove dependency of all objects in dependency_[conn].
  Status PopList(int conn, std::unordered_set<ID>& objects) {
    typename dependency_map_t::const_accessor accessor;
    if (!dependency_.find(accessor, conn)) {
      return Status::Invalid("connection not exist.");
    } else {
      objects = accessor->second;
      auto status = Status::OK();
      for (auto& elem : objects) {
        // try our best to remove dependency.
        auto _status = RemoveDependency(elem, conn);
        if (!_status.ok()) {
          status = _status;
        }
      }
      return status;
    }
  }

  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t changes) {
    auto status = Self().FetchAndModify(id, ref_cnt, changes);
    return status;
  }

  Status OnRelease(ID const& id) { return Self().OnRelease(id); }

  Status OnDelete(ID const& id) { return Self().OnDelete(id); }

 private:
  inline Der& Self() { return static_cast<Der&>(*this); }
  dependency_map_t dependency_;
};

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

class BulkStore : public BulkStoreBase<ObjectID, Payload> {
 public:
  Status Create(const size_t size, ObjectID& object_id,
                std::shared_ptr<Payload>& object);
};

class PlasmaBulkStore
    : public BulkStoreBase<PlasmaID, PlasmaPayload>,
      public DependencyTracker<PlasmaID, PlasmaPayload, PlasmaBulkStore> {
 public:
  Status Create(size_t const data_size, size_t const plasma_size,
                PlasmaID const& plasma_id, ObjectID& object_id,
                std::shared_ptr<PlasmaPayload>& object);

  Status FetchAndModify(PlasmaID const& id, int64_t& ref_cnt, int64_t changes);

  Status OnRelease(PlasmaID const& id);

  Status OnDelete(PlasmaID const& id);

  Status Release(PlasmaID const& id, int conn);

  Status Delete(PlasmaID const& id);
};

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_MEMORY_H_
