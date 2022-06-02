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

#ifndef SRC_SERVER_MEMORY_USAGE_H_
#define SRC_SERVER_MEMORY_USAGE_H_

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

namespace detail {

/**
 * @brief DependencyTracker is a CRTP class provides the dependency tracking for
 * its derived classes. It record which blobs is been used by each
 * `SocketConnection`. It requires the derived class to implement the:
 *  - `OnRelease(ID)` method to describe what will happens when `ref_count`
 * reaches zero.
 *  - `OnDelete(ID)` method to describe what will happens when `ref_count`
 * reaches zero and the object is marked as to be deleted.
 *  - `FetchAndModify(ID, int, int)` method to fetch the current `ref_count` and
 * modify it by the given value.
 */
template <typename ID, typename P, typename Der>
class DependencyTracker
    : public LifeCycleTracker<ID, P, DependencyTracker<ID, P, Der>> {
 public:
  using base_t = LifeCycleTracker<ID, P, DependencyTracker<ID, P, Der>>;
  using dependency_map_t = tbb::concurrent_hash_map</*socket_connection*/ int,
                                                    std::unordered_set<ID>>;

  DependencyTracker() {}

  /**
   * @brief Add dependency of blobs to the socket connection. Also increase the
   * reference count of blobs by one if they are not tracked before. Each socket
   * connection can increase the reference count of a blob at most by one.
   *
   * @param ids The blob IDs.
   * @param socket_connection The socket connection ID.
   */
  Status AddDependency(std::unordered_set<ID> const& ids, int conn) {
    for (auto const& id : ids) {
      RETURN_ON_ERROR(AddDependency(id, conn));
    }
    return Status::OK();
  }

  Status AddDependency(ID const& id, int conn) {
    typename dependency_map_t::accessor accessor;
    if (!dependency_.find(accessor, conn)) {
      dependency_.emplace(conn, std::unordered_set<ID>());
    }

    if (dependency_.find(accessor, conn)) {
      auto& objects = accessor->second;
      if (objects.find(id) == objects.end()) {
        objects.emplace(id);
        RETURN_ON_ERROR(this->IncreaseReferenceCount(id));
      }
    }
    return Status::OK();
  }

  /**
   * @brief Remove the dependency of the blob from the socket connection.
   * Decrease the reference count of the blob by one.
   *
   * @param ids The blob IDs.
   * @param socket_connection The socket connection ID.
   */
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
        if (objects.empty()) {
          dependency_.erase(accessor);
        }
      }
    }

    RETURN_ON_ERROR(this->DecreaseReferenceCount(id));

    return Status::OK();
  }

  /**
   * @brief delete a object lazily, this will add the object to a delete queue
   * and do the actual deletion when their reference count reaches zero.
   */
  Status PreDelete(ID const& id) { return base_t::PreDelete(id); }

  /**
   * @brief Remove all the dependency of all objects in given socket connection.
   */
  Status ReleaseConnection(int conn) {
    typename dependency_map_t::const_accessor accessor;
    if (!dependency_.find(accessor, conn)) {
      return Status::Invalid("connection not exist.");
    } else {
      auto& objects = accessor->second;
      for (auto& elem : objects) {
        // try our best to remove dependency.
        RETURN_ON_ERROR(this->DecreaseReferenceCount(elem));
      }
      dependency_.erase(accessor);
      return Status::OK();
    }
  }

 public:
  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t changes) {
    return Self().FetchAndModify(id, ref_cnt, changes);
  }

  Status OnRelease(ID const& id) { return Self().OnRelease(id); }

  Status OnDelete(ID const& id) { return Self().OnDelete(id); }

 private:
  inline Der& Self() { return static_cast<Der&>(*this); }
  dependency_map_t dependency_;
};

/**
 * @brief ColdObjectTracker is a CRTP class record non-in-use object in a list
 * for its derived classes. It requires the derived class to implement the:
 *  - `OnRelease(ID)` method to describe what will happens when `ref_count`
 * reaches zero.
 *  - `OnDelete(ID)` method to describe what will happens when `ref_count`
 * reaches zero and the object is marked as to be deleted.
 *  - `FetchAndModify(ID, int, int)` method to fetch the current `ref_count` and
 * modify it by the given value.
 */
template <typename ID, typename P, typename Der>
class ColdObjectTracker
    : public DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>> {
 public:
  using cold_object_map_t = tbb::concurrent_hash_map<ID, std::shared_ptr<P>>;
  using base_t = DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>>;

  ColdObjectTracker() {}

  /**
   * @brief remove a blob from the cold object list.
   *
   * @param id The object ID.
   */
  Status RemoveFromColdList(ID const& id) {
    typename cold_object_map_t::const_accessor accessor;
    if (cold_objects_.find(accessor, id)) {
      cold_objects_.erase(accessor);
    }
    return Status::OK();
  }

  using base_t::RemoveDependency;

  Status AddDependency(std::unordered_set<ID> const& ids, int conn) {
    for (auto const& id : ids) {
      RETURN_ON_ERROR(AddDependency(id, conn));
    }
    return Status::OK();
  }

  /**
   * @brief Remove this blob from cold object list if it accessed again.
   */
  Status AddDependency(ID const& id, int conn) {
    RETURN_ON_ERROR(base_t::AddDependency(id, conn));
    RETURN_ON_ERROR(this->RemoveFromColdList(id));
    return Status::OK();
  }

  /**
   * @brief Add a blob to the cold object list.
   */
  Status MarkAsCold(ID const& id, std::shared_ptr<P> payload) {
    typename cold_object_map_t::const_accessor accessor;
    if (payload->IsSealed()) {
      if (!cold_objects_.find(accessor, id)) {
        cold_objects_.emplace(id, payload);
      }
    }
    return Status::OK();
  }

  /**
   * @brief check if a blob is in-use. Return true if it is in-use.
   */
  Status IsInUse(ID const& id, bool& is_in_use) {
    typename cold_object_map_t::const_accessor accessor;
    if (cold_objects_.find(accessor, id)) {
      is_in_use = false;
    } else {
      is_in_use = true;
    }
    return Status::OK();
  }

 public:
  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t changes) {
    return Self().FetchAndModify(id, ref_cnt, changes);
  }

  Status OnRelease(ID const& id) { return Self().OnRelease(id); }

  Status OnDelete(ID const& id) { return Self().OnDelete(id); }

 private:
  inline Der& Self() { return static_cast<Der&>(*this); }
  cold_object_map_t cold_objects_;
};

}  // namespace detail

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_USAGE_H_
