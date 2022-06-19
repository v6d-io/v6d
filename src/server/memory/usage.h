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

#include <list>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
  /*
   * @brief LRU is a tracker of least recent used blob ID, it has two methods:
   * - `Ref(ID id)` Add the id if not exists, or refresh usage if exists
   * - `PopLeastUsed()` Get the least used blob id. If no object in structure,
   * then statu will be Invalids
   */
  class LRU {
   public:
    using value_t = std::pair<ID, std::shared_ptr<P>>;
    using lru_map_t =
        std::unordered_map<ID, typename std::list<value_t>::iterator>;
    using lru_list_t = std::list<value_t>;
    LRU() = default;
    ~LRU() = default;
    void Ref(ID id, std::shared_ptr<P> payload) {
      std::unique_lock<decltype(mu_)> locked;
      auto it = map_.find(id);
      if (it == map_.end()) {
        list_.emplace_front(id, payload);
        map_.emplace(id, list_.begin());
      } else {
        list_.splice(list_.begin(), list_, it->second);
      }
    }

    bool CheckExist(ID id) const {
      std::shared_lock<decltype(mu_)> shared_locked;
      auto it = map_.find(id);
      if (it == map_.end()) {
        return false;
      }
      return true;
    }

    Status UnRef(const ID& id) {
      std::unique_lock<decltype(mu_)> locked;
      auto it = map_.find(id);
      if (it == map_.end()) {
        return Status::OK();
      }
      list_.erase(it->second);
      map_.erase(it);
      return Status::OK();
    }

    std::pair<Status, value_t> PopLeastUsed() {
      std::unique_lock<decltype(mu_)> locked;
      if (list_.empty()) {
        return {Status::Invalid(), -1};
      }
      auto back = list_.back();
      map_.erase(back.first);
      list_.pop_back();
      return {Status::OK(), back};
    }

   private:
    mutable std::shared_timed_mutex mu_;
    // protect by mu_
    lru_map_t map_;
    lru_list_t list_;
  };

 public:
  using cold_object_map_t = tbb::concurrent_hash_map<ID, std::shared_ptr<P>>;
  using base_t = DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>>;
  using lru_t = LRU;

  ColdObjectTracker() {}

  /**
   * @brief remove a blob from the cold object list.
   *
   * @param id The object ID.
   */
  Status RemoveFromColdList(ID const& id) {
    cold_obj_lru_.UnRef(id);
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
    if (payload->IsSealed()) {
      cold_obj_lru_.Ref(id, payload);
    }
    return Status::OK();
  }

  /**
   * @brief check if a blob is in-use. Return true if it is in-use.
   */
  Status IsInUse(ID const& id, bool& is_in_use) {
    if (cold_obj_lru_.CheckExist(id)) {
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
  lru_t cold_obj_lru_;
};

}  // namespace detail

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_USAGE_H_
