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

template <typename ID, typename P, typename Der>
class DependencyTracker
    : public LifeCycleTracker<ID, P, DependencyTracker<ID, P, Der>> {
 public:
  using base_t = LifeCycleTracker<ID, P, DependencyTracker<ID, P, Der>>;
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
   * Note:
   * If object_id exists in dependency_, then its reference count should
   * determinately greater than 0, thus it will determinately not be deleted.
   * Delete will not remove dependency.
   */
  Status PreDelete(ID const& id) { return base_t::PreDelete(id); }

  /// Remove the dependency of all objects in given connection.
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

  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t changes) {
    return Self().FetchAndModify(id, ref_cnt, changes);
  }

  Status OnRelease(ID const& id) { return Self().OnRelease(id); }

  Status OnDelete(ID const& id) { return Self().OnDelete(id); }

 private:
  inline Der& Self() { return static_cast<Der&>(*this); }
  dependency_map_t dependency_;
};

template <typename ID, typename P, typename Der>
class ColdObjectTracker
    : public DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>> {
 public:
  using cold_object_map_t = tbb::concurrent_hash_map<ID, std::shared_ptr<P>>;
  using base_t = DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>>;

  ColdObjectTracker() {}

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

  Status AddDependency(ID const& id, int conn) {
    RETURN_ON_ERROR(base_t::AddDependency(id, conn));
    RETURN_ON_ERROR(this->RemoveFromColdList(id));
    return Status::OK();
  }

  Status MarkAsCold(ID const& id, std::shared_ptr<P> payload) {
    typename cold_object_map_t::const_accessor accessor;
    if (payload->IsSealed()) {
      if (!cold_objects_.find(accessor, id)) {
        cold_objects_.emplace(id, payload);
      }
    }
    return Status::OK();
  }

  Status IsInUse(ID const& id, bool& is_in_use) {
    typename cold_object_map_t::const_accessor accessor;
    if (cold_objects_.find(accessor, id)) {
      is_in_use = false;
    } else {
      is_in_use = true;
    }
    return Status::OK();
  }

  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t changes) {
    return Self().FetchAndModify(id, ref_cnt, changes);
  }

  Status OnRelease(ID const& id) { return Self().OnRelease(id); }

  Status OnDelete(ID const& id) { return Self().OnDelete(id); }

 private:
  inline Der& Self() { return static_cast<Der&>(*this); }
  cold_object_map_t cold_objects_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_USAGE_H_
