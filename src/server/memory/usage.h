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

#include <glog/logging.h>
#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "server/util/spill_file.h"

#ifdef __APPLE__
#include "boost/thread/shared_mutex.hpp"
#endif

#include "oneapi/tbb/concurrent_hash_map.h"

#include "common/memory/payload.h"
#include "common/util/lifecycle.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "server/memory/allocator.h"
#include "server/util/file_io_adaptor.h"

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
 *    reaches zero.
 *  - `OnDelete(ID)` method to describe what will happens when `ref_count`
 *    reaches zero and the object is marked as to be deleted.
 *  - `FetchAndModify(ID, int, int)` method to fetch the current `ref_count` and
 *    modify it by the given value.
 */
template <typename ID, typename P, typename Der>
class ColdObjectTracker
    : public DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>> {
 public:
  /*
   * @brief LRU is a tracker of least recent used blob ID, it has two methods:
   * - `Ref(ID id)` Add the id if not exists. (Actually here we shouldn't expect
   *    a redundant Ref, because no Object will be insert twice). But in current
   *    implementation, we will overwrite the previous one.
   * - `Unref(ID id)` Remove the designated id from lru.
   * - `PopLeastUsed()` Get the least used blob id. If no object in structure,
   * then statu will be Invalids.
   * - `CheckExist(ID id)` Check the existence of id.
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
        list_.erase(it->second);
        list_.emplace_front(id, payload);
        it->second = list_.begin();
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

    /**
     * @brief Here we have two actions: 1. delete from lru_list
     *        2. delete from spilled_obj_
     * @param id is the objectID
     * @param fast_delete indicates if we directly remove the spilled object
     * without reload
     * @param store_ptr is used for spill
     * @return * Status
     */
    Status Unref(const ID& id, bool fast_delete,
                 std::shared_ptr<Der> store_ptr) {
      std::unique_lock<decltype(mu_)> locked;
      auto it = map_.find(id);
      if (it == map_.end()) {
        auto it = spilled_obj_.find(id);
        if (it == spilled_obj_.end()) {
          return Status::OK();
        }
        if (!fast_delete) {
          RETURN_ON_ERROR(store_ptr->ReloadPayload_(id, it->second));
        } else {
          store_ptr->DeletePayloadFile_(id);
        }
        spilled_obj_.erase(it);
        return Status::OK();
      }
      list_.erase(it->second);
      map_.erase(it);
      return Status::OK();
    }

    Status Spill(size_t sz, std::shared_ptr<Der> bulk_store_ptr) {
      std::unique_lock<decltype(mu_)> locked;
      size_t spilled_sz = 0;
      auto st = Status::OK();
      auto it = list_.rbegin();
      while (it != list_.rend()) {
        st = bulk_store_ptr->SpillPayload_(it->second);
        if (!st.ok()) {
          LOG(ERROR) << st.ToString();
          break;
        }
        spilled_sz += it->second->data_size;
        spilled_obj_.emplace(it->first, it->second);
        map_.erase(it->first);
        it++;
        if (sz <= spilled_sz) {
          break;
        }
      }
      auto poped_size = std::distance(list_.rbegin(), it);
      while (poped_size-- > 0) {
        list_.pop_back();
      }
      if (st.ok() && spilled_sz == 0) {
        return Status::NotEnoughMemory("Nothing spilled");
      }
      return st;
    }

    bool CheckSpilled(const ID& id) {
      std::shared_lock<decltype(mu_)> lock;
      return spilled_obj_.find(id) != spilled_obj_.end();
    }

   private:
#if __APPLE__
    mutable boost::shared_mutex mu_;
#else
    mutable std::shared_timed_mutex mu_;
#endif
    // protected by mu_
    lru_map_t map_;
    lru_list_t list_;
    std::unordered_map<ID, std::shared_ptr<P>> spilled_obj_;
  };

 public:
  using cold_object_map_t = tbb::concurrent_hash_map<ID, std::shared_ptr<P>>;
  using base_t = DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>>;
  using lru_t = LRU;

  ColdObjectTracker() {}
  ~ColdObjectTracker() {
    if (!spill_path.empty()) {
      util::FileIOAdaptor io_adaptor(spill_path);
      io_adaptor.DeleteDir();
    }
  }

  /**
   * @brief remove a blob from the cold object list.
   *
   * @param id The object ID.
   * @param is_delete Indicates if is to delete or for later reference.
   */
  Status RemoveFromColdList(ID const& id, bool is_delete) {
    RETURN_ON_ERROR(
        cold_obj_lru_.Unref(id, is_delete, Self().shared_from_this()));
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
    RETURN_ON_ERROR(this->RemoveFromColdList(id, false));
    return Status::OK();
  }

  /**
   * @brief Add a blob to the cold object list.
   */
  Status MarkAsCold(ID const& id, std::shared_ptr<P> payload) {
    if (payload->IsSealed()) {
      cold_obj_lru_.Ref(id, payload);
      return Status::OK();
    }
    std::string err_msg = "Not Sealed, Object ID: " + std::to_string(id);
    LOG(WARNING) << err_msg;
    return Status::ObjectNotSealed(err_msg);
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

  /**
   * @brief check if a blob is spilled out. Return true if it is spilled.
   */
  Status IsSpilled(ID const& id, bool& is_spilled) {
    if (cold_obj_lru_.CheckSpilled(id)) {
      is_spilled = true;
    } else {
      is_spilled = false;
    }
    return Status::OK();
  }

  /**
   * @brief Only triggered when detected OOM, this function will spill cold-obj
   * to disk till memory usage back to allowed watermark.
   * @param sz spilled size
   */
  Status SpillColdObject(int64_t sz) {
    if (sz <= 0) {
      return Status::NotEnoughMemory("Nothing will be spilled");
    }
    return cold_obj_lru_.Spill(sz, Self().shared_from_this());
  }

  /**
   * @brief If spill_path is set, then spill will be conducted if memory
   * threshold is triggered or we got an empty pointer
   *
   * @return - If spill is disable, then just allocate memory and return
   * whatever we got
   *  - If spill is allowed, then we shall conduct spilling and trying to give a
   * non-nullptr pointer
   */
  uint8_t* AllocateMemoryWithSpill(size_t size, int* fd, int64_t* map_size,
                                   ptrdiff_t* offset) {
    uint8_t* pointer = nullptr;
    pointer = Self().AllocateMemory(size, fd, map_size, offset);
    // no spill will be conducted
    if (spill_path.empty()) {
      return pointer;
    }
    if (pointer == nullptr ||
        BulkAllocator::Allocated() >=
            static_cast<int64_t>(Self().mem_spill_upper_bound_)) {
      std::unique_lock<std::mutex> locked(spill_mu_);
      // if already got someone spilled, then we should allocate normally
      if (pointer == nullptr)
        pointer = Self().AllocateMemory(size, fd, map_size, offset);

      if (pointer == nullptr ||
          BulkAllocator::Allocated() >=
              static_cast<int64_t>(Self().mem_spill_upper_bound_)) {
        int64_t spill_size =
            BulkAllocator::Allocated() - Self().mem_spill_lower_bound_;
        if (SpillColdObject(spill_size).ok()) {
          pointer = pointer ? pointer
                            : Self().AllocateMemory(size, fd, map_size, offset);
        }
      }
    }
    return pointer;
  }

 public:
  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t changes) {
    return Self().FetchAndModify(id, ref_cnt, changes);
  }

  Status OnRelease(ID const& id) { return Self().OnRelease(id); }

  Status OnDelete(ID const& id) { return Self().OnDelete(id); }

 protected:
  Status SpillPayload_(std::shared_ptr<P>& payload) {
    assert(payload->is_sealed);
    util::SpillWriteFile write_file(spill_path);
    RETURN_ON_ERROR(write_file.Write(payload));
    RETURN_ON_ERROR(write_file.Sync());
    BulkAllocator::Free(payload->pointer, payload->data_size);
    payload->store_fd = -1;
    payload->pointer = nullptr;
    payload->is_spilled = true;
    return Status::OK();
  }

  Status ReloadPayload_(const ID& id, std::shared_ptr<P>& payload) {
    assert(payload->is_spilled == true);
    util::SpillReadFile read_file(spill_path);
    RETURN_ON_ERROR(read_file.Read(payload, Self().shared_from_this()));
    return Status::OK();
  }

  Status DeletePayloadFile_(const ID& id) {
    util::FileIOAdaptor io_adaptor(spill_path);
    RETURN_ON_ERROR(io_adaptor.RemoveFile(spill_path + std::to_string(id)));
    return Status::OK();
  }

  void SetSpillPath(const std::string& spill_path) {
    this->spill_path = spill_path;
    if(spill_path.empty()){
      LOG(ERROR) << "No spill path set, disable spill...";
      return;
    }
    if (spill_path.back() != '/') {
      this->spill_path.push_back('/');
    }
    util::FileIOAdaptor io_adaptor(this->spill_path + "test");
    if (io_adaptor.Open("w").ok()) {
      io_adaptor.RemoveFile(this->spill_path + "test");
    } else {
      LOG(ERROR) << "No such directory " << this->spill_path;
      LOG(ERROR) << "So disable spill...";
      this->spill_path.clear();
    }
  }

 private:
  inline Der& Self() { return static_cast<Der&>(*this); }
  lru_t cold_obj_lru_;
  std::string spill_path;
  std::mutex spill_mu_;
};

}  // namespace detail

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_USAGE_H_
