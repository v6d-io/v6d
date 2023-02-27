/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

#include <algorithm>
#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "libcuckoo/cuckoohash_map.hh"

#include "common/memory/payload.h"
#include "common/util/arrow.h"
#include "common/util/lifecycle.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "server/memory/allocator.h"
#include "server/util/file_io_adaptor.h"
#include "server/util/spill_file.h"

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
  using dependency_map_t = libcuckoo::cuckoohash_map</*socket_connection*/ int,
                                                     ska::flat_hash_set<ID>>;
  DependencyTracker() {}

  /**
   * @brief Add dependency of blobs to the socket connection. Also increase the
   * reference count of blobs by one if they are not tracked before. Each socket
   * connection can increase the reference count of a blob at most by one.
   *
   * @param ids The blob IDs.
   * @param socket_connection The socket connection ID.
   */
  Status AddDependency(std::unordered_set<ID> const& ids, const int conn) {
    for (auto const& id : ids) {
      RETURN_ON_ERROR(AddDependency(id, conn));
    }
    return Status::OK();
  }

  Status AddDependency(const ID id, const int conn) {
    Status status;
    auto fn = [this, id, &status](ska::flat_hash_set<ID>& objects) -> bool {
      if (objects.find(id) == objects.end()) {
        objects.emplace(id);
        status = this->IncreaseReferenceCount(id);
      }
      return false;
    };
    if (dependency_.upsert(conn, fn, ska::flat_hash_set<ID>{})) {
      // newly inserted, needs updating it again
      dependency_.update_fn(conn, fn);
    }
    return status;
  }

  /**
   * @brief Remove the dependency of the blob from the socket connection.
   * Decrease the reference count of the blob by one.
   *
   * @param ids The blob IDs.
   * @param socket_connection The socket connection ID.
   */
  Status RemoveDependency(std::unordered_set<ID> const& ids, const int conn) {
    for (auto const& id : ids) {
      RETURN_ON_ERROR(RemoveDependency(id, conn));
    }
    return Status::OK();
  }

  Status RemoveDependency(const ID id, const int conn) {
    Status status;
    bool accessed = dependency_.erase_fn(
        conn, [id, &status](ska::flat_hash_set<ID>& objects) -> bool {
          if (objects.find(id) == objects.end()) {
            status = Status::ObjectNotExists(
                "DependencyTracker: failed to find object during remove "
                "dependency: " +
                ObjectIDToString(id));
            return false;
          } else {
            objects.erase(id);
            return objects.empty();
          }
        });
    if (!accessed) {
      return Status::KeyError("DependencyTracker: connection not exist.");
    }
    RETURN_ON_ERROR(status);
    return this->DecreaseReferenceCount(id);
  }

  /**
   * @brief delete a object lazily, this will add the object to a delete queue
   * and do the actual deletion when their reference count reaches zero.
   */
  Status PreDelete(ID const& id) { return base_t::PreDelete(id); }

  /**
   * @brief Remove all the dependency of all objects in given socket connection.
   */
  Status ReleaseConnection(const int conn) {
    Status status;
    bool accessed = dependency_.erase_fn(
        conn, [this, &status](ska::flat_hash_set<ID>& objects) -> bool {
          for (auto& elem : objects) {
            // try our best to remove dependency.
            status = this->DecreaseReferenceCount(elem);
            if (!status.ok()) {
              return false;  // early return, don't delete
            }
          }
          return true;
        });
    if (!accessed) {
      return Status::KeyError("connection doesn't exist.");
    }
    return status;
  }

 public:
  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t changes) {
    return self().FetchAndModify(id, ref_cnt, changes);
  }

  Status OnRelease(ID const& id) { return self().OnRelease(id); }

  Status OnDelete(ID const& id) { return self().OnDelete(id); }

 private:
  inline Der& self() { return static_cast<Der&>(*this); }

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
   * then status will be Invalids.
   * - `CheckExist(ID id)` Check the existence of id.
   */
  class LRU {
   public:
    using value_t = std::pair<ID, std::shared_ptr<P>>;
    using lru_map_t =
        ska::flat_hash_map<ID, typename std::list<value_t>::iterator>;
    using lru_list_t = std::list<value_t>;

    LRU() = default;
    ~LRU() = default;

    void Ref(const ID id, const std::shared_ptr<P>& payload) {
      std::lock_guard<decltype(mu_)> locked(mu_);
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

    bool CheckExist(const ID id) const {
      std::lock_guard<decltype(mu_)> locked(mu_);
      return map_.find(id) != map_.end();
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
    Status Unref(const ID id, const bool fast_delete,
                 const std::shared_ptr<Der>& bulk_store) {
      std::lock_guard<decltype(mu_)> locked(mu_);
      auto it = map_.find(id);
      if (it == map_.end()) {
        auto spilled = spilled_obj_.find(id);
        if (spilled == spilled_obj_.end()) {
          return Status::OK();
        }
        if (!fast_delete) {
          // NB: explicitly copy the std::shared_ptr as the iterator is not
          // stable.
          auto payload = spilled->second;
          RETURN_ON_ERROR(bulk_store->ReloadPayload(id, payload));
        } else {
          RETURN_ON_ERROR(bulk_store->DeletePayloadFile(id));
        }
        spilled_obj_.erase(spilled);
        return Status::OK();
      } else {
        list_.erase(it->second);
        map_.erase(it);
        return Status::OK();
      }
    }

    Status SpillFor(const size_t sz, const std::shared_ptr<Der>& bulk_store) {
      std::lock_guard<decltype(mu_)> locked(mu_);
      size_t spilled_sz = 0;
      auto status = Status::OK();
      auto it = list_.rbegin();
      std::map<ObjectID, std::shared_ptr<Payload>> pinned_objects;
      while (it != list_.rend()) {
        if (it->second->IsPinned()) {
          // bypass pinned
          pinned_objects.emplace(it->first, it->second);
          it++;
          continue;
        }

        auto s = this->spill(it->first, it->second, bulk_store);
        if (s.ok()) {
          spilled_sz += it->second->data_size;
          map_.erase(it->first);
        } else if (s.IsObjectSpilled()) {
          map_.erase(it->first);
        } else {
          status += s;
          break;
        }
        it++;
        if (sz <= spilled_sz) {
          break;
        }
      }
      auto popped_size = std::distance(list_.rbegin(), it);
      while (popped_size-- > 0) {
        list_.pop_back();
      }
      // restore pinned objects
      for (auto const& item : pinned_objects) {
        Ref(item.first, item.second);
      }
      if (!status.ok() || (status.ok() && spilled_sz == 0)) {
        auto s =
            Status::NotEnoughMemory("Still not enough memory after spilling");
        s += status;
        return s;
      }
      return Status::OK();
    }

    Status SpillObjects(
        const std::map<ObjectID, std::shared_ptr<Payload>>& objects,
        const std::shared_ptr<Der>& bulk_store) {
      std::lock_guard<decltype(mu_)> locked(mu_);
      auto status = Status::OK();
      for (auto const& item : objects) {
        if (item.second->IsPinned()) {
          // bypass pinned objects
          continue;
        }
        status += this->spill(item.first, item.second, bulk_store);
      }
      return status;
    }

    Status ReloadObjects(
        const std::map<ObjectID, std::shared_ptr<Payload>>& objects,
        const bool pin, const std::shared_ptr<Der>& bulk_store) {
      std::lock_guard<decltype(mu_)> locked(mu_);
      auto status = Status::OK();
      for (auto const& item : objects) {
        status += this->reload(item.first, item.second, pin, bulk_store);
      }
      return status;
    }

    bool CheckSpilled(const ID& id) {
      std::lock_guard<decltype(mu_)> locked(mu_);
      return spilled_obj_.find(id) != spilled_obj_.end();
    }

   private:
    Status spill(const ObjectID object_id,
                 const std::shared_ptr<Payload>& payload,
                 const std::shared_ptr<Der>& bulk_store) {
      std::lock_guard<decltype(mu_)> locked(mu_);
      if (payload->is_spilled) {
        return Status::ObjectSpilled(object_id);
      }
      spilled_obj_.emplace(object_id, payload);
      return bulk_store->SpillPayload(payload);
    }

    Status reload(const ObjectID object_id,
                  const std::shared_ptr<Payload>& payload, const bool pin,
                  const std::shared_ptr<Der>& bulk_store) {
      std::lock_guard<decltype(mu_)> locked(mu_);
      if (pin) {
        payload->Pin();
      }
      if (!payload->is_spilled) {
        return Status::OK();
      }
      {
        auto loc = spilled_obj_.find(object_id);
        if (loc != spilled_obj_.end()) {
          spilled_obj_.erase(loc);
        }
      }
      return bulk_store->ReloadPayload(object_id, payload);
    }

    mutable std::recursive_mutex mu_;
    // protected by mu_
    lru_map_t map_;
    lru_list_t list_;
    ska::flat_hash_map<ID, std::shared_ptr<P>> spilled_obj_;
  };

 public:
  using base_t = DependencyTracker<ID, P, ColdObjectTracker<ID, P, Der>>;
  using lru_t = LRU;

  ColdObjectTracker() {}
  ~ColdObjectTracker() {
    if (!spill_path_.empty()) {
      io::FileIOAdaptor io_adaptor(spill_path_);
      DISCARD_ARROW_ERROR(io_adaptor.DeleteDir());
    }
  }

  /**
   * @brief remove a blob from the cold object list.
   *
   * @param id The object ID.
   * @param is_delete Indicates if is to delete or for later reference.
   */
  Status RemoveFromColdList(const ID id, const bool is_delete) {
    RETURN_ON_ERROR(cold_obj_lru_.Unref(id, is_delete, shared_from_self()));
    return Status::OK();
  }

  using base_t::RemoveDependency;

  Status AddDependency(std::unordered_set<ID> const& ids, const int conn) {
    for (auto const& id : ids) {
      RETURN_ON_ERROR(AddDependency(id, conn));
    }
    return Status::OK();
  }

  /**
   * @brief Remove this blob from cold object list if it accessed again.
   */
  Status AddDependency(const ID id, const int conn) {
    RETURN_ON_ERROR(base_t::AddDependency(id, conn));
    RETURN_ON_ERROR(this->RemoveFromColdList(id, false));
    return Status::OK();
  }

  /**
   * @brief Add a blob to the cold object list.
   */
  Status MarkAsCold(const ID id, const std::shared_ptr<P>& payload) {
    if (payload->IsSealed()) {
      cold_obj_lru_.Ref(id, payload);
    }
    // n.b.: unseal blobs shouldn't be spilled, as will be re-get by clients
    // with a "unsafe" argument.
    return Status::OK();
  }

  /**
   * @brief check if a blob is in-use. Return true if it is in-use.
   */
  Status IsInUse(const ID id, bool& is_in_use) {
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
  Status IsSpilled(const ID id, bool& is_spilled) {
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
  Status SpillColdObjectFor(const int64_t sz) {
    if (spill_path_.empty()) {
      return Status::Invalid("Spill path is not set");
    }
    if (sz == 0) {
      return Status::OK();
    } else if (sz < 0) {
      return Status::Invalid("The expected spill size is invalid");
    }
    return cold_obj_lru_.SpillFor(sz, shared_from_self());
  }

  /**
   * @brief Triggered when been requested to spill specified objects to disk.
   * @param objects spilled blobs
   */
  Status SpillColdObjects(
      const std::map<ObjectID, std::shared_ptr<Payload>>& objects) {
    if (spill_path_.empty()) {
      return Status::Invalid("Spill path is not set");
    }
    return cold_obj_lru_.SpillObjects(objects, shared_from_self());
  }

  /**
   * @brief Triggered when been requested to spill specified objects to disk.
   * @param objects reloaded blobs
   */
  Status ReloadColdObjects(
      const std::map<ObjectID, std::shared_ptr<Payload>>& objects,
      const bool pin) {
    if (spill_path_.empty()) {
      return Status::OK();  // bypass, as spill is not enabled
    }
    return cold_obj_lru_.ReloadObjects(objects, pin, shared_from_self());
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
  uint8_t* AllocateMemoryWithSpill(const size_t size, int* fd,
                                   int64_t* map_size, ptrdiff_t* offset) {
    uint8_t* pointer = nullptr;
    pointer = self().AllocateMemory(size, fd, map_size, offset);
    // no spill will be conducted
    if (spill_path_.empty()) {
      return pointer;
    }

    // spill will be triggered when:
    //
    //  1. unable to allocate memory
    //  2. memory usage is above upper bound
    if (pointer == nullptr ||
        BulkAllocator::Allocated() >= self().mem_spill_upper_bound_) {
      std::unique_lock<std::mutex> locked(spill_mu_);

      int64_t min_spill_size = 0;
      if (pointer == nullptr) {
        min_spill_size = size - (BulkAllocator::GetFootprintLimit() -
                                 BulkAllocator::Allocated());
      }
      if (BulkAllocator::Allocated() > self().mem_spill_lower_bound_) {
        min_spill_size =
            std::max(min_spill_size, BulkAllocator::Allocated() -
                                         self().mem_spill_lower_bound_);
      }

      auto s = SpillColdObjectFor(min_spill_size);
      if (!s.ok()) {
        DLOG(ERROR) << "Error during spilling cold object: " << s.ToString();
      }

      // try to allocate again if needed
      if (pointer == nullptr) {
        pointer = self().AllocateMemory(size, fd, map_size, offset);
      }
    }
    return pointer;
  }

 public:
  Status FetchAndModify(const ID id, int64_t& ref_cnt, int64_t changes) {
    return self().FetchAndModify(id, ref_cnt, changes);
  }

  Status OnRelease(const ID id) { return self().OnRelease(id); }

  Status OnDelete(const ID id) { return self().OnDelete(id); }

 protected:
  Status SpillPayload(const std::shared_ptr<P>& payload) {
    if (!payload->is_sealed) {
      return Status::ObjectNotSealed(
          "payload is not sealed and cannot be spilled: " +
          ObjectIDToString(payload->object_id));
    }
    if (payload->is_spilled) {
      return Status::ObjectSpilled(payload->object_id);
    }
    {
      io::SpillFileWriter writer(spill_path_);
      RETURN_ON_ERROR(writer.Write(payload));
      RETURN_ON_ERROR(writer.Sync());
    }
    BulkAllocator::Free(payload->pointer, payload->data_size);
    payload->store_fd = -1;
    payload->pointer = nullptr;
    payload->is_spilled = true;
    return Status::OK();
  }

  Status ReloadPayload(const ID id, const std::shared_ptr<P>& payload) {
    if (!payload->is_spilled) {
      return Status::ObjectNotSpilled(payload->object_id);
    }
    {
      io::SpillFileReader reader(spill_path_);
      RETURN_ON_ERROR(reader.Read(payload, shared_from_self()));
    }
    return this->DeletePayloadFile(id);
  }

  Status DeletePayloadFile(const ID id) {
    io::FileIOAdaptor io_adaptor(spill_path_);
    RETURN_ON_ERROR(io_adaptor.RemoveFile(spill_path_ + std::to_string(id)));
    return Status::OK();
  }

  void SetSpillPath(const std::string& spill_path) {
    spill_path_ = spill_path;
    if (spill_path.empty()) {
      LOG(INFO) << "No spill path set, spill has been disabled ...";
      return;
    }
    if (spill_path.back() != '/') {
      spill_path_.push_back('/');
    }
    io::FileIOAdaptor io_adaptor(spill_path_ + "test");
    if (io_adaptor.Open("w").ok()) {
      DISCARD_ARROW_ERROR(io_adaptor.RemoveFile(spill_path_ + "test"));
    } else {
      LOG(WARNING)
          << "Disabling spilling as the specified spill directory '"
          << spill_path_
          << "' doesn't exist, or vineyardd doesn't have the write permission";
      spill_path_.clear();
    }
  }

 private:
  inline Der& self() { return static_cast<Der&>(*this); }
  virtual std::shared_ptr<Der> shared_from_self() = 0;

  lru_t cold_obj_lru_;
  std::string spill_path_;
  std::mutex spill_mu_;
};

}  // namespace detail

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_USAGE_H_
