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

#ifndef MODULES_BASIC_DS_HASHMAP_MVCC_H_
#define MODULES_BASIC_DS_HASHMAP_MVCC_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "wyhash/wyhash.hpp"

#include "basic/ds/hashmap.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

namespace detail {}  // namespace detail

/**
 * @brief HashmapMVCC is a semi-persistent hashmap data structure which
 *
 *  - each insert returns a (possible) new HashMap instance
 *    - insert is not thread safe
 *  - doesn't support delete
 *  - query on old instance may see the future value (similar to
 *    READ UNCOMMITTED)
 *  - RAII style
 *
 * The blob or blob write is not freed along with the hashmap object, the
 * caller is responsible for the lifecycle of the blob.
 *
 * @tparam K The type for the key.
 * @tparam V The type for the value.
 * @tparam std::hash<K> The hash function for the key.
 * @tparam std::equal_to<K> The compare function for the key.
 */
template <typename K, typename V, typename H = prime_number_hash_wy<K>,
          typename E = std::equal_to<K>>
class HashmapMVCC
    : public H,
      public E,
      public std::enable_shared_from_this<HashmapMVCC<K, V, H, E>> {
  static_assert(
      std::is_pod<K>::value && std::is_pod<V>::value,
      "K and V must be POD as they will be copied to vineyard's shared memory");

 public:
  using hashmap_t = HashmapMVCC<K, V, H, E>;
  using value_type = std::pair<K, V>;
  using Entry = ska::detailv3::sherwood_v3_entry<value_type>;
  using EntryPointer = const Entry*;
  using MutEntryPointer = Entry*;

  /**
   * @param initial_capacity The initial capacity of the hashmap.
   */
  static Status Make(Client& client, const size_t initial_capacity,
                     std::shared_ptr<hashmap_t>& out) {
    std::unique_ptr<BlobWriter> blob_writer;
    // expect 40% load factor
    // and minimal max lookup (ska::detailv3::min_lookups) is 4
    size_t estimated_bucket_count = std::max(size_t(2), initial_capacity) * 2.5;
    size_t num_slots =
        estimated_bucket_count + compute_max_lookups(estimated_bucket_count);
    RETURN_ON_ERROR(client.CreateBlob(num_slots * sizeof(Entry), blob_writer));
    memset(blob_writer->data(), 0xff /* -1 */, blob_writer->size());
    (reinterpret_cast<MutEntryPointer>(blob_writer->data()) + num_slots - 1)
        ->distance_from_desired = Entry::special_end_value;
    out = std::shared_ptr<hashmap_t>(
        new hashmap_t(client, std::move(blob_writer)));
    return Status::OK();
  }

  static Status View(Client& client, const std::shared_ptr<Blob>& blob,
                     std::shared_ptr<const hashmap_t>& out) {
    out = std::shared_ptr<hashmap_t>(new hashmap_t(client, blob));
    return Status::OK();
  }

  const ObjectID id() const { return blob_id_; }

  const std::shared_ptr<Blob>& blob() const { return blob_; }

  const std::unique_ptr<BlobWriter>& blob_writer() const {
    return blob_writer_;
  }

  /**
   * @brief The iterator to iterate key-value mappings in the HashMap.
   *
   */
  struct iterator {
    iterator() = default;
    explicit iterator(EntryPointer current) : current(current) {}
    EntryPointer current = EntryPointer();

    friend bool operator==(const iterator& lhs, const iterator& rhs) {
      return lhs.current == rhs.current;
    }

    friend bool operator!=(const iterator& lhs, const iterator& rhs) {
      return lhs.current != rhs.current;
    }

    iterator& operator++() {
      do {
        ++current;
      } while (current->is_empty());
      return *this;
    }

    iterator operator++(int) {
      iterator copy(*this);
      ++*this;
      return copy;
    }

    const value_type& operator*() const { return current->value; }

    const value_type* operator->() const {
      return std::addressof(current->value);
    }
  };

  /**
   * @brief The beginning iterator.
   *
   */
  iterator begin() const {
    for (EntryPointer it = entries_;; ++it) {
      if (it->has_value()) {
        return iterator(it);
      }
    }
  }

  /**
   * @brief The ending iterator.
   *
   */
  iterator end() const {
    return iterator(entries_ +
                    static_cast<ptrdiff_t>(num_buckets_ - 1 + max_lookups_));
  }

  /**
   * @brief Find the iterator by key.
   *
   */
  iterator find(const K& key) {
    size_t index = hash_policy_.index_for_hash(hash_object(key));
    EntryPointer it = entries_ + static_cast<ptrdiff_t>(index);
    for (int8_t distance = 0; max_lookups_ > distance && it->has_value();
         ++distance, ++it) {
      if (compares_equal(key, it->value.first)) {
        return iterator(it);
      }
    }
    int8_t distance = 0;
    while (true) {
      if (max_lookups_ <= distance || it->is_empty()) {
        return end();
      }
      if (compares_equal(key, it->value.first)) {
        return iterator(it);
      }
      ++distance;
      ++it;
    }
    return end();
  }

  /**
   * @brief Return the const iterator by key.
   *
   */
  const iterator find(const K& key) const {
    return const_cast<hashmap_t*>(this)->find(key);
  }

  /**
   * @brief Return the number of occurancies of the key.
   *
   */
  size_t count(const K& key) const { return find(key) == end() ? 0 : 1; }

  /**
   * @brief Get the value by key.
   * Here the existence of the key is checked.
   */
  const V& at(const K& key) const {
    auto found = this->find(key);
    if (found == this->end()) {
      throw std::out_of_range("Argument passed to at() was not in the map.");
    }
    return found->second;
  }

  const size_t bucket_count() const { return num_buckets_; }

  /**
   * @brief Reserve the size for the hashmap.
   *
   */
  Status reserve(std::shared_ptr<hashmap_t>& out, const size_t size) const {
    // expect 40% load factor
    size_t estimated_bucket_count = size * 2.5;
    if (estimated_bucket_count > num_buckets_) {
      out = this->shared_from_this();
      return Status::OK();
    }
    return rehash(out, estimated_bucket_count);
  }

  /**
   * @brief Emplace key-value pair into the hashmap.
   *
   * @param out If the emplace finished without rehash, the `out` will
   *        be nullptr, otherwise will be the new hashmap after rehash.
   */
  template <class... Args>
  inline Status emplace(std::shared_ptr<hashmap_t>& out, Args&&... args) {
    if (try_emplace(const_cast<MutEntryPointer>(entries_), hash_policy_,
                    num_buckets_, std::forward<Args>(args)...)) {
      out = nullptr;
      return Status::OK();
    }
    out = this->shared_from_this();
    size_t new_num_buckets = compute_next_num_buckets(out->num_buckets_);
    while (true) {
      DVLOG(100) << "trigger rehash when emplace: " << new_num_buckets;
      RETURN_ON_ERROR(this->rehash(out, new_num_buckets));
      if (try_emplace(const_cast<MutEntryPointer>(out->entries_),
                      out->hash_policy_, out->num_buckets_,
                      std::forward<Args>(args)...)) {
        return Status::OK();
      } else {
        RETURN_ON_ERROR(out->blob_writer_->Abort(client_));
        new_num_buckets = compute_next_num_buckets(out->num_buckets_);
      }
    }
    return Status::OK();
  }

 private:
  HashmapMVCC(Client& client, const std::shared_ptr<Blob>& blob)
      : client_(client), blob_(blob), blob_id_(blob_->id()) {
    num_buckets_ = compute_num_buckets(blob_->size() / sizeof(Entry));
    max_lookups_ = compute_max_lookups(num_buckets_);
    hash_policy_.set_prime(num_buckets_ - 1);
    entries_ = reinterpret_cast<EntryPointer>(blob_->data());
  }

  HashmapMVCC(Client& client, std::unique_ptr<BlobWriter>&& blob_writer)
      : client_(client),
        blob_writer_(std::move(blob_writer)),
        blob_id_(blob_writer_->id()) {
    num_buckets_ = compute_num_buckets(blob_writer_->size() / sizeof(Entry));
    max_lookups_ = compute_max_lookups(num_buckets_);
    hash_policy_.set_prime(num_buckets_ - 1);
    entries_ = reinterpret_cast<EntryPointer>(blob_writer_->data());
  }

  HashmapMVCC(const HashmapMVCC<K, V, H, E>&) = delete;
  HashmapMVCC(HashmapMVCC<K, V, H, E>&&) = delete;
  HashmapMVCC& operator=(const HashmapMVCC<K, V, H, E>&) = delete;
  HashmapMVCC& operator=(HashmapMVCC<K, V, H, E>&&) = delete;

  Status rehash(std::shared_ptr<hashmap_t>& out, const size_t num_buckets) {
    size_t new_num_buckets = num_buckets;
    while (true) {
      std::unique_ptr<BlobWriter> new_blob_writer;
      size_t num_slots = new_num_buckets + compute_max_lookups(new_num_buckets);
      RETURN_ON_ERROR(
          client_.CreateBlob(num_slots * sizeof(Entry), new_blob_writer));
      memset(new_blob_writer->data(), 0xff /* -1 */, new_blob_writer->size());
      MutEntryPointer new_entries =
          reinterpret_cast<MutEntryPointer>(new_blob_writer->data());
      (new_entries + num_slots - 1)->distance_from_desired =
          Entry::special_end_value;
      prime_hash_policy new_hash_policy;
      new_hash_policy.set_prime(new_num_buckets - 1);

      bool succeed = true;
      for (EntryPointer begin = entries_,
                        end = entries_ + num_buckets_ - 1 + max_lookups_;
           begin != end; ++begin) {
        if (begin->is_empty()) {
          continue;
        }
        succeed &= try_emplace(new_entries, new_hash_policy, new_num_buckets,
                               begin->value.first, begin->value.second);
        if (!succeed) {
          break;
        }
      }

      if (succeed) {
        out = std::shared_ptr<hashmap_t>(
            new hashmap_t(client_, std::move(new_blob_writer)));
        break;
      } else {
        // try next round with more buckets
        VINEYARD_DISCARD(new_blob_writer->Abort(client_));
        new_num_buckets = compute_next_num_buckets(new_num_buckets);
      }
    }
    return Status::OK();
  }

  /**
   * @return The the emplace succeed, false means rehash is required.
   */
  template <typename Key, typename... Args>
  bool try_emplace(MutEntryPointer entries, prime_hash_policy& hash_policy,
                   const size_t num_buckets, Key&& key, Args&&... args) {
    size_t index = hash_policy.index_for_hash(hash_object(key));
    MutEntryPointer current_entry = entries + ptrdiff_t(index);
    int8_t distance_from_desired = 0;
    for (; max_lookups_ > distance_from_desired && current_entry->has_value();
         ++current_entry, ++distance_from_desired) {
      if (compares_equal(key, current_entry->value.first)) {
        return true;
      }
    }
    return emplace_new_key(entries, hash_policy, num_buckets,
                           distance_from_desired, current_entry,
                           std::forward<Key>(key), std::forward<Args>(args)...);
  }

  template <typename Key, typename... Args>
  bool emplace_new_key(MutEntryPointer entries, prime_hash_policy& hash_policy,
                       const size_t num_buckets, int8_t distance_from_desired,
                       MutEntryPointer current_entry, Key&& key,
                       Args&&... args) {
    const uint8_t max_lookups = compute_max_lookups(num_buckets);
    if (num_buckets - 1 == 0 || distance_from_desired == max_lookups) {
      return false;
    } else if (current_entry->is_empty()) {
      current_entry->emplace(distance_from_desired, std::forward<Key>(key),
                             std::forward<Args>(args)...);
      return true;
    }
    for (++distance_from_desired, ++current_entry;; ++current_entry) {
      if (current_entry->is_empty()) {
        current_entry->emplace(distance_from_desired, std::forward<Key>(key),
                               std::forward<Args>(args)...);
        return true;
      } else {
        ++distance_from_desired;
        if (distance_from_desired == max_lookups) {
          return false;
        }
      }
    }
  }

  static const int8_t compute_max_lookups(const size_t num_buckets) {
    int8_t desired = ska::detailv3::log2(num_buckets);
    return std::max(ska::detailv3::min_lookups, desired);
  }

  // num_slots = num_buckets + compute_max_lookups(num_buckets)
  static const size_t compute_num_buckets(const size_t num_slots) {
    size_t num_buckets = num_slots - compute_max_lookups(num_slots);
    while (num_buckets + compute_max_lookups(num_buckets) != num_slots) {
      num_buckets -= 1;
    }
    return num_buckets;
  }

  static const size_t compute_next_num_buckets(const size_t num_buckets) {
    return std::max(size_t(4), num_buckets * 2);
  }

  const size_t hash_object(const K& key) const {
    return static_cast<const H&>(*this)(key);
  }

  const bool compares_equal(const K& lhs, const K& rhs) const {
    return static_cast<const E&>(*this)(lhs, rhs);
  }

  Client& client_;

  std::shared_ptr<Blob> blob_;
  std::unique_ptr<BlobWriter> blob_writer_;
  ObjectID blob_id_ = InvalidObjectID();

  size_t num_buckets_ = 0;
  int8_t max_lookups_ = 0;
  prime_hash_policy hash_policy_;
  EntryPointer entries_;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_HASHMAP_MVCC_H_
