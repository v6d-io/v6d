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

#ifndef MODULES_BASIC_DS_HASHMAP_H_
#define MODULES_BASIC_DS_HASHMAP_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "wyhash/wyhash.hpp"

#include "basic/ds/array.h"
#include "basic/ds/hashmap.vineyard.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/arrow.h"
#include "common/util/uuid.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "BBHash/BooPHF.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace vineyard {

/**
 * @brief HashmapBuilder is used for constructing hashmaps that supported by
 * vineyard.
 *
 * @tparam K The type for the key.
 * @tparam V The type for the value.
 * @tparam std::hash<K> The hash function for the key.
 * @tparam std::equal_to<K> The compare function for the key.
 */
template <typename K, typename V, typename H = prime_number_hash_wy<K>,
          typename E = std::equal_to<K>>
class HashmapBuilder : public HashmapBaseBuilder<K, V, H, E> {
 public:
  explicit HashmapBuilder(Client& client)
      : HashmapBaseBuilder<K, V, H, E>(client) {}

  HashmapBuilder(Client& client, ska::flat_hash_map<K, V, H, E>&& hashmap)
      : HashmapBaseBuilder<K, V, H, E>(client), hashmap_(std::move(hashmap)) {}

  /**
   * @brief Get the mapping value of the given key.
   *
   */
  inline V& operator[](const K& key) { return hashmap_[key]; }

  /**
   * @brief Get the mapping value of the given key.
   *
   */
  inline V& operator[](K&& key) { return hashmap_[std::move(key)]; }

  /**
   * @brief Emplace key-value pair into the hashmap.
   *
   */
  template <class... Args>
  inline bool emplace(Args&&... args) {
    return hashmap_.emplace(std::forward<Args>(args)...).second;
  }

  /**
   * @brief Get the mapping value of the given key.
   *
   */
  V& at(const K& key) { return hashmap_.at(key); }

  /**
   * @brief Get the const mapping value of the given key.
   *
   */
  const V& at(const K& key) const { return hashmap_.at(key); }

  /**
   * @brief Get the size of the hashmap.
   *
   */
  size_t size() const { return hashmap_.size(); }

  /**
   * @brief Reserve the size for the hashmap.
   *
   */
  void reserve(size_t size) { hashmap_.reserve(size); }

  /**
   * @brief Return the maximum possible size of the HashMap, i.e., the number
   * of elements that can be stored in the HashMap.
   *
   */
  size_t bucket_count() const { return hashmap_.bucket_count(); }

  /**
   * @brief Return the load factor of the HashMap.
   *
   */
  float load_factor() const { return hashmap_.load_factor(); }

  /**
   * @brief Check whether the hashmap is empty.
   *
   */
  bool empty() const { return hashmap_.empty(); }

  /**
   * @brief Return the beginning iterator.
   *
   */
  typename ska::flat_hash_map<K, V, H, E>::iterator begin() {
    return hashmap_.begin();
  }

  /**
   * @brief Return the const beginning iterator.
   *
   */
  typename ska::flat_hash_map<K, V, H, E>::const_iterator begin() const {
    return hashmap_.begin();
  }

  /**
   * @brief Return the const beginning iterator.
   *
   */
  typename ska::flat_hash_map<K, V, H, E>::const_iterator cbegin() const {
    return hashmap_.cbegin();
  }

  /**
   * @brief Return the ending iterator
   *
   */
  typename ska::flat_hash_map<K, V, H, E>::iterator end() {
    return hashmap_.end();
  }

  /**
   * @brief Return the const ending iterator.
   *
   */
  typename ska::flat_hash_map<K, V, H, E>::const_iterator end() const {
    return hashmap_.end();
  }

  /**
   * @brief Return the const ending iterator.
   *
   */
  typename ska::flat_hash_map<K, V, H, E>::const_iterator cend() const {
    return hashmap_.cend();
  }

  /**
   * @brief Find the value by key.
   *
   */
  typename ska::flat_hash_map<K, V, H, E>::iterator find(const K& key) {
    return hashmap_.find(key);
  }

  /**
   * @brief Associated with a given data buffer
   */
  void AssociateDataBuffer(std::shared_ptr<Blob> data_buffer) {
    this->data_buffer_ = data_buffer;
  }

  /**
   * @brief Build the hashmap object.
   *
   */
  Status Build(Client& client) override {
    using entry_t = typename Hashmap<K, V, H, E>::Entry;

    // shrink the size of hashmap
    hashmap_.shrink_to_fit();

    size_t entry_size =
        hashmap_.get_num_slots_minus_one() + hashmap_.get_max_lookups() + 1;
    auto entries_builder = std::make_shared<ArrayBuilder<entry_t>>(
        client, hashmap_.get_entries(), entry_size);

    this->set_num_slots_minus_one_(hashmap_.get_num_slots_minus_one());
    this->set_max_lookups_(hashmap_.get_max_lookups());
    this->set_num_elements_(hashmap_.size());
    this->set_entries_(std::static_pointer_cast<ObjectBase>(entries_builder));

    if (this->data_buffer_ != nullptr) {
      this->set_data_buffer_(
          reinterpret_cast<uintptr_t>(this->data_buffer_->data()));
      this->set_data_buffer_mapped_(this->data_buffer_);
    } else {
      this->set_data_buffer_(reinterpret_cast<uintptr_t>(nullptr));
      this->set_data_buffer_mapped_(Blob::MakeEmpty(client));
    }
    return Status::OK();
  }

 private:
  ska::flat_hash_map<K, V, H, E> hashmap_;
  std::shared_ptr<Blob> data_buffer_;
};

template <typename K, typename V>
class PerfectHashmapBuilder : public PerfectHashmapBaseBuilder<K, V> {
 public:
  static_assert(std::is_pod<V>::value, "V in perfect hashmap must be POD type");

  typedef boomphf::SingleHashFunctor<K> hasher_t;

  explicit PerfectHashmapBuilder(Client& client)
      : PerfectHashmapBaseBuilder<K, V>(client) {}

  Status ComputeHash(Client& client, const K* keys, const V* values,
                     const size_t n_elements) {
    std::shared_ptr<Blob> blob;
    RETURN_ON_ERROR(this->allocateKeys(client, keys, n_elements, blob));
    return ComputeHash(client, blob, values, n_elements);
  }

  /**
   * Using existing blobs for keys.
   */
  Status ComputeHash(Client& client, const std::shared_ptr<Blob> keys,
                     const V* values, const size_t n_elements) {
    this->set_num_elements_(n_elements);
    this->set_ph_keys_(keys);
    RETURN_ON_ERROR(detail::boomphf::build_keys(
        bphf_, reinterpret_cast<const K*>(keys->data()), n_elements));
    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return detail::boomphf::build_values(
              bphf_, reinterpret_cast<const K*>(keys->data()), n_elements,
              values, shuffled_values);
        });
  }

  /**
   * Using existing arrow array for keys.
   */
  Status ComputeHash(Client& client,
                     const std::shared_ptr<ArrowVineyardArrayType<K>>& keys,
                     const V* values, const size_t n_elements) {
    this->set_num_elements_(n_elements);
    this->set_ph_keys_(keys);
    RETURN_ON_ERROR(detail::boomphf::build_keys(bphf_, keys->GetArray()));
    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return detail::boomphf::build_values(bphf_, keys->GetArray(), values,
                                               shuffled_values);
        });
    return Status::OK();
  }

  Status ComputeHash(Client& client, const K* keys, const V begin_value,
                     const size_t n_elements) {
    std::shared_ptr<Blob> blob;
    RETURN_ON_ERROR(this->allocateKeys(client, keys, n_elements, blob));
    return ComputeHash(client, blob, begin_value, n_elements);
  }

  /**
   * Using existing blobs for keys.
   */
  Status ComputeHash(Client& client, const std::shared_ptr<Blob> keys,
                     const V begin_value, const size_t n_elements) {
    this->set_num_elements_(n_elements);
    this->set_ph_keys_(keys);
    RETURN_ON_ERROR(detail::boomphf::build_keys(
        bphf_, reinterpret_cast<const K*>(keys->data()), n_elements));
    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return detail::boomphf::build_values(
              bphf_, reinterpret_cast<const K*>(keys->data()), n_elements,
              begin_value, shuffled_values);
        });
  }

  /**
   * Using existing arrow array for keys.
   */
  Status ComputeHash(Client& client,
                     const std::shared_ptr<ArrowVineyardArrayType<K>>& keys,
                     const V begin_value, const size_t n_elements) {
    this->set_num_elements_(n_elements);
    this->set_ph_keys_(keys);
    RETURN_ON_ERROR(detail::boomphf::build_keys(bphf_, keys->GetArray()));
    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return detail::boomphf::build_values(bphf_, keys->GetArray(),
                                               begin_value, shuffled_values);
        });
    return Status::OK();
  }

  size_t size() const { return this->num_elements_; }

  /**
   * @brief Build the hashmap object.
   *
   */
  Status Build(Client& client) override {
    size_t size = detail::boomphf::bphf_serde::compute_size(bphf_);
    std::unique_ptr<BlobWriter> blob_writer;
    RETURN_ON_ERROR(client.CreateBlob(size, blob_writer));
    char* dst = detail::boomphf::bphf_serde::ser(blob_writer->data(), bphf_);
    RETURN_ON_ASSERT(dst == blob_writer->data() + size,
                     "boomphf serialization error: buffer size mismatched");
    std::shared_ptr<Object> blob;
    RETURN_ON_ERROR(blob_writer->Seal(client, blob));
    this->set_ph_(std::dynamic_pointer_cast<Blob>(blob));
    return Status::OK();
  }

 private:
  Status allocateKeys(Client& client, const K* keys, const size_t n_elements,
                      std::shared_ptr<Blob>& out) {
    std::unique_ptr<BlobWriter> blob_writer;
    RETURN_ON_ERROR(client.CreateBlob(n_elements * sizeof(K), blob_writer));
    memcpy(blob_writer->data(), keys, n_elements * sizeof(K));
    std::shared_ptr<Object> blob;
    RETURN_ON_ERROR(blob_writer->Seal(client, blob));
    out = std::dynamic_pointer_cast<Blob>(blob);
    return Status::OK();
  }

  template <typename Func>
  Status allocateValues(Client& client, const size_t n_elements, Func func) {
    std::unique_ptr<BlobWriter> blob_writer;
    RETURN_ON_ERROR(client.CreateBlob(n_elements * sizeof(V), blob_writer));
    V* values = reinterpret_cast<V*>(blob_writer->data());
    RETURN_ON_ERROR(func(values));
    std::shared_ptr<Object> blob;
    RETURN_ON_ERROR(blob_writer->Seal(client, blob));
    this->set_ph_values_(blob);
    return Status::OK();
  }

  boomphf::mphf<K, hasher_t> bphf_;

  const int concurrency_ = std::thread::hardware_concurrency();
  const double gamma_ = 2.5f;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_HASHMAP_H_
