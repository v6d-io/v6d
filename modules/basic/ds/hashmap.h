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
#include <utility>

#include "flat_hash_map/flat_hash_map.hpp"
#include "wyhash/wyhash.hpp"  // IWYU pragma: keep

#include "basic/ds/array.h"
#include "basic/ds/hashmap.vineyard.h"
#include "basic/ds/perfect_hash_indexer.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/arrow.h"  // IWYU pragma: keep

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

// GRAPE PERFECT HASH
template <typename K, typename V>
class GrapePerfectHashMapBuilder;
using detail::boomphf::arrow_array_iterator;

template <typename K, typename V>
class GrapePerfectHashMapBuilder;
using detail::boomphf::arrow_array_iterator;

template <typename K, typename V>
class GrapePerfectHashMap : public Registered<GrapePerfectHashMap<K, V>> {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<GrapePerfectHashMap>(new GrapePerfectHashMap<K, V>()));
  }

  void Construct(const ObjectMeta& meta) override {
    Object::Construct(meta);

    std::string typeName = type_name<GrapePerfectHashMap<K, V>>();

    VINEYARD_ASSERT(typeName == meta.GetTypeName(),
                    "Type dismatch, expect " + typeName + ", but got " +
                        meta.GetTypeName());
    this->v_buffer_ =
        std::dynamic_pointer_cast<Blob>(meta.GetMember("v_buffer_"));
    this->hash_data_ =
        std::dynamic_pointer_cast<Blob>(meta.GetMember("hash_data_"));
    this->n_elements = meta.GetKeyValue<size_t>("n_elements");

    this->idxer_.Init(this->hash_data_);
  }

  size_t size() const { return this->n_elements; }

  const V& at(const K& key) const {
    uint64_t value_index = 0;
    this->idxer_.get_index(key, value_index);
    return reinterpret_cast<const V*>(this->v_buffer_->data())[value_index];
  }

 private:
  grape::ImmPHIdxer<K, uint64_t> idxer_;
  std::shared_ptr<Blob> v_buffer_;
  std::shared_ptr<Blob> hash_data_;
  size_t n_elements;

  friend class Client;
  friend class GrapePerfectHashMapBuilder<K, V>;
};

template <typename V>
class GrapePerfectHashMap<std::string_view, V>
    : public Registered<GrapePerfectHashMap<std::string_view, V>> {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<GrapePerfectHashMap>(
            new GrapePerfectHashMap<std::string_view, V>()));
  }

  void Construct(const ObjectMeta& meta) override {
    Object::Construct(meta);

    std::string typeName = type_name<GrapePerfectHashMap<std::string_view, V>>();

    VINEYARD_ASSERT(typeName == meta.GetTypeName(),
                    "Type dismatch, expect " + typeName + ", but got " +
                        meta.GetTypeName());
    this->v_buffer_ =
        std::dynamic_pointer_cast<Blob>(meta.GetMember("v_buffer_"));
    this->hash_data_ =
        std::dynamic_pointer_cast<Blob>(meta.GetMember("hash_data_"));
    this->n_elements = meta.GetKeyValue<size_t>("n_elements");

    this->idxer_.Init(this->hash_data_);
  }

  size_t size() const { return this->n_elements; }

  const V& at(const std::string_view& key) const {
    uint64_t value_index = 0;
    nonstd::string_view key_view(key.data(), key.size());
    this->idxer_.get_index(key_view, value_index);
    return reinterpret_cast<const V*>(this->v_buffer_->data())[value_index];
  }

 private:
  grape::ImmPHIdxer<nonstd::string_view, uint64_t> idxer_;
  std::shared_ptr<Blob> v_buffer_;
  std::shared_ptr<Blob> hash_data_;
  size_t n_elements;

  friend class Client;
  friend class GrapePerfectHashMapBuilder<std::string_view, V>;
};

template <typename K, typename V>
class GrapePerfectHashMapBuilder : public ObjectBuilder {
 public:
  explicit GrapePerfectHashMapBuilder(Client& client) {}

  explicit GrapePerfectHashMapBuilder(
      GrapePerfectHashMap<K, V> const& __value) {
    VINEYARD_ASSERT(false, "Not implemented yet");
  }

  explicit GrapePerfectHashMapBuilder(
      std::shared_ptr<GrapePerfectHashMap<K, V>> const& __value)
      : GrapePerfectHashMapBuilder(*__value) {}

  Status _Seal(Client& client, std::shared_ptr<Object>& object) override {
    // prepare member
    std::shared_ptr<GrapePerfectHashMap<K, V>> value =
        std::make_shared<GrapePerfectHashMap<K, V>>();
    object = value;
    this->hash_data_ = this->idxer_.buffer();
    value->meta_.AddMember("v_buffer_", this->v_buffer_);
    value->meta_.AddMember("hash_data_", this->hash_data_);
    value->meta_.AddKeyValue("n_elements", this->n_elements);
    value->meta_.SetTypeName(type_name<GrapePerfectHashMap<K, V>>());
    value->idxer_ = this->idxer_;
    value->n_elements = this->n_elements;
    value->v_buffer_ = std::dynamic_pointer_cast<Blob>(this->v_buffer_);
    value->hash_data_ = std::dynamic_pointer_cast<Blob>(this->hash_data_);

    VINEYARD_CHECK_OK(client.CreateMetaData(value->meta_, value->id_));
    this->set_sealed(true);

    return Status::OK();
  }

  Status Build(Client& client) override {
    // TBD
    return Status::OK();
  }

  Status ComputeHash(Client& client, const K* keys, const V* values,
                     const size_t n_elements) {
    std::shared_ptr<Blob> blob;
    RETURN_ON_ERROR(this->allocateKeys(client, keys, n_elements, blob));
    return ComputeHash(client, blob, values, n_elements);
  }

  Status ComputeHash(Client& client, const std::shared_ptr<Blob> keys,
                     const V* values, const size_t n_elements) {
    this->n_elements = n_elements;

    // do create hash(build key)
    for (size_t i = 0; i < n_elements; ++i) {
      this->builder_.add((reinterpret_cast<const K*>(keys->data()))[i]);
    }
    this->idxer_ = this->builder_.finish(client);

    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return this->build_values(reinterpret_cast<const K*>(keys->data()),
                                    n_elements, values, shuffled_values);
        });
  }

  Status ComputeHash(Client& client,
                     const std::shared_ptr<ArrowVineyardArrayType<K>>& keys,
                     const V begin_value, const size_t n_elements) {
    this->n_elements = n_elements;

    for (auto iter = arrow_array_iterator<K, ArrowArrayType<K>>(
             keys->GetArray()->begin());
         iter !=
         arrow_array_iterator<K, ArrowArrayType<K>>(keys->GetArray()->end());
         iter++) {
      this->builder_.add(*iter);
    }
    this->idxer_ = this->builder_.finish(client);

    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return this->build_values(keys->GetArray(), begin_value,
                                    shuffled_values);
        });
  }

  Status ComputeHash(Client& client,
                     const std::shared_ptr<ArrowVineyardArrayType<K>>& keys,
                     const V* values, const size_t n_elements) {
    this->n_elements = n_elements;

    for (auto iter = arrow_array_iterator<K, ArrowArrayType<K>>(
             keys->GetArray()->begin());
         iter !=
         arrow_array_iterator<K, ArrowArrayType<K>>(keys->GetArray()->end());
         iter++) {
      this->builder_.add(*iter);
    }
    this->idxer_ = this->builder_.finish(client);

    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return this->build_values(keys->GetArray(), values,
                                    shuffled_values);
        });
  }

  Status build_values(
      const K* keys, const size_t n_elements, const V* begin_value, V* values,
      const size_t concurrency = std::thread::hardware_concurrency()) {
    RETURN_ON_ASSERT(std::is_integral<K>::value, "K must be integral type.");
    parallel_for(
        static_cast<size_t>(0), n_elements,
        [&](const size_t index) {
          uint64_t v_index_ = 0;
          this->idxer_.get_index(keys[index], v_index_);
          values[v_index_] = begin_value[index];
        },
        concurrency);
    return Status::OK();
  }

  Status build_values(
      const std::shared_ptr<ArrowArrayType<K>>& keys, const V begin_value,
      V* values,
      const size_t concurrency = std::thread::hardware_concurrency()) {
    parallel_for(
        static_cast<size_t>(0), static_cast<size_t>(keys->length()),
        [&](const size_t index) {
          uint64_t v_index_ = 0;
          this->idxer_.get_index(keys->GetView(index), v_index_);
          values[v_index_] = begin_value + index;
        },
        concurrency);
    return Status::OK();
  }

  Status build_values(
      const std::shared_ptr<ArrowArrayType<K>>& keys, const V* begin_value,
      V* values,
      const size_t concurrency = std::thread::hardware_concurrency()) {
    parallel_for(
        static_cast<size_t>(0), static_cast<size_t>(keys->length()),
        [&](const size_t index) {
          uint64_t v_index_ = 0;
          this->idxer_.get_index(keys->GetView(index), v_index_);
          values[v_index_] = begin_value[index];
        },
        concurrency);
    return Status::OK();
  }

  size_t size() const { return this->n_elements; }

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
    RETURN_ON_ERROR(blob_writer->Seal(client, v_buffer_));
    return Status::OK();
  }

  grape::PHIdxerViewBuilder<K, uint64_t> builder_;
  grape::ImmPHIdxer<K, uint64_t> idxer_;
  std::shared_ptr<Object> v_buffer_;
  std::shared_ptr<Object> hash_data_;
  size_t n_elements;
};

template <typename V>
class GrapePerfectHashMapBuilder<std::string_view, V> : public ObjectBuilder {
 public:
  explicit GrapePerfectHashMapBuilder(Client& client) {}

  explicit GrapePerfectHashMapBuilder(
      GrapePerfectHashMap<std::string_view, V> const& __value) {
    VINEYARD_ASSERT(false, "Not implemented yet");
  }

  explicit GrapePerfectHashMapBuilder(
      std::shared_ptr<GrapePerfectHashMap<std::string_view, V>> const& __value)
      : GrapePerfectHashMapBuilder(*__value) {}
  // auto io_adaptor = vineyard::IOFactory::CreateIOAdaptor(expanded);

  Status _Seal(Client& client, std::shared_ptr<Object>& object) override {
    // prepare member
    std::shared_ptr<GrapePerfectHashMap<std::string_view, V>> value =
        std::make_shared<GrapePerfectHashMap<std::string_view, V>>();
    object = value;
    this->hash_data_ = this->idxer_.buffer();
    value->meta_.AddMember("v_buffer_", this->v_buffer_);
    value->meta_.AddMember("hash_data_", this->hash_data_);
    value->meta_.AddKeyValue("n_elements", this->n_elements);
    value->meta_.SetTypeName(type_name<GrapePerfectHashMap<std::string_view, V>>());
    value->idxer_ = this->idxer_;
    value->n_elements = this->n_elements;
    value->v_buffer_ = std::dynamic_pointer_cast<Blob>(this->v_buffer_);
    value->hash_data_ = std::dynamic_pointer_cast<Blob>(this->hash_data_);

    VINEYARD_CHECK_OK(client.CreateMetaData(value->meta_, value->id_));
    this->set_sealed(true);

    return Status::OK();
  }

  Status Build(Client& client) override {
    // TBD
    return Status::OK();
  }

  Status ComputeHash(
      Client& client,
      const std::shared_ptr<ArrowVineyardArrayType<std::string_view>>& keys,
      const V begin_value, const size_t n_elements) {
    this->n_elements = n_elements;

    for (auto iter = arrow_array_iterator<std::string_view,
                                          ArrowArrayType<std::string_view>>(
             keys->GetArray()->begin());
         iter != arrow_array_iterator<std::string_view,
                                      ArrowArrayType<std::string_view>>(
                     keys->GetArray()->end());
         iter++) {
      nonstd::string_view key_view((*iter).data(), (*iter).size());
      this->builder_.add(key_view);
    }
    this->idxer_ = this->builder_.finish(client);

    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return this->build_values(keys->GetArray(), begin_value,
                                    shuffled_values);
        });
  }

  Status ComputeHash(
      Client& client,
      const std::shared_ptr<ArrowVineyardArrayType<std::string_view>>& keys,
      const V* begin_value, const size_t n_elements) {
    this->n_elements = n_elements;

    for (auto iter = arrow_array_iterator<std::string_view,
                                          ArrowArrayType<std::string_view>>(
             keys->GetArray()->begin());
         iter != arrow_array_iterator<std::string_view,
                                      ArrowArrayType<std::string_view>>(
                     keys->GetArray()->end());
         iter++) {
      nonstd::string_view key_view((*iter).data(), (*iter).size());
      this->builder_.add(key_view);
    }
    this->idxer_ = this->builder_.finish(client);

    return this->allocateValues(
        client, n_elements, [&](V* shuffled_values) -> Status {
          return this->build_values(keys->GetArray(), begin_value,
                                    shuffled_values);
        });
  }

  Status build_values(
      const std::shared_ptr<ArrowArrayType<std::string_view>>& keys,
      const V begin_value, V* values,
      const size_t concurrency = std::thread::hardware_concurrency()) {
    parallel_for(
        static_cast<size_t>(0), static_cast<size_t>(keys->length()),
        [&](const size_t index) {
          uint64_t v_index_ = 0;
          nonstd::string_view key_view(keys->GetView(index).data(),
                                       keys->GetView(index).size());
          this->idxer_.get_index(key_view, v_index_);
          values[v_index_] = begin_value + index;
        },
        concurrency);
    return Status::OK();
  }

  Status build_values(
      const std::shared_ptr<ArrowArrayType<std::string_view>>& keys,
      const V* begin_value, V* values,
      const size_t concurrency = std::thread::hardware_concurrency()) {
    parallel_for(
        static_cast<size_t>(0), static_cast<size_t>(keys->length()),
        [&](const size_t index) {
          uint64_t v_index_ = 0;
          nonstd::string_view key_view(keys->GetView(index).data(),
                                       keys->GetView(index).size());
          this->idxer_.get_index(key_view, v_index_);
          values[v_index_] = begin_value[index];
        },
        concurrency);
    return Status::OK();
  }

  size_t size() const { return this->n_elements; }

 private:
  template <typename Func>
  Status allocateValues(Client& client, const size_t n_elements, Func func) {
    std::unique_ptr<BlobWriter> blob_writer;
    RETURN_ON_ERROR(client.CreateBlob(n_elements * sizeof(V), blob_writer));
    V* values = reinterpret_cast<V*>(blob_writer->data());
    RETURN_ON_ERROR(func(values));
    RETURN_ON_ERROR(blob_writer->Seal(client, v_buffer_));
    return Status::OK();
  }

  grape::PHIdxerViewBuilder<nonstd::string_view, uint64_t> builder_;
  grape::ImmPHIdxer<nonstd::string_view, uint64_t> idxer_;
  std::shared_ptr<Object> v_buffer_;
  std::shared_ptr<Object> hash_data_;
  size_t n_elements;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_HASHMAP_H_
