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

#ifndef MODULES_BASIC_DS_PERFECT_HASH_INDEXER_H_
#define MODULES_BASIC_DS_PERFECT_HASH_INDEXER_H_

#include <memory>
#include <utility>
#include <vector>

#include "basic/ds/perfect_hash/hashmap_indexer_impl.h"
#include "basic/ds/perfect_hash/single_phf_view.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "common/memory/memcpy.h"
#include "common/util/arrow.h"

namespace vineyard {
namespace perfect_hash {

using vineyard::Blob;
using vineyard::BlobWriter;
using vineyard::Client;
using vineyard::Object;

template <typename KEY_T, typename INDEX_T>
class PHIdxerView {
 public:
  PHIdxerView() {}
  ~PHIdxerView() {}

  void init(const void* buffer, size_t size) {
    mem_loader loader(reinterpret_cast<const char*>(buffer), size);
    phf_view_.load(loader);
    keys_view_.load(loader);
  }

  size_t entry_num() const { return keys_view_.size(); }

  bool empty() const { return keys_view_.empty(); }

  bool get_key(INDEX_T lid, KEY_T& oid) const {
    if (lid >= keys_view_.size()) {
      return false;
    }
    oid = keys_view_.get(lid);
    return true;
  }

  bool get_index(const KEY_T& oid, INDEX_T& lid) const {
    auto idx = phf_view_(oid);
    if (idx < keys_view_.size() && keys_view_.get(idx) == oid) {
      lid = idx;
      return true;
    }
    return false;
  }

  size_t size() const { return keys_view_.size(); }

 private:
  SinglePHFView<murmurhasher> phf_view_;
  hashmap_indexer_impl::KeyBufferView<KEY_T> keys_view_;
};

template <typename INDEX_T>
class PHIdxerView<arrow_string_view, INDEX_T> {
 public:
  PHIdxerView() {}
  ~PHIdxerView() {}

  void init(const void* buffer, size_t size) {
    mem_loader loader(reinterpret_cast<const char*>(buffer), size);
    phf_view_.load(loader);
    keys_view_.load(loader);
  }

  size_t entry_num() const { return keys_view_.size(); }

  bool empty() const { return keys_view_.size() == 0; }

  bool get_key(INDEX_T lid, nonstd::string_view& oid) const {
    if (lid >= keys_view_.size()) {
      return false;
    }
    oid = keys_view_.get(lid);
    return true;
  }

  bool get_index(const nonstd::string_view& oid, INDEX_T& lid) const {
    auto idx = phf_view_(oid);
    if (idx < keys_view_.size() && keys_view_.get(idx) == oid) {
      lid = idx;
      return true;
    }
    return false;
  }

  bool get_index(const arrow_string_view& oid, INDEX_T& lid) const {
    nonstd::string_view oid_view(oid.data(), oid.size());
    return get_index(oid_view, lid);
  }

  size_t size() const { return keys_view_.size(); }

 private:
  SinglePHFView<murmurhasher> phf_view_;
  hashmap_indexer_impl::KeyBuffer<nonstd::string_view> keys_view_;
};

template <typename KEY_T, typename INDEX_T>
class ImmPHIdxer {
 public:
  void Init(std::shared_ptr<Blob> buf) {
    buffer_ = buf;
    idxer_.init(buffer_->data(), buffer_->size());
  }

  size_t entry_num() const { return idxer_.entry_num(); }

  bool empty() const { return idxer_.empty(); }

  bool get_key(INDEX_T lid, KEY_T& oid) const {
    return idxer_.get_key(lid, oid);
  }

  bool get_index(const KEY_T& oid, INDEX_T& lid) const {
    return idxer_.get_index(oid, lid);
  }

  size_t size() const { return idxer_.size(); }

  const std::shared_ptr<Blob>& buffer() const { return buffer_; }

 private:
  std::shared_ptr<Blob> buffer_;
  PHIdxerView<KEY_T, INDEX_T> idxer_;
};

template <typename KEY_T, typename INDEX_T>
class PHIdxerViewBuilder {
 public:
  PHIdxerViewBuilder() = default;
  ~PHIdxerViewBuilder() = default;

  void add(const KEY_T& oid) { keys_.push_back(oid); }

  void add(KEY_T&& oid) { keys_.push_back(std::move(oid)); }

  Status Finish(Client& client, ImmPHIdxer<KEY_T, INDEX_T>& idxer) {
    pthash::single_phf<murmurhasher, pthash::dictionary_dictionary, true> phf;
    size_t serialize_size = 0;

    SinglePHFView<murmurhasher>::build(keys_.begin(), keys_.size(), phf, 1);
    std::unique_ptr<BlobWriter> writer;

    hashmap_indexer_impl::KeyBuffer<KEY_T> key_buffer;
    std::vector<KEY_T> ordered_keys(keys_.size());
    for (auto& key : keys_) {
      size_t idx = phf(key);
      ordered_keys[idx] = key;
    }
    for (auto& key : ordered_keys) {
      key_buffer.push_back(key);
    }
    /**
     * Because the num_bits function uses the essentials::vec_bytes to calculate
     * the size of the buffer. So the returned value is the same as the size of
     * the dumped buffer.
     */
    serialize_size += phf.num_bits() / 8;
    serialize_size += key_buffer.dump_size();

    RETURN_ON_ERROR(client.CreateBlob(serialize_size, writer));
    external_mem_dumper dumper(writer->data(), serialize_size);
    phf.dump(dumper);
    key_buffer.dump(dumper);

    std::shared_ptr<Object> buf;
    RETURN_ON_ERROR(writer->Seal(client, buf));
    idxer.Init(std::dynamic_pointer_cast<Blob>(buf));
    return Status::OK();
  }

 private:
  std::vector<KEY_T> keys_;
};

template <typename INDEX_T>
class PHIdxerViewBuilder<arrow_string_view, INDEX_T> {
 public:
  PHIdxerViewBuilder() = default;
  ~PHIdxerViewBuilder() = default;

  void add(const arrow_string_view& oid) {
    nonstd::string_view oid_view(oid.data(), oid.size());
    keys_.push_back(oid_view);
  }

  void add(arrow_string_view&& oid) {
    nonstd::string_view oid_view(oid.data(), oid.size());
    keys_.push_back(std::move(oid_view));
  }

  Status Finish(Client& client, ImmPHIdxer<arrow_string_view, INDEX_T>& idxer) {
    // mem_dumper dumper;
    pthash::single_phf<murmurhasher, pthash::dictionary_dictionary, true> phf;
    size_t serialize_size = 0;

    SinglePHFView<murmurhasher>::build(keys_.begin(), keys_.size(), phf, 1);
    std::unique_ptr<BlobWriter> writer;

    hashmap_indexer_impl::KeyBuffer<nonstd::string_view> key_buffer;
    std::vector<nonstd::string_view> ordered_keys(keys_.size());
    for (auto& key : keys_) {
      size_t idx = phf(key);
      ordered_keys[idx] = key;
    }
    for (auto& key : ordered_keys) {
      key_buffer.push_back(key);
    }
    serialize_size += phf.num_bits() / 8;
    serialize_size += key_buffer.dump_size();

    RETURN_ON_ERROR(client.CreateBlob(serialize_size, writer));

    external_mem_dumper dumper(writer->data(), serialize_size);
    phf.dump(dumper);
    key_buffer.dump(dumper);

    std::shared_ptr<Object> buf;
    RETURN_ON_ERROR(writer->Seal(client, buf));
    idxer.Init(std::dynamic_pointer_cast<Blob>(buf));
    return Status::OK();
  }

 private:
  std::vector<nonstd::string_view> keys_;
};

}  // namespace perfect_hash
}  // namespace vineyard

#endif  // MODULES_BASIC_DS_PERFECT_HASH_INDEXER_H_
