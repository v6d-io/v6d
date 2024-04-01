/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef MODULES_BASIC_DS_PERFECT_HASH_HASHMAP_INDEXER_IMPL_H_
#define MODULES_BASIC_DS_PERFECT_HASH_HASHMAP_INDEXER_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "basic/ds/perfect_hash/ref_vector.h"
#include "basic/ds/perfect_hash/string_view_vector.h"
#include "pthash/essentials/essentials.hpp"

namespace vineyard {
namespace perfect_hash {

namespace hashmap_indexer_impl {

static constexpr int8_t min_lookups = 4;
static constexpr double max_load_factor = 0.5f;

template <typename T>
struct KeyBuffer {
 public:
  KeyBuffer() = default;
  ~KeyBuffer() = default;

  const T& get(size_t idx) const { return inner_[idx]; }
  void set(size_t idx, const T& val) { inner_[idx] = val; }

  void push_back(const T& val) { inner_.push_back(val); }

  size_t size() const { return inner_.size(); }

  std::vector<T>& buffer() { return inner_; }
  const std::vector<T>& buffer() const { return inner_; }

  void swap(KeyBuffer& rhs) { inner_.swap(rhs.inner_); }

  void clear() { inner_.clear(); }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load_vec(inner_);
  }

  template <typename Dumper>
  void dump(Dumper& dumper) const {
    dumper.dump_vec(inner_);
  }

  size_t dump_size() { return essentials::vec_bytes(inner_); }

 private:
  std::vector<T> inner_;
};

template <>
struct KeyBuffer<nonstd::string_view> {
  KeyBuffer() = default;
  ~KeyBuffer() = default;

  nonstd::string_view get(size_t idx) const { return inner_[idx]; }

  void push_back(const nonstd::string_view& val) { inner_.push_back(val); }

  size_t size() const { return inner_.size(); }

  StringViewVector& buffer() { return inner_; }
  const StringViewVector& buffer() const { return inner_; }

  void swap(KeyBuffer& rhs) { inner_.swap(rhs.inner_); }

  void clear() { inner_.clear(); }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load_vec(inner_.content_buffer());
    loader.load_vec(inner_.offset_buffer());
  }

  template <typename Dumper>
  void dump(Dumper& dumper) const {
    dumper.dump_vec(inner_.content_buffer());
    dumper.dump_vec(inner_.offset_buffer());
  }

  size_t dump_size() {
    return sizeof(size_t) + inner_.content_buffer().size() + sizeof(size_t) +
           inner_.offset_buffer().size() * sizeof(size_t);
  }

 private:
  StringViewVector inner_;
};

template <typename T>
struct KeyBufferView {
 public:
  KeyBufferView() {}

  size_t init(const void* buffer, size_t size) {
    return inner_.init(buffer, size);
  }

  T get(size_t idx) const { return inner_.get(idx); }

  size_t size() const { return inner_.size(); }

  template <typename Loader>
  void load(Loader& loader) {
    inner_.load(loader);
  }

 private:
  ref_vector<T> inner_;
};

}  // namespace hashmap_indexer_impl

}  // namespace perfect_hash
}  // namespace vineyard

#endif  // MODULES_BASIC_DS_PERFECT_HASH_HASHMAP_INDEXER_IMPL_H_
