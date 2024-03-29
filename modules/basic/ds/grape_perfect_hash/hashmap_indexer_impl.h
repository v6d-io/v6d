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

#ifndef MODULES_BASIC_DS_GRAPE_PERFECT_HASH_HASHMAP_INDEXER_IMPL_H_
#define MODULES_BASIC_DS_GRAPE_PERFECT_HASH_HASHMAP_INDEXER_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "basic/ds/grape_perfect_hash/config.h"
#include "basic/ds/grape_perfect_hash/ref_vector.h"
#include "basic/ds/grape_perfect_hash/string_view_vector.h"

namespace grape_perfect_hash {

namespace hashmap_indexer_impl {

static constexpr int8_t min_lookups = 4;
static constexpr double max_load_factor = 0.5f;

inline int8_t log2(size_t value) {
  static constexpr int8_t table[64] = {
      63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48, 27, 54, 33, 42, 3,
      61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4,
      62, 57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21,
      56, 45, 25, 31, 35, 16, 9,  12, 44, 24, 15, 8,  23, 7,  6,  5};
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value |= value >> 32;
  return table[((value - (value >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
}

template <typename T>
struct KeyBuffer {
 public:
  KeyBuffer() = default;
  ~KeyBuffer() = default;

  const T& get(size_t idx) const { return inner_[idx]; }
  void set(size_t idx, const T& val) { inner_[idx] = val; }

  void push_back(const T& val) { inner_.push_back(val); }

  size_t size() const { return inner_.size(); }

  std::vector<T, Allocator<T>>& buffer() { return inner_; }
  const std::vector<T, Allocator<T>>& buffer() const { return inner_; }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) const {
    size_t size = inner_.size();
    CHECK(writer->Write(&size, sizeof(size_t)));
    if (size > 0) {
      CHECK(writer->Write(const_cast<T*>(inner_.data()), size * sizeof(T)));
    }
  }

  void serialize_to_mem(std::vector<char>& buf) const {
    encode_vec(inner_, buf);
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    size_t size;
    CHECK(reader->Read(&size, sizeof(size_t)));
    if (size > 0) {
      inner_.resize(size);
      CHECK(reader->Read(inner_.data(), size * sizeof(T)));
    }
  }

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

 private:
  std::vector<T, Allocator<T>> inner_;
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

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) const {
    inner_.serialize(writer);
  }

  void serialize_to_mem(std::vector<char>& buf) const {
    inner_.serialize_to_mem(buf);
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    inner_.deserialize(reader);
  }

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

}  // namespace grape_perfect_hash

#endif  // MODULES_BASIC_DS_GRAPE_PERFECT_HASH_HASHMAP_INDEXER_IMPL_H_
