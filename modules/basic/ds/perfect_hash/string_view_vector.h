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

#ifndef MODULES_BASIC_DS_PERFECT_HASH_STRING_VIEW_VECTOR_H_
#define MODULES_BASIC_DS_PERFECT_HASH_STRING_VIEW_VECTOR_H_

#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "basic/ds/perfect_hash/ref_vector.h"
#define nssv_CONFIG_SELECT_STRING_VIEW nssv_STRING_VIEW_NONSTD
#include "string_view/string_view.hpp"

namespace vineyard {
namespace perfect_hash {

class StringViewVector {
 public:
  StringViewVector() { offsets_.push_back(0); }
  ~StringViewVector() {}

  void push_back(const nonstd::string_view& val) {
    size_t old_size = buffer_.size();
    buffer_.resize(old_size + val.size());
    memcpy(&buffer_[old_size], val.data(), val.size());
    offsets_.push_back(buffer_.size());
  }

  void emplace_back(const nonstd::string_view& val) {
    size_t old_size = buffer_.size();
    buffer_.resize(old_size + val.size());
    memcpy(&buffer_[old_size], val.data(), val.size());
    offsets_.push_back(buffer_.size());
  }

  size_t size() const {
    assert(offsets_.size() > 0);
    return offsets_.size() - 1;
  }

  nonstd::string_view operator[](size_t index) const {
    size_t from = offsets_[index];
    size_t len = offsets_[index + 1] - from;
    return nonstd::string_view(&buffer_[from], len);
  }

  std::vector<char>& content_buffer() { return buffer_; }

  const std::vector<char>& content_buffer() const { return buffer_; }

  std::vector<size_t>& offset_buffer() { return offsets_; }

  const std::vector<size_t>& offset_buffer() const { return offsets_; }

  void clear() {
    buffer_.clear();
    offsets_.clear();
    offsets_.push_back(0);
  }

  void swap(StringViewVector& rhs) {
    buffer_.swap(rhs.buffer_);
    offsets_.swap(rhs.offsets_);
  }

 private:
  std::vector<char> buffer_;
  std::vector<size_t> offsets_;
};

template <>
struct ref_vector<nonstd::string_view> {
  ref_vector() {}
  ~ref_vector() {}

  size_t init(const void* buffer, size_t size) {
    size_t buffer_size = buffer_.init(buffer, size);
    const void* ptr = reinterpret_cast<const char*>(buffer) + buffer_size;
    size_t offset_size = offsets_.init(ptr, size - buffer_size);
    return buffer_size + offset_size;
  }

  ref_vector<char>& buffer() { return buffer_; }
  ref_vector<size_t>& offsets() { return offsets_; }

  const ref_vector<char>& buffer() const { return buffer_; }
  const ref_vector<size_t>& offsets() const { return offsets_; }

  size_t size() const {
    if (offsets_.size() == 0) {
      return 0;
    }
    return offsets_.size() - 1;
  }

  nonstd::string_view get(size_t idx) const {
    size_t from = offsets_.get(idx);
    size_t to = offsets_.get(idx + 1);
    return nonstd::string_view(buffer_.data() + from, to - from);
  }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load_ref_vec(buffer_);
    loader.load_ref_vec(offsets_);
  }

 private:
  ref_vector<char> buffer_;
  ref_vector<size_t> offsets_;
};

}  // namespace perfect_hash
}  // namespace vineyard

#endif  // MODULES_BASIC_DS_PERFECT_HASH_STRING_VIEW_VECTOR_H_
