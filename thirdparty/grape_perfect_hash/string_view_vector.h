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

#ifndef GRAPE_UTILS_STRING_VIEW_VECTOR_H_
#define GRAPE_UTILS_STRING_VIEW_VECTOR_H_

#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <memory>
#include <vector>

#include "ref_vector.h"
#define nssv_CONFIG_SELECT_STRING_VIEW nssv_STRING_VIEW_NONSTD
#include "string_view/string_view.hpp"

namespace grape {

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

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) const {
    size_t content_buffer_size = content_buffer().size();
    CHECK(writer->Write(&content_buffer_size, sizeof(size_t)));
    if (content_buffer_size > 0) {
      CHECK(writer->Write(const_cast<char*>(content_buffer().data()),
                          content_buffer_size * sizeof(char)));
    }
    size_t offset_buffer_size = offset_buffer().size();
    CHECK(writer->Write(&offset_buffer_size, sizeof(size_t)));
    if (offset_buffer_size > 0) {
      CHECK(writer->Write(const_cast<size_t*>(offset_buffer().data()),
                          offset_buffer_size * sizeof(size_t)));
    }
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    size_t content_buffer_size;
    CHECK(reader->Read(&content_buffer_size, sizeof(size_t)));
    if (content_buffer_size > 0) {
      content_buffer().resize(content_buffer_size);
      CHECK(reader->Read(content_buffer().data(),
                         content_buffer_size * sizeof(char)));
    }
    size_t offset_buffer_size;
    CHECK(reader->Read(&offset_buffer_size, sizeof(size_t)));
    if (offset_buffer_size > 0) {
      offset_buffer().resize(offset_buffer_size);
      CHECK(reader->Read(offset_buffer().data(),
                         offset_buffer_size * sizeof(size_t)));
    }
  }

  void serialize_to_mem(std::vector<char>& buf) const {
    encode_vec(buffer_, buf);
    encode_vec(offsets_, buf);
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

}  // namespace grape

#endif  // GRAPE_UTILS_STRING_VIEW_VECTOR_H_
