/** Copyright 2020 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRAPE_UTILS_REF_VECTOR_H_
#define GRAPE_UTILS_REF_VECTOR_H_

#include <vector>

namespace grape {

template <typename T>
struct ref_vector {
  static_assert(std::is_pod<T>::value, "T must be POD type");
  ref_vector() : buffer_(nullptr), size_(0) {}
  ~ref_vector() {}

  size_t init(const void* buffer, size_t size) {
    const void* ptr = buffer;
    size_ = *reinterpret_cast<const size_t*>(ptr);
    ptr = reinterpret_cast<const char*>(ptr) + sizeof(size_t);
    buffer_ = reinterpret_cast<const T*>(ptr);
    return size_ * sizeof(T) + sizeof(size_t);
  }

  size_t size() const { return size_; }

  T get(size_t idx) const { return buffer_[idx]; }

  const T* data() const { return buffer_; }

  const T& operator[](size_t idx) const { return buffer_[idx]; }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load_ref_vec(*this);
  }

 private:
  const T* buffer_;
  size_t size_;
};

template <typename T, typename ALLOC_T>
void encode_vec(const std::vector<T, ALLOC_T>& vec, std::vector<char>& buf) {
  size_t old_size = buf.size();
  size_t vec_size = vec.size();
  buf.resize(old_size + sizeof(size_t) + vec_size * sizeof(T));
  char* ptr = buf.data() + old_size;
  memcpy(ptr, &vec_size, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(ptr, vec.data(), sizeof(T) * vec_size);
}

template <typename T>
void encode_val(const T& val, std::vector<char>& buf) {
  size_t old_size = buf.size();
  buf.resize(old_size + sizeof(T));
  char* ptr = buf.data() + old_size;
  memcpy(ptr, &val, sizeof(T));
}

template <typename T>
const char* decode_val(T& val, const char* buf) {
  memcpy(&val, buf, sizeof(T));
  return buf + sizeof(T);
}

}  // namespace grape

#endif  // GRAPE_UTILS_REF_VECTOR_H_