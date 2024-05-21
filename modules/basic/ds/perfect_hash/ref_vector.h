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

#ifndef MODULES_BASIC_DS_PERFECT_HASH_REF_VECTOR_H_
#define MODULES_BASIC_DS_PERFECT_HASH_REF_VECTOR_H_

#include <vector>

namespace vineyard {
namespace perfect_hash {

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

}  // namespace perfect_hash
}  // namespace vineyard

#endif  // MODULES_BASIC_DS_PERFECT_HASH_REF_VECTOR_H_
