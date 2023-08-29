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

#ifndef MODULES_BASIC_DS_ARRAY_H_
#define MODULES_BASIC_DS_ARRAY_H_

#include <memory>
#include <utility>
#include <vector>

#include "basic/ds/array.vineyard.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"

namespace vineyard {

/**
 * @brief ArrayBuilder is designed for constructing arrays that supported by
 * vineyard.
 *
 * @tparam T The type for the elements.
 */
template <typename T>
class ArrayBuilder : public ArrayBaseBuilder<T> {
 public:
  ArrayBuilder(Client& client, size_t size)
      : ArrayBaseBuilder<T>(client), client_(client), size_(size) {
    VINEYARD_CHECK_OK(client.CreateBlob(size_ * sizeof(T), buffer_writer_));
    this->data_ = reinterpret_cast<T*>(buffer_writer_->data());
  }

  /**
   * @brief Initialize the ArrayBuilder from an existing std::vector of type T.
   *
   * @param client The client connected to the vineyard server.
   * @param vec The local std::vector of type T.
   */
  ArrayBuilder(Client& client, std::vector<T> const& vec)
      : ArrayBuilder(client, vec.size()) {
    memcpy(data_, vec.data(), size_ * sizeof(T));
  }

  /**
   * @brief Initialize the ArrayBuilder from an existing C array of type T.
   *
   * @param client The client connected to the vineyard server.
   * @param data The pointer to the array.
   * @param size The size of the array.
   */
  ArrayBuilder(Client& client, const T* data, size_t size)
      : ArrayBuilder(client, size) {
    memcpy(data_, data, size_ * sizeof(T));
  }

  ~ArrayBuilder() {
    if (!this->sealed() && buffer_writer_ != nullptr) {
      VINEYARD_DISCARD(buffer_writer_->Abort(client_));
    }
  }

  /**
   * @brief Get the size of the array, i.e., number of elements in the array.
   *
   * @return The size.
   */
  size_t const size() const { return size_; }

  /**
   * @brief Get the element located in the given index of the array.
   *
   * @param idx The give index.
   * @return The element at the given index.
   */
  T& operator[](size_t idx) { return data_[idx]; }

  /**
   * @brief Get the data pointer to the array.
   *
   * @return The data pointer.
   */
  T* data() noexcept { return data_; }

  /**
   * @brief Get the const data pointer to the array.
   *
   * @return The const data pointer.
   */
  const T* data() const noexcept { return data_; }

  /**
   * @brief Build the array object.
   *
   * @param client The client connected to the vineyard server.
   */
  Status Build(Client& client) override {
    this->set_size_(size_);
    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(buffer_writer_)));
    return Status::OK();
  }

 private:
  Client& client_;
  std::unique_ptr<BlobWriter> buffer_writer_;
  T* data_;
  size_t size_;
};

/**
 * @brief ResizableArrayBuilder is used for building resizable arrays.
 *
 * @tparam T The type for the elements.
 */
template <typename T>
class ResizableArrayBuilder : public ArrayBaseBuilder<T> {
 public:
  explicit ResizableArrayBuilder(Client& client, size_t size = 0)
      : ArrayBaseBuilder<T>(client), vec_(size) {}

  T& operator[](size_t idx) { return vec_[idx]; }

  void push_back(T const& v) { vec_.push_back(v); }

  void push_back(T&& v) { vec_.push_back(std::move(v)); }

  template <class... Args>
  void emplace_back(Args&&... args) {
    vec_.emplace_back(std::forward<Args>(args)...);
  }

  size_t const size() const { return vec_.size(); }
  void reserve(size_t size) { vec_.reserve(size); }
  void resize(size_t size) { vec_.resize(size); }
  void resize(size_t size, T const& value) { vec_.resize(size, value); }

  bool empty() const { return vec_.empty(); }

  void shrink_to_fit() { return vec_.shrink_to_fit(); }

  // FIXME: exposing the internal vec outside the builder is bad,
  // but we need to manipulate a vector builder in sync_comm.h
  std::vector<T>& payload() { return vec_; }

  Status Build(Client& client) override {
    std::unique_ptr<BlobWriter> buffer_writer;
    RETURN_ON_ERROR(client.CreateBlob(vec_.size() * sizeof(T), buffer_writer));
    memcpy(buffer_writer->data(), vec_.data(), vec_.size() * sizeof(T));

    this->set_size_(vec_.size());
    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
    return Status::OK();
  }

 private:
  std::vector<T> vec_;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_ARRAY_H_
