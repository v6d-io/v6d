/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#ifndef MODULES_BASIC_DS_TUPLE_H_
#define MODULES_BASIC_DS_TUPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic/ds/tuple.vineyard.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"
#include "common/util/uuid.h"

namespace vineyard {

/**
 * @brief TupleBuilder is designed for generating tuples
 *
 */
class TupleBuilder : public TupleBaseBuilder {
 public:
  explicit TupleBuilder(Client& client) : TupleBaseBuilder(client) {
    this->set_size_(0);
  }

  /**
   * @brief Initialize the TupleBuilder with a given size.
   *
   * @param client The client connected to the vineyard server.
   * @param size The size of the tuple to build.
   */
  explicit TupleBuilder(Client& client, size_t const size)
      : TupleBaseBuilder(client) {
    this->set_size_(size);
  }

  /**
   * @brief Get the size of the tuple, i.e., the number of elements it contains.
   *
   * @return The size of the tuple.
   */
  size_t const Size() const { return this->size_; }

  /**
   * @brief Set the size for the tuple.
   * Note that the size of a tuple can be set only once.
   *
   * @param size The size for the tuple.
   */
  void SetSize(size_t size) {
    if (this->size_ > 0) {
      LOG(ERROR) << "The size of a tuple cannot set for multiple times";
    } else {
      this->set_size_(size);
    }
  }

  /**
   * @brief Get the builder at the given index.
   * Here the index is bound-checked.
   *
   * @param index The given index.
   * @return The builder at the given index.
   */
  std::shared_ptr<ObjectBuilder> At(size_t index) {
    if (index >= size_) {
      LOG(ERROR) << "tuple builder::at(): out of range: " << index;
      return nullptr;
    }
    return std::dynamic_pointer_cast<ObjectBuilder>(this->elements_[index]);
  }

  /**
   * @brief Set the builder for the value at the given index.
   * When building the tuple, the builder will be invoked to
   * build the value.
   *
   * @param idx The index of the value.
   * @param value The builder for the value for the given index.
   */
  void SetValue(size_t idx, std::shared_ptr<ObjectBuilder> const& value) {
    if (idx >= size_) {
      LOG(ERROR) << "tuple::set(): out of range";
    } else {
      this->set_elements_(idx, value);
    }
  }

  /**
   * @brief Set the builder for the value at the given index.
   * When building the tuple, the builder will be invoked to
   * build the value.
   *
   * @param idx The index of the value.
   * @param value The value for the given index.
   */
  void SetValue(size_t idx, std::shared_ptr<Object> const& value) {
    if (idx >= size_) {
      LOG(ERROR) << "tuple::set(): out of range";
    } else {
      this->set_elements_(idx, value);
    }
  }
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_TUPLE_H_
