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

#ifndef MODULES_BASIC_DS_PAIR_H_
#define MODULES_BASIC_DS_PAIR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic/ds/pair.vineyard.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

/**
 * @brief PairBuilder is designed for building pairs
 *
 */
class PairBuilder : public PairBaseBuilder {
 public:
  explicit PairBuilder(Client& client) : PairBaseBuilder(client) {}

  /**
   * @brief Get the builder for the first element.
   *
   * @return The builder for the first element.
   */
  std::shared_ptr<ObjectBuilder> First() {
    return std::dynamic_pointer_cast<ObjectBuilder>(this->first_);
  }

  /**
   * @brief Get the builder for the second element.
   *
   * @return The builder for the second element.
   */
  std::shared_ptr<ObjectBuilder> Second() {
    return std::dynamic_pointer_cast<ObjectBuilder>(this->second_);
  }

  /**
   * @brief Set the builder for the first element.
   * When building the pair, the builder will be invoked to build
   * the first element.
   *
   * @param first The builder for the first object.
   */
  void SetFirst(std::shared_ptr<ObjectBuilder> const& first) {
    VINEYARD_ASSERT(this->first_ == nullptr);
    this->set_first_(first);
  }

  /**
   * @brief Set the builder for the first element.
   * When building the pair, the builder will be invoked to build
   * the first element.
   *
   * @param first The value for the first object.
   */
  void SetFirst(std::shared_ptr<Object> const& first) {
    VINEYARD_ASSERT(this->first_ == nullptr);
    this->set_first_(first);
  }

  /**
   * @brief Set the builder for the second element.
   * When building the pair, the builder will be invoked to build
   * the second element.
   *
   * @param second The builder for the second object.
   */
  void SetSecond(std::shared_ptr<ObjectBuilder> const& second) {
    VINEYARD_ASSERT(this->second_ == nullptr);
    this->set_second_(second);
  }

  /**
   * @brief Set the builder for the second element.
   * When building the pair, the builder will be invoked to build
   * the second element.
   *
   * @param second The value for the second object.
   */
  void SetSecond(std::shared_ptr<Object> const& second) {
    VINEYARD_ASSERT(this->second_ == nullptr);
    this->set_second_(second);
  }
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_PAIR_H_
