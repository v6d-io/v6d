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

#ifndef MODULES_BASIC_DS_SCALAR_H_
#define MODULES_BASIC_DS_SCALAR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "boost/lexical_cast.hpp"

#include "basic/ds/scalar.vineyard.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

/**
 * @brief ScalarBuilder is used for building scalars that supported by vineyard.
 *
 * @tparam T The type for the scalar.
 */
template <typename T>
class ScalarBuilder : public ScalarBaseBuilder<T> {
 public:
  explicit ScalarBuilder(Client& client) : ScalarBaseBuilder<T>(client) {
    this->set_type_(type_);
  }

  /**
   * @brief Initialize the scalar with the value.
   *
   * @param client The client connected to the vineyard server.
   * @param value The value for the scalar.
   */
  explicit ScalarBuilder(Client& client, T const& value)
      : ScalarBaseBuilder<T>(client) {
    this->set_value_(value);
  }

  /**
   * @brief Set the value of the scalar.
   *
   * @param value The value for the scalar.
   */
  void SetValue(T const& value) { this->set_value_(value); }

 private:
  const AnyType type_ = AnyTypeEnum<T>::value;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_SCALAR_H_
