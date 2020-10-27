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

#ifndef MODULES_GRAPH_UTILS_CONTEXT_PROTOCOLS_H_
#define MODULES_GRAPH_UTILS_CONTEXT_PROTOCOLS_H_

#include <memory>

#include "arrow/type.h"

namespace vineyard {

template <typename T>
struct TypeToInt {
  static constexpr int value = -1;
};

template <>
struct TypeToInt<int> {
  static constexpr int value = 1;
};

template <>
struct TypeToInt<double> {
  static constexpr int value = 2;
};

template <>
struct TypeToInt<int64_t> {
  static constexpr int value = 3;
};

template <>
struct TypeToInt<uint64_t> {
  static constexpr int value = 4;
};

inline int ArrowDataTypeToInt(std::shared_ptr<arrow::DataType> type) {
  if (type->Equals(arrow::int32())) {
    return 1;
  } else if (type->Equals(arrow::float64())) {
    return 2;
  } else if (type->Equals(arrow::int64())) {
    return 3;
  } else {
    return -1;
  }
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_CONTEXT_PROTOCOLS_H_
