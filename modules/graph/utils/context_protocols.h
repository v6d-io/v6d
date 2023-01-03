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

#ifndef MODULES_GRAPH_UTILS_CONTEXT_PROTOCOLS_H_
#define MODULES_GRAPH_UTILS_CONTEXT_PROTOCOLS_H_

#include <memory>
#include <string>

#include "arrow/api.h"
#include "arrow/io/api.h"

namespace vineyard {

template <typename T>
struct TypeToInt {
  static constexpr int value = -1;
};

template <>
struct TypeToInt<void> {
  static constexpr int value = 0;
};

template <>
struct TypeToInt<bool> {
  static constexpr int value = 1;
};

template <>
struct TypeToInt<int32_t> {
  static constexpr int value = 2;
};

template <>
struct TypeToInt<uint32_t> {
  static constexpr int value = 3;
};

template <>
struct TypeToInt<int64_t> {
  static constexpr int value = 4;
};

template <>
struct TypeToInt<uint64_t> {
  static constexpr int value = 5;
};

template <>
struct TypeToInt<float> {
  static constexpr int value = 6;
};

template <>
struct TypeToInt<double> {
  static constexpr int value = 7;
};

template <>
struct TypeToInt<std::string> {
  static constexpr int value = 8;
};

int ArrowDataTypeToInt(const std::shared_ptr<arrow::DataType>& type);

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_CONTEXT_PROTOCOLS_H_
