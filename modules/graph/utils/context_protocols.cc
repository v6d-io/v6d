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

#include "graph/utils/context_protocols.h"

#include <memory>
#include <string>

#include "arrow/api.h"
#include "arrow/io/api.h"

namespace vineyard {

constexpr int TypeToInt<void>::value;
constexpr int TypeToInt<bool>::value;
constexpr int TypeToInt<int32_t>::value;
constexpr int TypeToInt<uint32_t>::value;
constexpr int TypeToInt<int64_t>::value;
constexpr int TypeToInt<uint64_t>::value;
constexpr int TypeToInt<float>::value;
constexpr int TypeToInt<double>::value;
constexpr int TypeToInt<std::string>::value;

int ArrowDataTypeToInt(const std::shared_ptr<arrow::DataType>& type) {
  if (type->Equals(arrow::null())) {
    return TypeToInt<void>::value;
  } else if (type->Equals(arrow::boolean())) {
    return TypeToInt<bool>::value;
  } else if (type->Equals(arrow::int32())) {
    return TypeToInt<int32_t>::value;
  } else if (type->Equals(arrow::uint32())) {
    return TypeToInt<uint32_t>::value;
  } else if (type->Equals(arrow::int64())) {
    return TypeToInt<int64_t>::value;
  } else if (type->Equals(arrow::uint64())) {
    return TypeToInt<uint64_t>::value;
  } else if (type->Equals(arrow::float32())) {
    return TypeToInt<float>::value;
  } else if (type->Equals(arrow::float64())) {
    return TypeToInt<double>::value;
  } else if (type->Equals(arrow::utf8()) || type->Equals(arrow::large_utf8())) {
    return TypeToInt<std::string>::value;
  }

  return -1;
}

}  // namespace vineyard
