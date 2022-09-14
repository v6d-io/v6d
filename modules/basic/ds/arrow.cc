/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include "basic/ds/arrow.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "basic/ds/arrow.vineyard.h"
#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "client/ds/blob.h"

namespace vineyard {

namespace detail {

template <typename T>
inline std::shared_ptr<ObjectBuilder> BuildNumericArray(
    Client& client,
    std::shared_ptr<typename ConvertToArrowType<T>::ArrayType> arr) {
  return std::make_shared<NumericArrayBuilder<T>>(client, arr);
}

std::shared_ptr<ObjectBuilder> BuildSimpleArray(
    Client& client, std::shared_ptr<arrow::Array> array) {
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<int8_t>::ArrayType>(
              array)) {
    return BuildNumericArray<int8_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<uint8_t>::ArrayType>(
              array)) {
    return BuildNumericArray<uint8_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<int16_t>::ArrayType>(
              array)) {
    return BuildNumericArray<int16_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<uint16_t>::ArrayType>(
              array)) {
    return BuildNumericArray<uint16_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<int32_t>::ArrayType>(
              array)) {
    return BuildNumericArray<int32_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<uint32_t>::ArrayType>(
              array)) {
    return BuildNumericArray<uint32_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<int64_t>::ArrayType>(
              array)) {
    return BuildNumericArray<int64_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<uint64_t>::ArrayType>(
              array)) {
    return BuildNumericArray<uint64_t>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<float>::ArrayType>(
              array)) {
    return BuildNumericArray<float>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<ConvertToArrowType<double>::ArrayType>(
              array)) {
    return BuildNumericArray<double>(client, arr);
  }
  if (auto arr = std::dynamic_pointer_cast<arrow::BooleanArray>(array)) {
    return std::make_shared<BooleanArrayBuilder>(client, arr);
  }
  if (auto arr =
          std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(array)) {
    return std::make_shared<FixedSizeBinaryArrayBuilder>(client, arr);
  }
  if (auto arr = std::dynamic_pointer_cast<arrow::StringArray>(array)) {
    return std::make_shared<StringArrayBuilder>(client, arr);
  }
  if (auto arr = std::dynamic_pointer_cast<arrow::LargeStringArray>(array)) {
    return std::make_shared<LargeStringArrayBuilder>(client, arr);
  }
  if (auto arr = std::dynamic_pointer_cast<arrow::NullArray>(array)) {
    return std::make_shared<NullArrayBuilder>(client, arr);
  }
  VINEYARD_ASSERT(nullptr != nullptr,
                  "Unsupported array type: " + array->type()->ToString());
  return nullptr;
}

std::shared_ptr<ObjectBuilder> BuildArray(Client& client,
                                          std::shared_ptr<arrow::Array> array) {
  if (auto arr = std::dynamic_pointer_cast<arrow::ListArray>(array)) {
    return std::make_shared<ListArrayBuilder>(client, arr);
  }
  if (auto arr = std::dynamic_pointer_cast<arrow::LargeListArray>(array)) {
    return std::make_shared<LargeListArrayBuilder>(client, arr);
  }
  if (auto arr = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(array)) {
    return std::make_shared<FixedSizeListArrayBuilder>(client, arr);
  }
  return BuildSimpleArray(client, array);
}
}  // namespace detail

}  // namespace vineyard
