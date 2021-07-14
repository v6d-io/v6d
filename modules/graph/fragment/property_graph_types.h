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

#ifndef MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_TYPES_H_
#define MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_TYPES_H_

#include <string>

#include "basic/ds/arrow.h"
#include "grape/config.h"

namespace vineyard {

using fid_t = grape::fid_t;

template <typename T>
struct InternalType {
  using type = T;
  using vineyard_array_type = vineyard::NumericArray<T>;
  using vineyard_builder_type = vineyard::NumericArrayBuilder<T>;
};

template <>
struct InternalType<std::string> {
  using type = arrow::util::string_view;
  using vineyard_array_type = vineyard::LargeStringArray;
  using vineyard_builder_type = vineyard::LargeStringArrayBuilder;
};

template <>
struct InternalType<arrow::util::string_view> {
  using type = arrow::util::string_view;
  using vineyard_array_type = vineyard::LargeStringArray;
  using vineyard_builder_type = vineyard::LargeStringArrayBuilder;
};

namespace property_graph_types {

using OID_TYPE = int64_t;
using EID_TYPE = uint64_t;
using VID_TYPE = uint64_t;
// using VID_TYPE = uint32_t;

using PROP_ID_TYPE = int;
using LABEL_ID_TYPE = int;

}  // namespace property_graph_types

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_TYPES_H_
