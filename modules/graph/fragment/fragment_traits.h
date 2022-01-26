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

#ifndef MODULES_GRAPH_FRAGMENT_FRAGMENT_TRAITS_H_
#define MODULES_GRAPH_FRAGMENT_FRAGMENT_TRAITS_H_

#include <string>
#include <type_traits>
#include <typeinfo>

namespace vineyard {

// TODO(lxj): unify the types conversion or place them together? e.g.
// those in context_utils.h
template <typename T>
struct is_property_fragment {
  using type = std::false_type;
  static constexpr bool value = false;
};

class ArrowFragmentBase;

template <>
struct is_property_fragment<ArrowFragmentBase> {
  using type = std::true_type;
  static constexpr bool value = true;
};

template <typename OID_T, typename VID_T>
class ArrowFragment;

template <typename OID_T, typename VID_T>
struct is_property_fragment<ArrowFragment<OID_T, VID_T>> {
  using type = std::true_type;
  static constexpr bool value = true;
};

template <typename OID_T, typename VID_T>
class AppendOnlyArrowFragment;

template <typename OID_T, typename VID_T>
struct is_property_fragment<AppendOnlyArrowFragment<OID_T, VID_T>> {
  using type = std::true_type;
  static constexpr bool value = true;
};

template <typename T>
struct TypeName {
  static const char* Get() { return typeid(T).name(); }
};

template <>
struct TypeName<int> {
  static const char* Get() { return "int"; }
};

template <>
struct TypeName<double> {
  static const char* Get() { return "double"; }
};

template <>
struct TypeName<std::string> {
  static const char* Get() { return "string"; }
};

template <>
struct TypeName<int64_t> {
  static const char* Get() { return "int64_t"; }
};

template <>
struct TypeName<uint64_t> {
  static const char* Get() { return "uint64_t"; }
};

template <>
struct TypeName<grape::EmptyType> {
  static const char* Get() { return "grape::EmptyType"; }
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_FRAGMENT_TRAITS_H_
