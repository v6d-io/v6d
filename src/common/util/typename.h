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

#ifndef SRC_COMMON_UTIL_TYPENAME_H_
#define SRC_COMMON_UTIL_TYPENAME_H_

#include <string>

#include "ctti/detail/name_filters.hpp"
#include "ctti/nameof.hpp"

namespace vineyard {

template <typename T>
inline const std::string type_name();

namespace detail {

template <typename Arg>
inline const std::string typename_unpack_args() {
  return type_name<Arg>();
}

template <typename T, typename U, typename... Args>
inline const std::string typename_unpack_args() {
  return type_name<T>() + "," + typename_unpack_args<U, Args...>();
}

template <typename T>
inline const std::string typename_impl(T const&) {
  return ctti::nameof<T>().cppstring();
}

template <template <typename...> class C, typename... Args>
inline const std::string typename_impl(C<Args...> const&) {
  constexpr auto fullname = ctti::pretty_function::type<C<Args...>>();
  constexpr const char* index = ctti::detail::find(fullname, "<");
  if (index == fullname.end()) {
    return fullname(CTTI_VALUE_PRETTY_FUNCTION_LEFT - 1,
                    fullname.end() - fullname.begin() - 1)
        .cppstring();
  }
  constexpr auto class_name =
      fullname(CTTI_VALUE_PRETTY_FUNCTION_LEFT - 1, index - fullname.begin());
  return class_name.cppstring() + "<" + typename_unpack_args<Args...>() + ">";
}

}  // namespace detail

template <typename T>
inline const std::string type_name() {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
  return detail::typename_impl(*(static_cast<T*>(nullptr)));
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

template <>
inline const std::string type_name<std::string>() {
  return "std::string";
}

template <>
inline const std::string type_name<int32_t>() {
  return "int";
}

template <>
inline const std::string type_name<int64_t>() {
  return "int64";
}

template <>
inline const std::string type_name<uint32_t>() {
  return "uint";
}

template <>
inline const std::string type_name<uint64_t>() {
  return "uint64";
}

}  // namespace vineyard

// for backwards compatiblity.
using vineyard::type_name;

#endif  // SRC_COMMON_UTIL_TYPENAME_H_
