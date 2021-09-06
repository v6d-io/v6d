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

#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define __VINEYARD_GCC_VERSION \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#if defined(__VINEYARD_GCC_VERSION)
#pragma GCC push_options
#pragma GCC optimize ("O0")
#endif
#include "ctti/detail/name_filters.hpp"
#include "ctti/nameof.hpp"
#if defined(__VINEYARD_GCC_VERSION)
#pragma GCC pop_options
#endif

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

#if defined(__VINEYARD_GCC_VERSION) && (__VINEYARD_GCC_VERSION <= 50100 || __VINEYARD_GCC_VERSION >= 70200)

#if defined(__clang__)
#define __TYPENAME_FROM_FUNCTION_PREFIX \
  "const std::string vineyard::detail::__typename_from_function() [T = "
#define __TYPENAME_FROM_FUNCTION_SUFFIX "]"
#elif defined(__GNUC__) && !defined(__clang__)
#define __TYPENAME_FROM_FUNCTION_PREFIX \
  "const string vineyard::detail::__typename_from_function() [with T = "
#define __TYPENAME_FROM_FUNCTION_SUFFIX \
  "; std::string = std::basic_string<char>]"
#else
#error "No support for this compiler."
#endif

#define __TYPENAME_FROM_FUNCTION_LEFT \
  (sizeof(__TYPENAME_FROM_FUNCTION_PREFIX) - 1)

template <typename T>
inline const std::string __typename_from_function() {
  std::string name = CTTI_PRETTY_FUNCTION;
  return name.substr(__TYPENAME_FROM_FUNCTION_LEFT,
                     name.length() - (__TYPENAME_FROM_FUNCTION_LEFT - 1) -
                         sizeof(__TYPENAME_FROM_FUNCTION_SUFFIX));
}
#endif

template <typename T>
inline const std::string typename_impl(T const&) {
#if defined(__VINEYARD_GCC_VERSION) && (__VINEYARD_GCC_VERSION <= 50100 || __VINEYARD_GCC_VERSION >= 70200)
  return ctti::nameof<T>().cppstring();
#else
  return __typename_from_function<T>();
#endif
}

template <template <typename...> class C, typename... Args>
inline const std::string typename_impl(C<Args...> const&) {
#if defined(__VINEYARD_GCC_VERSION) && (__VINEYARD_GCC_VERSION <= 50100 || __VINEYARD_GCC_VERSION >= 70200)
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
#else
  const auto fullname = __typename_from_function<C<Args...>>();
  const auto index = fullname.find('<');
  if (index == std::string::npos) {
    return fullname;
  }
  const auto class_name = fullname.substr(0, index);
  return class_name + "<" + typename_unpack_args<Args...>() + ">";
#endif
}

}  // namespace detail

template <typename T>
inline const std::string type_name() {
#if (__GNUC__ >= 6) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
  return detail::typename_impl(*(static_cast<T*>(nullptr)));
#if (__GNUC__ >= 6) || defined(__clang__)
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
