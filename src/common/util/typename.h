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

#ifndef SRC_COMMON_UTIL_TYPENAME_H_
#define SRC_COMMON_UTIL_TYPENAME_H_

#include <string>
#include <vector>

#if defined(__clang__)
#define __THE_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(__GNUC__) && !defined(__clang__)
#define __THE_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define __THE_PRETTY_FUNCTION __FUNCSIG__
#else
#error "No support for this compiler."
#endif

// See also: https://stackoverflow.com/a/55926503/5080177
//
#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define __VINEYARD_GCC_VERSION \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

namespace vineyard {

template <typename T>
inline const std::string __type_name();

namespace detail {

template <typename Arg>
inline const std::string typename_unpack_args() {
  return __type_name<Arg>();
}

template <typename T, typename U, typename... Args>
inline const std::string typename_unpack_args() {
  return __type_name<T>() + "," + typename_unpack_args<U, Args...>();
}

#if defined(__clang__)
#define __TYPENAME_FROM_FUNCTION_PREFIX \
  "const std::string vineyard::detail::__typename_from_function() [T = "
#define __TYPENAME_FROM_FUNCTION_SUFFIX "]"
#elif defined(__GNUC__) && !defined(__clang__)
#define __TYPENAME_FROM_FUNCTION_PREFIX \
  "const string vineyard::detail::__typename_from_function() [with T = "
#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI
#if defined(__VINEYARD_GCC_VERSION) && __VINEYARD_GCC_VERSION <= 90100
#define __TYPENAME_FROM_FUNCTION_SUFFIX \
  "; std::__cxx11::string = std::__cxx11::basic_string<char>]"
#else
#define __TYPENAME_FROM_FUNCTION_SUFFIX \
  "; std::string = std::__cxx11::basic_string<char>]"
#endif
#else
#define __TYPENAME_FROM_FUNCTION_SUFFIX \
  "; std::string = std::basic_string<char>]"
#endif
#else
#error "No support for this compiler."
#endif

#define __TYPENAME_FROM_FUNCTION_LEFT \
  (sizeof(__TYPENAME_FROM_FUNCTION_PREFIX) - 1)

template <typename T>
inline const std::string __typename_from_function() {
  const std::string name = __THE_PRETTY_FUNCTION;
  return name.substr(__TYPENAME_FROM_FUNCTION_LEFT,
                     name.length() - (__TYPENAME_FROM_FUNCTION_LEFT - 1) -
                         sizeof(__TYPENAME_FROM_FUNCTION_SUFFIX));
}

template <typename T>
inline const std::string typename_impl(T const&) {
  return __typename_from_function<T>();
}

template <template <typename...> class C, typename... Args>
inline const std::string typename_impl(C<Args...> const&) {
  const auto fullname = __typename_from_function<C<Args...>>();
  const auto index = fullname.find('<');
  if (index == std::string::npos) {
    return fullname;
  }
  const auto class_name = fullname.substr(0, index);
  return class_name + "<" + typename_unpack_args<Args...>() + ">";
}

}  // namespace detail

template <typename T>
struct typename_t {
  inline static const std::string name() {
#if (__GNUC__ >= 6) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
    return detail::typename_impl(*(static_cast<T*>(nullptr)));
#if (__GNUC__ >= 6) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }
};

template <typename T>
inline const std::string __type_name() {
  return typename_t<T>::name();
}

template <typename T>
inline const std::string type_name() {
  std::string name = __type_name<T>();
  // drop the `std::__1::` namespace for libc++ and `std::__cxx11::` for some
  // version of libstdc++
  //
  // erase std::__1::/std::__cxx11:: and std:: difference: to make the object
  // can be get by clients that linked against different STL libraries.
  static const std::vector<std::string> stdmarkers{"std::__1::",
                                                   "std::__cxx11::"};
  for (auto const& marker : stdmarkers) {
    for (std::string::size_type p = name.find(marker); p != std::string::npos;
         p = name.find(marker)) {
      name.replace(p, marker.size(), "std::");
    }
  }
  return name;
}

template <>
struct typename_t<std::string> {
  inline static const std::string name() { return "std::string"; }
};

template <>
struct typename_t<int32_t> {
  inline static const std::string name() { return "int"; }
};

template <>
struct typename_t<int64_t> {
  inline static const std::string name() { return "int64"; }
};

template <>
struct typename_t<uint32_t> {
  inline static const std::string name() { return "uint"; }
};

template <>
struct typename_t<uint64_t> {
  inline static const std::string name() { return "uint64"; }
};

}  // namespace vineyard

// for backwards compatibility.
using vineyard::type_name;

#endif  // SRC_COMMON_UTIL_TYPENAME_H_
