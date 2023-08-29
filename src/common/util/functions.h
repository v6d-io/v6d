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

#ifndef SRC_COMMON_UTIL_FUNCTIONS_H_
#define SRC_COMMON_UTIL_FUNCTIONS_H_

#include <sys/time.h>

#include <iostream>
#include <regex>
#include <string>
#include <utility>

#include "boost/algorithm/string/replace.hpp"

#include "common/util/env.h"

namespace vineyard {

// Expand the path with environment variable
inline std::string ExpandEnvironmentVariables(const std::string& text) {
  auto text_copy = text;
  try {
    std::regex env("\\$[{]?([^/:=@#,|}]*)[}]?");
    std::smatch match;
    while (std::regex_search(text_copy, match, env)) {
      std::string var = read_env(match.str(1).c_str());
      boost::replace_first(text_copy, match[0].str(), var);
    }
    return text_copy;
  } catch (std::exception& e) {
    std::clog << "[error] failed to expand environment variable: " << e.what()
              << std::endl;
    return text;
  }
}

/**
 * @brief Get the current timestamp in seconds.
 */
inline double GetCurrentTime() {
  timeval t;
  gettimeofday(&t, 0);
  return static_cast<double>(t.tv_sec) +
         static_cast<double>(t.tv_usec) / 1000000;
}

/**
 * @brief Generate a 64bit unique timestamp, no actual meanings.
 */
inline int64_t GetTimestamp() {
  timeval t;
  gettimeofday(&t, 0);
  return ((int64_t) t.tv_sec << sizeof(int32_t) * 8) + (int64_t) t.tv_usec;
}

/**
 * @brief Get the current timestamp in microseconds.
 */
inline int64_t GetMicroTimestamp() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

namespace detail {

// Private implementation for std::exchange, as it is unavailable on ArchLinux.
//
// See also: https://en.cppreference.com/w/cpp/utility/exchange
//
template <class T, class U = T>
inline T exchange_value(T& obj, U&& new_value) {
  T old_value = std::move(obj);
  obj = std::forward<U>(new_value);
  return old_value;
}
}  // namespace detail

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_FUNCTIONS_H_
