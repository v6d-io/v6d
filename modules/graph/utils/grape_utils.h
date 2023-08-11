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

#ifndef MODULES_GRAPH_UTILS_GRAPE_UTILS_H_
#define MODULES_GRAPH_UTILS_GRAPE_UTILS_H_

#include <algorithm>
#include <string>
#include <vector>

// FIXME(lxj):unify the namespace;
namespace grape {

struct RefString;
class RSVector;

template <typename T>
struct InternalType {
  using value_type = T;
  using buffer_type = std::vector<T>;
};

template <>
struct InternalType<std::string> {
  using value_type = RefString;
  using buffer_type = RSVector;
};
}  // namespace grape

inline bool is_number(const std::string& s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

namespace vineyard {

inline std::string normalize_datatype(const std::string& str) {
  if (str == "null" || str == "NULL") {
    return "null";
  }
  if (str == "bool" || str == "boolean") {
    return "bool";
  }
  if (str == "int" || str == "int32_t" || str == "int32") {
    return "int32_t";
  }
  if (str == "int64_t" || str == "int64") {
    return "int64_t";
  }
  if (str == "uint32_t" || str == "uint32" || str == "uint") {
    return "uint32_t";
  }
  if (str == "uint64_t" || str == "uint64") {
    return "uint64_t";
  }
  if (str == "empty" || str == "EmptyType" || str == "grape::EmptyType") {
    return "grape::EmptyType";
  }
  if (str == "string" || str == "std::string" || str == "str") {
    return "std::string";
  }
  return str;
}

inline std::string random_string(size_t length) {
  srand(rand() ^ time(0));  // NOLINT(runtime/threadsafe_fn)
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}
}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_GRAPE_UTILS_H_
