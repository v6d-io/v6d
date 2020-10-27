/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef SRC_COMMON_UTIL_PTREE_H_
#define SRC_COMMON_UTIL_PTREE_H_

#include <sstream>
#include <string>

#include "boost/exception/diagnostic_information.hpp"

#include "common/util/boost.h"

namespace vineyard {

template <typename T>
void print_json_value(std::stringstream& ss, T const& value) {
  ss << value;
}

template <>
void print_json_value(std::stringstream& ss, std::string const& value);

template <>
void print_json_value(std::stringstream& ss, char const& value);

template <typename Container>
void put_container(ptree& tree, std::string const& path,
                   Container const& container) {
  std::stringstream ss;
  ss << "[";
  auto iter = std::begin(container);
  if (iter != container.end()) {
    print_json_value(ss, *iter);
    ++iter;
  }
  while (iter != container.end()) {
    ss << ", ";
    print_json_value(ss, *iter);
    ++iter;
  }
  ss << "]";
  tree.put(path, ss.str());
}

template <typename Container>
void get_container(ptree const& tree, std::string const& path,
                   Container& container) {
  // FIXME: parse the JSON array directly
  ptree body;
  std::istringstream body_iss(tree.get<std::string>(path));
  bpt::read_json(body_iss, body);
  using T = typename Container::value_type;
  for (auto const& kv : body) {
    container.insert(std::end(container), kv.second.get_value<T>());
  }
}

// Any operation on meta_tree (ptree) shouldn't break down the vineyard server
#ifndef CATCH_PTREE_ERROR
#define CATCH_PTREE_ERROR(expr)                                        \
  [&]() {                                                              \
    try {                                                              \
      return (expr);                                                   \
    } catch (bpt::ptree_error const& err) {                            \
      LOG(ERROR) << "ptree: " << err.what();                           \
      LOG(ERROR) << boost::current_exception_diagnostic_information(); \
      return Status::MetaTreeInvalid();                                \
    }                                                                  \
  }()
#endif  // CATCH_PTREE_ERROR

}  //  namespace vineyard

#endif  // SRC_COMMON_UTIL_PTREE_H_
