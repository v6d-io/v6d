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

#ifndef SRC_COMMON_UTIL_JSON_H_
#define SRC_COMMON_UTIL_JSON_H_

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "nlohmann/json.hpp"

namespace vineyard {

using json = nlohmann::json;

// Any operation on meta_tree (json) shouldn't break down the vineyard server
#ifndef CATCH_JSON_ERROR
#if defined(NDEBUG)
#define CATCH_JSON_ERROR(expr)                                  \
  [&]() {                                                       \
    try {                                                       \
      return (expr);                                            \
    } catch (std::out_of_range const& err) {                    \
      std::clog << "[error] json: " << err.what() << std::endl; \
      return vineyard::Status::MetaTreeInvalid();               \
    } catch (vineyard::json::exception const& err) {            \
      std::clog << "[error] json: " << err.what() << std::endl; \
      return vineyard::Status::MetaTreeInvalid();               \
    }                                                           \
  }()
#else
#define CATCH_JSON_ERROR(expr) expr
#endif  // NDEBUG
#endif  // CATCH_JSON_ERROR

template <typename... Args>
auto json_to_string(Args&&... args)
    -> decltype(nlohmann::to_string(std::forward<Args>(args)...)) {
  return nlohmann::to_string(std::forward<Args>(args)...);
}

template <typename T>
void print_json_value(std::stringstream& ss, T const& value) {
  ss << value;
}

template <>
void print_json_value(std::stringstream& ss, std::string const& value);

template <>
void print_json_value(std::stringstream& ss, char const& value);

template <typename Container>
void put_container(json& tree, std::string const& path,
                   Container const& container) {
  tree[path] = json_to_string(json(container));
}

template <typename Container>
void get_container(json const& tree, std::string const& path,
                   Container& container) {
  // FIXME: parse the JSON array directly
  json body = json::parse(tree[path].get_ref<std::string const&>());
  using T = typename Container::value_type;
  for (auto const& item : body.items()) {
    container.insert(std::end(container), item.value().get<T>());
  }
}

}  //  namespace vineyard

#endif  // SRC_COMMON_UTIL_JSON_H_
