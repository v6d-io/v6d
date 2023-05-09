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

#ifndef SRC_COMMON_UTIL_JSON_H_
#define SRC_COMMON_UTIL_JSON_H_

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

// make sure bundled one is used
#include "single_include/nlohmann/json.hpp"

#include "common/util/macros.h"

namespace vineyard {

using json = nlohmann::json;

// Any operation on meta_tree (json) shouldn't break down the vineyard server
#ifndef CATCH_JSON_ERROR
#define CATCH_JSON_ERROR_RETURN_ANY(var, status, expr)                  \
  do {                                                                  \
    try {                                                               \
      var = expr;                                                       \
    } catch (std::out_of_range const& err) {                            \
      std::clog << "[error] json: out of range: " << err.what()         \
                << ("in '" #expr "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (std::invalid_argument const& err) {                        \
      std::clog << "[error] json: invalid argument: " << err.what()     \
                << ("in '" #expr "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (vineyard::json::exception const& err) {                    \
      std::clog << "[error] json: " << err.what() << ("in '" #expr "'") \
                << std::endl;                                           \
      status = vineyard::Status::MetaTreeInvalid();                     \
    }                                                                   \
  } while (0)
#define CATCH_JSON_ERROR_STATEMENT(status, stmt)                        \
  do {                                                                  \
    try {                                                               \
      stmt;                                                             \
    } catch (std::out_of_range const& err) {                            \
      std::clog << "[error] json: out of range: " << err.what()         \
                << ("in '" #stmt "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (std::invalid_argument const& err) {                        \
      std::clog << "[error] json: invalid argument: " << err.what()     \
                << ("in '" #stmt "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (vineyard::json::exception const& err) {                    \
      std::clog << "[error] json: " << err.what() << ("in '" #stmt "'") \
                << std::endl;                                           \
      status = vineyard::Status::MetaTreeInvalid();                     \
    }                                                                   \
  } while (0)

#define CATCH_JSON_ERROR_RETURN_STATUS(var, expr) \
  CATCH_JSON_ERROR_RETURN_ANY(var, var, expr)
#define CATCH_JSON_ERROR(...)                          \
  GET_MACRO2(__VA_ARGS__, CATCH_JSON_ERROR_RETURN_ANY, \
             CATCH_JSON_ERROR_RETURN_STATUS)           \
  (__VA_ARGS__)
#endif  // CATCH_JSON_ERROR

#ifndef VCATCH_JSON_ERROR
#define VCATCH_JSON_ERROR_RETURN_ANY(data, var, status, expr)           \
  do {                                                                  \
    try {                                                               \
      var = expr;                                                       \
    } catch (std::out_of_range const& err) {                            \
      std::clog << "[error] json: out of range: " << err.what()         \
                << ("in '" #expr "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (std::invalid_argument const& err) {                        \
      std::clog << "[error] json: invalid argument: " << err.what()     \
                << ("in '" #expr "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (vineyard::json::exception const& err) {                    \
      std::clog << "[error] json: " << err.what() << ("in '" #expr "'") \
                << std::endl;                                           \
      status = vineyard::Status::MetaTreeInvalid();                     \
    }                                                                   \
    if (VLOG_IS_ON(100) && !status.ok()) {                              \
      std::clog << "[error] json: " << data.dump(2) << std::endl;       \
    }                                                                   \
  } while (0)
#define VCATCH_JSON_ERROR_STATEMENT(data, status, stmt)                 \
  do {                                                                  \
    try {                                                               \
      stmt;                                                             \
    } catch (std::out_of_range const& err) {                            \
      std::clog << "[error] json: out of range: " << err.what()         \
                << ("in '" #stmt "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (std::invalid_argument const& err) {                        \
      std::clog << "[error] json: invalid argument: " << err.what()     \
                << ("in '" #stmt "'") << std::endl;                     \
      status = vineyard::Status::MetaTreeInvalid();                     \
    } catch (vineyard::json::exception const& err) {                    \
      std::clog << "[error] json: " << err.what() << ("in '" #stmt "'") \
                << std::endl;                                           \
      status = vineyard::Status::MetaTreeInvalid();                     \
    }                                                                   \
    if (VLOG_IS_ON(100) && !status.ok()) {                              \
      std::clog << "[error] json: " << data.dump(2) << std::endl;       \
    }                                                                   \
  } while (0)

#define VCATCH_JSON_ERROR_RETURN_STATUS(data, var, expr) \
  VCATCH_JSON_ERROR_RETURN_ANY(data, var, var, expr)
#define VCATCH_JSON_ERROR(...)                          \
  GET_MACRO3(__VA_ARGS__, VCATCH_JSON_ERROR_RETURN_ANY, \
             VCATCH_JSON_ERROR_RETURN_STATUS)           \
  (__VA_ARGS__)
#endif  // VCATCH_JSON_ERROR

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
