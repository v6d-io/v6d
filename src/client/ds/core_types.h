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

#ifndef SRC_CLIENT_DS_CORE_TYPES_H_
#define SRC_CLIENT_DS_CORE_TYPES_H_

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/util/json.h"

namespace vineyard {

#ifndef DECLARE_DEFAULT_WARPPER
#define DECLARE_DEFAULT_WARPPER(T)              \
  T() : base_type() {}                          \
  T(base_type const& v) : base_type(v) {}       \
  T(base_type&& v) : base_type(std::move(v)) {} \
  T& operator=(const base_type& v) {            \
    base_type::operator=(v);                    \
    return *this;                               \
  }                                             \
  T& operator=(base_type&& v) {                 \
    base_type::operator=(std::move(v));         \
    return *this;                               \
  }

#endif

template <typename T>
class Tuple final : public std::vector<T> {
 public:
  using base_type = std::vector<T>;
  DECLARE_DEFAULT_WARPPER(Tuple)
};

template <typename T>
class List final : public std::vector<T> {
 public:
  using base_type = std::vector<T>;
  DECLARE_DEFAULT_WARPPER(List)
};

class String final : public std::string {
 public:
  using base_type = std::string;
  DECLARE_DEFAULT_WARPPER(String)
};

inline void to_json(json& j, const String& str) { j = json(std::string(str)); }

inline void from_json(const json& j, String& str) {
  str = j.get_ref<std::string const&>();
}

template <typename Key, typename T>
class Map final : public std::map<Key, T> {
 public:
  using base_type = std::map<Key, T>;
  DECLARE_DEFAULT_WARPPER(Map)

  using __other_type = std::unordered_map<Key, T>;

  Map(__other_type const& v) : base_type(v.begin(), v.end()) {}

  Map& operator=(const __other_type& v) {
    this->clear();
    __other_type::insert(v.begin(), v.end());
    return *this;
  }
};

template <typename Key, typename T>
class UnorderedMap final : public std::unordered_map<Key, T> {
 public:
  using base_type = std::unordered_map<Key, T>;
  DECLARE_DEFAULT_WARPPER(UnorderedMap)

  using __other_type = std::map<Key, T>;

  UnorderedMap(__other_type const& v) : base_type(v.begin(), v.end()) {}
  UnorderedMap& operator=(const __other_type& v) {
    this->clear();
    __other_type::insert(v.begin(), v.end());
    return *this;
  }
};

#undef DECLARE_DEFAULT_WARPPER

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_CORE_TYPES_H_
