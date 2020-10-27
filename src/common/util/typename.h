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

#ifndef SRC_COMMON_UTIL_TYPENAME_H_
#define SRC_COMMON_UTIL_TYPENAME_H_

#include <string>
#include <typeinfo>

#include "boost/core/demangle.hpp"

// FIXME don't depends on boost, but currently we just needs the
// backwards compatibility.

template <typename T>
std::string type_name() {
  std::string name = boost::core::demangle(typeid(T).name());

  // strip all spaces between `<`, `>` and `,`, to ensure consistency
  std::string compact_name;
  for (size_t idx = 0; idx < name.size(); ++idx) {
    if (name[idx] != ' ') {
      compact_name.append(1, name[idx]);
    } else {
      if (idx > 0 && idx < (name.size() - 1) && std::isalnum(name[idx - 1]) &&
          std::isalnum(name[idx + 1])) {
        compact_name.append(1, name[idx]);
      }
    }
  }

  // erase std::__1:: and std:: difference: to make the object can be get by
  // clients that linked against different STL libraries.
  const std::string marker = "std::__1::";
  for (std::string::size_type p = compact_name.find(marker);
       p != std::string::npos; p = compact_name.find(marker)) {
    compact_name.replace(p, marker.size(), "std::");
  }
  return compact_name;
}

#endif  // SRC_COMMON_UTIL_TYPENAME_H_
