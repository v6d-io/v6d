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

#ifndef MODULES_IO_IO_UTILS_H_
#define MODULES_IO_IO_UTILS_H_

#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include "common/util/boost.h"

namespace vineyard {

template <typename T>
void ReportStatus(const std::string& kind, T const& value) {
  LOG(INFO) << "kind: " << kind;
  ptree result;
  result.put("type", kind);
  result.put("content", value);
  std::stringstream ss;
  bpt::write_json(ss, result, false);
  std::cout << ss.str() << std::flush;
}

}  // namespace vineyard

#endif  // MODULES_IO_IO_UTILS_H_
