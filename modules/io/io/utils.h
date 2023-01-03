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

#ifndef MODULES_IO_IO_UTILS_H_
#define MODULES_IO_IO_UTILS_H_

#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include "common/util/json.h"
#include "common/util/logging.h"

namespace vineyard {

template <typename T>
void ReportStatus(const std::string& kind, T const& value) {
  json result;
  result["type"] = kind;
  result["content"] = value;
  std::cout << json_to_string(result) << std::endl;
}

#ifndef CHECK_AND_REPORT
#define CHECK_AND_REPORT(status)                       \
  do {                                                 \
    if (!status.ok()) {                                \
      LOG(ERROR) << "IO Error: " << status.ToString(); \
      ReportStatus("error", status.ToString());        \
      VINEYARD_CHECK_OK(status);                       \
    }                                                  \
  } while (0)
#endif  // CHECK_AND_REPORT

}  // namespace vineyard

#endif  // MODULES_IO_IO_UTILS_H_
