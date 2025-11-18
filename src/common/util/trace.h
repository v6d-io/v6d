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

#ifndef SRC_COMMON_UTIL_TRACE_H_
#define SRC_COMMON_UTIL_TRACE_H_

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "common/util/env.h"

namespace vineyard {

class Logger {
 public:
  Logger() : enable_(false) {}

  explicit Logger(bool enable) : enable_(enable) {}

  explicit Logger(int log_level) {
    int glog_level = 0;
    try {
      glog_level = stoi(read_env("GLOG_v", "0"));
    } catch (...) { glog_level = 0; }
    enable_ = log_level <= glog_level ? true : false;
  }

  template <typename T>
  Logger& operator<<(const T& msg) {
    if (!enable_) {
      return *this;
    }
    ss_ << msg;
    return *this;
  }

  using endl_type = std::ostream& (*) (std::ostream&);

  Logger& operator<<(endl_type e) {
    if (!enable_) {
      return *this;
    }
    std::cout << ss_.str() << std::endl;
    ss_.str("");
    return *this;
  }

 private:
  bool enable_;
  std::stringstream ss_;
};

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_TRACE_H_
