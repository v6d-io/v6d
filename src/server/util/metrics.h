/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#ifndef SRC_SERVER_UTIL_METRICS_H_
#define SRC_SERVER_UTIL_METRICS_H_

#include <string>

#include "common/util/env.h"
#include "common/util/logging.h"
#include "server/util/spec_resolvers.h"

namespace vineyard {

#ifndef LOG_COUNTER
#define LOG_COUNTER(metric_name, label)                                   \
  do {                                                                    \
    static const std::string __METRIC_USER = read_env("USER", "v6d");     \
    LOG_IF_EVERY_N(INFO, FLAGS_prometheus, 1)                             \
        << __METRIC_USER << " " << (label) << " " << (metric_name) << " " \
        << logging::COUNTER;                                              \
  } while (0)
#endif

#ifndef LOG_SUMMARY
#define LOG_SUMMARY(metric_name, label, metric_val)                          \
  do {                                                                       \
    static const std::string __METRIC_USER = read_env("USER", "v6d");        \
    LOG_IF(INFO, FLAGS_prometheus) << __METRIC_USER << " " << (label) << " " \
                                   << (metric_name) << " " << (metric_val);  \
  } while (0)
#endif

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_METRICS_H_
