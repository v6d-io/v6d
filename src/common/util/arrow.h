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

#ifndef SRC_COMMON_UTIL_ARROW_H_
#define SRC_COMMON_UTIL_ARROW_H_

#include <utility>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "common/util/status.h"

namespace vineyard {

#ifndef CHECK_ARROW_ERROR
#define CHECK_ARROW_ERROR(expr) \
  VINEYARD_CHECK_OK(::vineyard::Status::ArrowError(expr))
#endif  // CHECK_ARROW_ERROR

// discard and ignore the error status.
#ifndef DISCARD_ARROW_ERROR
#define DISCARD_ARROW_ERROR(expr)                               \
  do {                                                          \
    auto status = (expr);                                       \
    if (!status.ok()) {} /* NOLINT(whitespace/empty_if_body) */ \
  } while (0)
#endif  // DISCARD_ARROW_ERROR

#ifndef CHECK_ARROW_ERROR_AND_ASSIGN
#define CHECK_ARROW_ERROR_AND_ASSIGN(lhs, expr) \
  do {                                          \
    auto status = (expr);                       \
    CHECK_ARROW_ERROR(status.status());         \
    lhs = std::move(status).ValueOrDie();       \
  } while (0)
#endif  // CHECK_ARROW_ERROR_AND_ASSIGN

#ifndef RETURN_ON_ARROW_ERROR
#define RETURN_ON_ARROW_ERROR(expr)                  \
  do {                                               \
    auto status = (expr);                            \
    if (!status.ok()) {                              \
      return ::vineyard::Status::ArrowError(status); \
    }                                                \
  } while (0)
#endif  // RETURN_ON_ARROW_ERROR

#ifndef RETURN_ON_ARROW_ERROR_AND_ASSIGN
#define RETURN_ON_ARROW_ERROR_AND_ASSIGN(lhs, expr)           \
  do {                                                        \
    auto result = (expr);                                     \
    if (!result.status().ok()) {                              \
      return ::vineyard::Status::ArrowError(result.status()); \
    }                                                         \
    lhs = std::move(result).ValueOrDie();                     \
  } while (0)
#endif  // RETURN_ON_ARROW_ERROR_AND_ASSIGN

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_ARROW_H_
