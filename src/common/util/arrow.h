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

#ifndef SRC_COMMON_UTIL_ARROW_H_
#define SRC_COMMON_UTIL_ARROW_H_

#include <memory>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "arrow/api.h"     // IWYU pragma: keep
#include "arrow/io/api.h"  // IWYU pragma: keep

#include "common/util/status.h"

namespace vineyard {

#if ARROW_VERSION_MAJOR >= 10
using arrow_string_view = std::string_view;
#else
using arrow_string_view = arrow::util::string_view;
#endif

using table_vec_t = std::vector<std::shared_ptr<arrow::Table>>;
using array_vec_t = std::vector<std::shared_ptr<arrow::Array>>;

}  // namespace vineyard

#if !nssv_USES_STD_STRING_VIEW && \
    ((!__cpp_lib_string_view) || ARROW_VERSION_MAJOR < 10)
#include "wyhash/wyhash.hpp"  // IWYU pragma: keep
namespace wy {
template <>
struct hash<vineyard::arrow_string_view>
    : public internal::hash_string_base<vineyard::arrow_string_view> {
  using hash_string_base::hash_string_base;  // Inherit constructors
};
}  // namespace wy
#endif

#if !nssv_USES_STD_STRING_VIEW && \
    ((!__cpp_lib_string_view) || ARROW_VERSION_MAJOR < 10)
#include "cityhash/cityhash.hpp"  // IWYU pragma: keep
namespace city {
template <>
class hash<vineyard::arrow_string_view> {
  inline uint64_t operator()(
      const vineyard::arrow_string_view& data) const noexcept {
    return detail::CityHash64(reinterpret_cast<const char*>(data.data()),
                              data.size());
  }
};
}  // namespace city
#endif

namespace vineyard {

/// Return an error when we meet an errorous status from apache-arrow APIs.
///
/// Don't put this function in `status.h` to avoid dependency on
/// apache-arrow.
static inline Status ArrowError(const arrow::Status& status) {
  if (status.ok()) {
    return Status::OK();
  } else {
    return Status(StatusCode::kArrowError, status.ToString());
  }
}

#ifndef CHECK_ARROW_ERROR
#define CHECK_ARROW_ERROR(expr) VINEYARD_CHECK_OK(::vineyard::ArrowError(expr))
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
#define RETURN_ON_ARROW_ERROR(expr)          \
  do {                                       \
    auto status = (expr);                    \
    if (!status.ok()) {                      \
      return ::vineyard::ArrowError(status); \
    }                                        \
  } while (0)
#endif  // RETURN_ON_ARROW_ERROR

#ifndef RETURN_ON_ARROW_ERROR_AND_ASSIGN
#define RETURN_ON_ARROW_ERROR_AND_ASSIGN(lhs, expr)   \
  do {                                                \
    auto result = (expr);                             \
    if (!result.status().ok()) {                      \
      return ::vineyard::ArrowError(result.status()); \
    }                                                 \
    lhs = std::move(result).ValueOrDie();             \
  } while (0)
#endif  // RETURN_ON_ARROW_ERROR_AND_ASSIGN

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_ARROW_H_
