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

#ifndef MODULES_GRAPH_UTILS_ERROR_H_
#define MODULES_GRAPH_UTILS_ERROR_H_

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "boost/leaf/all.hpp"

#include "graph/utils/mpi_utils.h"

namespace vineyard {

enum class ErrorCode {
  kOk,
  kIOError,
  kArrowError,
  kVineyardError,
  kUnspecificError,
  kNetworkError,
  kCommandError,
  kDataTypeError,
  kIllegalStateError,
  kInvalidValueError,
  kInvalidOperationError,
  kUnsupportedOperationError,
  kUnimplementedMethod,
};

inline const char* ErrorCodeToString(ErrorCode ec) {
  switch (ec) {
  case ErrorCode::kOk:
    return "Ok";
  case ErrorCode::kIOError:
    return "IOError";
  case ErrorCode::kArrowError:
    return "ArrowError";
  case ErrorCode::kVineyardError:
    return "VineyardError";
  case ErrorCode::kUnspecificError:
    return "UnspecificError";
  case ErrorCode::kNetworkError:
    return "NetworkError";
  case ErrorCode::kCommandError:
    return "CommandError";
  case ErrorCode::kDataTypeError:
    return "DataTypeError";
  case ErrorCode::kIllegalStateError:
    return "IllegalStateError";
  case ErrorCode::kInvalidOperationError:
    return "InvalidOperationError";
  case ErrorCode::kInvalidValueError:
    return "InvalidValueError";
  case ErrorCode::kUnsupportedOperationError:
    return "UnsupportedOperationError";
  case ErrorCode::kUnimplementedMethod:
    return "UnimplementedMethod";
  default:
    CHECK(false);
    return "UndefinedErrorCode";
  }
}

struct GSError {
  ErrorCode error_code;
  std::string error_msg;
  GSError() : error_code(ErrorCode::kOk), error_msg() {}

  GSError(ErrorCode code, std::string msg)
      : error_code(code), error_msg(std::move(msg)) {}

  explicit GSError(ErrorCode code) : GSError(code, "") {}

  explicit operator bool() const { return error_code != ErrorCode::kOk; }

  bool ok() const { return error_code == ErrorCode::kOk; }
};

inline grape::InArchive& operator<<(grape::InArchive& archive,
                                    const GSError& e) {
  archive << e.error_code;
  archive << e.error_msg;
  return archive;
}

inline grape::OutArchive& operator>>(grape::OutArchive& archive, GSError& e) {
  archive >> e.error_code;
  archive >> e.error_msg;
  return archive;
}

#define RETURN_GS_ERROR(code, msg)                                            \
  return ::boost::leaf::new_error(vineyard::GSError(                          \
      (code), std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + \
                  std::string(__FUNCTION__) + " -> " + (msg)))

inline GSError all_gather_error(const GSError& e,
                                const grape::CommSpec& comm_spec) {
  std::stringstream ss;
  ss << ErrorCodeToString(e.error_code) << " " << e.error_msg;
  ss << " occurred on Worker " << comm_spec.worker_id();
  auto msg = ss.str();

  std::vector<std::string> error_msgs(comm_spec.worker_num());
  GlobalAllGatherv<std::string>(msg, error_msgs, comm_spec);
  auto msgs = std::accumulate(
      error_msgs.begin(), error_msgs.end(), std::string(),
      [](const std::string& a, const std::string& b) -> std::string {
        return a + (!a.empty() ? ";" : "") + b;
      });

  return {ErrorCode::kUnspecificError, msgs};
}

inline GSError all_gather_error(const grape::CommSpec& comm_spec) {
  std::vector<std::string> error_msgs(comm_spec.worker_num());
  std::string msg;

  GlobalAllGatherv<std::string>(msg, error_msgs, comm_spec);
  auto msgs = std::accumulate(
      error_msgs.begin(), error_msgs.end(), std::string(),
      [](const std::string& a, const std::string& b) -> std::string {
        return a + (!a.empty() ? ";" : "") + b;
      });

  if (!msgs.empty()) {
    return {ErrorCode::kUnspecificError, msgs};
  }
  return {ErrorCode::kOk, ""};
}

template <class F_T, class... ARGS_T>
inline typename std::result_of<
    typename std::decay<F_T>::type(typename std::decay<ARGS_T>::type...)>::type
sync_gs_error(const grape::CommSpec& comm_spec, F_T&& f, ARGS_T&&... args) {
  using return_t = typename std::result_of<typename std::decay<F_T>::type(
      typename std::decay<ARGS_T>::type...)>::type;
  auto f_wrapper = [](F_T&& _f, ARGS_T&&... _args) -> return_t {
    try {
      return _f(std::forward<ARGS_T>(_args)...);
    } catch (std::runtime_error& e) {
      return boost::leaf::new_error(
          GSError(ErrorCode::kUnspecificError, e.what()));
    }
  };

  return boost::leaf::try_handle_some(
      [&]() -> return_t {
        auto&& r = f_wrapper(f, args...);
        if (!r) {
          return r.error();
        }
        auto e = all_gather_error(comm_spec);
        if (e) {
          // throw a new error type to prevent invoking AllGatherError twice
          return boost::leaf::new_error(e, std::string());
        }
        return r.value();
      },
      [](const GSError& e, const std::string& dummy) {
        return boost::leaf::new_error(e);
      },
      [&comm_spec](const GSError& e) {
        return boost::leaf::new_error(all_gather_error(e, comm_spec));
      });
}

#define CHECK_OR_RAISE(condition)                    \
  do {                                               \
    if (!(condition)) {                              \
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError, \
                      "Check failed: " #condition);  \
    }                                                \
  } while (0)

#define VY_OK_OR_RAISE(expr)                                                \
  do {                                                                      \
    auto status_name = (expr);                                              \
    if (!(status_name).ok()) {                                              \
      RETURN_GS_ERROR(ErrorCode::kVineyardError, (status_name).ToString()); \
    }                                                                       \
  } while (0)

#define ARROW_OK_OR_RAISE(expr)                                          \
  do {                                                                   \
    auto status_name = (expr);                                           \
    if (!(status_name).ok()) {                                           \
      RETURN_GS_ERROR(ErrorCode::kArrowError, (status_name).ToString()); \
    }                                                                    \
  } while (0)

#define ARROW_OK_ASSIGN_OR_RAISE(lhs, expr)               \
  do {                                                    \
    auto status_name = (expr);                            \
    if (!(status_name).ok()) {                            \
      RETURN_GS_ERROR(ErrorCode::kArrowError,             \
                      (status_name).status().ToString()); \
    }                                                     \
    (lhs) = std::move(status_name).ValueOrDie();          \
  } while (0)

#define ARROW_CHECK_OK(expr)                                     \
  do {                                                           \
    auto status = (expr);                                        \
    if (!status.ok()) {                                          \
      LOG(FATAL) << "Arrow check failed: " << status.ToString(); \
    }                                                            \
  } while (0)

#define ARROW_CHECK_OK_AND_ASSIGN(lhs, expr)                                   \
  do {                                                                         \
    auto status_name = (expr);                                                 \
    if (!status_name.ok()) {                                                   \
      LOG(FATAL) << "Arrow check failed: " << status_name.status().ToString(); \
    }                                                                          \
    (lhs) = std::move(status_name).ValueOrDie();                               \
  } while (0)

}  // namespace vineyard
#endif  // MODULES_GRAPH_UTILS_ERROR_H_
