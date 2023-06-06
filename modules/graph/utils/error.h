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

#ifndef MODULES_GRAPH_UTILS_ERROR_H_
#define MODULES_GRAPH_UTILS_ERROR_H_

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(__has_include) && __has_include(<version>)
#include <version>
#endif

#include "boost/leaf.hpp"

#include "common/backtrace/backtrace.hpp"
#include "graph/utils/mpi_utils.h"

#ifdef __cpp_lib_is_invocable
template <class T, typename... Args>
using result_of_t = std::invoke_result_t<T, Args...>;
#else
template <class T, typename... Args>
using result_of_t = typename std::result_of<T(Args...)>::type;
#endif

namespace vineyard {

enum class ErrorCode {
  kOk,
  kIOError,
  kArrowError,
  kVineyardError,
  kUnspecificError,
  kDistributedError,
  kNetworkError,
  kCommandError,
  kDataTypeError,
  kIllegalStateError,
  kInvalidValueError,
  kInvalidOperationError,
  kUnsupportedOperationError,
  kUnimplementedMethod,
  kGraphArError,
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
  case ErrorCode::kDistributedError:
    return "DistributedError";
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
  case ErrorCode::kGraphArError:
    return "GraphArError";
  default:
    return "UndefinedErrorCode";
  }
}

struct GSError {
  ErrorCode error_code;
  std::string error_msg;
  std::string backtrace;
  GSError() : error_code(ErrorCode::kOk) {}

  explicit GSError(ErrorCode code) : GSError(code, "") {}

  GSError(ErrorCode code, std::string msg)
      : error_code(code), error_msg(std::move(msg)) {}

  GSError(ErrorCode code, std::string msg, std::string bt)
      : error_code(code), error_msg(std::move(msg)), backtrace(std::move(bt)) {}

  explicit operator bool() const { return error_code != ErrorCode::kOk; }

  bool ok() const { return error_code == ErrorCode::kOk; }
};

inline grape::InArchive& operator<<(grape::InArchive& archive,
                                    const GSError& e) {
  archive << e.error_code;
  archive << e.error_msg;
  archive << e.backtrace;
  return archive;
}

inline grape::OutArchive& operator>>(grape::OutArchive& archive, GSError& e) {
  archive >> e.error_code;
  archive >> e.error_msg;
  archive >> e.backtrace;
  return archive;
}

#define TOKENPASTE(x, y) x##y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)

#ifndef RETURN_GS_ERROR
#define RETURN_GS_ERROR(code, msg)                                         \
  do {                                                                     \
    std::stringstream TOKENPASTE2(_ss, __LINE__);                          \
    vineyard::backtrace_info::backtrace(TOKENPASTE2(_ss, __LINE__), true); \
    return ::boost::leaf::new_error(vineyard::GSError(                     \
        (code),                                                            \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " +    \
            std::string(__FUNCTION__) + " -> " + (msg),                    \
        TOKENPASTE2(_ss, __LINE__).str()));                                \
  } while (0)
#endif

#ifndef BOOST_LEAF_ASSIGN
#define BOOST_LEAF_ASSIGN(v, r)                                     \
  static_assert(::boost::leaf::is_result_type<                      \
                    typename std::decay<decltype(r)>::type>::value, \
                "BOOST_LEAF_ASSIGN requires a result type");        \
  auto&& TOKENPASTE2(_leaf_r, __LINE__) = r;                        \
  if (!TOKENPASTE2(_leaf_r, __LINE__))                              \
    return TOKENPASTE2(_leaf_r, __LINE__).error();                  \
  v = TOKENPASTE2(_leaf_r, __LINE__).value()
#endif

inline GSError all_gather_error(const GSError& e,
                                const grape::CommSpec& comm_spec) {
  std::stringstream ss;
  ss << ErrorCodeToString(e.error_code) << " occurred on worker "
     << comm_spec.worker_id();
  ss << ": " << e.error_msg;

  std::vector<GSError> error_objs(comm_spec.worker_num());
  GlobalAllGatherv<GSError>(const_cast<GSError&>(e), error_objs, comm_spec);

  return {e.error_code, ss.str(), e.backtrace};
}

inline GSError all_gather_error(const grape::CommSpec& comm_spec) {
  std::vector<GSError> error_objs(comm_spec.worker_num());
  GSError ok;

  GlobalAllGatherv<GSError>(ok, error_objs, comm_spec);

  // Return a distributed error, the error messages will be aggregated by
  // upstream
  for (auto& e : error_objs) {
    if (!e.ok()) {
      return {ErrorCode::kDistributedError, e.error_msg, e.backtrace};
    }
  }
  return {ErrorCode::kOk, ""};
}

template <class F_T, class... ARGS_T>
inline result_of_t<typename std::decay<F_T>::type,
                   typename std::decay<ARGS_T>::type...>
sync_gs_error(const grape::CommSpec& comm_spec, F_T&& f, ARGS_T&&... args) {
  using return_t = result_of_t<typename std::decay<F_T>::type,
                               typename std::decay<ARGS_T>::type...>;
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
        // We have to return here. Then the concrete error object(GSError) will
        // be caught by the third lambda.
        if (!r) {  // throw #1
          return r.error();
        }
        auto e = all_gather_error(comm_spec);
        if (e) {
          // throw a new error type to prevent invoking all_gather_error twice
          return boost::leaf::new_error(e, std::string());  // throw #2
        }
        return r.value();
      },
      [](const GSError& e, const std::string& dummy) {  // catch #2
        return boost::leaf::new_error(e);
      },
      [&comm_spec](const GSError& e) {  // catch #1
        return boost::leaf::new_error(all_gather_error(e, comm_spec));
      });
}

#define CHECK_OR_RAISE(condition)                              \
  do {                                                         \
    if (!(condition)) {                                        \
      RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError, \
                      "Check failed: " #condition);            \
    }                                                          \
  } while (0)

#define VY_OK_OR_RAISE(expr)                               \
  do {                                                     \
    auto status_name = (expr);                             \
    if (!(status_name).ok()) {                             \
      RETURN_GS_ERROR(vineyard::ErrorCode::kVineyardError, \
                      (status_name).ToString());           \
    }                                                      \
  } while (0)

#define ARROW_OK_OR_RAISE(expr)                         \
  do {                                                  \
    auto status_name = (expr);                          \
    if (!(status_name).ok()) {                          \
      RETURN_GS_ERROR(vineyard::ErrorCode::kArrowError, \
                      (status_name).ToString());        \
    }                                                   \
  } while (0)

#define ARROW_OK_ASSIGN_OR_RAISE(lhs, expr)               \
  do {                                                    \
    auto status_name = (expr);                            \
    if (!(status_name).ok()) {                            \
      RETURN_GS_ERROR(vineyard::ErrorCode::kArrowError,   \
                      (status_name).status().ToString()); \
    }                                                     \
    (lhs) = std::move(status_name).ValueOrDie();          \
  } while (0)

#ifndef ARROW_CHECK_OK
#define ARROW_CHECK_OK(expr)                                     \
  do {                                                           \
    auto status = (expr);                                        \
    if (!status.ok()) {                                          \
      LOG(FATAL) << "Arrow check failed: " << status.ToString(); \
    }                                                            \
  } while (0)
#endif  // ARROW_CHECK_OK

#ifndef ARROW_CHECK_OK_AND_ASSIGN
#define ARROW_CHECK_OK_AND_ASSIGN(lhs, expr)                                   \
  do {                                                                         \
    auto status_name = (expr);                                                 \
    if (!status_name.ok()) {                                                   \
      LOG(FATAL) << "Arrow check failed: " << status_name.status().ToString(); \
    }                                                                          \
    (lhs) = std::move(status_name).ValueOrDie();                               \
  } while (0)
#endif  // ARROW_CHECK_OK_AND_ASSIGN

}  // namespace vineyard
#endif  // MODULES_GRAPH_UTILS_ERROR_H_
