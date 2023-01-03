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

#ifndef SRC_COMMON_BACKTRACE_BACKTRACE_ON_TERMINATE_HPP_
#define SRC_COMMON_BACKTRACE_BACKTRACE_ON_TERMINATE_HPP_

#include <cxxabi.h>

#include <exception>
#include <iostream>
#include <memory>
#include <type_traits>
#include <typeinfo>

#include "common/backtrace/backtrace.hpp"

namespace vineyard {

#if __cplusplus >= 201703L
[[noreturn]] void backtrace_on_terminate();
#else
[[noreturn]] void backtrace_on_terminate() noexcept;
#endif

static_assert(
    std::is_same<std::terminate_handler, decltype(&backtrace_on_terminate)>{},
    "invalid terminate handler: type is mismatched");

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
std::unique_ptr<std::remove_pointer_t<std::terminate_handler>,
                decltype(std::set_terminate)&>
    terminate_handler{std::set_terminate(backtrace_on_terminate),
                      std::set_terminate};
#pragma clang diagnostic pop

#if __cplusplus >= 201703L
[[noreturn]] void backtrace_on_terminate() {
#else
[[noreturn]] void backtrace_on_terminate() noexcept {
#endif
  std::set_terminate(
      terminate_handler.release());  // to avoid infinite looping if any
  if (std::exception_ptr ep = std::current_exception()) {
    try {
      std::rethrow_exception(ep);
    } catch (std::exception const& e) {
      std::clog << std::endl
                << "Unhandled exception:" << std::endl
                << "  "
                << "std::exception:what(): " << e.what() << std::endl
                << std::endl;
    } catch (...) {
      if (std::type_info* et = abi::__cxa_current_exception_type()) {
        std::unique_ptr<char, decltype(std::free)&> demangled_name{nullptr,
                                                                   std::free};
        size_t demangled_size = 0;
        std::clog << std::endl
                  << "Unhandled exception:" << std::endl
                  << "  "
                  << "exception type: "
                  << backtrace_info::get_demangled_name(
                         et->name(), demangled_name, demangled_size)
                  << std::endl
                  << std::endl;
      } else {
        std::clog << std::endl
                  << "Unhandled exception:" << std::endl
                  << "  "
                  << "unknown exception type" << std::endl
                  << std::endl;
      }
    }
  }
#ifdef WITH_LIBUNWIND
  std::clog << "backtrace:" << std::endl;
  backtrace_info::backtrace(std::clog, true, 2);
#endif
  std::_Exit(EXIT_FAILURE);
}

}  // namespace vineyard

#endif  // SRC_COMMON_BACKTRACE_BACKTRACE_ON_TERMINATE_HPP_
