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

#ifndef WITHOUT_LIBUNWIND

#pragma once

#include "demangle.hpp"
#include "backtrace.hpp"

#include <iostream>
#include <type_traits>
#include <exception>
#include <memory>
#include <typeinfo>

#include <cxxabi.h>

namespace
{

[[noreturn]]
void
backtrace_on_terminate() noexcept;

static_assert(std::is_same< std::terminate_handler, decltype(&backtrace_on_terminate) >{});

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
std::unique_ptr< std::remove_pointer_t< std::terminate_handler >, decltype(std::set_terminate) & > terminate_handler{std::set_terminate(backtrace_on_terminate), std::set_terminate};
#pragma clang diagnostic pop

[[noreturn]]
void
backtrace_on_terminate() noexcept
{
    std::set_terminate(terminate_handler.release()); // to avoid infinite looping if any
    backtrace(std::clog);
    if (std::exception_ptr ep = std::current_exception()) {
        try {
            std::rethrow_exception(ep);
        } catch (std::exception const & e) {
            std::clog << "backtrace: unhandled exception std::exception:what(): " << e.what() << std::endl;
        } catch (...) {
            if (std::type_info * et = abi::__cxa_current_exception_type()) {
                std::clog << "backtrace: unhandled exception type: " << get_demangled_name(et->name()) << std::endl;
            } else {
                std::clog << "backtrace: unhandled unknown exception" << std::endl;
            }
        }
    }
    std::_Exit(EXIT_FAILURE);
}

}

#endif

