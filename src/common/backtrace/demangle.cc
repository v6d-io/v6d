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

#ifdef WITH_LIBUNWIND

#include "demangle.hpp"

#include <cxxabi.h>

#include <memory>

namespace {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
std::unique_ptr<char, decltype(std::free)&> demangled_name{nullptr, std::free};
#pragma clang diagnostic pop

}  // namespace

char const* get_demangled_name(char const* const symbol) noexcept {
  if (!symbol) {
    return "<null>";
  }
  int status = -4;
  demangled_name.reset(
      abi::__cxa_demangle(symbol, demangled_name.release(), nullptr, &status));
  return ((status == 0) ? demangled_name.get() : symbol);
}

#endif
