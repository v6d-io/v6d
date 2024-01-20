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

#ifndef SRC_COMMON_UTIL_MACROS_H_
#define SRC_COMMON_UTIL_MACROS_H_

namespace vineyard {

#define VINEYARD_STRINGIFY(x) #x

#define VINEYARD_TO_STRING(x) VINEYARD_STRINGIFY(x)

#if (defined(__GNUC__) || defined(__APPLE__))
#define VINEYARD_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define VINEYARD_MUST_USE_RESULT
#endif

#if defined(__clang__)
#define VINEYARD_MUST_USE_TYPE VINEYARD_MUST_USE_RESULT
#else
#define VINEYARD_MUST_USE_TYPE
#endif

#ifndef GET_MACRO
#define GET_MACRO(_1, _2, NAME, ...) NAME
#endif

#ifndef GET_MACRO2
#define GET_MACRO2(_1, _2, _3, NAME, ...) NAME
#endif

#ifndef GET_MACRO3
#define GET_MACRO3(_1, _2, _3, _4, NAME, ...) NAME
#endif

// Allow comma in macro's argument, see also:
// https://stackoverflow.com/a/13842612/5080177
#define ARG(...) __VA_ARGS__

}  // namespace vineyard
#endif  // SRC_COMMON_UTIL_MACROS_H_
