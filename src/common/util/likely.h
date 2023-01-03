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

#ifndef SRC_COMMON_UTIL_LIKELY_H_
#define SRC_COMMON_UTIL_LIKELY_H_

#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__IBMC__) || \
    defined(__INTEL_COMPILER) || defined(__clang__)
#ifndef unlikely
#define unlikely(x_) __builtin_expect(!!(x_), 0)
#endif
#ifndef likely
#define likely(x_) __builtin_expect(!!(x_), 1)
#endif
#else
#ifndef unlikely
#define unlikely(x_) (x_)
#endif
#ifndef likely
#define likely(x_) (x_)
#endif
#endif

#endif  // SRC_COMMON_UTIL_LIKELY_H_
