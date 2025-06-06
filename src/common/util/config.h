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

#ifndef SRC_COMMON_UTIL_CONFIG_H_
#define SRC_COMMON_UTIL_CONFIG_H_

#define VINEYARD_VERSION_MAJOR 0
#define VINEYARD_VERSION_MINOR 24
#define VINEYARD_VERSION_PATCH 4

#define VINEYARD_VERSION                                              \
  ((VINEYARD_VERSION_MAJOR * 1000) + VINEYARD_VERSION_MINOR) * 1000 + \
      VINEYARD_VERSION_PATCH
#define VINEYARD_VERSION_STRING "0.24.4"

#endif  // SRC_COMMON_UTIL_CONFIG_H_
