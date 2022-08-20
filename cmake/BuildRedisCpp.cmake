# Copyright 2020-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tell redis-plus-plus that the hiredis has been installed
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

# build redis-plus-plus
set(REDIS_PLUS_PLUS_BUILD_TEST OFF CACHE BOOL "Build tests.")
set(REDIS_PLUS_PLUS_BUILD_ASYNC libuv CACHE STRING "Support async interface.")
set(REDIS_PLUS_PLUS_ASYNC_FUTURE boost CACHE STRING "Use boost future.")
add_subdirectory_static(thirdparty/redis-plus-plus EXCLUDE_FROM_ALL)
set(REDIS_PLUS_PLUS_LIBRARIES redis++)
set(REDIS_PLUS_PLUS_INCLUDE_DIR thirdparty/redis-plus-plus/src/sw)
