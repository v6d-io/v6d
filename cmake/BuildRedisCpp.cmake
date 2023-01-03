# Copyright 2020-2023 Alibaba Group Holding Limited.
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

# find libuv
include("${CMAKE_CURRENT_LIST_DIR}/FindLibUV.cmake")
if(NOT LIBUV_FOUND)
    message(FATAL "libuv is required to build bundled redis-plus-plus")
endif()

# build redis-plus-plus
set(REDIS_PLUS_PLUS_BUILD_TEST OFF CACHE BOOL "Don't build tests.")
set(REDIS_PLUS_PLUS_BUILD_SHARED OFF CACHE BOOL "Don't build shared library.")
set(REDIS_PLUS_PLUS_BUILD_ASYNC libuv CACHE STRING "Support async interface.")
set(REDIS_PLUS_PLUS_ASYNC_FUTURE std CACHE STRING "Don't use boost future.")
set(REDIS_PLUS_PLUS_CXX_STANDARD 11 CACHE STRING "Building redis++ with C++11.")
set(REDIS_PLUS_PLUS_USE_TLS OFF CACHE STRING "Don't build with TLS")
add_subdirectory_static(thirdparty/redis-plus-plus EXCLUDE_FROM_ALL)
set(REDIS_PLUS_PLUS_LIBRARIES redis++::redis++_static)
set(REDIS_PLUS_PLUS_INCLUDE_DIR thirdparty/redis-plus-plus/src/sw
                                thirdparty/redis-plus-plus/src/sw/redis++/cxx11
                                thirdparty/redis-plus-plus/src/sw/redis++/future/std
                                thirdparty/redis-plus-plus/src/sw/redis++/no_tls
)

# requires libuv
if(POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW)
endif()

target_link_libraries(redis++_static PRIVATE ${LIBUV_LIBRARIES})
target_include_directories(redis++_static PRIVATE ${LIBUV_INCLUDE_DIRS})
