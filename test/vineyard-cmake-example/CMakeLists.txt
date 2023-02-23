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

cmake_minimum_required(VERSION 3.3)

project(vineyard-cmake-example LANGUAGES C CXX)

find_package(vineyard REQUIRED)

add_executable(vineyard-cmake-example ${CMAKE_CURRENT_SOURCE_DIR}/example.cc)
target_include_directories(vineyard-cmake-example PRIVATE ${VINEYARD_INCLUDE_DIRS})
target_link_libraries(vineyard-cmake-example PRIVATE ${VINEYARD_LIBRARIES})
