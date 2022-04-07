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
#
# A cmake function to help users use vineyard's code genenrator.
#

macro(find_python_executable)
  if(NOT DEFINED PYTHON_EXECUTABLE)
    if(DEFINED ENV{VIRTUAL_ENV})
      find_program(
        PYTHON_EXECUTABLE python
        PATHS "$ENV{VIRTUAL_ENV}" "$ENV{VIRTUAL_ENV}/bin"
        NO_DEFAULT_PATH)
    elseif(DEFINED ENV{CONDA_PREFIX})
      find_program(
        PYTHON_EXECUTABLE python
        PATHS "$ENV{CONDA_PREFIX}" "$ENV{CONDA_PREFIX}/bin"
        NO_DEFAULT_PATH)
    elseif(DEFINED ENV{pythonLocation})
      find_program(
        PYTHON_EXECUTABLE python
        PATHS "$ENV{pythonLocation}" "$ENV{pythonLocation}/bin"
        NO_DEFAULT_PATH)
    else()
      set(PYBIND11_PYTHON_VERSION 3)
      find_package(PythonInterp)
    endif()
    if(NOT PYTHON_EXECUTABLE)
      message(FATAL_ERROR "Failed to find a valid python interpreter, try speicifying `PYTHON_EXECUTABLE` instead")
    endif()
    message(STATUS "Use Python executable: ${PYTHON_EXECUTABLE}")
  endif()
endmacro()
