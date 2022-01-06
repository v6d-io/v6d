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

include("${CMAKE_CURRENT_LIST_DIR}/DetermineImplicitIncludes.cmake")
determine_implicit_includes(CXX CXX_IMPLICIT_INCLUDE_DIRECTORIES)

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
  endif()
endmacro()

function(vineyard_generate)
  set(_options)
  set(_singleargs LANGUAGE OUT_VAR VINEYARD_OUT_DIR CMAKE_BUILD_DIR)
  set(_multiargs VINEYARD_MODULES SYSTEM_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES GENERATE_EXTENSIONS)

  cmake_parse_arguments(vineyard_generate "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

  if(NOT vineyard_generate_VINEYARD_MODULES)
    message(SEND_ERROR "Error: vineyard_generate called without any source files")
    return()
  endif()

  if(NOT vineyard_generate_OUT_VAR)
    message(SEND_ERROR "Error: vineyard_generate called without a output variable")
    return()
  endif()

  if(NOT vineyard_generate_LANGUAGE)
    set(vineyard_generate_LANGUAGE cpp)
  endif()
  string(TOLOWER ${vineyard_generate_LANGUAGE} vineyard_generate_LANGUAGE)

  if(NOT vineyard_generate_CMAKE_BUILD_DIR)
    set(vineyard_generate_CMAKE_BUILD_DIR "${CMAKE_BINARY_DIR}")
  endif()

  if(NOT vineyard_generate_VINEYARD_OUT_DIR)
    set(vineyard_generate_VINEYARD_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  if(NOT vineyard_generate_SYSTEM_INCLUDE_DIRECTORIES)
    set(vineyard_generate_SYSTEM_INCLUDE_DIRECTORIES ${CXX_IMPLICIT_INCLUDE_DIRECTORIES})
  endif()

  if(NOT vineyard_generate_INCLUDE_DIRECTORIES)
    get_property(CXX_EXPLICIT_INCLUDE_DIRECTORIES
                 DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                 PROPERTY INCLUDE_DIRECTORIES)
    set(vineyard_generate_INCLUDE_DIRECTORIES ${CXX_EXPLICIT_INCLUDE_DIRECTORIES})
  endif()

  if(NOT vineyard_generate_GENERATE_EXTENSIONS)
    if(vineyard_generate_LANGUAGE STREQUAL cpp)
      set(vineyard_generate_GENERATE_EXTENSIONS .vineyard.h)
    elseif(vineyard_generate_LANGUAGE STREQUAL python)
      set(vineyard_generate_GENERATE_EXTENSIONS _vineyard.py)
    else()
      message(SEND_ERROR "Error: vineyard_generate given unknown Language ${LANGUAGE}, please provide a value for GENERATE_EXTENSIONS")
      return()
    endif()
  endif()

  if(NOT vineyard_generate_VINEYARD_MODULES)
    message(SEND_ERROR "Error: vineyard_generate could not find any .vineyard-module files")
    return()
  endif()

  find_python_executable()
  message(STATUS "Use Python executable: ${PYTHON_EXECUTABLE}")

  set(_generated_srcs_all)
  foreach(_vineyard_module ${vineyard_generate_VINEYARD_MODULES})
    get_filename_component(_abs_file ${_vineyard_module} ABSOLUTE)
    get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
    get_filename_component(_basename ${_vineyard_module} NAME_WE)
    file(RELATIVE_PATH _rel_dir ${CMAKE_CURRENT_SOURCE_DIR} ${_abs_dir})

    set(_possible_rel_dir ${_rel_dir}/)

    set(_generated_srcs)
    foreach(_ext ${vineyard_generate_GENERATE_EXTENSIONS})
      list(APPEND _generated_srcs "${vineyard_generate_VINEYARD_OUT_DIR}/${_possible_rel_dir}${_basename}${_ext}")
    endforeach()

    list(APPEND _generated_srcs_all ${_generated_srcs})

    # parse dependencies
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}"
              -m
              codegen
              --dump-dependencies "True"
              --root-directory "${CMAKE_CURRENT_SOURCE_DIR}"
              --system-includes "${vineyard_generate_SYSTEM_INCLUDE_DIRECTORIES}"
              --includes "${vineyard_generate_INCLUDE_DIRECTORIES}"
              --build-directory "${vineyard_generate_CMAKE_BUILD_DIR}"
              --source ${_abs_file}
              --target ${_generated_srcs}
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/python/vineyard/core/"
      OUTPUT_VARIABLE DEPS_OUTPUT
      ERROR_VARIABLE DEPS_ERROR
      RESULT_VARIABLE CODEGEN_EXIT_CODE
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(DEPS_OUTPUT MATCHES ".*pip3\ install.*" OR DEPS_ERROR MATCHES ".*pip3\ install.*" OR NOT CODEGEN_EXIT_CODE EQUAL 0)
      message(FATAL_ERROR "${DEPS_OUTPUT} ${DEPS_ERROR}")
    endif()

    string(REGEX REPLACE "\r*\n" ";" output_lines "${DEPS_OUTPUT}")
    set(_generated_srcs_depends_all)
    if(output_lines)
      foreach(line in ${output_lines})
        if(${line} MATCHES "Depends:.+")
          string(SUBSTRING ${line} 8 -1 inc_path)
          list(APPEND _generated_srcs_depends_all ${inc_path})
        endif()
      endforeach()
    endif()

    file(GLOB _codegen_scripts "${PROJECT_SOURCE_DIR}/python/vineyard/core/codegen/*.py")
    add_custom_command(
      OUTPUT ${_generated_srcs}
      COMMAND "${PYTHON_EXECUTABLE}"
      ARGS -m
      ARGS codegen
      ARGS --root-directory "${CMAKE_CURRENT_SOURCE_DIR}"
      ARGS --system-includes "${vineyard_generate_SYSTEM_INCLUDE_DIRECTORIES}"
      ARGS --includes "${vineyard_generate_INCLUDE_DIRECTORIES}"
      ARGS --build-directory "${vineyard_generate_CMAKE_BUILD_DIR}"
      ARGS --source ${_abs_file}
      ARGS --target ${_generated_srcs}
      ARGS --verbose
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/python/vineyard/core/"
      DEPENDS ${_codegen_scripts}
              ${_abs_file}
              ${_generated_srcs_depends_all}
      IMPLICIT_DEPENDS CXX ${_abs_file}
      COMMENT "Running ${vineyard_generate_LANGUAGE} vineyard module compiler on ${_vineyard_module}"
      VERBATIM)
  endforeach()

  set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
  if(vineyard_generate_OUT_VAR)
    set(${vineyard_generate_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
  endif()
endfunction()
