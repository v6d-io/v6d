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
include("${CMAKE_CURRENT_LIST_DIR}/FindPythonExecutable.cmake")

set(JAVA_GENERATOR_ARTIFACT "com.alibaba.fastffi:binding-generator:0.1:jar:jar-with-dependencies")
set(JAVA_GENERATOR_JAR "binding-generator-0.1-jar-with-dependencies.jar")
set(JAVA_GENERATOR_JAR_ABS "${PROJECT_BINARY_DIR}/${JAVA_GENERATOR_JAR}")

add_custom_command(
  OUTPUT ${JAVA_GENERATOR_JAR_ABS}
  COMMAND mvn dependency:get "-Dartifact=${JAVA_GENERATOR_ARTIFACT}"
  COMMAND mvn dependency:copy "-Dartifact=${JAVA_GENERATOR_ARTIFACT}" "-DoutputDirectory=${PROJECT_BINARY_DIR}"
  COMMENT "Fetching the java bindings generator")
set_source_files_properties(${JAVA_GENERATOR_JAR_ABS} PROPERTIES GENERATED TRUE)

function(vineyard_generate_java)
  set(_options)
  set(_singleargs OUT_VAR CMAKE_BUILD_DIR VINEYARD_OUT_DIR PACKAGE_DIR ROOT_PACKAGE FFI_LIBRARY_NAME EXCLUDES FORWARDS)
  set(_multiargs VINEYARD_MODULES SYSTEM_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES GENERATE_EXTENSIONS DEPENDS)

  cmake_parse_arguments(vineyard_generate_java "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

  if(NOT vineyard_generate_java_VINEYARD_MODULES)
    message(SEND_ERROR "Error: vineyard_generate called without any source files")
    return()
  endif()

  if(NOT vineyard_generate_java_OUT_VAR)
    message(SEND_ERROR "Error: vineyard_generate called without a output variable")
    return()
  endif()

  if(NOT vineyard_generate_java_CMAKE_BUILD_DIR)
    set(vineyard_generate_java_CMAKE_BUILD_DIR "${CMAKE_BINARY_DIR}")
  endif()

  if(NOT vineyard_generate_java_VINEYARD_OUT_DIR)
    set(vineyard_generate_java_VINEYARD_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  if(NOT vineyard_generate_java_PACKAGE_DIR)
    set(vineyard_generate_java_PACKAGE_DIR "${PROJECT_SOURCE_DIR}/packages-java")
  endif()

  if(NOT vineyard_generate_java_ROOT_PACKAGE)
    set(vineyard_generate_java_ROOT_PACKAGE "io.v6d")
  endif()

  if(NOT vineyard_generate_java_FFI_LIBRARY_NAME)
    set(vineyard_generate_java_FFI_LIBRARY_NAME "_vineyard_java")
  endif()

  if(NOT vineyard_generate_java_EXCLUDES)
    message("excludes.txt = ${vineyard_generate_java_PACKAGE_DIR}/excludes.txt")
    if(EXISTS ${vineyard_generate_java_PACKAGE_DIR}/excludes.txt)
      set(vineyard_generate_java_EXCLUDES "")
      # set(vineyard_generate_java_EXCLUDES ${vineyard_generate_java_PACKAGE_DIR}/excludes.txt)
    else()
      set(vineyard_generate_java_EXCLUDES "")
    endif()
  endif()

  if(NOT vineyard_generate_java_FORWARDS)
    if(EXISTS ${vineyard_generate_java_PACKAGE_DIR}/forward-headers.txt)
      set(vineyard_generate_java_FORWARDS ${vineyard_generate_java_PACKAGE_DIR}/forward-headers.txt)
    else()
      set(vineyard_generate_java_FORWARDS "")
    endif()
  endif()

  if(NOT vineyard_generate_java_SYSTEM_INCLUDE_DIRECTORIES)
    determine_implicit_includes(CXX CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    set(vineyard_generate_java_SYSTEM_INCLUDE_DIRECTORIES ${CXX_IMPLICIT_INCLUDE_DIRECTORIES})
  endif()

  if(NOT vineyard_generate_java_INCLUDE_DIRECTORIES)
    get_property(CXX_EXPLICIT_INCLUDE_DIRECTORIES
                 DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                 PROPERTY INCLUDE_DIRECTORIES)
    set(vineyard_generate_java_INCLUDE_DIRECTORIES ${CXX_EXPLICIT_INCLUDE_DIRECTORIES})
  endif()

  set(vineyard_generate_java_GENERATE_EXTENSIONS .java.touch)

  if(NOT vineyard_generate_java_VINEYARD_MODULES)
    message(SEND_ERROR "Error: vineyard_generate could not find any .vineyard-module files")
    return()
  endif()

  find_python_executable()

  set(_generated_srcs_all)
  foreach(_vineyard_module ${vineyard_generate_java_VINEYARD_MODULES})
    get_filename_component(_abs_file ${_vineyard_module} ABSOLUTE)
    get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
    get_filename_component(_basename ${_vineyard_module} NAME_WLE)
    get_filename_component(_baseext ${_vineyard_module} LAST_EXT)
    file(RELATIVE_PATH _rel_dir ${CMAKE_CURRENT_SOURCE_DIR} ${_abs_dir})

    set(_possible_rel_dir ${_rel_dir}/)

    set(_generated_srcs)
    foreach(_ext ${vineyard_generate_java_GENERATE_EXTENSIONS})
      list(APPEND _generated_srcs "${vineyard_generate_java_VINEYARD_OUT_DIR}/${_possible_rel_dir}${_basename}${_ext}")
    endforeach()

    list(APPEND _generated_srcs_all ${_generated_srcs})

    file(GLOB _codegen_scripts "${PROJECT_SOURCE_DIR}/python/vineyard/core/codegen/*.py")
    add_custom_command(
      OUTPUT ${_generated_srcs}
      COMMAND ${CMAKE_COMMAND}
      ARGS -E
      ARGS env
      ARGS FFI_BINDING_GENERATOR=${JAVA_GENERATOR_JAR_ABS}
      ARGS "${PYTHON_EXECUTABLE}"
      ARGS -m
      ARGS codegen
      ARGS --root-directory "${PROJECT_SOURCE_DIR}"
      ARGS --system-includes "${vineyard_generate_java_SYSTEM_INCLUDE_DIRECTORIES}"
      ARGS --includes "${vineyard_generate_java_INCLUDE_DIRECTORIES}"
      ARGS --build-directory "${vineyard_generate_java_CMAKE_BUILD_DIR}"
      ARGS --source ${_abs_file}
      ARGS --target ${_generated_srcs}
      ARGS --langauge java
      ARGS --package ${vineyard_generate_java_PACKAGE_DIR}/src/main/java
      ARGS --package-name ${vineyard_generate_java_ROOT_PACKAGE}
      ARGS --ffilibrary-name ${vineyard_generate_java_FFI_LIBRARY_NAME}
      ARGS --excludes "${vineyard_generate_java_EXCLUDES}"
      ARGS --forwards "${vineyard_generate_java_FORWARDS}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/python/vineyard/core/"
      DEPENDS ${JAVA_GENERATOR_DIST}
              ${_codegen_scripts}
              ${_abs_file}
              ${_generated_srcs_depends_all}
              ${vineyard_generate_java_DEPENDS}
      IMPLICIT_DEPENDS CXX ${_abs_file}
      COMMENT "Running java vineyard module compiler on ${_vineyard_module}"
      VERBATIM)
  endforeach()

  set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
  if(vineyard_generate_java_OUT_VAR)
    set(${vineyard_generate_java_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
  endif()
endfunction()
