cmake_minimum_required(VERSION 3.3)

if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()
project(vineyard LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
set(PLATFORM_NATIVE_LIBRARY_PATH "" CACHE STRING "Target directory for JNI native libraries")
set(PLATFORM_NATIVE_LIBRARY_VERSION "" CACHE STRING "Version string for JNI native libraries, e.g., 0.1-SNAPSHOT")

if("${PLATFORM_NATIVE_LIBRARY_PATH} " STREQUAL " ")
    message(FATAL_ERROR "The property 'PLATFORM_NATIVE_LIBRARY_PATH' is not set")
endif()

include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include(ExternalProject)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(ProcessorCount)
ProcessorCount(N)

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
        STRING "Choose the type of build." FORCE
    )
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
    )
else()
    message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}'.")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND
        ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
            "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    # avoid the llvm-strip: error: unsupported load command (cmd=0x80000034) error
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if(NOT (CMAKE_CXX_COMPILER_LAUNCHER MATCHES "ccache") AND NOT (CMAKE_C_COMPILER_LAUNCHER MATCHES "ccache"))
    find_program(ccache_EXECUTABLE ccache)
    if(ccache_EXECUTABLE)
        set(CMAKE_C_COMPILER_LAUNCHER ${ccache_EXECUTABLE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${ccache_EXECUTABLE})
        add_custom_target(ccache-stats
            COMMAND ${ccache_EXECUTABLE} --show-stats
        )
    else()
        add_custom_target(ccache-stats
            COMMAND echo "ccache not found."
        )
    endif(ccache_EXECUTABLE)
endif()

# enable colored diagnostics
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    add_compile_options(-fcolor-diagnostics)
endif()

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
if(APPLE)
    # the LC_RPATH on Mac seems doesn't support multiple path (separated with `:`)
    # fortunately, we just need to take care `lib` on Mac.
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
else()
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64:${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC \
                                        -Wall \
                                        -Wno-attributes \
                                        -Wno-unknown-pragmas \
                                        -Wno-unused-variable \
                                        -Wno-deprecated-declarations"
)

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/../lib")
endif()

check_cxx_compiler_flag(-std=c++17 HAVE_FLAG_STD_CXX17)
check_cxx_compiler_flag(-std=c++14 HAVE_FLAG_STD_CXX14)
if(HAVE_FLAG_STD_CXX17)
    set(CMAKE_CXX_STANDARD 17)
elseif(HAVE_FLAG_STD_CXX14)
    set(CMAKE_CXX_STANDARD 14)
else()
    set(CMAKE_CXX_STANDARD 11)
endif()

if(CMAKE_VERSION VERSION_LESS "3.1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
endif()

# find vineyard
find_package(vineyard REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/static-lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/shared-lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

macro(target_add_link_options target scope)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs OPTIONS)
    cmake_parse_arguments(target_add_link_options "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(${CMAKE_VERSION} VERSION_LESS "3.13")
        target_link_libraries(${target} INTERFACE ${target_add_link_options_OPTIONS})
    else()
        target_link_options(${target} ${scope} ${target_add_link_options_OPTIONS})
    endif()
endmacro()

# JNI
find_package(JNI COMPONENTS JVM REQUIRED)
include_directories(SYSTEM ${JAVA_INCLUDE_PATH})
include_directories(SYSTEM ${JAVA_INCLUDE_PATH2})

# generated JNI stuffs
set(GENERATED_JNI_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/target/jni/javah-include)

# build the vineyard_core_jni library
file(GLOB_RECURSE CLIENT_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/main/cpp/*.cc")
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)

add_library(vineyard-core_jni ${CLIENT_SRC_FILES})
set_target_properties(vineyard-core_jni PROPERTIES OUTPUT_NAME vineyard-core_jni-${PLATFORM_NATIVE_LIBRARY_VERSION})
target_link_libraries(vineyard-core_jni PUBLIC ${JAVA_JVM_LIBRARY}
                                               ${JNI_LIBRARIES}
)
target_include_directories(vineyard-core_jni PRIVATE ${VINEYARD_INCLUDE_DIRS}
                                                     ${GENERATED_JNI_INCLUDE_DIRS})

add_custom_target(vineyard-core_jni_resources
        ALL
        COMMAND mkdir -p "${PLATFORM_NATIVE_LIBRARY_PATH}"
        COMMAND cp "$<TARGET_FILE:vineyard-core_jni>" "${PLATFORM_NATIVE_LIBRARY_PATH}"
        DEPENDS vineyard-core_jni
        COMMENT "Copying vineyard core JNI libraries."
        VERBATIM)

file(GLOB_RECURSE FILES_NEED_FORMAT "src/main/cpp/*.cc" "src/main/cpp/*.h")
add_custom_target(vineyard_clformat
        COMMAND clang-format --style=file -i ${FILES_NEED_FORMAT}
        COMMENT "Running clang-format."
        VERBATIM)
