cmake_minimum_required(VERSION 3.3)

set(VINEYARD_MAJOR_VERSION 0)
set(VINEYARD_MINOR_VERSION 21)
set(VINEYARD_PATCH_VERSION 5)
set(VINEYARD_VERSION ${VINEYARD_MAJOR_VERSION}.${VINEYARD_MINOR_VERSION}.${VINEYARD_PATCH_VERSION})

message(STATUS "Configuring and building vineyard version '${VINEYARD_VERSION}'.")

if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0025 NEW)
endif()
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
endif()
if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
endif()
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
    set(CMAKE_POLICY_DEFAULT_CMP0104 OLD)
endif()
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
    set(CMAKE_POLICY_DEFAULT_CMP0146 OLD)
endif()

project(vineyard LANGUAGES C CXX VERSION ${VINEYARD_VERSION})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(USE_STATIC_BOOST_LIBS "Build with static-linked boost libraries" OFF)
option(USE_EXTERNAL_ETCD_LIBS "Build with external etcd-cpp-apiv3 library rather than the submodule one" OFF)
option(USE_EXTERNAL_REDIS_LIBS "Build with external redis-plus-plus library rather than the submodule one" ON)
option(USE_EXTERNAL_HIREDIS_LIBS "Build with external hiredis library rather than the submodule one" ON)
option(VINEYARD_USE_ASAN "Using address sanitizer to check memory accessing" OFF)
option(VINEYARD_USE_LTO "Using IPO/LTO support for link-time optimization" OFF)
option(USE_LIBUNWIND "Using libunwind to retrieve the stack backtrace when exception occurs" ON)
option(USE_INCLUDE_WHAT_YOU_USE "Simply the intra-module dependencies with iwyu" OFF)
option(USE_JSON_DIAGNOSTICS "Using json diagnostics to check the validity of metadata" OFF)
option(USE_CUDA "Enabling GPU (CUDA) support" OFF)

option(BUILD_VINEYARD_SERVER "Build vineyard's server" ON)
option(BUILD_VINEYARD_SERVER_REDIS "Enable redis as the metadata backend" OFF)
option(BUILD_VINEYARD_SERVER_ETCD "Enable redis as the metadata backend" ON)
option(BUILD_VINEYARD_SERVER_SPILLING "Enable spilling functionalities in vineyardd" ON)
option(BUILD_VINEYARD_CLIENT "Build vineyard's client" ON)
option(BUILD_VINEYARD_CLIENT_VERBOSE "Build vineyard's client with the most verbose logs" OFF)
option(BUILD_VINEYARD_PYTHON_BINDINGS "Build vineyard's python bindings" ON)
option(BUILD_VINEYARD_PYPI_PACKAGES "Build vineyard's python bindings" OFF)

option(BUILD_VINEYARD_JAVA "Build vineyard's java SDK" OFF)

option(BUILD_VINEYARD_BASIC "Build vineyard's basic data structures" ON)
option(BUILD_VINEYARD_IO "Enable vineyard's IOAdaptor support" ON)
option(BUILD_VINEYARD_GRAPH "Enable vineyard's graph data structures" ON)
option(BUILD_VINEYARD_GRAPH_WITH_GAR "Building vineyard's graph data with GraphAr support" OFF)
option(BUILD_VINEYARD_MALLOC "Build vineyard's implementation for client-side malloc" ON)
option(BUILD_VINEYARD_MALLOC_OVERRIDE "Using the client-side malloc to override the default ones" OFF)
option(BUILD_VINEYARD_FUSE "Enable vineyard's fuse support" OFF)
option(BUILD_VINEYARD_FUSE_PARQUET "Enable vineyard's fuse parquet support" OFF)
option(BUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME "Enable hosseinmoein dataframe support" OFF)
option(BUILD_VINEYARD_LLM_CACHE "Enable kv-state cache support" ON)

option(BUILD_VINEYARD_TESTS "Generate make targets for vineyard tests" ON)
option(BUILD_VINEYARD_TESTS_ALL "Include make targets for vineyard tests to ALL" OFF)
option(BUILD_VINEYARD_BENCHMARKS "Generate make targets for vineyard benchmarks" ON)
option(BUILD_VINEYARD_BENCHMARKS_ALL "Include make targets for vineyard benchmarks to ALL" OFF)
option(BUILD_VINEYARD_COVERAGE "Build vineyard with coverage information, requires build with Debug" OFF)
option(BUILD_VINEYARD_PROFILING "Build vineyard with profiling information" OFF)

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
        ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    # avoid the llvm-strip: error: unsupported load command (cmd=0x80000034) error
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if(VINEYARD_USE_LTO AND CMAKE_VERSION VERSION_GREATER "3.9" AND
        (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lpo_supported OUTPUT lpo_supported_error)
    if(lpo_supported)
        message(STATUS "IPO / LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO / LTO not supported: '${lpo_supported_error}'")
    endif()
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
else()
    add_custom_target(ccache-stats
        COMMAND ${ccache_EXECUTABLE} --show-stats
    )
endif()

# enable colored diagnostics
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    add_compile_options(-fcolor-diagnostics)
endif()

if(USE_INCLUDE_WHAT_YOU_USE AND NOT CMAKE_CXX_INCLUDE_WHAT_YOU_USE)
    find_program(iwyu_EXECUTABLE include-what-you-use)
    if(iwyu_EXECUTABLE)
        set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${iwyu_EXECUTABLE})
    else()
        message(WARNING "Failed to locate iwyu, please specify with -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE instead")
    endif()
endif()

set(DEFAULT_ALLOCATOR "dlmalloc")
set(WITH_ALLOCATOR "${DEFAULT_ALLOCATOR}" CACHE
    STRING "Choose the allocator for vineyard server (vineyardd)."
)
set_property(CACHE WITH_ALLOCATOR PROPERTY STRINGS
    "dlmalloc" "mimalloc"
)
message(STATUS "Setting the allocator to vineyardd as '${WITH_ALLOCATOR}'.")

# check the cxx abi
include("cmake/CheckGCCABI.cmake")
check_gcc_cxx11abi()

get_directory_property(CURRENT_DEFINITIONS COMPILE_DEFINITIONS)
if(NOT ("${CURRENT_DEFINITIONS} ${CMAKE_CXX_FLAGS}" MATCHES ".*_GLIBCXX_USE_CXX11_ABI.*"))
    if(GCC_USE_CXX11_ABI EQUAL 1)
       add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
    endif()
endif()

# disables messages from sub_directory, see also: https://stackoverflow.com/a/38983571/5080177
function(message)
    if (NOT MESSAGE_QUIET)
        _message(${ARGN})
    endif()
endfunction()

message(STATUS "Vineyard will be installed to: ${CMAKE_INSTALL_PREFIX}")

# reference: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath
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
                                        -Wno-unknown-pragmas"
)

check_cxx_compiler_flag(-Wno-class-conversion W_NO_CLASS_CONVERSION)
check_cxx_compiler_flag(-Wno-class-memaccess W_NO_CLASS_MEMACCESS)
check_cxx_compiler_flag(-Wno-defaulted-function-deleted W_NO_DEFAULTED_FUNCTION_DELETED)
check_cxx_compiler_flag(-Wno-deprecated-declarations W_NO_DEPRECATED_DECLARATIONS)
check_cxx_compiler_flag(-Wno-error=c++11-narrowing W_NO_ERROR_CXX11_NARROWING)
check_cxx_compiler_flag(-Wno-format-truncation W_NO_FORMAT_TRUNCATION)
check_cxx_compiler_flag(-Wno-noexcept-type W_NO_NOEXCEPT_TYPE)
check_cxx_compiler_flag(-Wno-unused-but-set-parameter W_NO_UNUSED_BUT_SET_PARAMETER)
check_cxx_compiler_flag(-Wno-unused-private-field W_NO_UNUSED_PRIVATE_FIELD)
check_cxx_compiler_flag(-Wno-unknown-warning-option W_NO_UNKNOWN_WARNING_OPTION)
check_cxx_compiler_flag(-Wno-enum-constexpr-conversion W_NO_ENUM_CONSTEXPR_CONVERSION)
if(W_NO_CLASS_CONVERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-conversion")
endif()
if(W_NO_CLASS_MEMACCESS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-memaccess")
endif()
if(W_NO_DEFAULTED_FUNCTION_DELETED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-defaulted-function-deleted")
endif()
if(W_NO_DEPRECATED_DECLARATIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()
if(W_NO_ERROR_CXX11_NARROWING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=c++11-narrowing")
endif()
if(W_NO_FORMAT_TRUNCATION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation")
endif()
if(W_NO_NOEXCEPT_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-noexcept-type")
endif()
if(W_NO_UNUSED_BUT_SET_PARAMETER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-parameter")
endif()
if(W_NO_UNUSED_PRIVATE_FIELD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
endif()
if(W_NO_UNKNOWN_WARNING_OPTION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
endif()
if(W_NO_ENUM_CONSTEXPR_CONVERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-enum-constexpr-conversion")
endif()

if(BUILD_VINEYARD_COVERAGE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
endif()

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/../shared-lib")
endif()

if(NOT "${CMAKE_CXX_STANDARD}")
    check_cxx_compiler_flag(-std=c++14 HAVE_FLAG_STD_CXX14)
    check_cxx_compiler_flag(-std=c++17 HAVE_FLAG_STD_CXX17)
    if(HAVE_FLAG_STD_CXX17)
        set(CMAKE_CXX_STANDARD 17)
    elseif(HAVE_FLAG_STD_CXX14)
        set(CMAKE_CXX_STANDARD 14)
    else()
        set(CMAKE_CXX_STANDARD 11)
    endif()
endif()

macro(add_subdirectory_shared directory)
    set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_BUILD_TYPE_SAVED "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(${directory} ${ARGN})
    set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")
    set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE_SAVED}")
endmacro()

macro(add_subdirectory_static directory)
    set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_BUILD_TYPE_SAVED "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(${directory} ${ARGN})
    set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")
    set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE_SAVED}")
endmacro()

macro(add_subdirectory_shared_debug directory)
    set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_BUILD_TYPE_SAVED "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE Debug)
    add_subdirectory(${directory} ${ARGN})
    set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")
    set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE_SAVED}")
endmacro()

macro(add_subdirectory_static_debug directory)
    set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_BUILD_TYPE_SAVED "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE Debug)
    add_subdirectory(${directory} ${ARGN})
    set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")
    set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE_SAVED}")
endmacro()

macro(find_apache_arrow)
    # workaround for: https://issues.apache.org/jira/browse/ARROW-11836
    if(NOT BUILD_VINEYARD_PYPI_PACKAGES)
      find_package(Arrow QUIET)
    endif()
    if(Arrow_FOUND)
        set(ARROW_INCLUDE_DIR)
        if (TARGET arrow_shared)
            set(ARROW_SHARED_LIB arrow_shared)
        endif()
        if (TARGET arrow_static)
            set(ARROW_STATIC_LIB arrow_static)
        endif()
    else()
        include("cmake/FindArrow.cmake")
        if(NOT ARROW_FOUND)
            message(FATAL_ERROR "apache-arrow is required, please install it and retry")
        endif()
    endif()
endmacro(find_apache_arrow)

macro(find_boost)
    if(NOT BUILD_SHARED_LIBS AND USE_STATIC_BOOST_LIBS)
        set(Boost_USE_STATIC_LIBS ON CACHE BOOL "Use static version of boost libraries.")
        set(Boost_USE_STATIC_RUNTIME ON CACHE BOOL "Use static version of boost runtimes.")
    endif()
    find_package(Boost COMPONENTS system filesystem)
    # Make boost::property_tree thread safe.
    add_compile_options(-DBOOST_SPIRIT_THREADSAFE)
    # Don't depends on the boost::system library.
    add_compile_options(-DBOOST_ERROR_CODE_HEADER_ONLY)
    add_compile_options(-DBOOST_SYSTEM_NO_DEPRECATED)
    # eliminate a lot of warnings for newer version of boost library.
    add_compile_options(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)
    # fixes the std::unary_function error
    add_compile_options(-DBOOST_NO_CXX98_FUNCTION_BASE)
    add_compile_options(-D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
endmacro()

macro(find_etcd_cpp_apiv3)
    if(USE_EXTERNAL_ETCD_LIBS)
        find_package(etcd-cpp-api 0.2.9 QUIET)
        if(NOT etcd-cpp-api_FOUND)
            message(WARNING "etcd-cpp-apiv3 not found, will use the bundled one from git submodules.")
            include("cmake/BuildEtcdCpp.cmake")
        endif()
    else()
        include("cmake/BuildEtcdCpp.cmake")
    endif()
endmacro()

macro(find_hiredis)
    if(USE_EXTERNAL_HIREDIS_LIBS)
    # not use find_package, because maybe we don't have hiredis-config.cmake
        find_path(HIREDIS_HEADER_PATH hiredis)
        find_library(HIREDIS_LIBRARY hiredis)
        if(HIREDIS_HEADER_PATH AND HIREDIS_LIBRARY)
            set(HIREDIS_INCLUDE_DIR ${HIREDIS_HEADER_PATH})
            set(HIREDIS_LIBRARIES ${HIREDIS_LIBRARY})
        else()
            message(WARNING "hiredis not found, will use the bundled one from git submodules.")
            include("cmake/BuildHiredis.cmake")
        endif()
    else()
        include("cmake/BuildHiredis.cmake")
    endif()
endmacro()

macro(find_redis_plus_plus)
    if(USE_EXTERNAL_REDIS_LIBS)
        # not use find_package, because of hiredis
        find_path(REDIS_PLUS_PLUS_HEADER sw)
        find_library(REDIS_PLUS_PLUS_LIB redis++)
        if(REDIS_PLUS_PLUS_HEADER AND REDIS_PLUS_PLUS_LIB)
            set(REDIS_PLUS_PLUS_INCLUDE_DIR "${REDIS_PLUS_PLUS_HEADER}/sw")
            set(REDIS_PLUS_PLUS_LIBRARIES ${REDIS_PLUS_PLUS_LIB})
        else()
            message(WARNING "redis-plus-plus not found, will use the bundled one from git submodules.")
            include("cmake/BuildRedisCpp.cmake")
        endif()
    else()
        include("cmake/BuildRedisCpp.cmake")
    endif()
endmacro()

macro(find_fuse)
    include("cmake/FindFUSE3.cmake")
    if(BUILD_VINEYARD_FUSE_PARQUET)
        if(NOT ARROW_PARQUET)
            message(FATAL_ERROR "Parquet is not enabled in the installed arrow.")
        endif()
        find_package(Parquet REQUIRED HINTS ${Arrow_DIR})
    endif()
endmacro()

macro(find_mimalloc)
    set(MI_OVERRIDE OFF CACHE INTERNAL "")
    set(MI_OSX_INTERPOSE OFF CACHE INTERNAL "")
    set(MI_OSX_ZONE OFF CACHE INTERNAL "")
    set(MI_DEBUG_FULL OFF CACHE INTERNAL "")
    set(MI_BUILD_SHARED OFF CACHE INTERNAL "")
    set(MI_BUILD_STATIC ON CACHE INTERNAL "")
    set(MI_BUILD_TESTS OFF CACHE INTERNAL "")
    if(NOT TARGET mimalloc-static)
        add_subdirectory_static(thirdparty/mimalloc EXCLUDE_FROM_ALL)
        # Its a hack that expose the internal API symbols inside the mimalloc library, even when
        # we are building a static library.
        #
        # I have checked the source of all usages of the following two macros and it is safe to
        # enable them on Linux and MacOS.
        target_compile_options(mimalloc-static PRIVATE -DMI_SHARED_LIB -DMI_SHARED_LIB_EXPORT)
    endif()
endmacro()

macro(find_glog)
    include("cmake/FindGlog.cmake")
endmacro()

macro(find_gflags)
    include("cmake/FindGFlags.cmake")
endmacro()

macro(find_libcuckoo)
    # install libcuckoo
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/thirdparty/libcuckoo/libcuckoo"
            DESTINATION ${CMAKE_INSTALL_PREFIX}/include/vineyard/contrib
            PATTERN "*.hh"
    )

    include_directories($<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/libcuckoo>
                        $<INSTALL_INTERFACE:include/vineyard/contrib>
    )
endmacro()

macro(find_libunwind)
    if(USE_LIBUNWIND)
        include("cmake/FindLibUnwind.cmake")
    endif()
endmacro()

macro(find_nlohmann_json)
    # include nlohmann/json
    add_definitions(-DJSON_USE_GLOBAL_UDLS=1)
    add_definitions(-DJSON_USE_IMPLICIT_CONVERSIONS=0)
    add_definitions(-DJSON_DISABLE_ENUM_SERIALIZATION=0)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND USE_JSON_DIAGNOSTICS)
        add_definitions(-DJSON_DIAGNOSTICS=1)
    else()
        add_definitions(-DJSON_DIAGNOSTICS=0)
    endif()
    add_definitions(-DJSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON=0)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/thirdparty/nlohmann-json/single_include"
            DESTINATION ${CMAKE_INSTALL_PREFIX}/include/vineyard/contrib
            PATTERN "*.hpp"
    )
    include_directories(BEFORE SYSTEM
                        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/nlohmann-json>
                        $<INSTALL_INTERFACE:include/vineyard/contrib>
    )
endmacro()

macro(find_openssl_libraries)
    if (APPLE)
        # If we're on OS X check for Homebrew's copy of OpenSSL instead of Apple's
        if (NOT OpenSSL_DIR)
            find_program(HOMEBREW brew)
            if (HOMEBREW STREQUAL "HOMEBREW-NOTFOUND")
                message(WARNING "Homebrew not found: not using Homebrew's OpenSSL")
                if (NOT OPENSSL_ROOT_DIR)
                    message(WARNING "Use -DOPENSSL_ROOT_DIR for non-Apple OpenSSL")
                endif()
            else()
                execute_process(COMMAND brew --prefix openssl
                    OUTPUT_VARIABLE OPENSSL_ROOT_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
            endif()
        endif()
    endif()
    find_package(OpenSSL ${ARGN})
    if (OPENSSL_FOUND)
        include_directories(SYSTEM ${OPENSSL_INCLUDE_DIR})
    endif()
endmacro()

macro(find_pthread)
    # find pthread
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads)
endmacro()

macro(find_cuda)
    # find cuda runtime library
    set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
    find_package(CUDA REQUIRED)
endmacro(find_cuda)

macro(find_zstd)
    # include zstd
    set(ZSTD_LEGACY_SUPPORT OFF CACHE INTERNAL "")
    set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_PROGRAMS OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_CONTRIB OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_TESTS OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_SHARED OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_STATIC ON CACHE INTERNAL "")
    set(ZSTD_BUILD_COMPRESSION ON CACHE INTERNAL "")
    set(ZSTD_BUILD_DECOMPRESSION ON CACHE INTERNAL "")
    set(ZSTD_BUILD_DICTBUILDER OFF CACHE INTERNAL "")
    set(ZSTD_BUILD_DEPRECATED OFF CACHE INTERNAL "")

    set(ZSTDLIB_VISIBLE "hidden" CACHE STRING "" FORCE)
    set(ZSTDERRORLIB_VISIBLE "hidden" CACHE STRING "" FORCE)
    set(ZDICTLIB_VISIBLE "hidden" CACHE STRING "" FORCE)
    set(ZSTDLIB_STATIC_API "hidden" CACHE STRING "" FORCE)
    set(ZDICTLIB_STATIC_API "hidden" CACHE STRING "" FORCE)

    if (NOT TARGET libzstd_static)
        set(CMAKE_MODULE_PATH_ORIGINAL ${CMAKE_MODULE_PATH})
        if(${CMAKE_VERSION} VERSION_LESS "3.18")
            list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
        endif()
        add_subdirectory_static(thirdparty/zstd/build/cmake EXCLUDE_FROM_ALL)
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_ORIGINAL})
    endif()
endmacro()

macro(find_common_libraries)
    find_glog()
    find_libcuckoo()
    find_libunwind()
    find_pthread()
    find_nlohmann_json()
    find_zstd()

    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    include_directories(SYSTEM ${GLOG_INCLUDE_DIR})
endmacro()

# find openssl first since apache-arrow may requires that (on MacOS, installed by brew)
find_openssl_libraries(QUIET)
find_common_libraries()

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/modules)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/thirdparty)

# build profiling library
if(BUILD_VINEYARD_PROFILING)
    add_definitions(-DWITH_PROFILING)
    include("cmake/FindGperftools.cmake")
endif()

if(${LIBUNWIND_FOUND})
    add_definitions(-DWITH_LIBUNWIND)
endif ()

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

macro(target_enable_sanitizer target visibility)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND VINEYARD_USE_ASAN)
        message(STATUS "Vineyardd will be built with -fsanitize=address")
        target_compile_options(${target} ${visibility} -fsanitize=address)
        target_add_link_options(${target} ${visibility} OPTIONS -fsanitize=address)
    endif()
endmacro()

macro(target_add_debuginfo target)
    target_compile_options(${target} PRIVATE -g)
endmacro()

macro(install_vineyard_target target)
    install(TARGETS ${target}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endmacro()

macro(install_export_vineyard_target target)
    install(TARGETS ${target}
        EXPORT vineyard-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endmacro()

macro(install_vineyard_headers header_path)
    if(${ARGC} GREATER_EQUAL 2)
        set(install_headers_destination "${ARGV1}")
    else()
        set(install_headers_destination "include/vineyard")
    endif()
    install(DIRECTORY ${header_path}
            DESTINATION ${install_headers_destination}          # target directory
            FILES_MATCHING                                      # install only matched files
            PATTERN "*.h"                                       # select header files
            PATTERN "*.hpp"                                     # select C++ template header files
            PATTERN "*.vineyard-mod"                            # select vineyard template files
            PATTERN "*/thirdparty/*" EXCLUDE                    # exclude thirdparty
    )
endmacro()

install_vineyard_headers("${PROJECT_SOURCE_DIR}/src/common")
set(VINEYARD_INSTALL_LIBS)

# resolve targets dependencies
if(BUILD_VINEYARD_PYPI_PACKAGES)
    set(BUILD_VINEYARD_PYTHON_BINDINGS ON)
endif()

if(BUILD_VINEYARD_FUSE_PARQUET)
    set(BUILD_VINEYARD_FUSE ON)
endif()

if(BUILD_VINEYARD_FUSE)
    set(BUILD_VINEYARD_BASIC ON)
endif()

if(BUILD_VINEYARD_GRAPH)
    set(BUILD_VINEYARD_BASIC ON)
    set(BUILD_VINEYARD_IO ON)
endif()

if(BUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME)
    set(BUILD_VINEYARD_BASIC ON)
endif()

if(BUILD_VINEYARD_IO)
    set(BUILD_VINEYARD_BASIC ON)
endif()

if(BUILD_VINEYARD_BASIC OR BUILD_VINEYARD_IO OR BUILD_VINEYARD_MIGRATION OR BUILD_VINEYARD_PYTHON_BINDINGS)
    set(BUILD_VINEYARD_CLIENT ON)

    find_apache_arrow()
    include_directories(SYSTEM ${ARROW_INCLUDE_DIR})
endif()

if(BUILD_VINEYARD_MALLOC)
    find_mimalloc()
    set(BUILD_VINEYARD_CLIENT ON)
endif()

add_custom_target(vineyard_tests)
add_custom_target(vineyard_benchmarks)
add_custom_target(vineyard_codegen)
add_custom_target(vineyard_codegen_java)

include("cmake/GenerateVineyard.cmake")
include("cmake/GenerateVineyardJava.cmake")

# boost is only required by some components
if(BUILD_VINEYARD_SERVER OR BUILD_VINEYARD_IO OR BUILD_VINEYARD_GRAPH)
    find_boost()
endif()

# build vineyardd
if(BUILD_VINEYARD_SERVER)
    find_gflags()
    find_mimalloc()

    if(BUILD_VINEYARD_SERVER_REDIS)
        find_hiredis()
        find_redis_plus_plus()
    endif()
    if(BUILD_VINEYARD_SERVER_ETCD)
        find_etcd_cpp_apiv3()
        find_openssl_libraries(REQUIRED)
    endif()
    if(BUILD_VINEYARD_SERVER_SPILLING)
        find_apache_arrow()
    endif()
    if(USE_CUDA)
        find_cuda()
    endif()

    file(GLOB_RECURSE SERVER_SRC_FILES "src/server/*.cc"
                                       "src/common/compression/*.cc"
                                       "src/common/memory/*.cc"
                                       "src/common/util/*.cc"
                                       "src/common/memory/gpu/*.cc"
    )
    if(BUILD_VINEYARD_SERVER_REDIS)
        list(APPEND SERVER_SRC_FILES "thirdparty/redis-plus-plus-shim/recipes/redlock.cpp")
    endif()
    add_executable(vineyardd ${SERVER_SRC_FILES})
    target_add_debuginfo(vineyardd)
    target_compile_options(vineyardd PRIVATE -DBUILD_VINEYARDD)
    target_link_libraries(vineyardd PRIVATE ${Boost_LIBRARIES}
                                            ${CMAKE_DL_LIBS}
                                            ${GFLAGS_LIBRARIES}
                                            ${GLOG_LIBRARIES}
                                            libzstd_static
                                            mimalloc-static
    )
    target_include_directories(vineyardd PRIVATE ${GFLAGS_INCLUDE_DIR})

    if(BUILD_VINEYARD_SERVER_REDIS)
        target_compile_options(vineyardd PRIVATE -DBUILD_VINEYARDD_REDIS)
        target_include_directories(vineyardd PRIVATE ${HIREDIS_INCLUDE_DIR}
                                                     ${REDIS_PLUS_PLUS_INCLUDE_DIR}
        )
        target_link_libraries(vineyardd PRIVATE ${HIREDIS_LIBRARIES}
                                                ${REDIS_PLUS_PLUS_LIBRARIES}
        )
    endif()
    if(BUILD_VINEYARD_SERVER_ETCD)
        target_compile_options(vineyardd PRIVATE -DBUILD_VINEYARDD_ETCD)
        target_include_directories(vineyardd PRIVATE ${ETCD_CPP_INCLUDE_DIR})
        target_link_libraries(vineyardd PRIVATE ${CPPREST_LIB}
                                                ${ETCD_CPP_LIBRARIES}
                                                ${OPENSSL_LIBRARIES}
        )
    endif()
    if(BUILD_VINEYARD_SERVER_SPILLING)
        target_compile_options(vineyardd PRIVATE -DBUILD_VINEYARDD_SPILLING)
        target_include_directories(vineyardd PRIVATE ${ARROW_INCLUDE_DIR})
        if(ARROW_SHARED_LIB)
            target_link_libraries(vineyardd PRIVATE ${ARROW_SHARED_LIB})
        else()
            target_link_libraries(vineyardd PRIVATE ${ARROW_STATIC_LIB})
        endif()
    endif()

    if(USE_CUDA)
        target_compile_options(vineyardd PRIVATE -DENABLE_CUDA)
        target_include_directories(vineyardd PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(vineyardd PRIVATE ${CUDA_LIBRARIES})
    endif()

    if(${LIBUNWIND_FOUND})
        target_link_libraries(vineyardd PRIVATE ${LIBUNWIND_LIBRARIES})
    endif()

    install_vineyard_target(vineyardd)
    if(NOT BUILD_SHARED_LIBS)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            target_compile_options(vineyardd PRIVATE -static-libgcc -static-libstdc++ -Os)
            target_add_link_options(vineyardd PRIVATE OPTIONS -static-libgcc -static-libstdc++ -Os)
        endif()
        target_link_libraries(vineyardd PRIVATE ${GRPC_GRPC++_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY})
    endif()
    if(BUILD_VINEYARD_PROFILING)
        target_include_directories(vineyardd PRIVATE ${Gperftools_INCLUDE_DIRS})
        target_link_libraries(vineyardd PRIVATE ${Gperftools_LIBRARIES})
    endif()

    message(STATUS "Vineyard will use '${WITH_ALLOCATOR}' by default for shared memory allocation")
    # bundle dlmalloc and mimalloc at the same time
    target_compile_options(vineyardd PRIVATE -DWITH_DLMALLOC -DWITH_MIMALLOC)

    if(WITH_ALLOCATOR STREQUAL "dlmalloc")
        target_compile_options(vineyardd PRIVATE -DDEFAULT_ALLOCATOR=dlmalloc)
    endif()
    if(WITH_ALLOCATOR STREQUAL "mimalloc")
        target_compile_options(vineyardd PRIVATE -DDEFAULT_ALLOCATOR=mimalloc)
    endif()
endif()

# build vineyard-client
if(BUILD_VINEYARD_CLIENT)
    # build vineyard registry library, it can only be a shared library, as a singleton.
    add_library(vineyard_internal_registry SHARED "src/client/ds/factory/registry.cc")
    target_link_libraries(vineyard_internal_registry PRIVATE ${CMAKE_DL_LIBS})
    if(BUILD_VINEYARD_PYPI_PACKAGES)
        target_compile_options(vineyard_internal_registry PRIVATE -Os)
        target_add_link_options(vineyard_internal_registry PRIVATE OPTIONS -Os)
        target_add_link_options(vineyard_internal_registry PRIVATE OPTIONS -Os)
    endif()
    install_export_vineyard_target(vineyard_internal_registry)
    target_add_link_options(vineyard_internal_registry PRIVATE
                            OPTIONS -Wl,-rpath,.:$ORIGIN:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64
    )
    set_target_properties(vineyard_internal_registry PROPERTIES CXX_VISIBILITY_PRESET "hidden")

    # build the vineyard client library
    file(GLOB CLIENT_SRC_FILES "src/client/*.cc"
                               "src/client/ds/*.cc"
                               "src/common/compression/*.cc"
                               "src/common/memory/*.cc"
                               "src/common/util/*.cc"
                               "src/common/memory/gpu/*.cc"
    )
    # the vineyard_client can only be a shared library, since the ObjectFactory
    # is a singleton.
    add_library(vineyard_client ${CLIENT_SRC_FILES})
    target_add_debuginfo(vineyard_client)
    target_link_libraries(vineyard_client PUBLIC ${CMAKE_DL_LIBS}
                                                 Threads::Threads
    )
    target_link_libraries(vineyard_client PRIVATE libzstd_static)

    if(USE_CUDA)
        find_cuda()

        target_compile_options(vineyard_client PRIVATE -DENABLE_CUDA)
        target_include_directories(vineyard_client PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(vineyard_client PRIVATE ${CUDA_LIBRARIES})
    endif()

    # make sure `vineyard_internal_registry` been built.
    add_dependencies(vineyard_client vineyard_internal_registry)
    if(${LIBUNWIND_FOUND})
        target_link_libraries(vineyard_client PRIVATE ${LIBUNWIND_LIBRARIES})
    endif()

    if(BUILD_VINEYARD_CLIENT_VERBOSE)
        target_compile_options(vineyard_client PRIVATE -DWITH_VERBOSE)
    endif()

    if(BUILD_VINEYARD_PYPI_PACKAGES)
        target_compile_options(vineyard_client PRIVATE -Os)
        target_add_link_options(vineyard_client PRIVATE OPTIONS -Os)
        if(APPLE)
            target_compile_options(vineyard_client PRIVATE -fvisibility=hidden)
        endif()
    endif()

    file(GLOB_RECURSE VINEYARD_CLIENT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                              "client/*.h")
    if(BUILD_VINEYARD_JAVA AND VINEYARD_CLIENT_HEADERS)
        vineyard_generate_java(
            OUT_VAR VINEYARD_JAVA_GENERATES
            VINEYARD_MODULES ${VINEYARD_CLIENT_HEADERS}
        )
    else()
        set(VINEYARD_JAVA_GENERATES)
    endif()
    add_custom_target(vineyard_client_gen_java
        DEPENDS ${VINEYARD_JAVA_GENERATES}
        COMMENT "Running java code generation for vineyard_client."
    )
    if(VINEYARD_JAVA_GENERATES)
        add_dependencies(vineyard_codegen_java vineyard_client_gen_java)
    endif()

    install_export_vineyard_target(vineyard_client)
    install_vineyard_headers("${PROJECT_SOURCE_DIR}/src/client")
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_client)
endif()

if(BUILD_VINEYARD_PYTHON_BINDINGS)
    set(PYBIND11_PYTHON_VERSION 3)
    if(NOT (CMAKE_VERSION VERSION_LESS "3.27"))
        set(PYBIND11_FINDPYTHON ON)
    endif()
    add_subdirectory_static(thirdparty/pybind11)
    set(PYTHON_BIND_FILES "python/client.cc"
                          "python/core.cc"
                          "python/error.cc"
                          "python/pybind11_docs.cc"
                          "python/pybind11_utils.cc"
                          "python/vineyard.cc")
    pybind11_add_module(_C MODULE ${PYTHON_BIND_FILES})
    target_add_debuginfo(_C)
    target_link_libraries(_C PRIVATE vineyard_client)
    target_include_directories(_C PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11/include")
    target_compile_options(_C PRIVATE -Wno-unused-value)
    set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shared-lib")
    if(UNIX AND NOT APPLE)
        target_add_link_options(_C PRIVATE OPTIONS -Wl,--exclude-libs=ALL)
    endif()
    if(BUILD_VINEYARD_PYPI_PACKAGES AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(_C PRIVATE -static)
        target_add_link_options(_C PRIVATE OPTIONS -static)
    else()
        target_compile_options(_C PRIVATE -Os)
        target_add_link_options(_C PRIVATE OPTIONS -Os)
    endif()

    file(RELATIVE_PATH RELATIVE_BUILD_PATH "${PROJECT_SOURCE_DIR}/python/vineyard" "${CMAKE_BINARY_DIR}/shared-lib")
    if(UNIX AND NOT APPLE)
        set_target_properties(_C PROPERTIES
                              BUILD_WITH_INSTALL_RPATH TRUE
                              INSTALL_RPATH_USE_LINK_PATH TRUE
                              INSTALL_RPATH ".:\$ORIGIN:\$ORIGIN/${RELATIVE_BUILD_PATH}/:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64:${INSTALL_RPATH}")
    endif()
    if(APPLE)
        set_target_properties(_C PROPERTIES
                              BUILD_WITH_INSTALL_RPATH TRUE
                              INSTALL_RPATH_USE_LINK_PATH TRUE
                              INSTALL_RPATH ".;@loader_path;@loader_path/${RELATIVE_BUILD_PATH}/;${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_INSTALL_PREFIX}/lib64;${INSTALL_RPATH}")
    endif()

    add_custom_target(vineyard_client_python
        ALL
        COMMAND cp "$<TARGET_FILE:_C>" "${PROJECT_SOURCE_DIR}/python/vineyard/"
        COMMAND cp "$<TARGET_FILE:vineyard_internal_registry>" "${PROJECT_SOURCE_DIR}/python/vineyard/"
        DEPENDS _C vineyard_internal_registry
        COMMENT "Copying python extensions."
        VERBATIM)
endif()

if(BUILD_VINEYARD_PYTHON_BINDINGS AND BUILD_VINEYARD_LLM_CACHE)
    set(PYBIND11_PYTHON_VERSION 3)
    if(NOT (CMAKE_VERSION VERSION_LESS "3.27"))
        set(PYBIND11_FINDPYTHON ON)
    endif()
    file(GLOB PYTHON_BIND_FILES "python/vineyard/llm/kv_state_cache.cc"
                                "modules/llm-cache/ds/*.h"
    )
    pybind11_add_module(llm_C MODULE ${PYTHON_BIND_FILES})
    # make sure `vineyard_llm_cache` been built.
    add_dependencies(llm_C vineyard_llm_cache)
    target_link_libraries(llm_C PRIVATE vineyard_client vineyard_llm_cache)
    target_include_directories(llm_C PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11/include"
                                                   "modules/llm-cache/ds/*.h")
    target_compile_options(llm_C PRIVATE -Wno-unused-value)
    set_target_properties(llm_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shared-lib")
    if(UNIX AND NOT APPLE)
        target_add_link_options(llm_C PRIVATE OPTIONS -Wl,--exclude-libs=ALL)
    endif()
    if(BUILD_VINEYARD_PYPI_PACKAGES AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(llm_C PRIVATE -static)
        target_add_link_options(llm_C PRIVATE OPTIONS -static)
    else()
        target_compile_options(llm_C PRIVATE -Os)
        target_add_link_options(llm_C PRIVATE OPTIONS -Os)
    endif()

    file(RELATIVE_PATH RELATIVE_BUILD_PATH "${PROJECT_SOURCE_DIR}/python/vineyard" "${CMAKE_BINARY_DIR}/shared-lib")
    if(UNIX AND NOT APPLE)
        set_target_properties(llm_C PROPERTIES
                            BUILD_WITH_INSTALL_RPATH TRUE
                            INSTALL_RPATH_USE_LINK_PATH TRUE
                            INSTALL_RPATH ".:\$ORIGIN:\$ORIGIN/${RELATIVE_BUILD_PATH}/:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64:${INSTALL_RPATH}")
    endif()
    if(APPLE)
        set_target_properties(llm_C PROPERTIES
                            BUILD_WITH_INSTALL_RPATH TRUE
                            INSTALL_RPATH_USE_LINK_PATH TRUE
                            INSTALL_RPATH ".;@loader_path;@loader_path/${RELATIVE_BUILD_PATH}/;${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_INSTALL_PREFIX}/lib64;${INSTALL_RPATH}")
    endif()

    add_custom_target(vineyard_with_llm_python
        ALL
        COMMAND cp "$<TARGET_FILE:llm_C>" "${PROJECT_SOURCE_DIR}/python/vineyard/llm/"
        DEPENDS llm_C
        COMMENT "Copying llm kv cache python extensions."
        VERBATIM)
endif()

# add sub_directories
if(BUILD_VINEYARD_BASIC)
    add_subdirectory(modules/basic)
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_basic)
endif()

if(BUILD_VINEYARD_IO)
    add_subdirectory(modules/io)
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_io)
endif()

if(BUILD_VINEYARD_GRAPH)
    add_subdirectory(modules/graph)
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_graph)
endif()

if(BUILD_VINEYARD_MALLOC)
    add_subdirectory(modules/malloc)
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_malloc)
endif()

if(BUILD_VINEYARD_FUSE)
    find_fuse()
    add_subdirectory(modules/fuse)
    # don't includes vineyard_fuse to "VINEYARD_LIBRARIES"
endif()

if(BUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME)
    add_subdirectory(modules/hosseinmoein-dataframe)
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_hosseinmoein_dataframe)
endif()

if(BUILD_VINEYARD_LLM_CACHE)
    add_subdirectory(modules/llm-cache)
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_llm_cache)
endif()

if(BUILD_VINEYARD_TESTS)
    add_subdirectory(test)
endif()

if(BUILD_VINEYARD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

function(file_glob_recurse VAR)
    set(options)
    set(oneValueArgs PATTERNS EXCLUDE_PATTERNS)
    set(multiValueArgs DIRECTORIES)
    cmake_parse_arguments(FILE_GLOB_RECURSE
                          "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(_subdirs)
    set(_srcs)
    foreach(_base ${FILE_GLOB_RECURSE_DIRECTORIES})
        file(GLOB _items LIST_DIRECTORIES true ${_base}/*)
        foreach(_item ${_items})
            if(${_item} MATCHES ".*thirdparty.*")
                continue()
            endif()
            if(IS_DIRECTORY ${_item})
                list(APPEND _subdirs ${_item})
            elseif(${_item} MATCHES "${FILE_GLOB_RECURSE_PATTERNS}" AND
                    NOT (${_item} MATCHES "${FILE_GLOB_RECURSE_EXCLUDE_PATTERNS}"))
                list(APPEND _srcs ${_item})
            endif()
        endforeach()
    endforeach()

    if(_subdirs)
        file_glob_recurse(_subsrcs DIRECTORIES ${_subdirs}
                                   PATTERNS ${FILE_GLOB_RECURSE_PATTERNS}
                                   EXCLUDE_PATTERNS ${FILE_GLOB_RECURSE_EXCLUDE_PATTERNS})
        list(APPEND _srcs ${_subsrcs})
    endif()
    set(${VAR} ${_srcs} PARENT_SCOPE)
endfunction()

file_glob_recurse(FILES_NEED_FORMAT DIRECTORIES      "src" "modules" "python" "test" "benchmark"
                                    PATTERNS         ".*\\.(cc|cpp|h|hpp|vineyard-mod)$"
                                    EXCLUDE_PATTERNS "(.*\\.vineyard.h$)"
)

# the `memcpy.h` is borrowed from external project
list(REMOVE_ITEM FILES_NEED_FORMAT "${PROJECT_SOURCE_DIR}/src/common/memory/memcpy.h")

add_custom_target(vineyard_clformat
        COMMAND clang-format --style=file -i ${FILES_NEED_FORMAT}
        COMMENT "Running clang-format."
        VERBATIM)

add_custom_target(vineyard_cpplint
        COMMAND ${PROJECT_SOURCE_DIR}/misc/cpplint.py --root=vineyard ${FILES_NEED_FORMAT}
        COMMENT "Running cpplint check."
        VERBATIM)

add_custom_target(python_install
        COMMAND python3 -m pip install --user .
        COMMAND python3 setup_airflow.py install
        COMMAND python3 setup_ml.py install
        COMMAND python3 setup_dask.py install
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        VERBATIM)

# for python's setup.cfg
configure_file("${PROJECT_SOURCE_DIR}/setup.cfg.in"
               "${PROJECT_SOURCE_DIR}/setup.cfg" @ONLY
)
configure_file("${PROJECT_SOURCE_DIR}/python/vineyard/version.py.in"
               "${PROJECT_SOURCE_DIR}/python/vineyard/version.py" @ONLY
)
configure_file("${PROJECT_SOURCE_DIR}/src/common/util/config.h.in"
               "${PROJECT_SOURCE_DIR}/src/common/util/config.h" @ONLY
)

configure_file("${PROJECT_SOURCE_DIR}/vineyard-config.in.cmake"
               "${PROJECT_BINARY_DIR}/vineyard-config.cmake" @ONLY
)
configure_file("${PROJECT_SOURCE_DIR}/vineyard-config-version.in.cmake"
               "${PROJECT_BINARY_DIR}/vineyard-config-version.cmake" @ONLY
)
install(FILES "${PROJECT_BINARY_DIR}/vineyard-config.cmake"
              "${PROJECT_BINARY_DIR}/vineyard-config-version.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindArrow.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/CheckGCCABI.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindFUSE3.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindGFlags.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindGlog.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindGperftools.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindLibUnwind.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindRdkafka.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindPythonExecutable.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/GenerateVineyard.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/GenerateVineyardJava.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/DetermineImplicitIncludes.cmake"
              DESTINATION lib/cmake/vineyard
)
install(EXPORT vineyard-targets
        FILE vineyard-targets.cmake
        DESTINATION lib/cmake/vineyard
)

# build docs
find_program(doxygen_EXECUTABLE doxygen NO_CMAKE_SYSTEM_PATH)
find_program(sphinx_build_EXECUTABLE sphinx-build NO_CMAKE_SYSTEM_PATH)
if(doxygen_EXECUTABLE AND sphinx_build_EXECUTABLE)
    add_custom_target(vineyard_doc
        COMMAND ${CMAKE_COMMAND} -E make_directory _build
        COMMAND ${doxygen_EXECUTABLE}
        COMMAND ${sphinx_build_EXECUTABLE} . _build/html
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
        VERBATIM
    )
else()
    if(NOT doxygen_EXECUTABLE)
        message(STATUS "Cannot find the doxygen executable.")
    endif()
    if(NOT sphinx_build_EXECUTABLE)
        message(STATUS "Cannot find the sphinx-build executable.")
    endif()
endif()
