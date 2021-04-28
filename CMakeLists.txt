cmake_minimum_required(VERSION 2.8)

set(VINEYARD_MAJOR_VERSION 0)
set(VINEYARD_MINOR_VERSION 2)
set(VINEYARD_PATCH_VERSION 0)
set(VINEYARD_VERSION ${VINEYARD_MAJOR_VERSION}.${VINEYARD_MINOR_VERSION}.${VINEYARD_PATCH_VERSION})

if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()
project(vineyard LANGUAGES C CXX VERSION ${VINEYARD_VERSION})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(USE_STATIC_BOOST_LIBS "Build with static-linked boost libraries" OFF)
option(USE_EXTERNAL_ETCD_LIBS "Build with external etcd-cpp-apiv3 library rather than the submodule one" OFF)
option(USE_EXTERNAL_TBB_LIBS "Build with external tbb library rather than the submodule one" OFF)

option(BUILD_VINEYARD_SERVER "Build vineyard's server" ON)
option(BUILD_VINEYARD_CLIENT "Build vineyard's client" ON)
option(BUILD_VINEYARD_PYTHON_BINDINGS "Build vineyard's python bindings" ON)
option(BUILD_VINEYARD_PYPI_PACKAGES "Build vineyard's python bindings" OFF)

option(BUILD_VINEYARD_BASIC "Build vineyard's basic data structures" ON)
option(BUILD_VINEYARD_IO "Enable vineyard's IOAdaptor support" ON)
option(BUILD_VINEYARD_GRAPH "Enable vineyard's graph data structures" ON)
option(BUILD_VINEYARD_MALLOC "Build vineyard's implementation for client-side malloc" ON)
option(BUILD_VINEYARD_MIGRATION "Enable vineyard's object migration support" ON)

option(BUILD_VINEYARD_TESTS "Generate make targets for vineyard tests" ON)
option(BUILD_VINEYARD_TESTS_ALL "Include make targets for vineyard tests to ALL" OFF)
option(BUILD_VINEYARD_COVERAGE "Build vineyard with coverage information, requires build with Debug" OFF)
option(BUILD_VINEYARD_PROFILING "Build vineyard with profiling information" OFF)

include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include(ExternalProject)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(ProcessorCount)
ProcessorCount(N)

add_definitions(-DVINEYARD_VERSION=${VINEYARD_VERSION})
message(STATUS "Configuring and building vineyard version '${VINEYARD_VERSION}'.")

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

find_program(ccache_EXECUTABLE ccache)
if(ccache_EXECUTABLE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${ccache_EXECUTABLE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${ccache_EXECUTABLE})
    add_custom_target(ccache-stats
        COMMAND ${ccache_EXECUTABLE} --show-stats
    )
else()
    add_custom_target(ccache-stats
        COMMAND echo "ccache not found."
    )
endif(ccache_EXECUTABLE)

set(DEFAUTL_ALLOCATOR "dlmalloc")
set(WITH_ALLOCATOR "${DEFAUTL_ALLOCATOR}" CACHE
    STRING "Choose the allocator for vineyard server (vineyardd)."
)
set_property(CACHE WITH_ALLOCATOR PROPERTY STRINGS
    "dlmalloc" "jemalloc"
)
message(STATUS "Setting the allocator to vineyardd as '${WITH_ALLOCATOR}'.")

# reference: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC \
                                        -Wall \
                                        -Wno-attributes \
                                        -Wno-unknown-pragmas"
)

check_cxx_compiler_flag(-Wno-class-memaccess W_NO_CLASS_MEMACCESS)
check_cxx_compiler_flag(-Wno-deprecated-declarations W_NO_DEPRECATED_DECLARATIONS)
check_cxx_compiler_flag(-Wno-defaulted-function-deleted W_NO_DEFAULTED_FUNCTION_DELETED)
check_cxx_compiler_flag(-Wno-error=c++11-narrowing W_NO_ERROR_CXX11_NARROWING)
if(W_NO_CLASS_MEMACCESS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-memaccess")
endif()
if(W_NO_DEPRECATED_DECLARATIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()
if(W_NO_DEFAULTED_FUNCTION_DELETED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-defaulted-function-deleted")
endif()
if(W_NO_ERROR_CXX11_NARROWING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=c++11-narrowing")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
if(BUILD_VINEYARD_COVERAGE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/../lib")
endif()

macro(find_apache_arrow)
    # workaround for: https://issues.apache.org/jira/projects/ARROW/issues/ARROW-11836
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
    find_package(Boost COMPONENTS system)
    # Make boost::property_tree thread safe.
    add_compile_options(-DBOOST_SPIRIT_THREADSAFE)
    # Don't depends on the boost::system library.
    add_compile_options(-DBOOST_ERROR_CODE_HEADER_ONLY)
    add_compile_options(-DBOOST_SYSTEM_NO_DEPRECATED)
    # eliminate a lot of warnings for newer version of boost library.
    add_compile_options(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)
endmacro(find_boost)

macro(find_etcd_cpp_apiv3)
    if(USE_EXTERNAL_ETCD_LIBS)
        find_package(etcd-cpp-api QUIET)
        if(NOT etcd-cpp-api_FOUND)
            message(WARNING "etcd-cpp-apiv3 not found, will use the bundled one from git submodules.")
            include("cmake/BuildEtcdCpp.cmake")
        endif()
    else()
        include("cmake/BuildEtcdCpp.cmake")
    endif()
endmacro(find_etcd_cpp_apiv3)

macro(find_intel_tbb)
    set(TBB_TEST OFF CACHE BOOL "Disable tbb testing")
    set(TBB_EXAMPLES OFF CACHE BOOL "Disable tbb examples")
    set(TBB_STRICT OFF CACHE BOOL "Don't treat warnings as errors for tbb library")
    set(TBB4PY_BUILD OFF CACHE BOOL "Disable tbb4py build")
    set(TBB_CPF ON CACHE BOOL "Enable preview features of the library")

    if(USE_EXTERNAL_TBB_LIBS)
        find_package(TBB QUIET)
        if(NOT TBB_FOUND)
            message(WARNING "intel-tbb not found, will use the bundled one from git submodules.")
            add_subdirectory(thirdparty/intel-tbb)
        endif()
    else()
        add_subdirectory(thirdparty/intel-tbb)
    endif()
endmacro(find_intel_tbb)

macro(find_jemalloc)
    # ./configure --disable-shared --enable-static --disable-cxx --enable-stats --disable-initial-exec-tls --with-jemalloc-prefix=vineyard_je_
    #
    # see: https://github.com/jemalloc/jemalloc/issues/1237
    ExternalProject_Add(libjemalloc
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/jemalloc"
        BUILD_IN_SOURCE 1
        BUILD_ALWAYS 1
        CONFIGURE_COMMAND "${PROJECT_SOURCE_DIR}/thirdparty/jemalloc/autogen.sh"
            --prefix=${CMAKE_INSTALL_PREFIX}
            --disable-shared
            --enable-static
            --disable-cxx
            --enable-stats
            --disable-initial-exec-tls
            --with-jemalloc-prefix=vineyard_je_
        BUILD_COMMAND ${MAKE}
        INSTALL_COMMAND echo "No need to install for jemalloc."
    )

    add_library(jemalloc STATIC IMPORTED)
    add_dependencies(jemalloc libjemalloc)
    set_target_properties(jemalloc PROPERTIES IMPORTED_LOCATION
                                              ${PROJECT_SOURCE_DIR}/thirdparty/jemalloc/lib/libjemalloc_pic.a)
    set_target_properties(jemalloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                              ${PROJECT_SOURCE_DIR}/thirdparty)
endmacro(find_jemalloc)

macro(find_glog)
    include("cmake/FindGlog.cmake")
endmacro(find_glog)

macro(find_gflags)
    include("cmake/FindGFlags.cmake")
endmacro(find_gflags)

macro(find_libunwind)
    include("cmake/FindLibUnwind.cmake")
endmacro(find_libunwind)

macro(find_nlohmann_json)
    # include nlohmann/json
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install ON CACHE INTERNAL "")
    set(JSON_MultipleHeaders ON CACHE INTERNAL "")
    set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
    add_subdirectory(thirdparty/nlohmann-json)
endmacro(find_nlohmann_json)

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
endmacro(find_openssl_libraries)

macro(find_pthread)
    # find pthread
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads)
endmacro(find_pthread)

macro(find_common_libraries)
    find_apache_arrow()
    find_boost()
    find_glog()
    find_gflags()
    find_jemalloc()
    find_libunwind()
    find_pthread()
    find_nlohmann_json()

    include_directories(SYSTEM ${ARROW_INCLUDE_DIR})
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    include_directories(SYSTEM ${GLOG_INCLUDE_DIR})
    include_directories(SYSTEM ${GFLAGS_INCLUDE_DIR})
endmacro(find_common_libraries)

# find openssl first since apache-arrow may requires that (on MacOS, installed by brew)
find_openssl_libraries(QUIET)
find_common_libraries()

if(CMAKE_VERSION VERSION_LESS "3.1")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        check_cxx_compiler_flag(-std=c++14 HAVE_FLAG_STD_CXX14)
        if(BUILD_VINEYARD_PYPI_PACKAGES AND NOT HAVE_FLAG_STD_CXX14)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
        endif()
    endif()
else()
    set(CMAKE_CXX_STANDARD 14)
endif()

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

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

macro(add_target_link_options target scope)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs OPTIONS)
    cmake_parse_arguments(add_target_link_options "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(${CMAKE_VERSION} VERSION_LESS "3.13")
        target_link_libraries(${target} INTERFACE ${add_target_link_options_OPTIONS})
    else()
        target_link_options(${target} ${scope} ${add_target_link_options_OPTIONS})
    endif()
endmacro()

macro(install_vineyard_target target)
    install(TARGETS ${target}
        EXPORT vineyard-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endmacro()

macro(install_vineyard_headers header_path)
    install(DIRECTORY ${header_path}
            DESTINATION include/vineyard        # target directory
            FILES_MATCHING                      # install only matched files
            PATTERN "*.h"                       # select header files
            PATTERN "*.hpp"                     # select C++ template header files
            PATTERN "*.vineyard-mod"            # select vineyard template files
    )
endmacro()

install_vineyard_headers("${PROJECT_SOURCE_DIR}/src/common")
set(VINEYARD_INSTALL_LIBS)

# resolve targets dependencies
if(BUILD_VINEYARD_PYPI_PACKAGES)
    set(BUILD_VINEYARD_PYTHON_BINDINGS ON)
endif()

if(BUILD_VINEYARD_GRAPH)
    set(BUILD_VINEYARD_BASIC ON)
    set(BUILD_VINEYARD_IO ON)
endif()

if(BUILD_VINEYARD_IO)
    set(BUILD_VINEYARD_BASIC ON)
endif()

if(BUILD_VINEYARD_BASIC OR BUILD_VINEYARD_IO OR BUILD_VINEYARD_MIGRATION OR BUILD_VINEYARD_PYTHON_BINDINGS)
    set(BUILD_VINEYARD_CLIENT ON)
endif()

if(BUILD_VINEYARD_MALLOC)
    set(BUILD_VINEYARD_CLIENT ON)
endif()

# build vineyardd
if(BUILD_VINEYARD_SERVER)
    find_openssl_libraries(REQUIRED)
    find_etcd_cpp_apiv3()
    find_intel_tbb()
    file(GLOB_RECURSE SERVER_SRC_FILES "src/server/*.cc"
                                       "src/common/memory/*.cc"
                                       "src/common/util/*.cc"
    )
    add_executable(vineyardd ${SERVER_SRC_FILES})
    target_compile_options(vineyardd PRIVATE -DBUILD_VINEYARDD)
    target_link_libraries(vineyardd PUBLIC ${Boost_LIBRARIES}
                                           ${CPPREST_LIB}
                                           ${ETCD_CPP_LIBRARIES}
                                           ${GFLAGS_LIBRARIES}
                                           ${GLOG_LIBRARIES}
                                           ${OPENSSL_LIBRARIES}
                                           nlohmann_json::nlohmann_json
                                           TBB::tbb
    )
    target_include_directories(vineyardd PRIVATE ${ETCD_CPP_INCLUDE_DIR})
    if(${LIBUNWIND_FOUND})
        target_link_libraries(vineyardd PRIVATE ${LIBUNWIND_LIBRARIES})
    endif()
    install_vineyard_target(vineyardd)
    install_vineyard_headers("${PROJECT_SOURCE_DIR}/src/server")
    if(NOT BUILD_SHARED_LIBS)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            target_compile_options(vineyardd PRIVATE -static-libgcc -static-libstdc++ -Os)
            add_target_link_options(vineyardd PRIVATE OPTIONS -static-libgcc -static-libstdc++ -Os)
        endif()
        target_link_libraries(vineyardd PRIVATE ${GRPC_GRPC++_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY})
    endif()
    if(BUILD_VINEYARD_PROFILING)
        target_include_directories(vineyardd PRIVATE ${Gperftools_INCLUDE_DIRS})
        target_link_libraries(vineyardd PRIVATE ${Gperftools_LIBRARIES})
    endif()
    message(STATUS "Vineyard will use '${WITH_ALLOCATOR}' for shared memory allocation")
    if(WITH_ALLOCATOR STREQUAL "dlmalloc")
        # use dlmalloc
        target_compile_options(vineyardd PRIVATE -DWITH_DLMALLOC)
    elseif(WITH_ALLOCATOR STREQUAL "jemalloc")
        target_link_libraries(vineyardd PRIVATE jemalloc ${CMAKE_DL_LIBS})
        target_compile_options(vineyardd PRIVATE -DWITH_JEMALLOC)
    else()
        message(FATAL "The allocator '${WITH_ALLOCATOR}' is not supported")
    endif()
endif()

# build vineyard-client
if(BUILD_VINEYARD_CLIENT)
    file(GLOB_RECURSE CLIENT_SRC_FILES "src/client/*.cc"
                                    "src/common/memory/*.cc"
                                    "src/common/util/*.cc"
    )
    # the vineyard_client can only be a shared library, since the ObjectFactory
    # is a singleton.
    add_library(vineyard_client SHARED ${CLIENT_SRC_FILES})
    target_link_libraries(vineyard_client PUBLIC Threads::Threads
                                                 nlohmann_json::nlohmann_json
                                                 ${GLOG_LIBRARIES}
    )
    if(BUILD_SHARED_LIBS AND BUILD_VINEYARD_PYPI_PACKAGES)
        # required on mac, but cause problem on linux.
        target_link_libraries(vineyard_client PUBLIC ${GFLAGS_LIBRARIES})
    endif()
    if(ARROW_SHARED_LIB)
        target_link_libraries(vineyard_client PUBLIC ${ARROW_SHARED_LIB})
    else()
        target_link_libraries(vineyard_client PUBLIC ${ARROW_STATIC_LIB})
    endif()

    target_link_libraries(vineyard_client PRIVATE jemalloc ${CMAKE_DL_LIBS})
    target_compile_options(vineyard_client PUBLIC -DWITH_JEMALLOC)

    if(BUILD_SHARED_LIBS AND BUILD_VINEYARD_PYPI_PACKAGES)
        target_compile_options(vineyard_client PRIVATE -Os)
        add_target_link_options(vineyard_client PRIVATE OPTIONS -Os)
    endif()
    install_vineyard_target(vineyard_client)
    install_vineyard_headers("${PROJECT_SOURCE_DIR}/src/client")
    list(APPEND VINEYARD_INSTALL_LIBS vineyard_client)
endif()

include("cmake/GenerateVineyard.cmake")

if(BUILD_VINEYARD_PYTHON_BINDINGS)
    set(PYBIND11_PYTHON_VERSION 3)
    add_subdirectory(thirdparty/pybind11)
    set(PYTHON_BIND_FILES "python/client.cc"
                          "python/core.cc"
                          "python/error.cc"
                          "python/pybind11_utils.cc"
                          "python/vineyard.cc")
    if(BUILD_VINEYARD_BASIC)
        list(APPEND PYTHON_BIND_FILES "python/stream.cc")
    endif()
    pybind11_add_module(_C MODULE ${PYTHON_BIND_FILES})
    target_link_libraries(_C PRIVATE vineyard_client)
    if(BUILD_VINEYARD_BASIC)
        target_link_libraries(_C PRIVATE vineyard_basic)
        target_compile_options(_C PRIVATE -DBIND_STREAM)
    endif()
    target_include_directories(_C PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                         thirdparty/pybind11/include)
    target_compile_options(_C PRIVATE -Wno-unused-value)
    set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/vineyard")
    add_target_link_options(_C PRIVATE
                            OPTIONS -Wl,-rpath,.:$ORIGIN:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(_C PRIVATE -static)
        add_target_link_options(_C PRIVATE OPTIONS -static)
    else()
        target_compile_options(_C PRIVATE -Os)
        add_target_link_options(_C PRIVATE OPTIONS -Os)
    endif()

    add_custom_target(vineyard_client_python
        ALL
        COMMAND cp "$<TARGET_FILE:_C>" "${PROJECT_SOURCE_DIR}/python/vineyard/"
        DEPENDS _C
        COMMENT "Copying python extensions."
        VERBATIM)
endif()

add_custom_target(vineyard_tests)

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

if(BUILD_VINEYARD_MIGRATION)
    add_subdirectory(modules/migrate)
    # don't includes vineyard_migrate to "VINEYARD_LIBRARIES"
endif()

if(BUILD_VINEYARD_TESTS)
    enable_testing()
    file(GLOB TEST_FILES RELATIVE "${PROJECT_SOURCE_DIR}/test"
                                  "${PROJECT_SOURCE_DIR}/test/*.cc"
    )
    foreach(f ${TEST_FILES})
        string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${f})
        set(T_NAME ${CMAKE_MATCH_1})
        message(STATUS "Found unit_test - " ${T_NAME})
        if(BUILD_VINEYARD_TESTS_ALL)
            add_executable(${T_NAME} test/${T_NAME}.cc)
        else()
            add_executable(${T_NAME} EXCLUDE_FROM_ALL test/${T_NAME}.cc)
        endif()

        target_link_libraries(${T_NAME} PRIVATE
                              vineyard_client vineyard_basic vineyard_io vineyard_graph
        )
        if(ARROW_SHARED_LIB)
            target_link_libraries(${T_NAME} PRIVATE ${ARROW_SHARED_LIB})
        else()
            target_link_libraries(${T_NAME} PRIVATE ${ARROW_STATIC_LIB})
        endif()

        if(${LIBUNWIND_FOUND})
            target_link_libraries(${T_NAME} PRIVATE ${LIBUNWIND_LIBRARIES})
        endif()

        add_test(${T_NAME}, ${T_NAME})
        add_dependencies(vineyard_tests ${T_NAME})

        if(${T_NAME} STREQUAL "delete_test" OR ${T_NAME} STREQUAL "rpc_delete_test")
            target_compile_options(${T_NAME} PRIVATE "-fno-access-control")
        endif()
    endforeach()
endif()

file(GLOB_RECURSE FILES_NEED_FORMAT "src/*.cc" "src/*.h" "src/*.hpp" "src/*.vineyard-mod"
                                    "modules/*.cc" "modules/*.h" "modules/*.vineyard-mod"
                                    "test/*.cc"
)
file(GLOB_RECURSE FILES_NEED_LINT "src/*.cc" "src/*.h" "src/*.hpp"
                                  "modules/*.cc" "modules/*.h"
                                  "test/*.cc"
)

foreach (file_path ${FILES_NEED_FORMAT})
    if (${file_path} MATCHES ".*vineyard.h" OR ${file_path} MATCHES ".*thirdparty.*")
        list(REMOVE_ITEM FILES_NEED_FORMAT ${file_path})
    endif ()
endforeach ()

foreach (file_path ${FILES_NEED_LINT})
    if (${file_path} MATCHES ".*vineyard.h" OR ${file_path} MATCHES ".*thirdparty.*")
        list(REMOVE_ITEM FILES_NEED_LINT ${file_path})
    endif ()
endforeach ()

add_custom_target(vineyard_clformat
        COMMAND clang-format --style=file -i ${FILES_NEED_FORMAT}
        COMMENT "Running clang-format."
        VERBATIM)

add_custom_target(vineyard_cpplint
        COMMAND ${PROJECT_SOURCE_DIR}/misc/cpplint.py --root=vineyard ${FILES_NEED_LINT}
        COMMENT "Running cpplint check."
        VERBATIM)

add_custom_target(python_install
        COMMAND python3 setup.py install
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        VERBATIM)

# install bundled thirdparty: flat_hash_map
install(DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty/flat_hash_map
        DESTINATION include                 # target directory
        FILES_MATCHING                      # install only matched files
        PATTERN "*.h"                       # select header files
        PATTERN "*.hpp"                     # select C++ template header files
)

# for python's setup.cfg
configure_file(setup.cfg.in "${PROJECT_SOURCE_DIR}/setup.cfg" @ONLY)
configure_file(python/vineyard/version.py.in "${PROJECT_SOURCE_DIR}/python/vineyard/version.py" @ONLY)

configure_file(vineyard-config.in.cmake
               "${PROJECT_BINARY_DIR}/vineyard-config.cmake" @ONLY
)
configure_file(vineyard-config-version.in.cmake
               "${PROJECT_BINARY_DIR}/vineyard-config-version.cmake" @ONLY
)
# install vineyard-codegen script.
get_target_property(VINEYARD_CODEGEN_SCRIPT vineyard::codegen IMPORTED_LOCATION)
install(PROGRAMS "${VINEYARD_CODEGEN_SCRIPT}"
        DESTINATION bin
        RENAME vineyard-codegen
)
install(FILES "${PROJECT_BINARY_DIR}/vineyard-config.cmake"
              "${PROJECT_BINARY_DIR}/vineyard-config-version.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindArrow.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindGFlags.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindGlog.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/FindRdkafka.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/GenerateVineyard.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/DetermineImplicitIncludes.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vineyard
)
install(EXPORT vineyard-targets
        FILE vineyard-targets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vineyard
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
