# build vineyard-basic
file(GLOB_RECURSE VINEYARD_MOD_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                    "ds/*.vineyard-mod"
                                    "stream/*.vineyard-mod")

file(GLOB_RECURSE VINEYARD_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}"
                                   "ds/*.h"
                                   "stream/*.h")

if(VINEYARD_MOD_SRCS)
    vineyard_generate(
        OUT_VAR VINEYARD_GENERATES
        VINEYARD_MODULES ${VINEYARD_MOD_SRCS}
    )
else()
    set(VINEYARD_GENERATES)
endif()

if(BUILD_VINEYARD_JAVA AND (VINEYARD_HEADERS OR VINEYARD_GENERATES))
    vineyard_generate_java(
        OUT_VAR VINEYARD_JAVA_GENERATES
        LANGUAGE java
        VINEYARD_MODULES ${VINEYARD_HEADERS} ${VINEYARD_GENERATES}
        DEPENDS ${VINEYARD_GENERATES}
    )
else()
    set(VINEYARD_JAVA_GENERATES)
endif()

add_custom_target(vineyard_basic_gen
                  DEPENDS ${VINEYARD_GENERATES}
                  COMMENT "Running code generation for vineyard_basic."
)

add_custom_target(vineyard_basic_gen_java
                  DEPENDS vineyard_basic_gen
                          ${VINEYARD_JAVA_GENERATES}
                  COMMENT "Running java code generation for vineyard_basic."
)

file(GLOB_RECURSE BASIC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_basic ${BASIC_SRC_FILES})
target_add_debuginfo(vineyard_basic)
find_package(MPI REQUIRED)
target_link_libraries(vineyard_basic PUBLIC vineyard_client
                                            ${ARROW_SHARED_LIB}
                                            ${GLOG_LIBRARIES}
                                            ${MPI_CXX_LIBRARIES}
)
target_include_directories(vineyard_basic PUBLIC ${ARROW_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})

find_package(libgrapelite 0.3.4 QUIET)
if(LIBGRAPELITE_INCLUDE_DIRS)
    message(STATUS "-- Found libgrape-lite: ${LIBGRAPELITE_INCLUDE_DIRS}")
    target_include_directories(vineyard_basic PUBLIC ${LIBGRAPELITE_INCLUDE_DIRS})
else()
    # use bundled libgrape-lite
    message(STATUS "-- Building libgrape-lite from submodule: ${CMAKE_SOURCE_DIR}/thirdparty/libgrape-lite")
    set(BUILD_LIBGRAPELITE_DOCS OFF CACHE BOOL "no libgrape-lite docs")
    set(BUILD_LIBGRAPELITE_TESTS OFF CACHE BOOL "no libgrape-lite tests")
    # use `add_subdirectory` to use the same CMAKE_BUILD_TYPE with vineyard itself and
    # ensure the libgrapelite-targets-{debug/release}.cmake been generated during installation.
    add_subdirectory("${CMAKE_SOURCE_DIR}/thirdparty/libgrape-lite"
                     "${CMAKE_SOURCE_DIR}/thirdparty/libgrape-lite"
    )
    target_include_directories(vineyard_basic PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thirdparty/libgrape-lite>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thirdparty/libgrape-lite/thirdparty>
        $<INSTALL_INTERFACE:include>
    )
endif()

# install bundled thirdparty: flat_hash_map
install(DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty/flat_hash_map
        DESTINATION include/vineyard/contrib    # target directory
        FILES_MATCHING                          # install only matched files
        PATTERN "*.h"                           # select header files
        PATTERN "*.hpp"                         # select C++ template header files
)

# install bundled thirdparty: wyhash
install(DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty/wyhash
        DESTINATION include/vineyard/contrib    # target directory
        FILES_MATCHING                          # install only matched files
        PATTERN "*.h"                           # select header files
        PATTERN "*.hpp"                         # select C++ template header files
)

# install bundled thirdparty: cityhash
install(DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty/cityhash
        DESTINATION include/vineyard/contrib    # target directory
        FILES_MATCHING                          # install only matched files
        PATTERN "*.h"                           # select header files
        PATTERN "*.hpp"                         # select C++ template header files
)

target_include_directories(vineyard_basic PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty>
    $<INSTALL_INTERFACE:include/vineyard/contrib>
)

add_dependencies(vineyard_basic vineyard_basic_gen)

add_dependencies(vineyard_codegen vineyard_basic_gen)
if(VINEYARD_JAVA_GENERATES)
    add_dependencies(vineyard_codegen_java vineyard_basic_gen_java)
endif()

install_export_vineyard_target(vineyard_basic)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
