include(GNUInstallDirs)

# options to control several default behaviors of vineyard graph
option(VINEYARD_GRAPH_MAX_LABEL_ID "Maximum label id value of vineyard graphs, defaults to 128" OFF)

option(MY_OPTION "Description of My Option" OFF)
set(VINEYARD_GRAPH_MAX_LABEL_ID_DEFAULT_VALUE 128 CACHE STRING "Default value for maximum label if for vineyard graphs")
set_property(CACHE VINEYARD_GRAPH_MAX_LABEL_ID PROPERTY STRINGS "1;2;4;8;16;32;64;128")

if (VINEYARD_GRAPH_MAX_LABEL_ID)
    message(STATUS "Setting VINEYARD_GRAPH_MAX_LABEL_ID to '${VINEYARD_GRAPH_MAX_LABEL_ID}'")
else()
    message(STATUS "Setting VINEYARD_GRAPH_MAX_LABEL_ID to default value: '${VINEYARD_GRAPH_MAX_LABEL_ID_DEFAULT_VALUE}'")
endif()

# build vineyard-graph
file(GLOB_RECURSE VINEYARD_MOD_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                    "*.vineyard-mod")

file(GLOB_RECURSE VINEYARD_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}"
                                   "*.h")

find_package(MPI REQUIRED)

if(VINEYARD_MOD_SRCS)
    vineyard_generate(
        OUT_VAR VINEYARD_GENERATES
        VINEYARD_MODULES ${VINEYARD_MOD_SRCS}
    )
else()
    set(VINEYARD_GENERATES)
endif()

if(FALSE AND BUILD_VINEYARD_JAVA AND (VINEYARD_HEADERS OR VINEYARD_GENERATES))
    vineyard_generate_java(
        OUT_VAR VINEYARD_JAVA_GENERATES
        VINEYARD_MODULES ${VINEYARD_HEADERS} ${VINEYARD_GENERATES}
        DEPENDS ${VINEYARD_GENERATES}
    )
else()
    set(VINEYARD_JAVA_GENERATES)
endif()

add_custom_target(vineyard_graph_gen
                  DEPENDS ${VINEYARD_GENERATES}
                  COMMENT "Running code generation for vineyard_graph."
)

add_custom_target(vineyard_graph_java_gen
                  DEPENDS vineyard_graph_gen
                          ${VINEYARD_JAVA_GENERATES}
                  COMMENT "Running java code generation for vineyard_graph."
)

add_dependencies(vineyard_graph_gen vineyard_basic_gen)

# add powturbo library
set(POWTURBO_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/powturbo/lib/v8.c
)
add_library(powturbo-objects OBJECT ${POWTURBO_SRC_FILES})
set(powturbo-target-objects $<TARGET_OBJECTS:powturbo-objects>)
target_compile_options(powturbo-objects PRIVATE -Wp,-w -O3 -fPIC -falign-loops -fstrict-aliasing)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_options(powturbo-objects PRIVATE -march=haswell -mtune=haswell -mno-avx -mno-aes -DPLAIN)
else()
    target_compile_options(powturbo-objects PRIVATE "-D_NAVX2")
endif()

file(GLOB_RECURSE GRAPH_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "fragment/*.cc"
                                                                "loader/*.cc"
                                                                "utils/*.cc"
                                                                "vertex_map/*.cc"
                                                                "writer/*.cc"
)

add_library(vineyard_graph ${GRAPH_SRC_FILES} ${powturbo-target-objects})
target_add_debuginfo(vineyard_graph)
if(NOT APPLE)
    target_compile_options(vineyard_graph PUBLIC "-fopenmp")
    target_link_options(vineyard_graph PUBLIC "-fopenmp")
endif()
target_include_directories(vineyard_graph PUBLIC ${MPI_CXX_INCLUDE_PATH})

find_package(Boost COMPONENTS leaf)
if(Boost_LEAF_FOUND)
    target_include_directories(vineyard_graph PUBLIC ${Boost_INCLUDE_DIRS})
else()
    # boost::leaf for error_handling
    add_subdirectory_static("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/boost-leaf"
                            "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/boost-leaf"
    )
    # install boost leaf
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/boost-leaf/include/"
            DESTINATION ${CMAKE_INSTALL_PREFIX}/include/vineyard/contrib
            PATTERN "*.hpp"
    )
    target_include_directories(vineyard_graph PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/boost-leaf/include>
        $<INSTALL_INTERFACE:include/vineyard/contrib>
    )
endif()

find_package(libgrapelite 0.2.5 QUIET)
if(LIBGRAPELITE_INCLUDE_DIRS)
    message(STATUS "-- Found libgrape-lite: ${LIBGRAPELITE_INCLUDE_DIRS}")
    target_include_directories(vineyard_graph PUBLIC ${LIBGRAPELITE_INCLUDE_DIRS})
else()
    # use bundled libgrape-lite
    message(STATUS "-- Building libgrape-lite from submodule: ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite")
    set(BUILD_LIBGRAPELITE_DOCS OFF CACHE BOOL "no libgrape-lite docs")
    set(BUILD_LIBGRAPELITE_TESTS OFF CACHE BOOL "no libgrape-lite tests")
    # use `add_subdirectory` to use the same CMAKE_BUILD_TYPE with vineyard itself and
    # ensure the libgrapelite-targets-{debug/release}.cmake been generated during installation.
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite"
                     "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/libgrape-lite"
    )
    target_include_directories(vineyard_graph PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite/thirdparty>
        $<INSTALL_INTERFACE:include>
    )
endif()

# install headers for powturbo
target_include_directories(vineyard_graph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty>
    $<INSTALL_INTERFACE:include>
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/powturbo/include"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/vineyard/contrib/powturbo
    PATTERN "*.h"
)

if(BUILD_VINEYARD_GRAPH_WITH_GAR)
    target_compile_definitions(vineyard_graph PUBLIC -DENABLE_GAR)
    find_package(gar 0.11.3 QUIET)
    if (gar_FOUND)
        message(STATUS "-- Found GraphAr: ${GAR_LIBRARIES}")
        target_include_directories(vineyard_graph PRIVATE ${GAR_INCLUDE_DIRS})
        target_link_libraries(vineyard_graph PRIVATE ${GAR_LIBRARIES})
    else()
        # build and install gar
        message(STATUS "-- Building GraphAr from submodule: ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/GraphAr")
        add_subdirectory_static("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/GraphAr/cpp"
                                "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/GraphAr"
        )
        # depends on gar
        target_link_libraries(vineyard_graph PRIVATE gar)
    endif()
endif()

if (VINEYARD_GRAPH_MAX_LABEL_ID)
    target_compile_options(vineyard_graph PUBLIC -DVINEYARD_GRAPH_MAX_LABEL_ID=${VINEYARD_GRAPH_MAX_LABEL_ID})
endif()

target_link_libraries(vineyard_graph PUBLIC vineyard_client
                                            vineyard_basic
                                            vineyard_io
                                            ${ARROW_SHARED_LIB}
                                            ${MPI_CXX_LIBRARIES}
)
if(${LIBUNWIND_FOUND})
    target_link_libraries(vineyard_graph PUBLIC ${LIBUNWIND_LIBRARIES})
endif()
if(APPLE)
    target_compile_options(vineyard_graph PUBLIC -Wno-unused-result)
endif()

add_dependencies(vineyard_graph vineyard_graph_gen)

add_dependencies(vineyard_codegen vineyard_graph_gen)
if(VINEYARD_JAVA_GENERATES)
    add_dependencies(vineyard_codegen_java vineyard_graph_java_gen)
endif()

install_export_vineyard_target(vineyard_graph)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

# add the vineyard-graph-loader target
file(GLOB_RECURSE GRAPH_LOADER_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "tools/*.cc")
add_executable(vineyard-graph-loader ${GRAPH_LOADER_SRC_FILES})
target_link_libraries(vineyard-graph-loader PRIVATE vineyard_graph)
install_vineyard_target(vineyard-graph-loader)

add_custom_target(vineyard_graph_tests)
add_dependencies(vineyard_tests vineyard_graph_tests)

if(BUILD_VINEYARD_TESTS)
    enable_testing()
    file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/test" "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cc")
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
                              vineyard_graph
                              ${ARROW_SHARED_LIB}
                              ${MPI_CXX_LIBRARIES})
        if(${LIBUNWIND_FOUND})
            target_link_libraries(${T_NAME} PRIVATE ${LIBUNWIND_LIBRARIES})
        endif()
        add_test(${T_NAME}, ${T_NAME})
        add_dependencies(vineyard_graph_tests ${T_NAME})
    endforeach()
endif()
