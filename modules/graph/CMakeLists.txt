include(GNUInstallDirs)

# build vineyard-graph
file(GLOB_RECURSE VINEYARD_MOD_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                    "*.vineyard-mod")

file(GLOB_RECURSE VINEYARD_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}"
                                   "*.h")

find_package(MPI REQUIRED)
find_package(libgrapelite REQUIRED)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/grin/include/proto/graph.proto)


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

file(GLOB_RECURSE GRAPH_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "fragment/*.cc"
                                                                "loader/*.cc"
                                                                "utils/*.cc"
                                                                "vertex_map/*.cc"
                                                                "writer/*.cc"
)

message(${GRAPH_SRC_FILES})

add_library(vineyard_graph ${GRAPH_SRC_FILES})
target_add_debuginfo(vineyard_graph)
target_compile_options(vineyard_graph PUBLIC "-fopenmp")
target_link_options(vineyard_graph PUBLIC "-fopenmp")
target_include_directories(vineyard_graph PUBLIC
                                          ${MPI_CXX_INCLUDE_PATH}
)

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

find_package(libgrapelite QUIET)
if(LIBGRAPELITE_INCLUDE_DIRS)
    message(STATUS "-- Found libgrape-lite: ${LIBGRAPELITE_INCLUDE_DIRS}")
    target_include_directories(vineyard_graph PUBLIC ${LIBGRAPELITE_INCLUDE_DIRS})
else()
    # use bundled libgrape-lite
    message(STATUS "-- Building libgrape-lite from submodule: ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite")
    set(BUILD_LIBGRAPELITE_DOCS OFF CACHE BOOL "no libgrape-lite docs")
    set(BUILD_LIBGRAPELITE_TESTS OFF CACHE BOOL "no libgrape-lite tests")
    add_subdirectory_static("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite"
                            "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/libgrape-lite"
    )
    target_include_directories(vineyard_graph PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libgrape-lite/thirdparty>
        $<INSTALL_INTERFACE:include>
    )
endif()

if(BUILD_VINEYARD_GRAPH_WITH_GAR)
    target_compile_definitions(vineyard_graph PUBLIC -DENABLE_GAR)
    find_package(gar QUIET)
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

install_vineyard_target(vineyard_graph)
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
    list(REMOVE_ITEM TEST_FILES "grin_test.cc")
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

# grin lib
file(GLOB_RECURSE GRIN_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "grin/*.cc")
add_library(vineyard_grin ${GRIN_SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(vineyard_grin PRIVATE vineyard_graph ${Protobuf_LIBRARIES})
install_vineyard_target(vineyard_grin)


# grin test
if(BUILD_VINEYARD_TESTS)
    enable_testing()
    file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/test" "${CMAKE_CURRENT_SOURCE_DIR}/test/grin_*.cc")
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
                              vineyard_grin
                              ${Protobuf_LIBRARIES}
                              ${ARROW_SHARED_LIB}
                              ${MPI_CXX_LIBRARIES})
        if(${LIBUNWIND_FOUND})
            target_link_libraries(${T_NAME} PRIVATE ${LIBUNWIND_LIBRARIES})
        endif()
        add_test(${T_NAME}, ${T_NAME})
    endforeach()
endif()