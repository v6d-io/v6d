# build vineyard-graph
file(GLOB_RECURSE VINEYARD_MOD_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                    "*.vineyard-mod")

find_package(MPI REQUIRED)
find_package(libgrapelite REQUIRED)

if (VINEYARD_MOD_SRCS)
    vineyard_generate(
        OUT_VAR VINEYARD_GENERATES
        VINEYARD_MODULES ${VINEYARD_MOD_SRCS}
    )
    add_custom_target(vineyard_graph_gen
                      DEPENDS ${VINEYARD_GENERATES}
                      COMMENT "Running code generation for vineyard_graph."
    )
    add_dependencies(vineyard_graph_gen vineyard_basic_gen)
endif()

file(GLOB_RECURSE GRAPH_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "fragment/*.cc"
                                                                "loader/*.cc"
                                                                "utils/*.cc"
                                                                "vertex_map/*.cc"
)

option(BUILD_VINEYARD_GRAPH_SELECTOR "Enable vineyard's selector operators on graphs" ON)

# boost::leaf for error_handling
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/boost-leaf"
                 "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/boost-leaf"
)

add_library(vineyard_graph ${GRAPH_SRC_FILES})
target_include_directories(vineyard_graph PUBLIC
                                          ${LIBGRAPELITE_INCLUDE_DIRS}
                                          ${MPI_CXX_INCLUDE_PATH}
)
target_include_directories(vineyard_graph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/boost-leaf/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vineyard_graph vineyard_client
                                     vineyard_basic
                                     vineyard_io
                                     ${ARROW_SHARED_LIB}
                                     ${MPI_CXX_LIBRARIES}
)
if(APPLE)
    target_compile_options(vineyard_graph PUBLIC -Wno-unused-result)
endif()
if(VINEYARD_MOD_SRCS)
    add_dependencies(vineyard_graph vineyard_graph_gen)
endif()

if(BUILD_VINEYARD_GRAPH_SELECTOR)
    target_compile_definitions(vineyard_graph PUBLIC -DENABLE_SELECTOR)
endif()

install_vineyard_target(vineyard_graph)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

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
        target_link_libraries(${T_NAME}
                              vineyard_graph
                              ${ARROW_SHARED_LIB}
                              ${CPPNETLIB_LIBRARIES}
                              ${MPI_CXX_LIBRARIES})
        add_test(${T_NAME}, ${T_NAME})
        add_dependencies(vineyard_tests ${T_NAME})
    endforeach()
endif()
