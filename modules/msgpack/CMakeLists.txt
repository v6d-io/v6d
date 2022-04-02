# build vineyard-msgpack
file(GLOB MSGPACK_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_msgpack ${MSGPACK_SRC_FILES})
target_link_libraries(vineyard_msgpack PUBLIC vineyard_client
                                              vineyard_basic
                                              ${ARROW_SHARED_LIB}
)
target_link_libraries(vineyard_msgpack PRIVATE msgpackc-cxx)

install_vineyard_target(vineyard_msgpack)
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
        target_link_libraries(${T_NAME} PRIVATE
                              vineyard_msgpack
                              ${ARROW_SHARED_LIB}
                              ${MPI_CXX_LIBRARIES})
        if(${LIBUNWIND_FOUND})
            target_link_libraries(${T_NAME} PRIVATE ${LIBUNWIND_LIBRARIES})
        endif()
        add_test(${T_NAME}, ${T_NAME})
        add_dependencies(vineyard_tests ${T_NAME})
    endforeach()
endif()
