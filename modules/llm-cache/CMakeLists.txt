file(GLOB VINEYARD_LLM_CACHE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                  "ds/*.cc"
                                  "ds/*.h"
                                  "hash/*.h"
                                  "radix-tree/*.cc"
                                  "radix-tree/*.h"
                                  "storage/*.cc"
                                  "storage/*.h"
                                  "${PROJECT_SOURCE_DIR}/thirdparty/rax/*.cc"
                                  "${PROJECT_SOURCE_DIR}/thirdparty/rax/*.h"
                                  "${PROJECT_SOURCE_DIR}/thirdparty/MurmurHash3/*.cc"
                                  "${PROJECT_SOURCE_DIR}/thirdparty/MurmurHash3/*.h"
                                  "${PROJECT_SOURCE_DIR}/thirdparty/cityhash/*.hpp"
)

add_library(vineyard_llm_cache ${VINEYARD_LLM_CACHE_SRCS})
target_link_libraries(vineyard_llm_cache PRIVATE libzstd_static ${GLOG_LIBRARIES})
target_link_libraries(vineyard_llm_cache PUBLIC vineyard_client)

# install bundled thirdparty: rax and MurmurHash3
install(DIRECTORY
            ${PROJECT_SOURCE_DIR}/thirdparty/rax
            ${PROJECT_SOURCE_DIR}/thirdparty/MurmurHash3
            ${PROJECT_SOURCE_DIR}/thirdparty/cityhash
        DESTINATION include/vineyard/contrib    # target directory
        FILES_MATCHING                          # install only matched files
        PATTERN "*.h"                           # select header files
        PATTERN "*.hpp"                         # select C++ template header files
)

install_export_vineyard_target(vineyard_llm_cache)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(vineyard_llm_cache_tests)
add_dependencies(vineyard_tests vineyard_llm_cache_tests)

if(BUILD_VINEYARD_TESTS)
    enable_testing()
    file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/tests" "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc")
    foreach(f ${TEST_FILES})
        string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${f})
        set(T_NAME ${CMAKE_MATCH_1})
        message(STATUS "Found unit_test - " ${T_NAME})
        if(BUILD_VINEYARD_TESTS_ALL)
            add_executable(${T_NAME} tests/${T_NAME}.cc)
        else()
            add_executable(${T_NAME} EXCLUDE_FROM_ALL tests/${T_NAME}.cc)
        endif()
        target_link_libraries(${T_NAME} PRIVATE
                              libzstd_static
                              vineyard_basic
                              vineyard_llm_cache)
        if(${LIBUNWIND_FOUND})
            target_link_libraries(${T_NAME} PRIVATE ${LIBUNWIND_LIBRARIES})
        endif()
        add_test(${T_NAME}, ${T_NAME})
        add_dependencies(vineyard_llm_cache_tests ${T_NAME})
    endforeach()
endif()
