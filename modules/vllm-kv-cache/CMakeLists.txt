file(GLOB VINEYARD_VLLM_KV_CACHE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                   "ds/*.cc"
                                   "src/*.cc"
                                   "src/storage/*.cc"
                                   "src/io/*.cc"
                                   "*.cc"
)

add_library(vineyard_vllm_kv_cache ${VINEYARD_VLLM_KV_CACHE_SRCS})
target_link_libraries(vineyard_vllm_kv_cache PRIVATE ${GLOG_LIBRARIES} ${AIO_LIBRARIES})
target_link_libraries(vineyard_vllm_kv_cache PUBLIC vineyard_client )

install_export_vineyard_target(vineyard_vllm_kv_cache)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(vineyard_vllm_kv_cache_tests)
add_dependencies(vineyard_tests vineyard_vllm_kv_cache_tests)

if(BUILD_VINEYARD_TESTS)
    enable_testing()
    # Define common test sources
    set(VLLM_STORAGE_TEST_COMMON_SOURCES 
        tests/vllm_storage_test_common.cc)
        
    file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/tests" "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc")
    foreach(f ${TEST_FILES})
        if (${f} STREQUAL "vllm_kv_storage_unittest.cc")
            # Skip the vllm_kv_storage_unittest.cc. It need to be fixed.
            continue()
        endif()
        string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${f})
        set(T_NAME ${CMAKE_MATCH_1})
        message(STATUS "Found unit_test - " ${T_NAME})
        if(BUILD_VINEYARD_TESTS_ALL)
            if(${T_NAME} STREQUAL "vllm_storage_test_common")
                # Skip the common file as it's not a test executable
                continue()
            endif()
            add_executable(${T_NAME} tests/${T_NAME}.cc)
        else()
            if(${T_NAME} STREQUAL "vllm_storage_test_common")
                # Skip the common file as it's not a test executable
                continue()
            endif()
            add_executable(${T_NAME} EXCLUDE_FROM_ALL tests/${T_NAME}.cc)
        endif()
        
        # Add common sources to test executables
        if(${T_NAME} MATCHES "vllm_storage_disk_test")
            target_sources(${T_NAME} PRIVATE ${VLLM_STORAGE_TEST_COMMON_SOURCES})
        endif()
        
        # Add common sources to performance test executables
        if(${T_NAME} MATCHES "vllm_storage_disk_perftest")
            target_sources(${T_NAME} PRIVATE ${VLLM_STORAGE_TEST_COMMON_SOURCES})
        endif()
        target_include_directories(${T_NAME} PRIVATE
                                   ${GTEST_INCLUDE_DIRS}
                                   ${GLOG_INCLUDE_DIRS})

        target_link_libraries(${T_NAME} PRIVATE
                              libzstd_static
                              vineyard_vllm_kv_cache
                              ${GLOG_LIBRARIES}
                              ${GTEST_LIBRARIES})
        if(${LIBUNWIND_FOUND})
            target_link_libraries(${T_NAME} PRIVATE ${LIBUNWIND_LIBRARIES})
        endif()
        add_test(${T_NAME}, ${T_NAME})
        add_dependencies(vineyard_vllm_kv_cache_tests ${T_NAME})
    endforeach()
endif()
