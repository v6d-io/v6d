enable_testing()

macro(add_test_case testname testfile)
    if(BUILD_VINEYARD_TESTS_ALL)
        add_executable(${testname} ${testfile})
    else()
        add_executable(${testname} EXCLUDE_FROM_ALL ${testfile})
    endif()

    target_link_libraries(${testname} PRIVATE ${VINEYARD_INSTALL_LIBS})
    if(ARROW_SHARED_LIB)
        target_link_libraries(${testname} PRIVATE ${ARROW_SHARED_LIB})
    else()
        target_link_libraries(${testname} PRIVATE ${ARROW_STATIC_LIB})
    endif()

    if(${LIBUNWIND_FOUND})
        target_link_libraries(${testname} PRIVATE ${LIBUNWIND_LIBRARIES})
    endif()

    add_test(${testname} ${testname})
    add_dependencies(vineyard_tests ${testname})
endmacro()

file(GLOB TEST_FILES RELATIVE "${PROJECT_SOURCE_DIR}/test"
                              "${PROJECT_SOURCE_DIR}/test/*.cc"
)
foreach(testfile ${TEST_FILES})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${testfile})
    set(testname ${CMAKE_MATCH_1})

    message(STATUS "Found unit_test - " ${testname})
    add_test_case(${testname} ${testfile})

    if(${testname} STREQUAL "delete_test" OR ${testname} STREQUAL "rpc_delete_test")
        target_compile_options(${testname} PRIVATE "-fno-access-control")
    endif()

    if(${testname} STREQUAL "lru_test")
        target_link_libraries(${testname} PRIVATE TBB::tbb)
    endif()

    if(${testname} STREQUAL "fuse_cache_manager_test")
        message(STATUS "Found fuse_cache_manager_test - " ${testname} ${PROJECT_SOURCE_DIR})
        target_include_directories(${testname} BEFORE PUBLIC ${PROJECT_SOURCE_DIR})
    endif()
    if(${testname} STREQUAL "allocator_test" OR ${testname} STREQUAL "mimalloc_test")
        if(BUILD_VINEYARD_MALLOC)
            target_compile_options(${testname} PRIVATE -DWITH_MIMALLOC)
        endif()
    endif()

    if(${testname} STREQUAL "hosseinmoein_dataframe_test")
        if(BUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME)
            target_compile_options(${testname} PRIVATE -DWITH_HOSSEINMOEIN_DATAFRAME)
            set_property(TARGET ${testname} PROPERTY CXX_STANDARD 17)
        endif()
    endif()
endforeach()
