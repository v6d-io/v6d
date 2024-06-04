enable_testing()

if(USE_CUDA)
    include(CheckLanguage)

    enable_language(CUDA)
    check_language(CUDA)
endif()

macro(add_test_case testname testfile)
    if(BUILD_VINEYARD_TESTS_ALL)
        add_executable(${testname} ${testfile})
    else()
        add_executable(${testname} EXCLUDE_FROM_ALL ${testfile})
    endif()

    target_include_directories(${testname} PRIVATE ${GLOG_INCLUDE_DIRS})
    target_link_libraries(${testname} PRIVATE ${VINEYARD_INSTALL_LIBS} ${GLOG_LIBRARIES})
    if(NOT ("${VINEYARD_INSTALL_LIBS}" STREQUAL "vineyard_client"))
        if(ARROW_SHARED_LIB)
            target_link_libraries(${testname} PRIVATE ${ARROW_SHARED_LIB})
        else()
            target_link_libraries(${testname} PRIVATE ${ARROW_STATIC_LIB})
        endif()
    endif()

    if(${LIBUNWIND_FOUND})
        target_link_libraries(${testname} PRIVATE ${LIBUNWIND_LIBRARIES})
    endif()

    add_test(${testname} ${testname})
    add_dependencies(vineyard_tests ${testname})
endmacro()

file(GLOB TEST_FILES RELATIVE "${PROJECT_SOURCE_DIR}/test"
                              "${PROJECT_SOURCE_DIR}/test/*.cc"
                              "${PROJECT_SOURCE_DIR}/test/*.cu"
)
foreach(testfile ${TEST_FILES})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${testfile})
    set(testname ${CMAKE_MATCH_1})

    if(${testname} STREQUAL "gpumalloc_test" AND NOT USE_CUDA)
        continue()
    endif()

    message(STATUS "Found unit_test - " ${testname})
    add_test_case(${testname} ${testfile})

    if(USE_CUDA)
        target_compile_options(${testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe "--diag_suppress=284 --diag_suppress=815 --diag_suppress=997">)
    endif()

    if(${testname} STREQUAL "delete_test" OR ${testname} STREQUAL "rpc_delete_test")
        target_compile_options(${testname} PRIVATE "-fno-access-control")
    endif()

    if(${testname} STREQUAL "compressor_test")
        target_sources(${testname} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/compression/compressor.cc)
        target_link_libraries(${testname} PRIVATE libzstd_static)
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
