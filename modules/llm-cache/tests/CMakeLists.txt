enable_testing()

macro(add_test_case testname testfile)
    add_executable(${testname} ${testfile})

    target_include_directories(${testname} PRIVATE ${GLOG_INCLUDE_DIRS})
    target_link_libraries(${testname} PRIVATE vineyard_llm_cache)

    add_test(${testname} ${testname})
    add_dependencies(vineyard_tests ${testname})
endmacro()

file(GLOB LLM_TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "./*.cc")

foreach(testfile ${LLM_TEST_FILES})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${testfile})
    set(testname ${CMAKE_MATCH_1})

    if(${testname} STREQUAL "gpumalloc_test" AND NOT USE_CUDA)
        continue()
    endif()

    message(STATUS "Found unit_test - " ${testname})
    add_test_case(${testname} ${testfile})
endforeach()

