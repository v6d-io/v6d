set(BENCH_BLOB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/blob_test.cc)

if(BUILD_VINEYARD_BENCHMARKS_ALL)
    set(blob_benchmark_options "")
else()
    set(blob_benchmark_options "EXCLUDE_FROM_ALL")
endif()

add_executable(blob_benchmark ${blob_benchmark_options} ${BENCH_BLOB_SRCS})

target_include_directories(blob_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(blob_benchmark PRIVATE vineyard_client)

add_dependencies(vineyard_benchmarks blob_benchmark)
