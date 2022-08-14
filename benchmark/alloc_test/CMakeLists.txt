set(BENCH_ALLOCATOR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/bench_allocator.cpp)

macro(add_benchmark target)
    if(BUILD_VINEYARD_BENCHMARKS_ALL)
        add_executable(${target} ${BENCH_ALLOCATOR_SRCS})
    else()
        add_executable(${target} EXCLUDE_FROM_ALL ${BENCH_ALLOCATOR_SRCS})
    endif()
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE mimalloc-static
                                            vineyard_client
                                            vineyard_malloc)
    # disable all warnings
    target_compile_options(${target} PUBLIC "-w")
    add_dependencies(vineyard_benchmarks ${target})
endmacro()

add_benchmark(bench_allocator_system)
target_compile_options(bench_allocator_system PRIVATE -DBENCH_SYSTEM)

add_benchmark(bench_allocator_mimalloc)
target_compile_options(bench_allocator_mimalloc PRIVATE -DBENCH_MIMALLOC)

add_benchmark(bench_allocator_vineyard)
target_compile_options(bench_allocator_vineyard PRIVATE -DBENCH_VINEYARD)
