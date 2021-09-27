# alloc_test
Performance testing or system benchmarks for jemalloc

Referred from https://github.com/daanx/mimalloc-bench/blob/master/bench/alloc-test/allocator_tester.h

To run this benchmark, build with

- g++ -std=c++11 bench_allocator.cpp -D WITH_JEMALLOC -I ../../src/ -I ../../modules -I ../../thirdparty -I ../../thirdparty/ctti/include/ -lglog -lvineyard_client -lvineyard_malloc -o alloc_test

Then run with

 - ./alloc_test