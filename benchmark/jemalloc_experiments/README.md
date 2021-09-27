# jemalloc-experiments
Performance testing or system benchmarks for jemalloc

Referenced from https://github.com/jemalloc/jemalloc-experiments

This repository contains programs that are useful in jemalloc development, but
not portable or polished enough to live in the main repo. Here we have some
relaxed constraints on languages, dependencies, and the build environment.

Libraries that need to be installed:
  - gflags

To build this benchmark, run
 - g++ -std=c++11 stress_test/Main.cpp stress_test/Producers.cpp stress_test/Mixer.cpp stress_test/ThreadObject.cpp stress_test/Distribution.cpp stress_test/Allocation.cpp 
   -D WITH_JEMALLOC -I ../../src/ -I ../../modules -I ../../thirdparty -I ../../thirdparty/ctti/include/ -I stress_test/ -lglog -lvineyard_client -lvineyard_malloc -lgflags
   -o stress
   
To run this benchmark, run
 - ./stress --distribution_file=stress_test/distributions/adfinder.txt --num_threads=1
 - ./stress --distribution_file=stress_test/distributions/adindexer.txt --num_threads=1
 - ./stress --distribution_file=stress_test/distributions/multifeed.txt --num_threads=1