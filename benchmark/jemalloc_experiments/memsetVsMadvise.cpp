#include <sys/mman.h>

#include <stdlib.h>

#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#include <gflags/gflags.h>

#include "util.h"

DEFINE_int32(num_runs, 100,
    "Number of times to zero the pages (per page count)");
DEFINE_int32(num_pages_min, 1, "Minimum number of pages to zero");
DEFINE_int32(num_pages_max, 50, "Maximum number of pages to zero");
DEFINE_int32(num_threads, 1,
    "Number of threads on which to try the experiment at once.");
DEFINE_bool(touch_after_zero, false,
    "Whether to actually try touching the pages we zero.");

struct Result {
  std::uint64_t memsetCycles;
  std::uint64_t madviseDontneedCycles;
  std::uint64_t madviseDontneedWillneedCycles;

  Result()
    : memsetCycles(0),
      madviseDontneedCycles(0),
      madviseDontneedWillneedCycles(0) {}

  void accum(const Result& other) {
    memsetCycles += other.memsetCycles;
    madviseDontneedCycles += other.madviseDontneedCycles;
    madviseDontneedWillneedCycles += other.madviseDontneedWillneedCycles;
  }
};

void maybeTouchPages(void* beginv, std::size_t length) {
  char* begin = static_cast<char*>(beginv);
  if (FLAGS_touch_after_zero) {
    for (char* ptr = begin; ptr != begin + length; ptr += 4096) {
      *ptr = 0;
    }
  }
}

void zeroMemset(void* ptr, std::size_t size) {
  std::memset(ptr, 0, size);
}

void zeroMadviseDontneed(void* ptr, std::size_t size) {
  int err = madvise(ptr, size, MADV_DONTNEED);
  if (err != 0) {
    std::cerr << "Couldn't madvise(... MADV_DONTNEED); error was "
      << err << std::endl;
    exit(1);
  }
}

void zeroMadviseDontneedWillneed(void* ptr, std::size_t size) {
  int err = madvise(ptr, size, MADV_DONTNEED);
  if (err != 0) {
    std::cerr << "Couldn't madvise(..., MADV_DONTNEED); error was "
      << err << std::endl;
    exit(1);
  }
  err = madvise(ptr, size, MADV_WILLNEED);
  if (err != 0) {
    std::cerr << "Couldn't madvise(..., MAP_POPULATE); error was "
      << err << std::endl;
    exit(1);
  }
}

Result runTest(std::size_t size) {
  Result result;
  void *ptr;
  int err = posix_memalign(&ptr, 4096, size);
  if (err != 0) {
    std::cerr << "Couldn't allocate; error was " << err << std::endl;
    exit(1);
  }
  // Touch all the pages from this thread.
  std::memset(ptr, 0, size);
  // Touch all the pages from another thread.
  std::async(std::launch::async, std::memset, ptr, 0, size).get();

  // We'll probably be dealing with uncached memory here; we care about this
  // difference when pulling memory out of an inactive state.
  util::flushCache(ptr, size);
  result.memsetCycles = util::runTimed([&]() {
    zeroMemset(ptr, size);
    maybeTouchPages(ptr, size);
  });
  util::flushCache(ptr, size);
  result.madviseDontneedCycles = util::runTimed([&]() {
    zeroMadviseDontneed(ptr, size);
    maybeTouchPages(ptr, size);
  });
  util::flushCache(ptr, size);
  result.madviseDontneedWillneedCycles = util::runTimed([&]() {
    zeroMadviseDontneedWillneed(ptr, size);
    maybeTouchPages(ptr, size);
  });

  return result;
}

int main(int argc, char** argv) {
  std::string usage =
    "This program benchmarks memset vs madvise for zeroing memory.\n"
    "Sample usage:\n";
  usage += argv[0];
  usage += " --num_pages_min=20 --num_pagse_max=50 --num_runs=30 ";
  usage += "--num_threads=4 --touch_after_zero=true";

  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  for (int i = FLAGS_num_pages_min; i <= FLAGS_num_pages_max; ++i) {
    Result sum;
    for (int j = 0; j < FLAGS_num_runs; ++j) {
      std::vector<std::future<Result>> results;
      for (int k = 0; k < FLAGS_num_threads; ++k) {
        results.push_back(std::async(std::launch::async, runTest, 4096 * i));
      }
      for (int k = 0; k < FLAGS_num_threads; ++k) {
        sum.accum(results[k].get());
      }
    }
    std::cout << "When zeroing " << i << " pages (averaging across "
      << FLAGS_num_runs << " runs of " << FLAGS_num_threads << " threads:\n"
      << "    memset:  " << sum.memsetCycles / FLAGS_num_runs << " cycles\n"
      << "    madvise(..., MADV_DONTNEED): "
      << sum.madviseDontneedCycles / FLAGS_num_runs << " cycles\n"
      << "    madvise(..., MADV_DONTNEED); madvise(..., MADV_WILLNEED): "
      << sum.madviseDontneedWillneedCycles / FLAGS_num_runs << " cycles\n";
  }

  return 0;
}
