/* -------------------------------------------------------------------------------
 * Copyright (c) 2018, OLogN Technologies AG
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------------
 *
 * Memory allocator tester
 *
 * v.1.00    Jun-22-2018    Initial release
 *
 * -------------------------------------------------------------------------------*/

#include <sys/mman.h>

#include <time.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "alloc_test.h"  // NOLINT(build/include_subdir)

#include "common/util/likely.h"
#include "malloc/malloc_wrapper.h"

#if defined(BENCH_MIMALLOC)
#include "malloc/mimalloc_allocator.h"
#include "mimalloc/include/mimalloc.h"
#endif

// #define BENCH_SYSTEM
// #define BENCH_MIMALLOC
// #define BENCH_VINEYARD

size_t GetMillisecondCount() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

size_t iterCount = 100000000;

void bench() {
  size_t start = GetMillisecondCount();

  size_t maxItems = 1 << 18;   // 512 KB
  size_t maxItemSizeExp = 10;  // 1K

  size_t dummyCtr = 0;
  size_t rssMax = 0;
  size_t rss;
  size_t allocatedSz = 0;
  size_t allocatedSzMax = 0;

  uint32_t reincarnation = 0;

  Pareto_80_20_6_Data paretoData;
  assert(maxItems <= UINT32_MAX);
  Pareto_80_20_6_Init(paretoData, (uint32_t) maxItems);

  struct TestBin {
    uint8_t* ptr;
    uint32_t sz;
    uint32_t reincarnation;
  };

  TestBin* baseBuff = nullptr;
#if defined(BENCH_SYSTEM)
  baseBuff = reinterpret_cast<TestBin*>(malloc(maxItems * sizeof(TestBin)));
#elif defined(BENCH_MIMALLOC)
  baseBuff = reinterpret_cast<TestBin*>(mi_malloc(maxItems * sizeof(TestBin)));
#elif defined(BENCH_VINEYARD)
  baseBuff =
      reinterpret_cast<TestBin*>(vineyard_malloc(maxItems * sizeof(TestBin)));
#else
#error "One of BENCH_SYSTEM, BENCH_MIMALLOC, BENCH_VINEYARD must be defined"
#endif
  assert(baseBuff);
  allocatedSz += maxItems * sizeof(TestBin);
  memset(baseBuff, 0, maxItems * sizeof(TestBin));

  PRNG rng;

  for (size_t k = 0; k < 32; ++k) {
    for (size_t j = 0; j < iterCount >> 5; ++j) {
      uint32_t rnum1 = rng.rng32();
      uint32_t rnum2 = rng.rng32();
      size_t idx = Pareto_80_20_6_Rand(paretoData, rnum1, rnum2);
      if (baseBuff[idx].ptr) {
#if defined(BENCH_SYSTEM)
        free(baseBuff[idx].ptr);
#elif defined(BENCH_MIMALLOC)
        mi_free(baseBuff[idx].ptr);
#elif defined(BENCH_VINEYARD)
        vineyard_free(baseBuff[idx].ptr);
#else
#error "One of BENCH_SYSTEM, BENCH_MIMALLOC, BENCH_VINEYARD must be defined"
#endif
        baseBuff[idx].ptr = 0;
      } else {
        size_t sz = calcSizeWithStatsAdjustment(rng.rng64(), maxItemSizeExp);
        baseBuff[idx].sz = (uint32_t) sz;

#if defined(BENCH_SYSTEM)
        baseBuff[idx].ptr = reinterpret_cast<uint8_t*>(malloc(sz));
#elif defined(BENCH_MIMALLOC)
        baseBuff[idx].ptr = reinterpret_cast<uint8_t*>(mi_malloc(sz));
#elif defined(BENCH_VINEYARD)
        baseBuff[idx].ptr = reinterpret_cast<uint8_t*>(vineyard_malloc(sz));
#else
#error "One of BENCH_SYSTEM, BENCH_MIMALLOC, BENCH_VINEYARD must be defined"
#endif
        if (unlikely(baseBuff[idx].ptr == nullptr)) {
          std::cerr << "Failed to allocate memory" << std::endl;
          std::abort();
        }
        memset(baseBuff[idx].ptr, (uint8_t) sz, sz);
      }
    }
  }

  size_t elapsed = GetMillisecondCount() - start;
  std::cout << "usage: " << elapsed << " milliseconds" << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 1) {
    printf("usage ./bench_allocator [<iterations>]");
    return 1;
  }
  if (argc >= 2) {
    iterCount = static_cast<size_t>(atoll(argv[1]));
  }

  bench();

  std::cout << "Finish allocator benchmarks..." << std::endl;
  return 0;
}
