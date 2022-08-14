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

#ifndef BENCHMARK_ALLOC_TEST_ALLOC_TEST_H_
#define BENCHMARK_ALLOC_TEST_ALLOC_TEST_H_

#include <stdio.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#define FORCE_INLINE inline

class PRNG {
  uint64_t seedVal;

 public:
  PRNG() { seedVal = 0; }
  explicit PRNG(size_t seed_) { seedVal = seed_; }
  void seed(size_t seed_) { seedVal = seed_; }

  /*FORCE_INLINE uint32_t rng32( uint32_t x )
  {
          // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
          x ^= x << 13;
          x ^= x >> 17;
          x ^= x << 5;
          return x;
  }*/
  /*	FORCE_INLINE uint32_t rng32()
          {
                  unsigned long long x = (seedVal += 7319936632422683443ULL);
                  x ^= x >> 32;
                  x *= c;
                  x ^= x >> 32;
                  x *= c;
                  x ^= x >> 32;
          return uint32_t(x);
          }*/
  FORCE_INLINE uint32_t rng32() {
    // based on implementation of xorshift by Arvid Gerstmann
    // see, for instance, https://arvid.io/2018/07/02/better-cxx-prng/
    uint64_t ret = seedVal * 0xd989bcacc137dcd5ull;
    seedVal ^= seedVal >> 11;
    seedVal ^= seedVal << 31;
    seedVal ^= seedVal >> 18;
    return uint32_t(ret >> 32ull);
  }

  FORCE_INLINE uint64_t rng64() {
    uint64_t ret = rng32();
    ret <<= 32;
    return ret + rng32();
  }
};

FORCE_INLINE size_t calcSizeWithStatsAdjustment(uint64_t randNum,
                                                size_t maxSizeExp) {
  assert(maxSizeExp >= 3);
  maxSizeExp -= 3;
  uint32_t statClassBase = (randNum & ((1 << maxSizeExp) - 1)) +
                           1;  // adding 1 to avoid dealing with 0
  randNum >>= maxSizeExp;
  int32_t idx = __builtin_ctzll(statClassBase);
  assert(idx <= maxSizeExp);
  idx += 2;
  size_t szMask = (1 << idx) - 1;
  return (randNum & szMask) + 1 + (((size_t) 1) << idx);
}

inline void testDistribution() {
  constexpr size_t exp = 16;
  constexpr size_t testCnt = 0x100000;
  size_t bins[exp + 1];  // NOLINT(runtime/arrays)
  memset(bins, 0, sizeof(bins));
  size_t total = 0;

  PRNG rng;

  for (size_t i = 0; i < testCnt; ++i) {
    size_t val = calcSizeWithStatsAdjustment(rng.rng64(), exp);
    assert(val);
    if (val <= 8) {
      bins[3] += 1;
    } else {
      for (size_t j = 4; j <= exp; ++j) {
        if (val <= (((size_t) 1) << j) && val > (((size_t) 1) << (j - 1))) {
          bins[j] += 1;
        }
      }
    }
  }
  // printf( "<=3: %zd\n", bins[0] + bins[1] + bins[2] + bins[3] );
  total = 0;
  for (size_t j = 0; j <= exp; ++j) {
    total += bins[j];
    // printf( "%zd: %zd\n", j, bins[j] );
  }
  assert(total == testCnt);
}

constexpr double Pareto_80_20_6[7] = {
    0.262144000000, 0.393216000000, 0.245760000000, 0.081920000000,
    0.015360000000, 0.001536000000, 0.000064000000};

struct Pareto_80_20_6_Data {
  uint32_t probabilityRanges[6];
  uint32_t offsets[8];
};

FORCE_INLINE
void Pareto_80_20_6_Init(Pareto_80_20_6_Data& data, uint32_t itemCount) {
  data.probabilityRanges[0] = (uint32_t)(UINT32_MAX * Pareto_80_20_6[0]);
  data.probabilityRanges[5] = (uint32_t)(UINT32_MAX * (1. - Pareto_80_20_6[6]));
  for (size_t i = 1; i < 5; ++i) {
    data.probabilityRanges[i] = data.probabilityRanges[i - 1] +
                                (uint32_t)(UINT32_MAX * Pareto_80_20_6[i]);
  }
  data.offsets[0] = 0;
  data.offsets[7] = itemCount;
  for (size_t i = 0; i < 6; ++i) {
    data.offsets[i + 1] =
        data.offsets[i] + (uint32_t)(itemCount * Pareto_80_20_6[6 - i]);
  }
}

FORCE_INLINE
size_t Pareto_80_20_6_Rand(const Pareto_80_20_6_Data& data, uint32_t rnum1,
                           uint32_t rnum2) {
  size_t idx = 6;
  if (rnum1 < data.probabilityRanges[0])
    idx = 0;
  else if (rnum1 < data.probabilityRanges[1])
    idx = 1;
  else if (rnum1 < data.probabilityRanges[2])
    idx = 2;
  else if (rnum1 < data.probabilityRanges[3])
    idx = 3;
  else if (rnum1 < data.probabilityRanges[4])
    idx = 4;
  else if (rnum1 < data.probabilityRanges[5])
    idx = 5;
  uint32_t rangeSize = data.offsets[idx + 1] - data.offsets[idx];
  uint32_t offsetInRange = rnum2 % rangeSize;
  return data.offsets[idx] + offsetInRange;
}

#endif  // BENCHMARK_ALLOC_TEST_ALLOC_TEST_H_
