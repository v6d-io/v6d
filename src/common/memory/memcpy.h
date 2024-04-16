/**
 * NOLINT(legal/copyright)
 *
 * The file src/common/memory/memcpy.h is referred and derived from project
 * clickhouse,
 *
 *    https://github.com/ClickHouse/ClickHouse/blob/master/base/glibc-compatibility/memcpy/memcpy.h
 *
 * which has the following license:
 *
 * Copyright 2016-2022 ClickHouse, Inc.
 *
 *             Apache License
 *      Version 2.0, January 2004
 * http://www.apache.org/licenses/
 */

#ifndef SRC_COMMON_MEMORY_MEMCPY_H_
#define SRC_COMMON_MEMORY_MEMCPY_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

namespace vineyard {

namespace memory {

static inline void * inline_memcpy(void * __restrict dst_, const void * __restrict src_, size_t size) {
    return memcpy(dst_, src_, size);
}

// clang-format on

// use the same default concurrency as apache-arrow.
static constexpr size_t default_memcpy_concurrency = 6;

static inline void* concurrent_memcpy(void* __restrict dst_,
                                      const void* __restrict src_, size_t size,
                                      const size_t concurrency = default_memcpy_concurrency) {
  static constexpr size_t concurrent_memcpy_threshold = 1024 * 1024 * 4;
  if (size < concurrent_memcpy_threshold) {
    inline_memcpy(dst_, src_, size);
  } else if ((dst_ >= src_ &&
              dst_ <= static_cast<const uint8_t*>(src_) + size) ||
             (src_ >= dst_ && src_ <= static_cast<uint8_t*>(dst_) + size)) {
    inline_memcpy(dst_, src_, size);
  } else {
    static constexpr size_t alignment = 1024 * 1024 * 4;
    size_t chunk_size = (size / concurrency + alignment - 1) & ~(alignment - 1);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < concurrency; ++i) {
      if (size <= i * chunk_size) {
        break;
      }
      size_t chunk = std::min(chunk_size, size - i * chunk_size);
      threads.emplace_back([=]() {
        inline_memcpy(static_cast<uint8_t*>(dst_) + i * chunk_size,
                      static_cast<const uint8_t*>(src_) + i * chunk_size,
                      chunk);
      });
    }
    for (auto &thread: threads) {
      thread.join();
    }
  }
  return dst_;
}

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_MEMCPY_H_
