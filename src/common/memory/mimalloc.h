/** Copyright 2020-2022 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef SRC_COMMON_MEMORY_MIMALLOC_H_
#define SRC_COMMON_MEMORY_MIMALLOC_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "mimalloc/include/mimalloc.h"

#include "common/util/likely.h"
#include "common/util/logging.h"

#ifndef MIMALLOC_SEGMENT_ALIGNED_SIZE
#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)
#endif

namespace vineyard {

namespace memory {

/**
 * Use a header only class to avoid depends mimalloc library in vineyard_client.
 */
class Mimalloc {
 public:
  Mimalloc();
  ~Mimalloc();

  /**
   * @brief Manages a particular memory arena.
   */
  static void* Init(void* addr, const size_t size,
                    const bool is_committed = false,
                    const bool is_zero = true) {
    // does't consist of large OS pages
    bool is_large = false;
    // no associated numa node
    int numa_node = -1;

    void* new_addr = addr;
    size_t new_size = size;

    // the addr must be 64MB aligned (required by mimalloc)
    if ((reinterpret_cast<uintptr_t>(addr) % MIMALLOC_SEGMENT_ALIGNED_SIZE) !=
        0) {
      new_addr = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(addr) +
                                          MIMALLOC_SEGMENT_ALIGNED_SIZE - 1) &
                                         ~(MIMALLOC_SEGMENT_ALIGNED_SIZE - 1));
      new_size = size - ((size_t) new_addr - (size_t) addr);
    }

    // do not use OS memory for allocation (but only pre-allocated arena)
    mi_option_set(mi_option_limit_os_alloc, 1);

    bool success = mi_manage_os_memory(new_addr, new_size, is_committed,
                                       is_large, is_zero, numa_node);
    if (!success) {
      std::clog << "[error] mimalloc failed to create the arena at " << new_addr
                << std::endl;
      return nullptr;
    }
    return new_addr;
  }

  static void* Allocate(const size_t bytes, const size_t alignment = 0) {
    if (unlikely(alignment)) {
      return mi_malloc_aligned(bytes, alignment);
    } else {
      return mi_malloc(bytes);
    }
  }

  static void* Reallocate(void* pointer, size_t size) {
    return mi_realloc(pointer, size);
  }

  static void Free(void* pointer, size_t size = 0) {
    if (likely(pointer)) {
      if (unlikely(size)) {
        mi_free_size(pointer, size);
      } else {
        mi_free(pointer);
      }
    }
  }

  static size_t GetAllocatedSize(void* pointer) {
    return mi_usable_size(pointer);
  }
};

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_MIMALLOC_H_
