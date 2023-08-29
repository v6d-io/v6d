/** Copyright 2020-2023 Alibaba Group Holding Limited.

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
  /**
   * @brief Manages a particular memory arena.
   */
  Mimalloc(void* addr, const size_t size, const bool is_committed = false,
           const bool is_zero = true) {
    // does't consist of large OS pages
    bool is_large = false;
    // no associated numa node
    int numa_node = -1;

    aligned_size = size;
    // the addr must be 64MB aligned (required by mimalloc)
    if ((reinterpret_cast<uintptr_t>(addr) % MIMALLOC_SEGMENT_ALIGNED_SIZE) !=
        0) {
      aligned_address =
          reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(addr) +
                                   MIMALLOC_SEGMENT_ALIGNED_SIZE - 1) &
                                  ~(MIMALLOC_SEGMENT_ALIGNED_SIZE - 1));
      aligned_size = size - ((size_t) aligned_address - (size_t) addr);
    } else {
      aligned_address = addr;
    }

    bool success =
        mi_manage_os_memory_ex(aligned_address, aligned_size, is_committed,
                               is_large, is_zero, numa_node, true, &arena_id);
    if (!success) {
      std::clog << "[error] mimalloc failed to create the arena at "
                << aligned_address << std::endl;
      aligned_address = nullptr;
    }
    heap = mi_heap_new_in_arena(arena_id);
    if (heap == nullptr) {
      std::clog << "[error] mimalloc failed to create the heap at "
                << aligned_address << std::endl;
      aligned_address = nullptr;
    }

    // do not use OS memory for allocation (but only pre-allocated arena)
    mi_option_set(mi_option_limit_os_alloc, 1);
  }

  ~Mimalloc() {
    // leave it un-deleted to keep allocated blocks
  }

  size_t AlignedSize() const { return aligned_size; }

  void* AlignedAddress() const { return aligned_address; }

  void* Allocate(const size_t bytes, const size_t alignment = 0) {
    if (unlikely(alignment)) {
      return mi_heap_malloc_aligned(heap, bytes, alignment);
    } else {
      return mi_heap_malloc(heap, bytes);
    }
  }

  void* Reallocate(void* pointer, size_t size) {
    return mi_heap_realloc(heap, pointer, size);
  }

  void Free(void* pointer, size_t size = 0) {
    if (likely(pointer)) {
      if (unlikely(size)) {
        mi_free_size(pointer, size);
      } else {
        mi_free(pointer);
      }
    }
  }

  size_t GetAllocatedSize(void* pointer) { return mi_usable_size(pointer); }

 private:
  void* aligned_address = nullptr;
  size_t aligned_size = 0;
  mi_arena_id_t arena_id{};
  mi_heap_t* heap = nullptr;
};

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_MIMALLOC_H_
