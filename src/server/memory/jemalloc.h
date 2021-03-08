/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef SRC_SERVER_MEMORY_JEMALLOC_H_
#define SRC_SERVER_MEMORY_JEMALLOC_H_

#if defined(WITH_JEMALLOC)

#include <sys/mman.h>

#define JEMALLOC_NO_DEMANGLE
#include <jemalloc/jemalloc.h>

#include "server/memory/malloc.h"

namespace vineyard {

// static void *
// AllocHook(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
// 		size_t alignment, bool *zero, bool *commit, unsigned arena_index)
// {
//   char *ret = (char *) JeAllocator::pre_alloc;
//   if (ret + size >= pre_alloc_end) {
//     return NULL;
//   }
//   pre_alloc = (char *)pre_alloc + size;
//   return ret;
// }

class JeAllocator {
 public:
  static void* Init(const size_t size) {
    base_fd_ = plasma::create_buffer(size);
    base_pointer_ =
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, base_fd_, 0);
    if (base_pointer_ == nullptr) {
      return base_pointer_;
    }
    base_end_pointer_ = static_cast<char *>(base_pointer_) + size;

    // jemalloc_hooks_ = {AllocHook, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    // NULL}; extent_hooks_t *new_hooks = &jemalloc_hooks_;
    size_t index_size = sizeof(arena_index_);
    //  je_mallctl("arenas.create", (void *)&arena_index_, &index_size, (void
    //  *)&new_hooks, sizeof(extent_hooks_t *));
    mallctl("arenas.create", (void*) &arena_index_, &index_size, nullptr, 0);

    plasma::MmapRecord& record = plasma::mmap_records[base_pointer_];
    record.fd = base_fd_;
    record.size = size;

    return base_pointer_;
  }

  static void* Allocate(const size_t alignment, const size_t bytes) {
    return reinterpret_cast<uint8_t*>(
        mallocx(std::max(bytes, alignment), MALLOCX_ALIGN(alignment) |
                                                   MALLOCX_ARENA(arena_index_) |
                                                   MALLOCX_TCACHE_NONE));
  }

  static void Free(void* pointer) { free(pointer); }

 public:
  static int base_fd_;
  static void* base_pointer_;
  static void* base_end_pointer_;
  static int arena_index_;
};

}  // namespace vineyard

#endif  // WITH_JEMALLOC

#endif  // SRC_SERVER_MEMORY_JEMALLOC_H_
