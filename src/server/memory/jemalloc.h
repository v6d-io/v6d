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

#include "server/memory/malloc.h"

namespace vineyard {

namespace memory {

class JemallocAllocator {
 public:
  static void* Init(const size_t size);

  static void* Allocate(const size_t bytes, const size_t alignment);

  static void* Reallocate(void* pointer, size_t size);

  static void Free(void* pointer, size_t = 0);

  static constexpr size_t Alignment = 1 * 1024 * 1024;  // 1MB

 private:
  static uintptr_t base_pointer_;
  static uintptr_t base_end_pointer_;

  static int flags_;
  static uintptr_t pre_alloc_;

  static void* theAllocHook(extent_hooks_t* extent_hooks, void* new_addr,
                            size_t size, size_t alignment, bool* zero,
                            bool* commit, unsigned arena_index);
};

}  // namespace memory

}  // namespace vineyard

#endif  // WITH_JEMALLOC

#endif  // SRC_SERVER_MEMORY_JEMALLOC_H_
