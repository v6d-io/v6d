/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#ifndef SRC_COMMON_MEMORY_JEMALLOC_H_
#define SRC_COMMON_MEMORY_JEMALLOC_H_

#include <utility>

#include "server/memory/malloc.h"

// forward declarations, to avoid include jemalloc/jemalloc.h.
struct extent_hooks_s;
typedef struct extent_hooks_s extent_hooks_t;

namespace vineyard {

namespace memory {

class Jemalloc {
 public:
  Jemalloc();
  ~Jemalloc();

  void* Init(void* space, const size_t size);

  void* Allocate(const size_t bytes, const size_t alignment = Alignment);

  void* Reallocate(void* pointer, size_t size);

  void Free(void* pointer, size_t = 0);

  void Recycle(const bool force = false);

  size_t GetAllocatedSize(void* pointer);

  size_t EstimateAllocatedSize(const size_t size);

  void Traverse();

  static constexpr size_t Alignment = 1 * 1024 * 1024;  // 1MB

  struct arena_t {
    uintptr_t base_pointer_ = reinterpret_cast<uintptr_t>(nullptr);
    uintptr_t base_end_pointer_ = reinterpret_cast<uintptr_t>(nullptr);
    uintptr_t pre_alloc_ = reinterpret_cast<uintptr_t>(nullptr);
  };

 private:
  unsigned arena_index_;
  int flags_ = 0;
  extent_hooks_t* extent_hooks_ = nullptr;

  static void* theAllocHook(extent_hooks_t* extent_hooks, void* new_addr,
                            size_t size, size_t alignment, bool* zero,
                            bool* commit, unsigned arena_index);

  // maximum supported arenas, a global static array is used to record status
  // of memory allocation for each arenas for being used in the c-style
  // callback.
  static constexpr size_t MAXIMUM_ARENAS = 128;
  static arena_t arenas_[MAXIMUM_ARENAS];
};

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_JEMALLOC_H_
