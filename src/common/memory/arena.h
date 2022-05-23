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

#ifndef SRC_COMMON_MEMORY_ARENA_H_
#define SRC_COMMON_MEMORY_ARENA_H_

#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <utility>

#include "server/memory/malloc.h"

// forward declarations, to avoid include jemalloc/jemalloc.h.
struct extent_hooks_s;
typedef struct extent_hooks_s extent_hooks_t;

namespace vineyard {

namespace memory {

class ArenaAllocator {
 public:
  ArenaAllocator();

  ~ArenaAllocator();

  void* Init(void* space, const size_t size);

  void* Allocate(const size_t size, const size_t alignment = Alignment);

  void Free(void* ptr, size_t = 0);

  int LookUp(void* ptr);

  size_t GetAllocatedSize(void* pointer);

  unsigned ThreadTotalAllocatedBytes();

  unsigned ThreadTotalDeallocatedBytes();

  static constexpr size_t Alignment = 1 * 1024 * 1024;  // 1MB

  struct arena_t {
    arena_t() {}
    arena_t(uintptr_t base_pointer, uintptr_t base_end_pointer,
            uintptr_t pre_alloc)
        : base_pointer_(base_pointer),
          base_end_pointer_(base_end_pointer),
          pre_alloc_(pre_alloc) {}
    uintptr_t base_pointer_ = reinterpret_cast<uintptr_t>(nullptr);
    uintptr_t base_end_pointer_ = reinterpret_cast<uintptr_t>(nullptr);
    uintptr_t pre_alloc_ = reinterpret_cast<uintptr_t>(nullptr);
  };

 private:
  int doCreateArena();

  int requestArena();

  void returnArena(unsigned arena_index);

  /*
   * Destroying and recreating the arena is simpler than
   * specifying extent hooks that deallocate during reset.
   */
  int doDestroyArena(unsigned arena_index);

  int doResetArena(unsigned arena_index);

  void destroyAllArenas();

  void resetAllArenas();

  void preAllocateArena(void* space, const size_t size);

  static void* theAllocHook(extent_hooks_t* extent_hooks, void* new_addr,
                            size_t size, size_t alignment, bool* zero,
                            bool* commit, unsigned arena_index);

 private:
  std::mutex arena_mutex_;
  std::mutex thread_map_mutex_;
  int num_arenas_;
  std::deque<unsigned> empty_arenas_;
  std::unordered_map<std::thread::id, int> thread_arena_map_;
  extent_hooks_t* extent_hooks_ = nullptr;
  static std::unordered_map<unsigned, arena_t> arenas_;
};

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_ARENA_H_
