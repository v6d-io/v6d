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

// #if defined(WITH_JEMALLOC)

#include <sys/mman.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/util/logging.h"

#include "common/memory/jemalloc.h"
#include "jemalloc/include/jemalloc/jemalloc.h"

namespace vineyard {

namespace memory {

class ArenaAllocator {
 public:
  ArenaAllocator();

  ~ArenaAllocator();

  void* Allocate(size_t size);

  void Free(void* ptr, size_t);

  unsigned LookUp(void* ptr);

  unsigned ThreadTotalAllocatedBytes();

  unsigned ThreadTotalDeallocatedBytes();

 private:
  unsigned doCreateArena();

  unsigned requestArena();

  void returnArena(unsigned arena_index);

  /*
   * Destroying and recreating the arena is simpler than
   * specifying extent hooks that deallocate during reset.
   */
  int doDestroyArena(unsigned arena_index);

  int doResetArena(unsigned arena_index);

  void destroyAllArenas();

  void resetAllArenas();

  void preAllocateArena();

 private:
  std::mutex arena_mutex_;
  std::mutex thread_map_mutex_;
  int num_arenas_;
  std::deque<unsigned> empty_arenas_;
  std::unordered_map<std::thread::id, unsigned> thread_arena_map_;
  extent_hooks_t* extent_hooks_ = nullptr;
};

}  // namespace memory

}  // namespace vineyard
