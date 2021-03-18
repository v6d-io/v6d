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

#if defined(WITH_JEMALLOC)

#include <sys/mman.h>

#include <algorithm>
#include <string>

#define JEMALLOC_NO_DEMANGLE
#include "jemalloc/jemalloc.h"
#undef JEMALLOC_NO_DEMANGLE

#include "common/util/logging.h"
#include "server/memory/jemalloc.h"
#include "server/memory/malloc.h"

namespace vineyard {

namespace memory {

uintptr_t JemallocAllocator::base_pointer_ =
    reinterpret_cast<uintptr_t>(nullptr);
uintptr_t JemallocAllocator::base_end_pointer_ =
    reinterpret_cast<uintptr_t>(nullptr);

int JemallocAllocator::flags_ = 0;
uintptr_t JemallocAllocator::pre_alloc_ = reinterpret_cast<uintptr_t>(nullptr);
extent_hooks_t JemallocAllocator::extent_hooks_ = {};

void* JemallocAllocator::Init(const size_t size) {
  // mmap
  int fd = create_buffer(size);
  void* space = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (space == nullptr) {
    return space;
  }
  base_pointer_ = reinterpret_cast<uintptr_t>(space);
  base_end_pointer_ = base_pointer_ + size;

  MmapRecord& record = mmap_records[space];
  record.fd = fd;
  record.size = size;

  /** Notes [Initialize jemalloc arena]
   *
   * 1. don't use hook when creating arena.
   * 2. don't use a hook like {alloc, null, null, ...}, rather, get the
   *    original hook and then replace the alloc function.
   * 3. set `retain_grow_limit` doesn't work.
   *
   * ref: https://github.com/facebook/folly/blob/master/folly/experimental/
   *      JemallocHugePageAllocator.cpp
   */

  // create arenas
  unsigned arena_index = -1;
  size_t arena_index_size = sizeof(arena_index);
  if (auto ret = je_mallctl("arenas.create", &arena_index, &arena_index_size,
                            nullptr, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to create arena";
    errno = err;
    return nullptr;
  }

  // set hook
  std::string hooks_key =
      "arena." + std::to_string(arena_index) + ".extent_hooks";
  extent_hooks_t* hooks;
  size_t hooks_size = sizeof(hooks);
  if (auto ret =
          je_mallctl(hooks_key.c_str(), &hooks, &hooks_size, nullptr, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to read the arena hooks";
    errno = err;
    return nullptr;
  }

  // Set the custom hook
  extent_hooks_ = *hooks;
  extent_hooks_.alloc = &theAllocHook;
  extent_hooks_t* new_hooks = &extent_hooks_;
  if (auto ret = je_mallctl(hooks_key.c_str(), nullptr, nullptr, &new_hooks,
                            sizeof(new_hooks))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to set the arena hooks";
    errno = err;
    return nullptr;
  }

  // set dirty decay and muzzy decay time to -1 to never free memory to kernel.
  ssize_t decay_ms = -1;
  std::string dirty_decay_key =
      "arena." + std::to_string(arena_index) + ".dirty_decay_ms";
  if (auto ret = je_mallctl(dirty_decay_key.c_str(), nullptr, nullptr,
                            &decay_ms, sizeof(decay_ms))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to set the dirty decay time";
    errno = err;
    return nullptr;
  }
  std::string muzzy_decay_key =
      "arena." + std::to_string(arena_index) + ".muzzy_decay_ms";
  if (auto ret = je_mallctl(muzzy_decay_key.c_str(), nullptr, nullptr,
                            &decay_ms, sizeof(decay_ms))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to set the muzzy decay time";
    errno = err;
    return nullptr;
  }

  flags_ = MALLOCX_ALIGN(kBlockSize) | MALLOCX_ARENA(arena_index) |
           MALLOCX_TCACHE_NONE;
  pre_alloc_ = base_pointer_;
  return space;
}

void* JemallocAllocator::Allocate(const size_t bytes, const size_t alignment) {
  return je_mallocx(std::max(bytes, alignment), flags_);
}

void* JemallocAllocator::Reallocate(void* pointer, size_t size) {
  return je_rallocx(pointer, size, flags_);
}

void JemallocAllocator::Free(void* pointer, size_t) {
  if (pointer) {
    je_dallocx(pointer, flags_);
  }
}

void* JemallocAllocator::theAllocHook(extent_hooks_t* extent_hooks,
                                      void* new_addr, size_t size,
                                      size_t alignment, bool* zero,
                                      bool* commit, unsigned arena_index) {
  // align
  uintptr_t ret = (pre_alloc_ + alignment - 1) & ~(alignment - 1);
  if (ret + size > JemallocAllocator::base_end_pointer_) {
    return nullptr;
  }
  pre_alloc_ = ret + size;
  // N.B. no need to manipulate the `commit` bit.
  // *commit = false;
  return reinterpret_cast<void*>(ret);
}

}  // namespace memory

}  // namespace vineyard

#endif  // WITH_JEMALLOC
