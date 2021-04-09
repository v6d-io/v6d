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

#include <algorithm>
#include <string>

#define JEMALLOC_NO_DEMANGLE
#include "jemalloc/include/jemalloc/jemalloc.h"
#undef JEMALLOC_NO_DEMANGLE

#include "common/memory/jemalloc.h"
#include "common/util/logging.h"

extern const extent_hooks_t je_ehooks_default_extent_hooks;

namespace vineyard {

namespace memory {

Jemalloc::arena_t Jemalloc::arenas_[Jemalloc::MAXIMUM_ARENAS];

Jemalloc::Jemalloc() {
  extent_hooks_ = static_cast<extent_hooks_t*>(malloc(sizeof(extent_hooks_t)));
}

Jemalloc::~Jemalloc() {
  if (extent_hooks_) {
    free(extent_hooks_);
  }
}

void* Jemalloc::Init(void* space, const size_t size) {
  // obtain the current arena numbers
  unsigned narenas = -1;
  size_t size_of_narenas = sizeof(unsigned);
  if (auto ret = vineyard_je_mallctl("arenas.narenas", &narenas,
                                     &size_of_narenas, nullptr, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to get narenas";
    errno = err;
    return nullptr;
  }

  arena_index_ = narenas;  // starts from 0
  if (arena_index_ >= MAXIMUM_ARENAS) {
    LOG(ERROR) << "There can be " << MAXIMUM_ARENAS << " arenas at most";
    return nullptr;
  }

  arena_t& arena = arenas_[arena_index_];
  arena.base_pointer_ = reinterpret_cast<uintptr_t>(space);
  arena.base_end_pointer_ = arena.base_pointer_ + size;
  arena.pre_alloc_ = arena.base_pointer_;

  // prepare the custom hook
  *extent_hooks_ = je_ehooks_default_extent_hooks;
  extent_hooks_->alloc = &theAllocHook;

  // create arenas
  size_t arena_index_size = sizeof(arena_index_);
  if (auto ret =
          vineyard_je_mallctl("arenas.create", &arena_index_, &arena_index_size,
                              &extent_hooks_, sizeof(extent_hooks_))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to create arena";
    errno = err;
    return nullptr;
  }
  LOG(INFO) << "arena index = " << arena_index_;

  // set muzzy decay time to -1 to prevent jemalloc freeing the memory to the
  // pool, but leave dirty decay time untouched to still give the memory back
  // to the os kernel.
  // ssize_t decay_ms = 1;
  // std::string dirty_decay_key =
  //     "arena." + std::to_string(arena_index_) + ".dirty_decay_ms";
  // if (auto ret = vineyard_je_mallctl(dirty_decay_key.c_str(), nullptr,
  // nullptr,
  //                           &decay_ms, sizeof(decay_ms))) {
  //   int err = std::exchange(errno, ret);
  //   PLOG(ERROR) << "Failed to set the dirty decay time";
  //   errno = err;
  //   return nullptr;
  // }
  ssize_t decay_ms = -1;
  std::string muzzy_decay_key =
      "arena." + std::to_string(arena_index_) + ".muzzy_decay_ms";
  if (auto ret = vineyard_je_mallctl(muzzy_decay_key.c_str(), nullptr, nullptr,
                                     &decay_ms, sizeof(decay_ms))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to set the muzzy decay time";
    errno = err;
    return nullptr;
  }

  flags_ = MALLOCX_ARENA(arena_index_) | MALLOCX_TCACHE_NONE;
  return space;
}

void* Jemalloc::Allocate(const size_t bytes, const size_t alignment) {
  return vineyard_je_mallocx(std::max(bytes, alignment), flags_);
}

void* Jemalloc::Reallocate(void* pointer, size_t size) {
  return vineyard_je_rallocx(pointer, size, flags_);
}

void Jemalloc::Free(void* pointer, size_t) {
  if (pointer) {
    vineyard_je_dallocx(pointer, flags_);
  }
}

void Jemalloc::Recycle(const bool /* unused currently */) {
  std::string decay_key = "arena." + std::to_string(arena_index_) + ".decay";
  if (auto ret = vineyard_je_mallctl(decay_key.c_str(), nullptr, nullptr,
                                     nullptr, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to recycle arena " << arena_index_;
    errno = err;
  }
}

size_t Jemalloc::GetAllocatedSize(void* pointer) {
  return vineyard_je_sallocx(pointer, flags_);
}

size_t Jemalloc::EstimateAllocatedSize(const size_t size) {
  return vineyard_je_nallocx(size, flags_);
}

void Jemalloc::Traverse() {
  std::string traverse_key =
      "arena." + std::to_string(arena_index_) + ".traverse";
  if (auto ret = vineyard_je_mallctl(traverse_key.c_str(), nullptr, nullptr,
                                     nullptr, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to traverse arena";
    errno = err;
  }
}

void* Jemalloc::theAllocHook(extent_hooks_t* extent_hooks, void* new_addr,
                             size_t size, size_t alignment, bool* zero,
                             bool* commit, unsigned arena_index) {
  // align
  arena_t& arena = arenas_[arena_index];
  uintptr_t ret = (arena.pre_alloc_ + alignment - 1) & ~(alignment - 1);
  if (ret + size > arena.base_end_pointer_) {
    return nullptr;
  }
  arena.pre_alloc_ = ret + size;
  // N.B. the shared memory is not pre-committed.
  *commit = false;
  return reinterpret_cast<void*>(ret);
}

}  // namespace memory

}  // namespace vineyard

#endif  // WITH_JEMALLOC
