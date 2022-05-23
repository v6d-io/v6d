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

#if defined(WITH_JEMALLOC)

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

#include "common/memory/arena.h"
#include "common/util/env.h"
#include "common/util/functions.h"

#define JEMALLOC_NO_DEMANGLE
#include "jemalloc/include/jemalloc/jemalloc.h"
#undef JEMALLOC_NO_DEMANGLE

extern const extent_hooks_t je_ehooks_default_extent_hooks;

namespace vineyard {

namespace memory {

std::unordered_map<unsigned, ArenaAllocator::arena_t> ArenaAllocator::arenas_;

ArenaAllocator::ArenaAllocator()
    : num_arenas_(std::thread::hardware_concurrency()),
      empty_arenas_(num_arenas_, 0) {
  extent_hooks_ = static_cast<extent_hooks_t*>(malloc(sizeof(extent_hooks_t)));
}

ArenaAllocator::~ArenaAllocator() {
  if (extent_hooks_) {
    free(extent_hooks_);
  }

  destroyAllArenas();
}

void* ArenaAllocator::Init(void* space, const size_t size) {
  preAllocateArena(space, size);
  return space;
}

void* ArenaAllocator::Allocate(const size_t size, const size_t alignment) {
  std::thread::id id = std::this_thread::get_id();
  int arena_index = -1;
  // Do not need lock here, as current thread is the only thread with thread id
  // = id .find() would return a const iterator, which is thread safe
  if (thread_arena_map_.find(id) == thread_arena_map_.end()) {
    arena_index = requestArena();
    if (arena_index == -1)
      return nullptr;
  } else {
    arena_index = thread_arena_map_[id];
  }

  // TODO: check flag
  return vineyard_je_mallocx(std::max(size, alignment), 0);
}

int ArenaAllocator::LookUp(void* ptr) {
  unsigned arena_index = -1;
  size_t sz = sizeof(unsigned);
  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena_index, &sz, &ptr,
                                     sizeof(ptr))) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "failed to lookup arena" << std::endl;
    errno = err;
    return -1;
  }
  return arena_index;
}

void ArenaAllocator::Free(void* ptr, size_t) {
  if (ptr) {
    vineyard_je_dallocx(ptr, 0);
  }
}

size_t ArenaAllocator::GetAllocatedSize(void* pointer) {
  return vineyard_je_sallocx(pointer, 0);
}

unsigned ArenaAllocator::ThreadTotalAllocatedBytes() {
  uint64_t allocated;
  size_t sz = sizeof(allocated);
  if (auto ret = vineyard_je_mallctl("thread.allocated",
                                     reinterpret_cast<void*>(&allocated), &sz,
                                     NULL, 0)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to get allocated bytes" << std::endl;
    errno = err;
    return -1;
  }
  return allocated;
}

unsigned ArenaAllocator::ThreadTotalDeallocatedBytes() {
  uint64_t deallocated;
  size_t sz = sizeof(deallocated);
  if (auto ret = vineyard_je_mallctl("thread.deallocated",
                                     reinterpret_cast<void*>(&deallocated), &sz,
                                     NULL, 0)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to get deallocated bytes" << std::endl;
    errno = err;
    return -1;
  }
  return deallocated;
}

int ArenaAllocator::requestArena() {
  std::thread::id id = std::this_thread::get_id();

  int arena_index;
  {
    std::lock_guard<std::mutex> guard(arena_mutex_);
    if (empty_arenas_.empty()) {
      std::clog << "All arenas used." << std::endl;
      // TODO: recycle arena here
      return -1;
    }
    arena_index = empty_arenas_.front();
    empty_arenas_.pop_front();
  }
  std::clog << "Arena " << arena_index << " requested for thread " << id
            << std::endl;
  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    thread_arena_map_[id] = arena_index;
  }

  if (auto ret = vineyard_je_mallctl("thread.arena", NULL, NULL, &arena_index,
                                     sizeof(arena_index))) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to bind arena " << arena_index << "for thread " << id
              << std::endl;
    errno = err;
    return -1;
  }

  return arena_index;
}

void ArenaAllocator::returnArena(unsigned arena_index) {
  std::thread::id id = std::this_thread::get_id();
  {
    std::lock_guard<std::mutex> guard(arena_mutex_);
    empty_arenas_.push_back(arena_index);
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    if (thread_arena_map_.find(id) != thread_arena_map_.end())
      thread_arena_map_.erase(thread_arena_map_.find(id));
  }
}

int ArenaAllocator::doCreateArena() {
  int arena_index;

  size_t sz = sizeof(arena_index);
  if (auto ret =
          vineyard_je_mallctl("arenas.create", &arena_index, &sz, nullptr, 0)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to create arena" << std::endl;
    errno = err;
    return -1;
  }

  // set extent hooks
  std::ostringstream hooks_key;
  hooks_key << "arena." << std::to_string(arena_index) << ".extent_hooks";
  size_t len = sizeof(extent_hooks_);
  if (auto ret = vineyard_je_mallctl(hooks_key.str().c_str(), &extent_hooks_,
                                     &len, nullptr, 0)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to set extent hooks" << std::endl;
    errno = err;
    return -1;
  }
  return arena_index;
}

int ArenaAllocator::doDestroyArena(unsigned arena_index) {
  size_t mib[3];
  size_t miblen;

  miblen = sizeof(mib) / sizeof(size_t);
  if (auto ret =
          vineyard_je_mallctlnametomib("arena.0.destroy", mib, &miblen)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to destroy arena " << arena_index << std::endl;
    errno = err;
    return -1;
  }

  mib[1] = arena_index;
  if (auto ret = vineyard_je_mallctlbymib(mib, miblen, NULL, NULL, NULL, 0)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to destroy arena " << arena_index << std::endl;
    errno = err;
    return -1;
  }
  returnArena(arena_index);
  return 0;
}

int ArenaAllocator::doResetArena(unsigned arena_index) {
  size_t mib[3];
  size_t miblen;

  miblen = sizeof(mib) / sizeof(size_t);
  if (auto ret = vineyard_je_mallctlnametomib("arena.0.reset", mib, &miblen)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to reset arena " << arena_index << std::endl;
    errno = err;
    return -1;
  }

  mib[1] = (size_t) arena_index;
  if (auto ret = vineyard_je_mallctlbymib(mib, miblen, NULL, NULL, NULL, 0)) {
    int err = detail::exchange_value(errno, ret);
    std::clog << "Failed to reset arena " << arena_index << std::endl;
    errno = err;
    return -1;
  }
  return 0;
}

void ArenaAllocator::destroyAllArenas() {
  for (auto index : empty_arenas_) {
    doDestroyArena(index);
  }
  std::lock_guard<std::mutex> guard(arena_mutex_);
  empty_arenas_.clear();
  std::clog << "Arenas destroyed." << std::endl;
}

void ArenaAllocator::resetAllArenas() {
  for (auto index : empty_arenas_) {
    doResetArena(index);
  }

  std::clog << "Arenas reseted." << std::endl;
}

void ArenaAllocator::preAllocateArena(void* space, const size_t size) {
  int64_t shmmax = get_maximum_shared_memory();
  std::clog << "Size of each arena " << shmmax << std::endl;
  *extent_hooks_ = je_ehooks_default_extent_hooks;
  extent_hooks_->alloc = &theAllocHook;
  for (int i = 0; i < num_arenas_; i++) {
    int arena_index = doCreateArena();
    if (arena_index == -1) {
      return;
    }

    auto* arena =
        new arena_t(reinterpret_cast<uintptr_t>(space) + i * shmmax,
                    reinterpret_cast<uintptr_t>(space) + (i + 1) * shmmax,
                    reinterpret_cast<uintptr_t>(space) + i * shmmax);

    arenas_[arena_index] = *arena;
    empty_arenas_[i] = arena_index;
    std::clog << "Arena index " << arena_index << " created" << std::endl;
    // TODO: create TCACHE for each arena
  }
}

void* ArenaAllocator::theAllocHook(extent_hooks_t* extent_hooks, void* new_addr,
                                   size_t size, size_t alignment, bool* zero,
                                   bool* commit, unsigned int arena_index) {
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
