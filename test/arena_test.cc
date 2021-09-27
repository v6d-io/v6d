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

#include <sys/mman.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/util/logging.h"

#include "common/memory/jemalloc.h"
#include "jemalloc/include/jemalloc/jemalloc.h"

#define KB 1024
#define MB (KB * 1024)
#define NUM_THREAD 3
#define MULTITHREAD

#define HUGE_SZ (2 << 20)
#define SMALL_SZ (8)

#define NUM_CORES std::thread::hardware_concurrency()
#define NUM_ARENA NUM_CORES

struct extent_hooks_s;
typedef struct extent_hooks_s extent_hooks_t;
extern const extent_hooks_t je_ehooks_default_extent_hooks;

using namespace vineyard::memory;  // NOLINT(build/namespaces)

extent_hooks_t* extent_hooks_ = nullptr;

std::mutex arena_mutex;
std::mutex thread_map_mutex;
// limit the number of arenas

// start with NUM_ARENA, erase arena after requested
std::deque<unsigned> empty_arenas(NUM_ARENA, 0);

// when requesting arena succeed, add thread id - arena index pair to the map
std::unordered_map<std::thread::id, unsigned> thread_arena_map;

void TestAlloc(size_t size, int flags) {
  LOG(INFO) << "Allocate " << size << " bytes";
  vineyard_je_mallocx(size, flags);
}

void TestFree(void* ptr) {
  LOG(INFO) << "Free pointer ptr=0x" << ptr;
  vineyard_je_free(ptr);
}

unsigned requestArena() {
  std::thread::id id = std::this_thread::get_id();

  unsigned arena_index;
  {
    std::lock_guard<std::mutex> guard(arena_mutex);
    if (empty_arenas.empty()) {
      LOG(ERROR) << "All arenas used.";
      // TODO: recycle arena here
      return -1;
    }
    arena_index = empty_arenas.front();
    empty_arenas.pop_front();
  }
  LOG(INFO) << "Arena " << arena_index << " requested for thread " << id;
  {
    std::lock_guard<std::mutex> guard(thread_map_mutex);
    thread_arena_map[id] = arena_index;
  }

  if (auto ret = vineyard_je_mallctl("thread.arena", NULL, NULL, &arena_index,
                                     sizeof(arena_index))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to bind arena " << arena_index << "for thread "
                << id;
    errno = err;
    return -1;
  }

  return arena_index;
}

void returnArena(unsigned arena_index) {
  std::thread::id id = std::this_thread::get_id();
  {
    std::lock_guard<std::mutex> guard(arena_mutex);
    empty_arenas.push_back(arena_index);
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex);
    if (thread_arena_map.find(id) != thread_arena_map.end())
      thread_arena_map.erase(thread_arena_map.find(id));
  }
}

unsigned doCreateArena(extent_hooks_t* hooks) {
  unsigned arena_index;
  size_t sz = sizeof(unsigned);
  if (auto ret = vineyard_je_mallctl(
          "arenas.create", &arena_index, &sz,
          reinterpret_cast<void*>(hooks != NULL ? &hooks : NULL),
          (hooks != NULL ? sizeof(hooks) : 0))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to create arena";
    errno = err;
  }
  return arena_index;
}

/*
 * Destroying and recreating the arena is simpler than
 * specifying extent hooks that deallocate during reset.
 */
int doDestroyArena(unsigned arena_index) {
  size_t mib[3];
  size_t miblen;

  miblen = sizeof(mib) / sizeof(size_t);
  if (auto ret =
          vineyard_je_mallctlnametomib("arena.0.destroy", mib, &miblen)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Unexpected mallctlnametomib() failure";
    errno = err;
    return -1;
  }

  mib[1] = arena_index;
  if (auto ret = vineyard_je_mallctlbymib(mib, miblen, NULL, NULL, NULL, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to destroy arena " << arena_index;
    errno = err;
    return -1;
  }
  returnArena(arena_index);
  return 0;
}

int doResetArena(unsigned arena_index) {
  size_t mib[3];
  size_t miblen;

  miblen = sizeof(mib) / sizeof(size_t);
  if (auto ret = vineyard_je_mallctlnametomib("arena.0.reset", mib, &miblen)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Unexpected mallctlnametomib() failure";
    errno = err;
    return -1;
  }

  mib[1] = (size_t) arena_index;
  if (auto ret = vineyard_je_mallctlbymib(mib, miblen, NULL, NULL, NULL, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to destroy arena";
    errno = err;
    return -1;
  }
  return 0;
}

void destroyAllArenas(std::deque<unsigned>& arenas) {
  for (auto index : arenas) {
    doDestroyArena(index);
  }
  std::lock_guard<std::mutex> guard(arena_mutex);
  arenas.clear();

  LOG(INFO) << "Arenas destroyed.";
}

void resetAllArenas(std::deque<unsigned>& arenas) {
  for (auto index : arenas) {
    doResetArena(index);
  }

  LOG(INFO) << "Arenas reseted.";
}

void preAllocateArena(std::deque<unsigned>& arenas) {
  for (size_t i = 0; i < NUM_ARENA; i++) {
    unsigned arena = -1;
    size_t sz = sizeof(unsigned);
    if (auto ret = vineyard_je_mallctl("arenas.create", &arena, &sz, NULL, 0)) {
      int err = std::exchange(errno, ret);
      PLOG(ERROR) << "failed to create arena";
      errno = err;
    }
    arenas[i] = arena;
    LOG(INFO) << "Arena " << arena << " created";
    // TODO: create TCACHE for each arena
  }
}

unsigned arenaLookUp(void* ptr) {
  unsigned arena_index;
  size_t sz = sizeof(unsigned);
  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena_index, &sz, &ptr,
                                     sizeof(ptr))) {
    int err = std::exchange(errno, ret);
    LOG(ERROR) << "failed to lookup arena";
    errno = err;
  }
  return arena_index;
}

unsigned threadTotalAllocatedBytes() {
  uint64_t allocated;
  size_t sz = sizeof(allocated);
  if (auto ret = vineyard_je_mallctl("thread.allocated",
                                     reinterpret_cast<void*>(&allocated), &sz,
                                     NULL, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to read thread.allocated";
    errno = err;
    return -1;
  }
  return allocated;
}

unsigned threadTotalDeallocatedBytes() {
  uint64_t deallocated;
  size_t sz = sizeof(deallocated);
  if (auto ret = vineyard_je_mallctl("thread.deallocated",
                                     reinterpret_cast<void*>(&deallocated), &sz,
                                     NULL, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "Failed to read thread.deallocated";
    errno = err;
    return -1;
  }
  return deallocated;
}

/**
 * each thread would create an arena, bind themselves to it and allocate
 */
void CreateArenaTask() {
  std::thread::id tid = std::this_thread::get_id();
  LOG(INFO) << "Created new thread " << tid;
  unsigned arena1, arena2;
  size_t sz = sizeof(unsigned);
  if (auto ret = vineyard_je_mallctl("arenas.create", &arena1, &sz, NULL, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to create arena";
    errno = err;
  }
  LOG(INFO) << "arena created for thread " << tid << ", index " << arena1;
  if (auto ret = vineyard_je_mallctl("thread.arena", NULL, NULL, &arena1,
                                     sizeof(arena1))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to bind arena";
    errno = err;
  }

  void* small = vineyard_je_mallocx(2 * MB, 0);
  if (small == nullptr) {
    LOG(ERROR) << "Failed to allocate 2 MB";
  }

  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena2, &sz, &small,
                                     sizeof(small))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to lookup arena";
    errno = err;
  }

  LOG(INFO) << "arena lookup for address" << small << ", index " << arena1;

  if (arena1 != arena2) {
    LOG(ERROR) << "Wrong arena used to mallocx. ";
  }

  doDestroyArena(arena1);
}

/**
 * each thread would request for an arena and bind themselves to it
 */
void RequestArenaTask() {
  int arena_index = requestArena();
  if (arena_index == -1) {
    return;
  }
  returnArena(arena_index);
}

/**
 * Each thread would try to allocate on its arena, if no arena binded to this
 * thread, request for one and allocate on it
 */
void AllocateTask() {
  std::thread::id id = std::this_thread::get_id();
  int arena_index = -1;
  if (thread_arena_map.find(id) == thread_arena_map.end()) {
    arena_index = requestArena();
    if (arena_index == -1) {
      return;
    }
  } else {
    arena_index = thread_arena_map[id];
  }

  int flag = MALLOCX_TCACHE_NONE;

  /*
   * Allocate.  The main thread will reset the arena, so there's
   * no need to deallocate.
   */
  void* p = vineyard_je_mallocx(2 * MB, flag);

  if (p == nullptr) {
    LOG(ERROR) << "Thread " << id << "Failed to allocate 2 MB";
  }

  vineyard_je_dallocx(p, flag);

  /*
   * Return arena when exit.
   */
  returnArena(arena_index);
  LOG(INFO) << "Thread " << id << " allocate finished.";
}

/**
 * create arena for main thread and sub thread, and allocate on both arenas
 */
int BaseTest() {
  unsigned arena1, arena2;
  size_t sz = sizeof(unsigned);
  /* bind current thread to a manual arena,
   * make sure mallocx in arena specified */
  std::thread::id main_tid = std::this_thread::get_id();
  std::thread sub_thread(CreateArenaTask);
  if (auto ret = vineyard_je_mallctl("arenas.create", &arena1, &sz, NULL, 0)) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to create arena";
    errno = err;
    return -1;
  }
  LOG(INFO) << "arena created for thread " << main_tid << ", index " << arena1;
  if (auto ret = vineyard_je_mallctl("thread.arena", NULL, NULL, &arena1,
                                     sizeof(arena1))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to bind arena";
    errno = err;
    return -1;
  }
  void* small = vineyard_je_mallocx(2 * MB, 0);
  if (small == nullptr) {
    PLOG(ERROR) << "Failed to allocate 2 MB";
    return -1;
  }

  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena2, &sz, &small,
                                     sizeof(small))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to lookup arena";
    errno = err;
    return -1;
  }

  LOG(INFO) << "arena lookup for address" << small << ", index " << arena1;
  if (arena1 != arena2) {
    LOG(ERROR) << "Wrong arena used to mallocx. ";
    return -1;
  }

  if (sub_thread.joinable()) {
    sub_thread.join();
  }

  arena2 = 0;
  void* ptr = vineyard_je_mallocx(2 * MB, MALLOCX_ARENA(50));
  if (ptr == nullptr) {
    PLOG(ERROR) << "Failed to allocate 2 MB";
    return -1;
  }
  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena2, &sz, &ptr,
                                     sizeof(ptr))) {
    int err = std::exchange(errno, ret);
    PLOG(ERROR) << "failed to lookup arena";
    errno = err;
    return -1;
  }

  LOG(INFO) << "Malloc on arena " << arena2;
  return 0;
}

int CreateArenaTest() {
  std::thread threads[NUM_THREAD];
  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i] = std::thread(CreateArenaTask);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i].join();
  }
  return 0;
}

int RequestArenaTest() {
  std::thread threads[NUM_THREAD];
  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i] = std::thread(RequestArenaTask);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i].join();
  }

  return 0;
}

int AllocateArenaTest() {
  std::thread threads[NUM_THREAD];

  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i] = std::thread(AllocateTask);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i].join();
  }

  destroyAllArenas(empty_arenas);
  return 0;
}

int main(int argc, char** argv) {
  LOG(INFO) << "arena test starts...";

  LOG(INFO) << "Base test starts...";
  if (BaseTest() == -1) {
    LOG(ERROR) << "Base test failed.";
    exit(-1);
  }

#ifdef MULTITHREAD
  LOG(INFO) << NUM_ARENA << " arenas totally";
  LOG(INFO) << NUM_THREAD << " threads working concurrently";

  /* Create new arena for each thread and allocate */
  LOG(INFO) << "***************Create arena test starts***************";
  if (CreateArenaTest() == -1) {
    LOG(ERROR) << "Create arena test failed.";
    exit(-1);
  }
  preAllocateArena(empty_arenas);

  /* Request for arena in a fixed arena pool */
  LOG(INFO) << "***************Request arena test starts***************";
  if (RequestArenaTest() == -1) {
    LOG(ERROR) << "Request arena test failed.";
    exit(-1);
  }

  /* Allocate and free in requested arena */
  LOG(INFO) << "***************Allocate arena test starts***************";
  if (AllocateArenaTest() == -1) {
    LOG(ERROR) << "Allocate arena test failed.";
    exit(-1);
  }

#endif

  LOG(INFO) << "Arena test succeed";

  return 0;
}
