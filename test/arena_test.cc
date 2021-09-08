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
#include <string>
#include <thread>

#include "common/util/logging.h"

#include "common/memory/jemalloc.h"
#include "jemalloc/include/jemalloc/jemalloc.h"

#define KB 1024
#define MB (KB * 1024)
#define NUM_THREAD 3
#define MULTITHREAD
struct extent_hooks_s;
typedef struct extent_hooks_s extent_hooks_t;
extern const extent_hooks_t je_ehooks_default_extent_hooks;

using namespace vineyard::memory;  // NOLINT(build/namespaces)

extent_hooks_t* extent_hooks_ = nullptr;
int flags = 0;

void TestAlloc(size_t size) {
  LOG(INFO) << "Allocate " << size << " bytes";
  vineyard_je_mallocx(size, flags);
}
void TestFree(void* ptr) {
  LOG(INFO) << "Free pointer ptr=0x" << ptr;
  vineyard_je_free(ptr);
}


/**
 * each thread would create an arena, bind themselves to it and allocate
 */
void ArenaTask() {
  std::thread::id tid = std::this_thread::get_id();
  LOG(INFO) << "Created new thread " << tid;
  unsigned arena1, arena2;
  size_t sz = sizeof(unsigned);
  if (auto ret = vineyard_je_mallctl("arenas.create", &arena1, &sz, NULL, 0)) {
    LOG(ERROR) << "failed to create arena";
  }
  LOG(INFO) << "arena created for thread " << tid << ", index " << arena1;
  if (auto ret = vineyard_je_mallctl("thread.arena", NULL, NULL, &arena1,
                                     sizeof(arena1))) {
    LOG(ERROR) << "failed to bind arena";
  }
  void* small = vineyard_je_mallocx(2 * MB, 0);
  if (small == nullptr) {
    LOG(ERROR) << "Failed to allocate 2 MB";
  }

  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena2, &sz, &small, sizeof(small))) {
    LOG(ERROR) << "failed to lookup arena";
  }

  LOG(INFO) << "arena lookup for address" << small << ", index " << arena1;

  if (arena1 != arena2) {
    LOG(ERROR) << "Wrong arena used to mallocx. ";
  }
}

int main(int argc, char** argv) {
  LOG(INFO) << "arena test starts...";
  unsigned arena1, arena2;
  size_t sz = sizeof(unsigned);
  void* sys_base = malloc(20 * MB);
  // initial implementation
  // void* jemalloc_base = Jemalloc::Init(sys_base, 20 MB);

  /* bind current thread to a manual arena,
   * make sure mallocx in arena specified */
  std::thread::id main_tid = std::this_thread::get_id();
  std::thread sub_thread(ArenaTask);
  if (auto ret = vineyard_je_mallctl("arenas.create", &arena1, &sz, NULL, 0)) {
    LOG(ERROR) << "failed to create arena";
  }
  LOG(INFO) << "arena created for thread " << main_tid << ", index " << arena1;
  if (auto ret = vineyard_je_mallctl("thread.arena", NULL, NULL, &arena1,
                         sizeof(arena1))) {
      LOG(ERROR) << "failed to bind arena";
  }
  void* small = vineyard_je_mallocx(2 * MB, 0);
  if (small == nullptr) {
    LOG(ERROR) << "Failed to allocate 2 MB";
  }

  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena2, &sz, &small, sizeof(small))) {
    LOG(ERROR) << "failed to lookup arena";
  }
  LOG(INFO) << "arena lookup for address" << small << ", index " << arena1;
  if (arena1 != arena2) {
    LOG(ERROR) << "Wrong arena used to mallocx. ";
  }

  if (sub_thread.joinable()) {
    sub_thread.join();
  }

  arena2 = 0;
  void* ptr = vineyard_je_mallocx(2 * MB, MALLOCX_ARENA(50));
  if (auto ret = vineyard_je_mallctl("arenas.lookup", &arena2, &sz, &ptr, sizeof(ptr))) {
    LOG(ERROR) << "failed to lookup arena";
  }
  LOG(INFO) << "Malloc on arena " << arena2;

#ifdef MULTITHREAD
  LOG(INFO) << NUM_THREAD << " threads working concurrently";
  std::thread threads[NUM_THREAD];

  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i] = std::thread(ArenaTask);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    threads[i].join();
  }
#endif

  LOG(INFO) << "Arena test succeed";

  return 0;
}
