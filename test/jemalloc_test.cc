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

#include <sys/mman.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "client/allocator.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/memory/arena.h"
#include "common/memory/jemalloc.h"
#include "common/util/env.h"
#include "common/util/logging.h"

int main(int argc, char** argv) {
  void* base = malloc(20 * 1024 * 1024);
#ifdef VINEYARD_BENCH
  auto* jemalloc_allocator = new vineyard::memory::Jemalloc();
#else
  auto* jemalloc_allocator = new vineyard::memory::ArenaAllocator();
#endif
  void* space = jemalloc_allocator->Init(base, 20 * 1024 * 1024);
  if (nullptr == space) {
    LOG(INFO) << "init failed";
    return 0;
  }

  void* p1 = jemalloc_allocator->Allocate(1 * 1024 * 1024);

  LOG(INFO) << "p1 " << jemalloc_allocator->GetAllocatedSize(p1);
  jemalloc_allocator->Free(p1);

  void* p2 = jemalloc_allocator->Allocate(2 * 1024 * 1024);
  LOG(INFO) << "p2 " << jemalloc_allocator->GetAllocatedSize(p2);
  jemalloc_allocator->Free(p2);

  void* p3 = jemalloc_allocator->Allocate(4 * 1024 * 1024);
  LOG(INFO) << "p3 " << jemalloc_allocator->GetAllocatedSize(p3);
  jemalloc_allocator->Free(p3);

  void* p4 = jemalloc_allocator->Allocate(8 * 1024 * 1024);
  LOG(INFO) << "p4 " << jemalloc_allocator->GetAllocatedSize(p4);
  jemalloc_allocator->Free(p4);

  void* p5 = jemalloc_allocator->Allocate(16 * 1024 * 1024);
  LOG(INFO) << "p5 " << jemalloc_allocator->GetAllocatedSize(p5);
  jemalloc_allocator->Free(p5);

  LOG(INFO) << "Passed jemalloc tests...";

  return 0;
}
