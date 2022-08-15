/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include "server/memory/mimalloc.h"

#include <sys/mman.h>

#include <stdio.h>

#include "common/util/status.h"
#include "server/memory/malloc.h"

#if defined(WITH_MIMALLOC)

namespace vineyard {

namespace memory {

void* MimallocAllocator::Init(const size_t size) {
  // create memory using mmap
  int fd = create_buffer(size);
  void* space = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (space == nullptr) {
    return space;
  }

  // the addr must be 64MB aligned(required by mimalloc)
  // this will cause some loss of memory, and the maximum does not exceed 64MB
  void* addr = reinterpret_cast<void*>(
      (reinterpret_cast<uintptr_t>(space) + MIMALLOC_SEGMENT_ALIGNED_SIZE - 1) &
      ~(MIMALLOC_SEGMENT_ALIGNED_SIZE - 1));
  size_t real_size = size - ((size_t) addr - (size_t) space);

  MmapRecord& record = mmap_records[space];
  record.fd = fd;
  record.size = size;

  return Mimalloc::Init(addr, real_size);
}

}  // namespace memory

}  // namespace vineyard

#endif  // WITH_MIMALLOC
