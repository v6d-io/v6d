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

#if defined(WITH_JEMALLOC)

#include <sys/mman.h>

#include "server/memory/jemalloc.h"
#include "server/memory/malloc.h"

namespace vineyard {

namespace memory {

void* JemallocAllocator::Init(const size_t size) {
  // create memory using mmap
  int fd = create_buffer(size);
  void* space = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (space == nullptr) {
    return space;
  }

  MmapRecord& record = mmap_records[space];
  record.fd = fd;
  record.size = size;

  return Jemalloc::Init(space, size);
}

}  // namespace memory

}  // namespace vineyard

#endif  // WITH_JEMALLOC
