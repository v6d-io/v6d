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

#ifndef SRC_COMMON_MEMORY_MIMALLOC_H_
#define SRC_COMMON_MEMORY_MIMALLOC_H_

#include "server/memory/malloc.h"

#define MIMALLOC_SEGMENT_ALIGNED_SIZE ((uintptr_t) 1 << 26)

namespace vineyard {

namespace memory {

class Mimalloc {
 public:
  Mimalloc();
  ~Mimalloc();

  void* Init(void* addr, const size_t size);

  void* Allocate(const size_t bytes, const size_t alignment = 0);

  void* Reallocate(void* pointer, size_t size);

  void Free(void* pointer, size_t = 0);

  size_t GetAllocatedSize(void* pointer);
};

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_MIMALLOC_H_
