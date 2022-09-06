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

#include <stdio.h>

#include "common/util/status.h"
#include "server/memory/malloc.h"

#if defined(WITH_MIMALLOC)

namespace vineyard {

namespace memory {

void* MimallocAllocator::Init(const size_t size) {
  // create memory using mmap
  bool is_committed = false;
  bool is_zero = true;
  void* space = mmap_buffer(size, &is_committed, &is_zero);
  if (space == nullptr) {
    return space;
  }
  return Mimalloc::Init(space, size, is_committed, is_zero);
}

}  // namespace memory

}  // namespace vineyard

#endif  // WITH_MIMALLOC
