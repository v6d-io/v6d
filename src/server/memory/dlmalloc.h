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

#ifndef SRC_SERVER_MEMORY_DLMALLOC_H_
#define SRC_SERVER_MEMORY_DLMALLOC_H_

#include "common/util/status.h"

namespace vineyard {

namespace memory {

class DLmallocAllocator {
 public:
  static void* Init(const size_t size);

  static void* Allocate(const size_t bytes, const size_t alignment);

  static void Free(void* pointer, size_t = 0);

  static void SetMallocGranularity(int value);
};

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_DLMALLOC_H_
