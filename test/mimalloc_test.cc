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

#include "common/util/env.h"
#include "common/util/logging.h"

#if defined(WITH_MIMALLOC)
#include "common/memory/mimalloc.h"
#endif

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
#if defined(WITH_MIMALLOC)
  size_t mimalloc_meta_size =
      MIMALLOC_SEGMENT_ALIGNED_SIZE * (std::thread::hardware_concurrency() + 1);
  size_t expected_heap_size = 64 * 1024 * 1024;
  size_t reserved_size = expected_heap_size + mimalloc_meta_size;

  void* base = malloc(reserved_size);
  LOG(INFO) << "initializing mimalloc with preallocated pointer " << base;

  std::shared_ptr<memory::Mimalloc> allocator =
      std::make_shared<memory::Mimalloc>(base, reserved_size);
  void* space = allocator->AlignedAddress();
  if (nullptr == space) {
    LOG(FATAL) << "failed to initialize mimalloc";
    return -1;
  }

  for (int k = 1; k <= 16; k *= 2) {
    void* pointer = allocator->Allocate(k * 1024 * 1024);
    LOG(INFO) << "pointer " << pointer << ", of size "
              << allocator->GetAllocatedSize(pointer);
    if (pointer == nullptr) {
      LOG(FATAL) << "failed to allocate (" << k << ") " << k * 1024 * 1024
                 << " bytes";
      return -1;
    }
    allocator->Free(pointer);
  }

#endif  // WITH_MIMALLOC

  LOG(INFO) << "Passed mimalloc tests...";

  return 0;
}
