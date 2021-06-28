/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/allocator.cc is referred and derived from project
 * apache-arrow,
 *
 *
https://github.com/apache/arrow/blob/master/cpp/src/plasma/plasma_allocator.cc
 *
 * which has the following license:
 *
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
 */

#include "server/memory/allocator.h"

#include <stdio.h>

#include "common/util/env.h"
#include "common/util/logging.h"
#include "server/memory/malloc.h"

#if defined(WITH_DLMALLOC)
#include "server/memory/dlmalloc.h"
#endif

#if defined(WITH_JEMALLOC)
#include "server/memory/jemalloc.h"
#endif

namespace vineyard {

int64_t BulkAllocator::footprint_limit_ = 0;
int64_t BulkAllocator::allocated_ = 0;

#if defined(WITH_JEMALLOC)
BulkAllocator::Allocator BulkAllocator::allocator_{};
#endif

void* BulkAllocator::Init(const size_t size) {
  int64_t shmmax = get_maximum_shared_memory();
  if (shmmax < static_cast<float>(size)) {
    LOG(WARNING) << "'size' is greater than the maximum shared memory size ("
                 << shmmax << ")";
  }
#if defined(WITH_DLMALLOC)
  return Allocator::Init(size);
#endif
#if defined(WITH_JEMALLOC)
  return allocator_.Init(size);
#endif
}

void* BulkAllocator::Memalign(const size_t bytes, const size_t alignment) {
  if (allocated_ + static_cast<int64_t>(bytes) > footprint_limit_) {
    return nullptr;
  }

#if defined(WITH_DLMALLOC)
  void* mem = Allocator::Allocate(bytes, alignment);
#endif
#if defined(WITH_JEMALLOC)
  void* mem = allocator_.Allocate(bytes, alignment);
#endif
  allocated_ += bytes;
  return mem;
}

void BulkAllocator::Free(void* mem, size_t bytes) {
#if defined(WITH_DLMALLOC)
  Allocator::Free(mem);
#endif
#if defined(WITH_JEMALLOC)
  allocator_.Free(mem);
#endif
  allocated_ -= bytes;
}

void BulkAllocator::SetFootprintLimit(size_t bytes) {
  footprint_limit_ = static_cast<int64_t>(bytes);
}

int64_t BulkAllocator::GetFootprintLimit() { return footprint_limit_; }

int64_t BulkAllocator::Allocated() { return allocated_; }

}  // namespace vineyard
