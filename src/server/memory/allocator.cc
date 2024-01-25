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

#include <sys/mman.h>
#if defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <sys/mount.h>
#endif

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "server/memory/allocator.h"

#include "common/util/logging.h"
#include "server/memory/dlmalloc.h"
#include "server/memory/mimalloc.h"

namespace vineyard {

bool BulkAllocator::use_mimalloc_ = false;
int64_t BulkAllocator::footprint_limit_ = 0;
int64_t BulkAllocator::allocated_ = 0;

void* BulkAllocator::Init(const size_t size, std::string const& allocator) {
  if (allocator == "dlmalloc") {
    use_mimalloc_ = false;
    return DLmallocAllocator::Init(size);
  } else {
    use_mimalloc_ = true;
    // leave enough space for alignment
    size_t space_size = MIMALLOC_SEGMENT_ALIGNED_SIZE + size;
    // mimalloc requires 64MB (segment aligned) for each thread
    size_t mimalloc_meta_size = MIMALLOC_SEGMENT_ALIGNED_SIZE *
                                (std::thread::hardware_concurrency() + 1);
    // leave spaces for memory fragmentation
    return MimallocAllocator::Init(static_cast<size_t>(
        static_cast<double>(space_size + mimalloc_meta_size) * 2));
  }
}

void* BulkAllocator::Memalign(const size_t bytes, const size_t alignment) {
  if (allocated_ + static_cast<int64_t>(bytes) > footprint_limit_) {
    return nullptr;
  }

  void* mem = nullptr;
  if (use_mimalloc_) {
    mem = MimallocAllocator::Allocate(bytes, alignment);

  } else {
    mem = DLmallocAllocator::Allocate(bytes, alignment);
  }
  if (mem != nullptr) {
    allocated_ += bytes;

    // mark it as will need to pre-allocate physical memory
    // and improve the followed memory write performance
    madvise(mem, bytes, MADV_WILLNEED);
  }
  return mem;
}

void BulkAllocator::Free(void* mem, size_t bytes) {
  if (use_mimalloc_) {
    MimallocAllocator::Free(mem, bytes);
  } else {
    DLmallocAllocator::Free(mem, bytes);
  }
  allocated_ -= bytes;
}

void BulkAllocator::SetFootprintLimit(size_t bytes) {
  footprint_limit_ = static_cast<int64_t>(bytes);
}

int64_t BulkAllocator::GetFootprintLimit() { return footprint_limit_; }

int64_t BulkAllocator::Allocated() { return allocated_; }

}  // namespace vineyard
