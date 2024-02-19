/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/dlmalloc.cc is referred and derived from project
 * apache-arrow,
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/plasma/dlmalloc.cc
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

#if defined(WITH_DLMALLOC)

#include <cstddef>
#include <mutex>

#include "common/util/logging.h"  // IWYU pragma: keep
#include "server/memory/dlmalloc.h"
#include "server/memory/malloc.h"

namespace vineyard {

namespace memory {

void* fake_mmap(size_t);
int fake_munmap(void*, int64_t);

#define MMAP(s) fake_mmap(s)
#define MUNMAP(a, s) fake_munmap(a, s)
#define DIRECT_MMAP(s) fake_mmap(s)
#define DIRECT_MUNMAP(a, s) fake_munmap(a, s)
#define USE_DL_PREFIX
#define HAVE_MORECORE 0
#define DEFAULT_MMAP_THRESHOLD MAX_SIZE_T
#define DEFAULT_GRANULARITY ((size_t) 128U * 1024U)
#define USE_LOCKS 1 /* makes the dlmalloc thread safe (but is not scalable) */

// prevent dlmalloc from crash for unexpected reasons
#define PROCEED_ON_ERROR 1

#include "dlmalloc/dlmalloc.c"  // NOLINT

#undef MMAP
#undef MUNMAP
#undef DIRECT_MMAP
#undef DIRECT_MUNMAP
#undef USE_DL_PREFIX
#undef HAVE_MORECORE
#undef DEFAULT_GRANULARITY
#undef USE_LOCKS

// dlmalloc.c defined DEBUG which will conflict with ARROW_LOG(DEBUG).
#ifdef DEBUG
#undef DEBUG
#endif

constexpr int GRANULARITY_MULTIPLIER = 2;

void* fake_mmap(size_t size) {
  // Increase dlmalloc's allocation granularity directly.
  mparams.granularity *= GRANULARITY_MULTIPLIER;

  bool is_committed = false;
  bool is_zero = true;
  return mmap_buffer(size, &is_committed, &is_zero);
}

int fake_munmap(void* addr, int64_t size) { return munmap_buffer(addr, size); }

void* DLmallocAllocator::allocator_ = nullptr;

void* DLmallocAllocator::Init(const size_t size) {
  static std::once_flag init_flag;
  std::call_once(init_flag, [size]() {
    // We are using a single memory-mapped file by malloc-ing and freeing a
    // single large amount of space up front.
    void* pointer = dlmemalign(kBlockSize, size - 256 * sizeof(size_t));
    if (pointer != nullptr) {
      // This will unmap the file, but the next one created will be as large
      // as this one (this is an implementation detail of dlmalloc).
      dlfree(pointer);
      DLmallocAllocator::allocator_ = pointer;
    }
  });
  return DLmallocAllocator::allocator_;
}

void* DLmallocAllocator::Allocate(const size_t bytes, const size_t alignment) {
  return dlmemalign(alignment, bytes);
}

void DLmallocAllocator::Free(void* pointer, size_t) { dlfree(pointer); }

void DLmallocAllocator::SetMallocGranularity(int value) {
  change_mparam(M_GRANULARITY, value);
}

}  // namespace memory

}  // namespace vineyard

#endif  // WITH_DLMALLOC
