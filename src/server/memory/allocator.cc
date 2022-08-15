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

#if defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <sys/mount.h>
#endif

#include <cstdio>
#include <cstring>
#include <string>

#include "common/util/env.h"
#include "common/util/logging.h"
#include "server/memory/allocator.h"
#include "server/memory/malloc.h"

#if defined(WITH_DLMALLOC)
#include "server/memory/dlmalloc.h"
#endif
#if defined(WITH_MIMALLOC)
#include "server/memory/mimalloc.h"
#endif

namespace vineyard {

int64_t BulkAllocator::footprint_limit_ = 0;
int64_t BulkAllocator::allocated_ = 0;

void* BulkAllocator::Init(const size_t size) {
#if __linux__
  // Seems that mac os doesn't respect the sysctl configuration.
  //
  // Vineyard actually can use shared memory larger than the the sysctl result,
  // we disable the warning on mac for less inaccurate warnings.
  int64_t shmmax = get_maximum_shared_memory();
  LOG(INFO) << "shmax: " << shmmax;
  if (shmmax < static_cast<float>(size)) {
    LOG(WARNING)
        << "The 'size' is greater than the maximum shared memory size ("
        << shmmax << ")" << std::endl;
    LOG(WARNING)
        << "    If you are inside a Docker container, please pass the argument "
           "'--shm-size' when 'docker run'."
        << std::endl;

    // try remount
    //
    // shm on /dev/shm type tmpfs (rw,nosuid,nodev,noexec,relatime,size=65536k)
    std::string options = "size=" + std::to_string(size);
    int flags = MS_REMOUNT | MS_NOSUID | MS_NODEV | MS_NOEXEC | MS_RELATIME;
    if (mount("shm", "/dev/shm", "tmpfs", flags, options.c_str()) != 0) {
      VLOG(2) << "Failed to remount: " << strerror(errno);
    }
  }
#endif

#if defined(WITH_DLMALLOC)
  return Allocator::Init(size);
#endif
#if defined(WITH_MIMALLOC)
  // mimalloc requires 64MB (segment aligned) for each thread
  size_t mimalloc_meta_size =
      MIMALLOC_SEGMENT_ALIGNED_SIZE * (std::thread::hardware_concurrency() + 1);
  // leave spaces for memory fragmentation
  return Allocator::Init(static_cast<size_t>(static_cast<double>(size) * 1.2) +
                         mimalloc_meta_size);
#endif
}

void* BulkAllocator::Memalign(const size_t bytes, const size_t alignment) {
  if (allocated_ + static_cast<int64_t>(bytes) > footprint_limit_) {
    return nullptr;
  }

  void* mem = Allocator::Allocate(bytes, alignment);
  if (mem != nullptr) {
    allocated_ += bytes;
  }
  return mem;
}

void BulkAllocator::Free(void* mem, size_t bytes) {
  Allocator::Free(mem, bytes);
  allocated_ -= bytes;
}

void BulkAllocator::SetFootprintLimit(size_t bytes) {
  footprint_limit_ = static_cast<int64_t>(bytes);
}

int64_t BulkAllocator::GetFootprintLimit() { return footprint_limit_; }

int64_t BulkAllocator::Allocated() { return allocated_; }

}  // namespace vineyard
