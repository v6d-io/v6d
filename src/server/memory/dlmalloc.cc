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

#include <stddef.h>

#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "common/util/logging.h"
#include "server/memory/malloc.h"

namespace plasma {

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

#include "dlmalloc/dlmalloc.c"  // NOLINT

#undef MMAP
#undef MUNMAP
#undef DIRECT_MMAP
#undef DIRECT_MUNMAP
#undef USE_DL_PREFIX
#undef HAVE_MORECORE
#undef DEFAULT_GRANULARITY

// dlmalloc.c defined DEBUG which will conflict with ARROW_LOG(DEBUG).
#ifdef DEBUG
#undef DEBUG
#endif

/// Gap between two consecutive mmap regions allocated by fake_mmap.
/// This ensures that the segments of memory returned by
/// fake_mmap are never contiguous and dlmalloc does not coalesce it
/// (in the client we cannot guarantee that these mmaps are contiguous).
constexpr int64_t kMmapRegionsGap = sizeof(size_t);

constexpr int GRANULARITY_MULTIPLIER = 2;

// Fine-grained control for whether we need pre-populate the shared memory.
//
// Usually it causes a long wait time at the start up, but it could improved
// the performance of visiting shared memory.
//
// In cases that the startup time doesn't much matter, e.g., in kubernetes
// environment, pre-populate will archive a win.
DEFINE_bool(reserve_memory, false, "Pre-reserving enough memory pages");

static void* pointer_advance(void* p, ptrdiff_t n) {
  return (unsigned char*) p + n;
}

static void* pointer_retreat(void* p, ptrdiff_t n) {
  return (unsigned char*) p - n;
}

// Create a buffer. This is creating a temporary file and then
// immediately unlinking it so we do not leave traces in the system.
int create_buffer(int64_t size) {
  int fd;
#ifdef _WIN32
  if (!CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                         (DWORD)((uint64_t) size >> (CHAR_BIT * sizeof(DWORD))),
                         (DWORD)(uint64_t) size, NULL)) {
    fd = -1;
  }
#else
  // directory where to create the memory-backed file
#ifdef __linux__
  std::string file_template = "/dev/shm/vineyard-bulk-XXXXXX";
#else
  std::string file_template = "/tmp/vineyard-bulk-XXXXXX";
#endif
  std::vector<char> file_name(file_template.begin(), file_template.end());
  file_name.push_back('\0');
  fd = mkstemp(&file_name[0]);
  if (fd < 0) {
    LOG(FATAL) << "create_buffer failed to open file " << &file_name[0];
    return -1;
  }
  // Immediately unlink the file so we do not leave traces in the system.
  if (unlink(&file_name[0]) != 0) {
    LOG(FATAL) << "failed to unlink file " << &file_name[0];
    return -1;
  }
  if (true) {
    // Increase the size of the file to the desired size. This seems not to be
    // needed for files that are backed by the huge page fs, see also
    // http://www.mail-archive.com/kvm-devel@lists.sourceforge.net/msg14737.html
    if (ftruncate(fd, (off_t) size) != 0) {
      LOG(FATAL) << "failed to ftruncate file " << &file_name[0];
      return -1;
    }
  }
#endif
  return fd;
}

void* fake_mmap(size_t size) {
  // Add kMmapRegionsGap so that the returned pointer is deliberately not
  // page-aligned. This ensures that the segments of memory returned by
  // fake_mmap are never contiguous.
  size += kMmapRegionsGap;

  int fd = create_buffer(size);
  CHECK_GE(fd, 0) << "Failed to create buffer during mmap";
  // MAP_POPULATE can be used to pre-populate the page tables for this memory
  // region
  // which avoids work when accessing the pages later. However it causes long
  // pauses
  // when mmapping the files. Only supported on Linux.

  int mmap_flag = MAP_SHARED;
  if (FLAGS_reserve_memory) {
#ifdef __linux__
    mmap_flag |= MAP_POPULATE;
#endif
  }

  void* pointer = mmap(NULL, size, PROT_READ | PROT_WRITE, mmap_flag, fd, 0);
  if (pointer == MAP_FAILED) {
    LOG(ERROR) << "mmap failed with error: " << strerror(errno);
    return pointer;
  }

  // Increase dlmalloc's allocation granularity directly.
  mparams.granularity *= GRANULARITY_MULTIPLIER;

  MmapRecord& record = mmap_records[pointer];
  record.fd = fd;
  record.size = size;

  // We lie to dlmalloc about where mapped memory actually lives.
  pointer = pointer_advance(pointer, kMmapRegionsGap);
  return pointer;
}

int fake_munmap(void* addr, int64_t size) {
  addr = pointer_retreat(addr, kMmapRegionsGap);
  size += kMmapRegionsGap;

  auto entry = mmap_records.find(addr);

  if (entry == mmap_records.end() || entry->second.size != size) {
    // Reject requests to munmap that don't directly match previous
    // calls to mmap, to prevent dlmalloc from trimming.
    return -1;
  }

  int r = munmap(addr, size);
  if (r == 0) {
    close(entry->second.fd);
  }

  mmap_records.erase(entry);
  return r;
}

void SetMallocGranularity(int value) { change_mparam(M_GRANULARITY, value); }

}  // namespace plasma
