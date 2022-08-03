/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/malloc.cc is referred and derived from project
 * apache-arrow,
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/plasma/malloc.cc
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

#include "common/util/logging.h"
#include "server/memory/malloc.h"

namespace vineyard {

namespace memory {

std::unordered_map<void*, MmapRecord> mmap_records;

static void* pointer_advance(void* p, ptrdiff_t n) {
  return (unsigned char*) p + n;
}

static ptrdiff_t pointer_distance(void const* pfrom, void const* pto) {
  return (unsigned char const*) pto - (unsigned char const*) pfrom;
}

// Create a buffer. This is creating a temporary file and then
// immediately unlinking it so we do not leave traces in the system.
int create_buffer(int64_t size) {
  int fd = -1;
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

void GetMallocMapinfo(void* addr, int* fd, int64_t* map_size,
                      ptrdiff_t* offset) {
  // About the efficiences: the records size usually small, thus linear search
  // is enough.
  for (const auto& entry : mmap_records) {
    if (addr >= entry.first &&
        addr < pointer_advance(entry.first, entry.second.size)) {
      *fd = entry.second.fd;
      *map_size = entry.second.size;
      *offset = pointer_distance(entry.first, addr);
      return;
    }
  }
  LOG(INFO) << "fd not found for " << addr
            << ", mmap records size = " << mmap_records.size();
  *fd = -1;
  *map_size = 0;
  *offset = 0;
}

}  // namespace memory

}  // namespace vineyard
