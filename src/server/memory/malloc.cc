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

#include "server/memory/malloc.h"

#include <stddef.h>

#include "common/util/logging.h"

namespace plasma {

std::map<void*, MmapRecord> mmap_records;

static void* pointer_advance(void* p, ptrdiff_t n) {
  return (unsigned char*) p + n;
}

static ptrdiff_t pointer_distance(void const* pfrom, void const* pto) {
  return (unsigned char const*) pto - (unsigned char const*) pfrom;
}

void GetMallocMapinfo(void* addr, int* fd, int64_t* map_size,
                      ptrdiff_t* offset) {
  auto entry = mmap_records.lower_bound(addr);
  auto upper = mmap_records.upper_bound(addr);
  while (entry != upper) {
    if (entry->first <= addr &&
        addr < pointer_advance(entry->first, entry->second.size)) {
      *fd = entry->second.fd;
      *map_size = entry->second.size;
      *offset = pointer_distance(entry->first, addr);
      return;
    }
    ++entry;
  }
  *fd = -1;
  *map_size = 0;
  *offset = 0;
}

}  // namespace plasma
