/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/malloc.h is referred and derived from project
 * apache-arrow,
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/plasma/malloc.h
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

#ifndef SRC_SERVER_MEMORY_MALLOC_H_
#define SRC_SERVER_MEMORY_MALLOC_H_

#include <inttypes.h>
#include <stddef.h>

#include <string>
#include <unordered_map>

namespace vineyard {

namespace memory {

/// Memory alignment.
constexpr int64_t kBlockSize = 64;

void GetMallocMapinfo(void* addr, int* fd, int64_t* map_length,
                      ptrdiff_t* offset);

struct MmapRecord {
  int fd = -1;
  int64_t size = -1;
};

/// Hashtable that contains one entry per segment that we got from the OS
/// via mmap. Associates the address of that segment with its file descriptor
/// and size.
extern std::unordered_map<void*, MmapRecord> mmap_records;

// Create a buffer. This is creating a temporary file and then
// immediately unlinking it so we do not leave traces in the system.
//
// Returns a fd as expected.
int create_buffer(int64_t size, bool memory = true);

// Returns a fd of the corresponding path as expected.
int create_buffer(int64_t size, std::string const& path);

// Create a buffer, and mmap the buffer as the shared memory space.
void* mmap_buffer(int64_t size, bool* is_committed, bool* is_zero);

// Create a buffer, and mmap the buffer as the shared memory space.
void* mmap_buffer(int fd, int64_t size, bool* is_committed, bool* is_zero);

// Unmap the buffer.
int munmap_buffer(void* addr, int64_t size);

}  // namespace memory

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_MALLOC_H_
