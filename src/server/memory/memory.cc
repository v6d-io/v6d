/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/memory.cc is partially referred and derived
 * from project apache-arrow,
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/plasma/memory.cc
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

#include "server/memory/memory.h"

#include <sys/mman.h>

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "server/memory/allocator.h"
#include "server/memory/malloc.h"

namespace vineyard {

using memory::GetMallocMapinfo;
using memory::kBlockSize;

Status BulkStore::PreAllocate(const size_t size) {
  BulkAllocator::SetFootprintLimit(size);
  void* pointer = BulkAllocator::Init(size);

  if (pointer == nullptr) {
    return Status::NotEnoughMemory("mmap failed, size = " +
                                   std::to_string(size));
  }

  // insert a special marker for obtaining the whole shared memory range
  ObjectID object_id = GenerateBlobID(
      reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()));
  int fd = -1;
  int64_t map_size = 0;
  ptrdiff_t offset = 0;
  GetMallocMapinfo(pointer, &fd, &map_size, &offset);
  objects_.emplace(
      object_id,
      std::make_shared<Payload>(object_id, size, static_cast<uint8_t*>(pointer),
                                fd, map_size, offset));
  return Status::OK();
}

// Allocate memory
uint8_t* BulkStore::AllocateMemory(size_t size, int* fd, int64_t* map_size,
                                   ptrdiff_t* offset) {
  // Try to evict objects until there is enough space.
  uint8_t* pointer = nullptr;
  pointer =
      reinterpret_cast<uint8_t*>(BulkAllocator::Memalign(size, kBlockSize));
  if (pointer) {
    GetMallocMapinfo(pointer, fd, map_size, offset);
  }
  return pointer;
}

Status BulkStore::ProcessCreateRequest(const size_t data_size,
                                       ObjectID& object_id,
                                       std::shared_ptr<Payload>& object) {
  if (data_size == 0) {
    object_id = EmptyBlobID();
    object = Payload::MakeEmpty();
    return Status::OK();
  }
  int fd = -1;
  int64_t map_size = 0;
  ptrdiff_t offset = 0;
  uint8_t* pointer = nullptr;
  pointer = AllocateMemory(data_size, &fd, &map_size, &offset);
  if (pointer == nullptr) {
    return Status::NotEnoughMemory("size = " + std::to_string(data_size));
  }
  object_id = GenerateBlobID(pointer);
  objects_.emplace(object_id,
                   std::make_shared<Payload>(object_id, data_size, pointer, fd,
                                             map_size, offset));
  object = objects_[object_id];
#ifndef NDEBUG
  VLOG(10) << "after allocate: " << Footprint() << "(" << FootprintLimit()
           << ")";
#endif
  return Status::OK();
}

Status BulkStore::ProcessGetRequest(const ObjectID id,
                                    std::shared_ptr<Payload>& object) {
  if (id == EmptyBlobID()) {
    object = Payload::MakeEmpty();
    return Status::OK();
  } else if (objects_.find(id) != objects_.end()) {
    object = objects_[id];
    return Status::OK();
  } else {
    return Status::ObjectNotExists();
  }
}

Status BulkStore::ProcessGetRequest(
    const std::vector<ObjectID>& ids,
    std::vector<std::shared_ptr<Payload>>& objects) {
  for (auto object_id : ids) {
    if (object_id == EmptyBlobID()) {
      objects.push_back(Payload::MakeEmpty());
    } else if (objects_.find(object_id) != objects_.end()) {
      objects.push_back(objects_[object_id]);
    }
  }
  return Status::OK();
}

Status BulkStore::ProcessDeleteRequest(const ObjectID& object_id) {
  if (object_id == EmptyBlobID()) {
    return Status::OK();
  }
  if (objects_.find(object_id) == objects_.end()) {
    return Status::ObjectNotExists();
  }
  auto& object = objects_[object_id];
  auto buff_size = object->data_size;
  BulkAllocator::Free(object->pointer, buff_size);
  objects_.erase(object_id);
#ifndef NDEBUG
  VLOG(10) << "after free: " << Footprint() << "(" << FootprintLimit() << ")";
#endif
  return Status::OK();
}

size_t BulkStore::Footprint() const { return BulkAllocator::Allocated(); }

size_t BulkStore::FootprintLimit() const {
  return BulkAllocator::GetFootprintLimit();
}

Status BulkStore::MakeArena(size_t const size, int& fd, uintptr_t& base) {
  fd = memory::create_buffer(size);
  if (fd == -1) {
    return Status::NotEnoughMemory("Failed to allocate a new arena");
  }
  void* space = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  base = reinterpret_cast<uintptr_t>(space);
  arenas_.emplace(fd, Arena{.fd = fd,
                            .size = size,
                            .base = reinterpret_cast<uintptr_t>(space)});
  return Status::OK();
}

namespace memory {
static void recycle_arena(const uintptr_t base, const size_t size,
                          std::vector<size_t> const& offsets,
                          std::vector<size_t> const& sizes);
};

Status BulkStore::FinalizeArena(const int fd,
                                std::vector<size_t> const& offsets,
                                std::vector<size_t> const& sizes) {
  VLOG(2) << "finalizing arena (fd) " << fd << "...";
  auto arena = arenas_.find(fd);
  if (arena == arenas_.end()) {
    return Status::ObjectNotExists("Arena for fd " + std::to_string(fd) +
                                   " cannot be found");
  }
  if (offsets.size() != sizes.size()) {
    return Status::UserInputError(
        "The offsets and sizes of sealed blobs are not match");
  }
  for (size_t idx = 0; idx < offsets.size(); ++idx) {
    VLOG(2) << "blob in use: " << offsets[idx] << " of size " << sizes[idx];
  }
  // recycle memory
  {
    memory::recycle_arena(arena->second.base, arena->second.size, offsets,
                          sizes);
  }
  // make it available
  {
    memory::MmapRecord& record =
        memory::mmap_records[reinterpret_cast<void*>(arena->second.base)];
    record.fd = arena->second.fd;
    record.size = arena->second.size;
    arenas_.erase(arena);
  }
  return Status::OK();
}

namespace memory {
static inline uintptr_t align_up(const uintptr_t address,
                                 const size_t alignment) {
  return (address + alignment - 1) & ~(alignment - 1);
}

static inline uintptr_t align_down(const uintptr_t address,
                                   const size_t alignment) {
  return address & ~(alignment - 1);
}

static inline void recycle_resident_memory(const uintptr_t base, size_t left,
                                           size_t right) {
  static size_t page_size = (size_t) sysconf(_SC_PAGESIZE);
  uintptr_t aligned_left = align_up(base + left, page_size),
            aligned_right = align_down(base + right, page_size);
#ifndef NDEBUG
  VLOG(10) << "recycle memory: " << reinterpret_cast<void*>(base + left) << "("
           << reinterpret_cast<void*>(aligned_left) << ") to "
           << reinterpret_cast<void*>(base + right) << "("
           << reinterpret_cast<void*>(aligned_right) << ")";
#endif
  if (aligned_left != aligned_right) {
    /**
     * Notes [Recycle Pages with madvise]:
     *
     * 1. madvise(.., MADV_FREE) cannot be used for shared memory, thus we use
     * `MADV_DONTNEED`.
     * 2. madvise(...) requires alignment to PAGE size.
     *
     * See also: https://man7.org/linux/man-pages/man2/madvise.2.html
     */
    if (madvise(reinterpret_cast<void*>(aligned_left),
                aligned_right - aligned_left, MADV_DONTNEED)) {
      LOG(ERROR) << "madvise: " << errno << " -> " << strerror(errno);
    }
  }
}

/**
 * @brief Find non-covered intervals, and release the memory back to OS kernel.
 *
 * n.b.: the intervals may overlap.
 */
static void recycle_arena(const uintptr_t base, const size_t size,
                          std::vector<size_t> const& offsets,
                          std::vector<size_t> const& sizes) {
  std::map<size_t, int32_t> points;
  points[0] = 0;
  points[size] = 0;
  for (size_t idx = 0; idx < offsets.size(); ++idx) {
    points[offsets[idx]] += 1;
    points[offsets[idx] + sizes[idx]] -= 1;
  }
  auto head = points.begin();
  int markersum = 0;
  while (true) {
    markersum += head->second;
    auto next = std::next(head);
    if (next == points.end()) {
      break;
    }
    if (markersum == 0) {
      // release memory in the untouched interval.
      recycle_resident_memory(base, head->first, next->first);
    }
    head = next;
  }
}
}  // namespace memory

}  // namespace vineyard
