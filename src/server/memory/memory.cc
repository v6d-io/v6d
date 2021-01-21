/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/memory.cc is referred and derived from project
 * apache-arrow,
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

#include <memory>
#include <vector>

#include "server/memory/allocator.h"
#include "server/memory/malloc.h"

namespace vineyard {

using plasma::BulkAllocator;
using plasma::GetMallocMapinfo;
using plasma::kBlockSize;

Status BulkStore::PreAllocate(const size_t size) {
  BulkAllocator::SetFootprintLimit(size);
  // We are using a single memory-mapped file by mallocing and freeing a single
  // large amount of space up front.
  void* pointer =
      BulkAllocator::Memalign(kBlockSize, size - 256 * sizeof(size_t));
  if (pointer == nullptr) {
    return Status::NotEnoughMemory("size = " + std::to_string(size));
  }
  // This will unmap the file, but the next one created will be as large
  // as this one (this is an implementation detail of dlmalloc).
  BulkAllocator::Free(pointer, size - 256 * sizeof(size_t));
  return Status::OK();
}

// Allocate memory
uint8_t* BulkStore::AllocateMemory(size_t size, int* fd, int64_t* map_size,
                                   ptrdiff_t* offset) {
  // Try to evict objects until there is enough space.
  uint8_t* pointer = nullptr;
  pointer =
      reinterpret_cast<uint8_t*>(BulkAllocator::Memalign(kBlockSize, size));
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

}  // namespace vineyard
