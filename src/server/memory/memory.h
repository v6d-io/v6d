/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/memory.h is referred and derived from project
 * apache-arrow,
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/plasma/memory.h
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

#ifndef SRC_SERVER_MEMORY_MEMORY_H_
#define SRC_SERVER_MEMORY_MEMORY_H_

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "oneapi/tbb/concurrent_hash_map.h"

#include "common/memory/payload.h"
#include "common/util/status.h"

namespace vineyard {

class BulkStore {
 public:
  using object_map_t =
      tbb::concurrent_hash_map<ObjectID, std::shared_ptr<Payload>>;

  ~BulkStore();

  Status PreAllocate(const size_t size);

  Status Create(const size_t size, ObjectID& object_id,
                std::shared_ptr<Payload>& object);

  Status Get(const ObjectID id, std::shared_ptr<Payload>& object);

  /**
   * This methods only return available objects, and doesn't fail when object
   * does not exists.
   */
  Status Get(const std::vector<ObjectID>& ids,
             std::vector<std::shared_ptr<Payload>>& objects);

  Status Delete(const ObjectID& object_id);

  bool Exists(const ObjectID& object_id);

  object_map_t const& List() const { return objects_; }

  size_t Footprint() const;
  size_t FootprintLimit() const;

  Status MakeArena(const size_t size, int& fd, uintptr_t& base);

  Status FinalizeArena(const int fd, std::vector<size_t> const& offsets,
                       std::vector<size_t> const& sizes);

 private:
  uint8_t* AllocateMemory(size_t size, int* fd, int64_t* map_size,
                          ptrdiff_t* offset);
  struct Arena {
    int fd;
    size_t size;
    uintptr_t base;
    static std::set<ObjectID> spans;
  };

  std::unordered_map<int /* fd */, Arena> arenas_;

  object_map_t objects_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_MEMORY_H_
