/** Copyright 2020-2021 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef SRC_CLIENT_ARENA_ALLOCATOR_H_
#define SRC_CLIENT_ARENA_ALLOCATOR_H_

#include <stddef.h>

#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "common/memory/arena.h"

namespace vineyard {

template <typename T>
struct VineyardArenaAllocator : public memory::ArenaAllocator {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using difference_type =
      typename std::pointer_traits<pointer>::difference_type;

  explicit VineyardArenaAllocator(
      const size_t size = std::numeric_limits<size_t>::max())
      : client_(vineyard::Client::Default()) {
    VINEYARD_CHECK_OK(_initialize_arena(size));
  }

  VineyardArenaAllocator(Client& client,
                         const size_t size = std::numeric_limits<size_t>::max())
      : client_(client) {
    VINEYARD_CHECK_OK(_initialize_arena(size));
  }

  ~VineyardArenaAllocator() noexcept { VINEYARD_DISCARD(Release()); }

  template <typename U>
  VineyardArenaAllocator(const VineyardArenaAllocator<U>&) noexcept {}

  T* allocate(size_t size, const void* = nullptr) {
    return reinterpret_cast<T*>(ArenaAllocator::Allocate(size));
  }

  void deallocate(T* ptr, size_t size) {
    if (freezed_.find(reinterpret_cast<uintptr_t>(ptr)) != freezed_.end()) {
      ArenaAllocator::Free(ptr, size);
    }
  }

  std::shared_ptr<Blob> Freeze(T* ptr) {
    size_t allocated_size = ArenaAllocator::GetAllocatedSize(ptr);
    std::clog << "freeze the pointer " << ptr << " of size " << allocated_size
              << std::endl;
    offsets_.emplace_back(reinterpret_cast<uintptr_t>(ptr) - space_);
    sizes_.emplace_back(allocated_size);
    freezed_.emplace(reinterpret_cast<uintptr_t>(ptr));
    ObjectID id = base_ + (reinterpret_cast<uintptr_t>(ptr) - space_);
    return Blob::FromAllocator(client_, id, reinterpret_cast<uintptr_t>(ptr),
                               allocated_size);
  }

  Status Release() {
    std::clog << "jemalloc arena finalized: of " << offsets_.size()
              << " blocks are in use." << std::endl;
    return client_.ReleaseArena(fd_, offsets_, sizes_);
  }

  Status Renew() {
    RETURN_ON_ERROR(client_.ReleaseArena(fd_, offsets_, sizes_));
    return _initialize_arena(available_size_);
  }

  template <typename U>
  struct rebind {
    using other = VineyardArenaAllocator<U>;
  };

 private:
  Client& client_;
  int fd_;
  uintptr_t base_, space_;
  size_t available_size_;
  std::vector<size_t> offsets_, sizes_;
  std::set<uintptr_t> freezed_;

  Status _initialize_arena(size_t size) {
    std::clog << "make arena: " << size << std::endl;
    RETURN_ON_ERROR(
        client_.CreateArena(size, fd_, available_size_, base_, space_));
    ArenaAllocator::Init(reinterpret_cast<void*>(space_), available_size_);
    std::clog << "jemalloc arena initialized: " << available_size_ << ", at "
              << reinterpret_cast<void*>(space_) << std::endl;

    // reset the context
    offsets_.clear();
    sizes_.clear();
    freezed_.clear();
    return Status::OK();
  }
};

template <typename T, typename U>
constexpr bool operator==(const VineyardArenaAllocator<T>&,
                          const VineyardArenaAllocator<U>&) noexcept {
  return true;
}

template <typename T, typename U>
constexpr bool operator!=(const VineyardArenaAllocator<T>&,
                          const VineyardArenaAllocator<U>&) noexcept {
  return false;
}

}  // namespace vineyard

#endif  // SRC_CLIENT_ARENA_ALLOCATOR_H_
