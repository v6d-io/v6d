/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

#ifndef MODULES_MALLOC_MIMALLOC_ALLOCATOR_H_
#define MODULES_MALLOC_MIMALLOC_ALLOCATOR_H_

#include <stddef.h>

#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"

namespace vineyard {

namespace memory {

class Mimalloc;

namespace detail {

// avoid requires mimalloc's headers in the public interfaces of vineyard

Status _initialize(Client& client, int& fd, size_t& size, uintptr_t& base,
                   uintptr_t& space, const size_t requested_size,
                   std::shared_ptr<Mimalloc>& allocator);
void* _allocate(const std::shared_ptr<Mimalloc>& allocator, size_t size);
void* _reallocate(const std::shared_ptr<Mimalloc>& allocator, void* pointer,
                  size_t size);
void _deallocate(const std::shared_ptr<Mimalloc>& allocator, void* pointer,
                 size_t size);
size_t _allocated_size(const std::shared_ptr<Mimalloc>& allocator,
                       void* pointer);

}  // namespace detail
}  // namespace memory

template <typename T>
struct VineyardMimallocAllocator {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using difference_type =
      typename std::pointer_traits<pointer>::difference_type;

  // returns a singleton instance
  static VineyardMimallocAllocator<T>* Create(Client& client) {
    static VineyardMimallocAllocator<T>* allocator =
        new VineyardMimallocAllocator<T>(client);
    return allocator;
  }

  ~VineyardMimallocAllocator() noexcept { VINEYARD_DISCARD(Release()); }

  template <typename U>
  VineyardMimallocAllocator(const VineyardMimallocAllocator<U>&) noexcept {}

  T* allocate(size_t size, const void* hint = nullptr) {
    return reinterpret_cast<T*>(memory::detail::_allocate(allocator_, size));
  }

  T* reallocate(void* pointer, size_t size) {
    return reinterpret_cast<T*>(
        memory::detail::_reallocate(allocator_, pointer, size));
  }

  void deallocate(T* ptr, size_t size = 0) {
    memory::detail::_deallocate(allocator_, ptr, size);
  }

  std::shared_ptr<Blob> Freeze(T* ptr) {
    size_t size = memory::detail::_allocated_size(allocator_, ptr);
    std::clog << "freezing the pointer " << ptr << " of size " << size
              << std::endl;
    offsets_.emplace_back(reinterpret_cast<uintptr_t>(ptr) - space_);
    sizes_.emplace_back(size);
    freezed_.emplace(reinterpret_cast<uintptr_t>(ptr));
    ObjectID id = base_ + (reinterpret_cast<uintptr_t>(ptr) - space_);
    return Blob::FromAllocator(client_, id, reinterpret_cast<uintptr_t>(ptr),
                               size);
  }

  Status Release() {
    std::clog << "mimalloc arena finalized: of " << offsets_.size()
              << " blocks are in use." << std::endl;
    return client_.ReleaseArena(fd_, offsets_, sizes_);
  }

  template <typename U>
  struct rebind {
    using other = VineyardMimallocAllocator<U>;
  };

 private:
  Client& client_;
  int fd_;
  size_t size_;
  uintptr_t base_;
  uintptr_t space_;
  std::vector<size_t> offsets_;
  std::vector<size_t> sizes_;
  std::set<uintptr_t> freezed_;
  std::shared_ptr<memory::Mimalloc> allocator_;

  explicit VineyardMimallocAllocator(
      Client& client, const size_t size = std::numeric_limits<size_t>::max())
      : client_(client) {
    VINEYARD_CHECK_OK(memory::detail::_initialize(client_, fd_, size_, base_,
                                                  space_, size, allocator_));

    // reset the context
    offsets_.clear();
    sizes_.clear();
    freezed_.clear();
  }
};

template <typename T, typename U>
constexpr bool operator==(const VineyardMimallocAllocator<T>&,
                          const VineyardMimallocAllocator<U>&) noexcept {
  // singleton
  return true;
}

template <typename T, typename U>
constexpr bool operator!=(const VineyardMimallocAllocator<T>&,
                          const VineyardMimallocAllocator<U>&) noexcept {
  // singleton
  return false;
}

}  // namespace vineyard

#endif  // MODULES_MALLOC_MIMALLOC_ALLOCATOR_H_
