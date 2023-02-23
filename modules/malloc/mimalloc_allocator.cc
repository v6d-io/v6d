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

#include "malloc/mimalloc_allocator.h"

#include <mutex>
#include <utility>

#include "client/client.h"
#include "client/ds/blob.h"

#include "common/memory/mimalloc.h"

namespace vineyard {

namespace memory {
namespace detail {

Status _initialize(Client& client, int& fd, size_t& size, uintptr_t& base,
                   uintptr_t& space, const size_t requested_size,
                   std::shared_ptr<Mimalloc>& allocator) {
  std::clog << "making arena: " << size << std::endl;
  RETURN_ON_ERROR(client.CreateArena(requested_size, fd, size, base, space));

  allocator =
      std::make_shared<memory::Mimalloc>(reinterpret_cast<void*>(space), size);
  std::clog << "mimalloc arena initialized: " << size << ", at "
            << reinterpret_cast<void*>(space) << std::endl;

  return Status::OK();
}

void* _allocate(const std::shared_ptr<Mimalloc>& allocator, size_t size) {
  return allocator->Allocate(size);
}

void* _reallocate(const std::shared_ptr<Mimalloc>& allocator, void* pointer,
                  size_t size) {
  return allocator->Reallocate(pointer, size);
}

void _deallocate(const std::shared_ptr<Mimalloc>& allocator, void* pointer,
                 size_t size) {
  allocator->Free(pointer, size);
}

size_t _allocated_size(const std::shared_ptr<Mimalloc>& allocator,
                       void* pointer) {
  return allocator->GetAllocatedSize(pointer);
}

}  // namespace detail
}  // namespace memory

}  // namespace vineyard
