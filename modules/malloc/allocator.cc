/** Copyright 2020 Alibaba Group Holding Limited.

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

#include "malloc/allocator.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "client/allocator.h"
#include "client/client.h"
#include "client/ds/blob.h"

namespace vineyard {

namespace detail {

static VineyardAllocator<void>& _DefaultAllocator() {
  static VineyardAllocator<void>* default_allocator =
      new VineyardAllocator<void>{};
  return *default_allocator;
}

static std::mutex allocator_mutex;

}  // namespace detail

}  // namespace vineyard

void* vineyard_malloc(size_t size) {
  return vineyard::detail::_DefaultAllocator().Allocate(size);
}

void* vineyard_calloc(size_t num, size_t size) {
  return vineyard::detail::_DefaultAllocator().Allocate(num * size);
}

void* vineyard_realloc(void* pointer, size_t size) {
  return vineyard::detail::_DefaultAllocator().Reallocate(pointer, size);
}

void vineyard_free(void* pointer) {
  vineyard::detail::_DefaultAllocator().Free(pointer);
}

void vineyard_freeze(void* pointer) {
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::detail::_DefaultAllocator().Freeze(pointer);
}

void vineyard_allocator_finalize(int renew) {
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::VineyardAllocator<void>& default_allocator =
      vineyard::detail::_DefaultAllocator();
  if (renew) {
    VINEYARD_CHECK_OK(default_allocator.Renew());
  } else {
    VINEYARD_CHECK_OK(default_allocator.Release());
  }
}
