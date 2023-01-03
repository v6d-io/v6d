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

#include "malloc/malloc_wrapper.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "client/client.h"
#include "client/ds/blob.h"
#include "common/util/status.h"

#include "malloc/mimalloc_allocator.h"

namespace vineyard {

namespace detail {

static VineyardMimallocAllocator<void>& _DefaultAllocator() {
  static VineyardMimallocAllocator<void>* default_allocator =
      VineyardMimallocAllocator<void>::Create(Client::Default());
  return *default_allocator;
}

static std::mutex allocator_mutex;

}  // namespace detail

}  // namespace vineyard

void* vineyard_malloc(__attribute__((unused)) size_t size) {
  return vineyard::detail::_DefaultAllocator().allocate(size);
}

void* vineyard_calloc(__attribute__((unused)) size_t num,
                      __attribute__((unused)) size_t size) {
  return vineyard::detail::_DefaultAllocator().allocate(num * size);
}

void* vineyard_realloc(__attribute__((unused)) void* pointer,
                       __attribute__((unused)) size_t size) {
  return vineyard::detail::_DefaultAllocator().reallocate(pointer, size);
}

void vineyard_free(__attribute__((unused)) void* pointer) {
  vineyard::detail::_DefaultAllocator().deallocate(pointer, 0);
}

void vineyard_freeze(__attribute__((unused)) void* pointer) {
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::detail::_DefaultAllocator().Freeze(pointer);
}

void vineyard_allocator_finalize(__attribute__((unused)) int renew) {
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::VineyardMimallocAllocator<void>& default_allocator =
      vineyard::detail::_DefaultAllocator();
  VINEYARD_CHECK_OK(default_allocator.Release());
}
