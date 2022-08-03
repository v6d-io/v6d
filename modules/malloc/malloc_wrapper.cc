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

#include "malloc/malloc_wrapper.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "client/client.h"
#include "client/ds/blob.h"
#include "malloc/allocator.h"
#include "malloc/arena_allocator.h"
#include "malloc/mimalloc_allocator.h"

namespace vineyard {

namespace detail {

#if defined(WITH_JEMALLOC)
static VineyardAllocator<void>& _DefaultAllocator() {
  static VineyardAllocator<void>* default_allocator =
      new VineyardAllocator<void>{};
  return *default_allocator;
}

static VineyardArenaAllocator<void>& _ArenaAllocator() {
  static VineyardArenaAllocator<void>* default_allocator =
      new VineyardArenaAllocator<void>{};
  return *default_allocator;
}
#endif

#if defined(WITH_MIMALLOC)
static VineyardMimallocAllocator<void>& _DefaultAllocator() {
  static VineyardMimallocAllocator<void>* default_allocator =
      new VineyardMimallocAllocator<void>{};
  return *default_allocator;
}
#endif

static std::mutex allocator_mutex;

}  // namespace detail

}  // namespace vineyard

void* vineyard_malloc(__attribute__((unused)) size_t size) {
#if defined(WITH_JEMALLOC)
  return vineyard::detail::_DefaultAllocator().Allocate(size);
#elif defined(WITH_MIMALLOC)
  return vineyard::detail::_DefaultAllocator().Allocate(size);
#else
  return nullptr;
#endif
}

void* vineyard_calloc(__attribute__((unused)) size_t num,
                      __attribute__((unused)) size_t size) {
#if defined(WITH_JEMALLOC)
  return vineyard::detail::_DefaultAllocator().Allocate(num * size);
#elif defined(WITH_MIMALLOC)
  return vineyard::detail::_DefaultAllocator().Allocate(num * size);
#else
  return nullptr;
#endif
}

void* vineyard_realloc(__attribute__((unused)) void* pointer,
                       __attribute__((unused)) size_t size) {
#if defined(WITH_JEMALLOC)
  return vineyard::detail::_DefaultAllocator().Reallocate(pointer, size);
#elif defined(WITH_MIMALLOC)
  return vineyard::detail::_DefaultAllocator().Reallocate(pointer, size);
#else
  return nullptr;
#endif
}

void vineyard_free(__attribute__((unused)) void* pointer) {
#if defined(WITH_JEMALLOC)
  vineyard::detail::_DefaultAllocator().Free(pointer);
#elif defined(WITH_MIMALLOC)
  vineyard::detail::_DefaultAllocator().Free(pointer);
#else
  (void) pointer;
#endif
}

void vineyard_freeze(__attribute__((unused)) void* pointer) {
#if defined(WITH_JEMALLOC)
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::detail::_DefaultAllocator().Freeze(pointer);
#endif
#if defined(WITH_MIMALLOC)
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::detail::_DefaultAllocator().Freeze(pointer);
#endif
}

void vineyard_allocator_finalize(__attribute__((unused)) int renew) {
#if defined(WITH_JEMALLOC)
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::VineyardAllocator<void>& default_allocator =
      vineyard::detail::_DefaultAllocator();
  if (renew) {
    VINEYARD_CHECK_OK(default_allocator.Renew());
  } else {
    VINEYARD_CHECK_OK(default_allocator.Release());
  }
#endif
#if defined(WITH_MIMALLOC)
  std::lock_guard<std::mutex> lock(vineyard::detail::allocator_mutex);
  vineyard::VineyardMimallocAllocator<void>& default_allocator =
      vineyard::detail::_DefaultAllocator();
  if (renew) {
    VINEYARD_CHECK_OK(default_allocator.Renew());
  } else {
    VINEYARD_CHECK_OK(default_allocator.Release());
  }
#endif
}

void* vineyard_arena_malloc(__attribute__((unused)) size_t size) {
#if defined(WITH_JEMALLOC)
  return vineyard::detail::_ArenaAllocator().Allocate(size);
#else
  return nullptr;
#endif
}

void vineyard_arena_free(__attribute__((unused)) void* ptr) {
#if defined(WITH_JEMALLOC)
  vineyard::detail::_ArenaAllocator().Free(ptr, 0);
#endif
}
