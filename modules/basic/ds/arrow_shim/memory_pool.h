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

#ifndef MODULES_BASIC_DS_ARROW_SHIM_MEMORY_POOL_H_
#define MODULES_BASIC_DS_ARROW_SHIM_MEMORY_POOL_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "arrow/memory_pool.h"
#include "arrow/util/config.h"

#include "client/client.h"

namespace vineyard {

namespace memory {

class VineyardMemoryPool : public arrow::MemoryPool {
 public:
  explicit VineyardMemoryPool(Client& client);

  ~VineyardMemoryPool() override;

  arrow::Status Allocate(int64_t size, uint8_t** out)
#if defined(ARROW_VERSION) && ARROW_VERSION < 11000000
      override
#endif
      ;  // NOLINT(whitespace/semicolon)
  arrow::Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr)
#if defined(ARROW_VERSION) && ARROW_VERSION < 11000000
      override
#endif
      ;  // NOLINT(whitespace/semicolon)
  void Free(uint8_t* buffer, int64_t size)
#if defined(ARROW_VERSION) && ARROW_VERSION < 11000000
      override
#endif
      ;  // NOLINT(whitespace/semicolon)

#if defined(ARROW_VERSION) && ARROW_VERSION >= 11000000
  arrow::Status Allocate(int64_t size, int64_t alignment,
                         uint8_t** out) override;
  arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                           int64_t alignment, uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;
#endif

  Status Take(const uint8_t* buffer, std::unique_ptr<BlobWriter>& sbuffer);

  Status Take(const std::shared_ptr<arrow::Buffer>& buffer,
              std::unique_ptr<BlobWriter>& sbuffer);

  /// The number of bytes that were allocated and not yet free'd through
  /// this allocator.
  int64_t bytes_allocated() const override;

  /// Return peak memory allocation in this memory pool
  ///
  /// \return Maximum bytes allocated. If not known (or not implemented),
  /// returns -1
  int64_t max_memory() const override;

  /// The number of bytes that were allocated.
  int64_t total_bytes_allocated() const
#if defined(ARROW_VERSION) && ARROW_VERSION >= 12000000
      override
#endif
      ;  // NOLINT(whitespace/semicolon)

  /// The number of allocations or reallocations that were requested.
  int64_t num_allocations() const
#if defined(ARROW_VERSION) && ARROW_VERSION >= 12000000
      override
#endif
      ;  // NOLINT(whitespace/semicolon)

  std::string backend_name() const override;

 private:
  Client& client_;
  std::atomic_size_t bytes_allocated_;
  std::atomic_size_t total_bytes_allocated_;
  std::atomic_size_t num_allocations_;
  std::mutex mutex_;
  std::map<uintptr_t, std::unique_ptr<BlobWriter>> buffers_;
};

}  // namespace memory

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_ARROW_SHIM_MEMORY_POOL_H_
