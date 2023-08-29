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

#include "basic/ds/arrow_shim/memory_pool.h"

#include <memory>
#include <string>
#include <utility>

#include "arrow/buffer.h"

#include "client/ds/blob.h"
#include "common/memory/memcpy.h"

namespace vineyard {

namespace memory {

VineyardMemoryPool::VineyardMemoryPool(Client& client) : client_(client) {
  bytes_allocated_.store(0);
  total_bytes_allocated_.store(0);
  num_allocations_.store(0);
}

VineyardMemoryPool::~VineyardMemoryPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& it : buffers_) {
    VINEYARD_DISCARD(it.second->Abort(client_));
  }
}

arrow::Status VineyardMemoryPool::Allocate(int64_t size, uint8_t** out) {
  if (size > 0) {
    std::unique_ptr<BlobWriter> sbuffer;
    auto status = client_.CreateBlob(size, sbuffer);
    if (!status.ok()) {
      return arrow::Status(arrow::StatusCode::OutOfMemory, status.ToString());
    }

    *out = reinterpret_cast<uint8_t*>(sbuffer->Buffer()->mutable_data());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      bytes_allocated_.fetch_add(size);
      total_bytes_allocated_.fetch_add(size);
      num_allocations_.fetch_add(1);
      buffers_.emplace(reinterpret_cast<uintptr_t>(*out), std::move(sbuffer));
    }
    return arrow::Status::OK();
  } else {
    *out = nullptr;
    return arrow::Status::OK();
  }
}

arrow::Status VineyardMemoryPool::Reallocate(int64_t old_size, int64_t new_size,
                                             uint8_t** ptr) {
  if (old_size >= new_size) {
    return arrow::Status::OK();
  }
  std::unique_ptr<BlobWriter> sbuffer;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(reinterpret_cast<uintptr_t>(*ptr));
    if (it != buffers_.end()) {
      sbuffer = std::move(it->second);
      bytes_allocated_.fetch_sub(old_size);
      buffers_.erase(it);
    }
  }
  if (sbuffer == nullptr) {
    return arrow::Status(arrow::StatusCode::OutOfMemory,
                         "Reallocate from an unknown buffer");
  }
  std::unique_ptr<BlobWriter> nsbuffer;
  auto status = client_.CreateBlob(new_size, nsbuffer);
  if (!status.ok()) {
    {
      // fill the old buffer back
      std::lock_guard<std::mutex> lock(mutex_);
      bytes_allocated_.fetch_add(old_size);
      *ptr = reinterpret_cast<uint8_t*>(sbuffer->Buffer()->mutable_data());
      buffers_.emplace(reinterpret_cast<uintptr_t>(*ptr), std::move(sbuffer));
    }
    return arrow::Status(arrow::StatusCode::OutOfMemory, status.ToString());
  }

  // copy to the new one
  *ptr = reinterpret_cast<uint8_t*>(nsbuffer->Buffer()->mutable_data());
  inline_memcpy(*ptr, sbuffer->Buffer()->data(), sbuffer->Buffer()->size());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    bytes_allocated_.fetch_add(new_size);
    const auto diff = new_size - old_size;
    if (diff > 0) {
      total_bytes_allocated_.fetch_add(diff);
    }
    num_allocations_.fetch_add(1);
    buffers_.emplace(reinterpret_cast<uintptr_t>(*ptr), std::move(nsbuffer));
  }
  // remove the original buffer
  VINEYARD_CHECK_OK(sbuffer->Abort(client_));
  return arrow::Status::OK();
}

void VineyardMemoryPool::Free(uint8_t* buffer, int64_t size) {
  std::unique_ptr<BlobWriter> sbuffer;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(reinterpret_cast<uintptr_t>(buffer));
    if (it != buffers_.end()) {
      sbuffer = std::move(it->second);
      bytes_allocated_.fetch_sub(size);
      buffers_.erase(it);
    }
  }
  if (sbuffer) {
    VINEYARD_CHECK_OK(sbuffer->Abort(client_));
  }
}

#if defined(ARROW_VERSION) && ARROW_VERSION >= 11000000
arrow::Status VineyardMemoryPool::Allocate(int64_t size, int64_t alignment,
                                           uint8_t** out) {
  return this->Allocate(size, out);
}

arrow::Status VineyardMemoryPool::Reallocate(int64_t old_size, int64_t new_size,
                                             int64_t alignment, uint8_t** ptr) {
  return this->Reallocate(old_size, new_size, ptr);
}

void VineyardMemoryPool::Free(uint8_t* buffer, int64_t size,
                              int64_t alignment) {
  return this->Free(buffer, size);
}
#endif

Status VineyardMemoryPool::Take(const uint8_t* buffer,
                                std::unique_ptr<BlobWriter>& sbuffer) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(reinterpret_cast<uintptr_t>(buffer));
    if (it != buffers_.end()) {
      sbuffer = std::move(it->second);
      bytes_allocated_.fetch_sub(sbuffer->size());
      buffers_.erase(it);
      return Status::OK();
    }
  }
  return Status::ObjectNotExists(
      "cannot find the blob for pointer " +
      std::to_string(reinterpret_cast<uintptr_t>(buffer)));
}

Status VineyardMemoryPool::Take(const std::shared_ptr<arrow::Buffer>& buffer,
                                std::unique_ptr<BlobWriter>& sbuffer) {
  return Take(buffer->data(), sbuffer);
}

/// The number of bytes that were allocated and not yet free'd through
/// this allocator.
int64_t VineyardMemoryPool::bytes_allocated() const {
  return bytes_allocated_.load();
}

/// Return peak memory allocation in this memory pool
///
/// \return Maximum bytes allocated. If not known (or not implemented),
/// returns -1
int64_t VineyardMemoryPool::max_memory() const { return -1; }

/// The number of bytes that were allocated.
int64_t VineyardMemoryPool::total_bytes_allocated() const {
  return total_bytes_allocated_.load();
}

/// The number of allocations or reallocations that were requested.
int64_t VineyardMemoryPool::num_allocations() const {
  return num_allocations_.load();
}

std::string VineyardMemoryPool::backend_name() const { return "vineyard"; }

}  // namespace memory

}  // namespace vineyard
