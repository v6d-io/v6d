/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#ifndef SRC_SERVER_MEMORY_GPU_GPUALLOCATOR_H_
#define SRC_SERVER_MEMORY_GPU_GPUALLOCATOR_H_

#include <cstddef>
#include <cstdint>

#ifdef ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "common/memory/gpu/unified_memory.h"

namespace vineyard {

class GPUBulkAllocator {
 public:
  /**
   * @brief GPU memory allocator to be complished
   * for now, adopt a naive "allocate-on-request" solution
   *
   * @param size the maximum size of GPU memory can be allocated
   * @return void*
   */
  static void* Init(const size_t size);

  /**
   * @brief Allocates size bytes and returns a pointer to the allocated memory.
   * The memory address will be a multiple of alignment, which must be a power
   * of two.
   * TODO: memory alignment
   *
   * @param bytes size of memory in bytes
   * @param alignment memory alignment
   * @return void*  pointer to allocated GPU memory
   */
  static void* Memalign(size_t bytes, size_t alignment);

  /**
   * @brief Frees the memory space pointed to by mem, which must have been
   * returned by a previous call to Memalign()
   *
   * @param mem Pointer to memory to free.
   * @param bytes Number of bytes to be freed.
   */
  static void Free(void* mem, size_t bytes);

  /**
   * @brief Set the memory footprint limit for Plasma.
   *
   * @return int64_t Plasma memory footprint limit in bytes.
   */
  static void SetFootprintLimit(size_t bytes);

  /**
   * @brief Get the number of bytes allocated by Plasma so far.
   *
   * @return int64_t Number of bytes allocated by Plasma so far
   */
  static int64_t GetFootprintLimit();

  /**
   * @brief Get the number of bytes allocated by Plasma so far.
   *
   * @return int64_t Number of bytes allocated by Plasma so far.
   */
  static int64_t Allocated();

// GPU memory Allocator to be finished
#if defined(WITH_GPUALLOCATOR)
  using Allocator = vineyard::memory::GPUAllocator;
#endif

 private:
  static int64_t gpu_allocated_;
  static int64_t gpu_footprint_limit_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_GPU_GPUALLOCATOR_H_
