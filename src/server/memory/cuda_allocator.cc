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

#include "server/memory/cuda_allocator.h"

#include <cstdio>
#include <cstring>

#include "common/util/logging.h"   // IWYU pragma: keep
#include "server/memory/malloc.h"  // IWYU pragma: keep

#if defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <sys/mount.h>
#endif

namespace vineyard {

int64_t CUDABulkAllocator::cuda_allocated_ = 0;
int64_t CUDABulkAllocator::cuda_footprint_limit_ = 0;

// CUDABulkAllocator implementation
void* CUDABulkAllocator::Init(const size_t size) {
  cuda_allocated_ = 0;
  cuda_footprint_limit_ = size;
  return nullptr;
}

void* CUDABulkAllocator::Memalign(const size_t bytes, const size_t alignment) {
  void* mem = nullptr;
#ifdef ENABLE_CUDA
  cudaError_t result = cudaMalloc(&mem, bytes);
  if (result != cudaSuccess) {
    DVLOG(10) << "cudaMalloc Error: " << cudaGetErrorString(result);
    return nullptr;
  }
  cuda_allocated_ += bytes;
#endif
  return mem;
}

void CUDABulkAllocator::Free(void* mem, size_t bytes) {
#ifdef ENABLE_CUDA
  cudaError_t result = cudaFree(mem);
  if (result != cudaSuccess) {
    DVLOG(10) << "cudaFree Error: " << cudaGetErrorString(result);
  }
  cuda_allocated_ -= bytes;
#endif
}

void CUDABulkAllocator::SetFootprintLimit(size_t bytes) {
  cuda_footprint_limit_ = static_cast<int64_t>(bytes);
}

int64_t CUDABulkAllocator::GetFootprintLimit() { return cuda_footprint_limit_; }

int64_t CUDABulkAllocator::Allocated() { return cuda_allocated_; }

}  // namespace vineyard
