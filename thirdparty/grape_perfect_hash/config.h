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

#ifndef GRAPE_CONFIG_H_
#define GRAPE_CONFIG_H_

#ifdef USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#endif

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#endif

#include "default_allocator.h"
#include "hp_allocator.h"

namespace google {}
namespace grape {

#ifdef GFLAGS_NAMESPACE
namespace gflags = GFLAGS_NAMESPACE;
#else
namespace gflags = google;
#endif

// type alias
using fid_t = unsigned;

#ifdef USE_HUGEPAGES
template <typename T>
using Allocator = HpAllocator<T>;
#else
template <typename T>
using Allocator = DefaultAllocator<T>;
#endif

#ifdef __CUDACC__
#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#define MAX_BLOCK_SIZE 256
#define MAX_GRID_SIZE 768
#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)
#else
#define DEV_HOST
#define DEV_HOST_INLINE inline
#define DEV_INLINE
#endif

const int kCoordinatorRank = 0;

const char kSerializationVertexMapFilename[] = "vertex_map.s";
const char kSerializationFilenameFormat[] = "%s/frag_%d.s";

}  // namespace grape
#endif  // GRAPE_CONFIG_H_
