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
#include "common/memory/cuda_ipc.h"

#include <cstddef>
#include <cstdint>

#if defined(ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

int send_cuda_pointer(void* pointer, uint8_t* handle) {
#if defined(ENABLE_CUDA)
  return cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(handle),
                             pointer);
#else
  return -1;
#endif
}

int recv_cuda_pointer(const uint8_t* handle, void** pointer) {
#if defined(ENABLE_CUDA)
  return cudaIpcOpenMemHandle(
      pointer,
      *reinterpret_cast<cudaIpcMemHandle_t*>(const_cast<uint8_t*>(handle)),
      cudaIpcMemLazyEnablePeerAccess);
#else
  return -1;
#endif
}
