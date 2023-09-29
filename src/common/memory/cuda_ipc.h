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
#ifndef SRC_COMMON_MEMORY_CUDA_IPC_H_
#define SRC_COMMON_MEMORY_CUDA_IPC_H_

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Casting a CUDA memory pointer to its IPC handle.
//
// @param pointer: the CUDA memory pointer.
// @param handle: the IPC handle, expected to be at least of size
//                 CUDA_IPC_HANDLE_SIZE (64).
//
// @return 0 on success, other values (cudaError_t) otherwise.
int send_cuda_pointer(void* pointer, uint8_t* handle);

// Casting an IPC handle to a CUDA memory pointer.
//
// @param handle: the IPC handle, expected can be safely reinterpret
//                casted to `cudaIpcMemHandle_t`.
// @param pointer: the CUDA memory pointer.
//
// @return 0 on success, other values (cudaError_t) otherwise.
int recv_cuda_pointer(const uint8_t* handle, void** pointer);

#ifdef __cplusplus
}
#endif

#endif  // SRC_COMMON_MEMORY_CUDA_IPC_H_
