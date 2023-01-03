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
#ifndef SRC_COMMON_MEMORY_GPU_UNIFIED_MEMORY_H_
#define SRC_COMMON_MEMORY_GPU_UNIFIED_MEMORY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#else
// to pass the compiling anyway.
typedef uint64_t cudaIpcMemHandle_t;
#endif

namespace vineyard {
enum class GUAError_t {
  guaSuccess = 0,
  guaInvalidGpuAddress = 1,
  guaInvalidCpuAddress = 2,
  guaMallocGPUFailed = 3,
  guaMallocCPUFailed = 4,
  guaOpenHandleFailed = 5,
  guaGetHandleFailed = 6,
  guaSyncFailed = 7,
};

std::string guaErrorToString(GUAError_t error);

class GPUUnifiedAddress {
 public:
  GPUUnifiedAddress() {
    ipc_owner_ = false;
    has_gpu_ = false;
    has_cpu_ = false;
    cpu_ptr_ = nullptr;
    gpu_ptr_ = nullptr;
    data_size_ = 0;
    ref_ = 0;
  }

  explicit GPUUnifiedAddress(bool owner) {
    ipc_owner_ = owner;
    has_gpu_ = false;
    has_cpu_ = false;
    cpu_ptr_ = nullptr;
    gpu_ptr_ = nullptr;
    data_size_ = 0;
    ref_ = 0;
  }

  GPUUnifiedAddress(bool owner, void* GPU_ptr) {
    ipc_owner_ = owner;
    has_gpu_ = false;
    has_cpu_ = false;
    cpu_ptr_ = nullptr;
    gpu_ptr_ = GPU_ptr;
    data_size_ = 0;
    ref_ = 0;
  }

  GPUUnifiedAddress(bool owner, cudaIpcMemHandle_t handle) {
    ipc_owner_ = owner;
    has_gpu_ = true;
    has_cpu_ = false;
    cpu_ptr_ = nullptr;
    gpu_ptr_ = nullptr;
    data_size_ = 0;
    ref_ = 0;
    cuda_handle_.handle = handle;
  }

  /**
   * @brief get the cpu memry address
   *
   * @param ptr the return cpu data address
   * @return GUAError_t the error type
   */
  GUAError_t CPUData(void** ptr);

  /**
   * @brief get the gpu memory address
   *
   * @param ptr the return gpu data address
   * @return GUAError_t the error type
   */
  GUAError_t GPUData(void** ptr);

  /**
   * @brief sync data from GPU related to this gua
   *
   * @return GUAError_t the error type
   */
  GUAError_t syncFromCPU();

  /**
   * @brief sync data from CPU related to this gua
   *
   * @return GUAError_t the error type
   */
  GUAError_t syncFromGPU();

  /**
   * @brief  Malloc memory related to this gua if needed.
   *
   * @param size the memory size to be allocated
   * @param ptr the memory address on cpu or GPU
   * @param is_GPU allocate on GPU
   * @return GUAError_t the error type
   */
  GUAError_t ManagedMalloc(size_t size, void** ptr, bool is_GPU = false);
  /**
   * @brief Free the memory
   *
   */
  void ManagedFree();
  /**
   * @brief GUA to json
   *
   */
  void GUAToJSON();

  /**
   * @brief Get the Ipc Handle object
   *
   * @param handle the returned handle
   * @return GUAError_t the error type
   */
  GUAError_t getIpcHandle(cudaIpcMemHandle_t& handle);
  /**
   * @brief Set the IpcHandle of this GUA
   *
   * @param handle
   */
  void setIpcHandle(cudaIpcMemHandle_t handle);

  /**
   * @brief Get the IpcHandle of this GUA as vector
   *
   * @return std::vector<int64_t>
   */
  std::vector<int64_t> getIpcHandleVec();

  /**
   * @brief Set the IpcHandle vector of this GUA
   *
   * @param handle_vec
   */
  void setIpcHandleVec(std::vector<int64_t> handle_vec);

  /**
   * @brief Set the GPU Mem Ptr object
   *
   * @param ptr
   */
  void setGPUMemPtr(void* ptr);

  /**
   * @brief return the GPU memory pointer
   *
   * @return void* the GPU-side memory address
   */
  void* getGPUMemPtr();
  /**
   * @brief Set the Cpu Mem Ptr object
   *
   * @param ptr
   */
  void setCPUMemPtr(void* ptr);

  /**
   * @brief Get the Cpu Mem Ptr object
   *
   * @return void*
   */
  void* getCPUMemPtr();

  /**
   * @brief Get the Size object
   *
   * @return int64_t
   */
  int64_t getSize();

  /**
   * @brief Set the Size object
   *
   * @param data_size
   */
  void setSize(int64_t data_size);

 private:
  bool has_gpu_;
  bool has_cpu_;
  bool ipc_owner_;
  void* cpu_ptr_;
  void* gpu_ptr_;
  // use a union struct to present handle
  // since it is easy to get
  union {
    cudaIpcMemHandle_t handle;
    int64_t handle_vec[8];
  } cuda_handle_;

  int64_t data_size_;
  int64_t ref_;
};

}  // namespace vineyard

#endif  // SRC_COMMON_MEMORY_GPU_UNIFIED_MEMORY_H_
