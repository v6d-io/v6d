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

#include "common/memory/gpu/unified_memory.h"
namespace vineyard {

std::string guaErrorToString(GUAError_t error) {
  std::string res = "";
  switch (error) {
  case GUAError_t::guaInvalidGpuAddress:
    res = "GUA: Invalid GPU address.";
    break;
  case GUAError_t::guaInvalidCpuAddress:
    res = "GUA: Invalid CPU address.";
    break;
  case GUAError_t::guaMallocCPUFailed:
    res = "GUA: Malloc CPU memory failed.";
    break;
  case GUAError_t::guaMallocGPUFailed:
    res = "GUA: Malloc GPU memory failed.";
    break;
  case GUAError_t::guaGetHandleFailed:
    res = "GUA: Get Ipc Handle failed.";
    break;
  case GUAError_t::guaOpenHandleFailed:
    res = "GUA: Open Ipc Handle failed.";
    break;
  case GUAError_t::guaSyncFailed:
    res = "GUA: Sync data failed.";
    break;
  default:
    break;
  }
  return res;
}

GUAError_t GPUUnifiedAddress::ManagedMalloc(size_t size, void** ptr,
                                            bool is_GPU) {
#ifdef ENABLE_GPU
  data_size_ = size;
  if (is_GPU) {
    cudaError_t result = cudaMalloc(&gpu_ptr_, size);
    if (result != cudaSuccess) {
      *ptr = nullptr;
      return GUAError_t::guaMallocGPUFailed;
    }
    has_gpu_ = true;
    *ptr = gpu_ptr_;
  } else {
    if (cpu_ptr_)
      free(cpu_ptr_);
    cpu_ptr_ = malloc(data_size_);
    if (cpu_ptr_ != nullptr) {
      has_cpu_ = true;
      *ptr = cpu_ptr_;
      return GUAError_t::guaSuccess;
    } else {
      *ptr = cpu_ptr_ = nullptr;
      return GUAError_t::guaMallocGPUFailed;
    }
  }
#endif
  return GUAError_t::guaSuccess;
}

void GPUUnifiedAddress::ManagedFree() {
  if (has_cpu_) {
    free(cpu_ptr_);
  }
#ifdef ENABLE_GPU
  if (ipc_owner_ && has_gpu_) {
    cudaFree(gpu_ptr_);
  }
#endif
}

GUAError_t GPUUnifiedAddress::CPUData(void** ptr) {
  if (has_cpu_) {
    *ptr = cpu_ptr_;
    return GUAError_t::guaSuccess;
  } else if (!has_cpu_) {
    cpu_ptr_ = malloc(data_size_);
    if (cpu_ptr_ != nullptr) {
      has_cpu_ = true;
      *ptr = cpu_ptr_;
      return GUAError_t::guaSuccess;
    } else {
      return GUAError_t::guaMallocCPUFailed;
    }
  }
  *ptr = nullptr;
  return GUAError_t::guaSuccess;
}

GUAError_t GPUUnifiedAddress::GPUData(void** ptr) {
#ifdef ENABLE_GPU
  // server process that mallocs the GPU memory
  if (ipc_owner_ && has_gpu_ && gpu_ptr_ != nullptr) {
    *ptr = gpu_ptr_;
    return GUAError_t::guaSuccess;
  } else if (!ipc_owner_) {
    // client has opened the handle before, return the GPU address directly
    if (has_gpu_ && gpu_ptr_ != nullptr) {
      *ptr = gpu_ptr_;
      return GUAError_t::guaSuccess;
    }
    // open the handle and get the GPU address.
    cudaError_t result = cudaIpcOpenMemHandle(&gpu_ptr_, cuda_handle_.handle,
                                              cudaIpcMemLazyEnablePeerAccess);
    if (result != cudaSuccess) {
      *ptr = gpu_ptr_ = nullptr;
      return GUAError_t::guaOpenHandleFailed;
    }
    has_gpu_ = true;
    *ptr = gpu_ptr_;
    return GUAError_t::guaSuccess;
  }
#endif
  *ptr = nullptr;
  return GUAError_t::guaSuccess;
}

GUAError_t GPUUnifiedAddress::syncFromCPU() {
#ifdef ENABLE_GPU
  void* ptr = nullptr;
  GPUData(&ptr);
  if (gpu_ptr_ == nullptr) {
    return GUAError_t::guaInvalidGpuAddress;
  }
  if (cpu_ptr_ == nullptr) {
    return GUAError_t::guaInvalidCpuAddress;
  }
  if (has_gpu_ && has_cpu_) {
    cudaError_t result =
        cudaMemcpy(gpu_ptr_, cpu_ptr_, data_size_, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      return GUAError_t::guaSyncFailed;
    }
    return GUAError_t::guaSuccess;
  }
#endif
  return GUAError_t::guaSuccess;
}

GUAError_t GPUUnifiedAddress::syncFromGPU() {
#ifdef ENABLE_GPU
  if (has_gpu_ && has_gpu_) {
    cudaError_t result =
        cudaMemcpy(cpu_ptr_, gpu_ptr_, data_size_, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      return GUAError_t::guaSyncFailed;
    }
    return GUAError_t::guaSuccess;
  } else if (!has_gpu_) {
    return GUAError_t::guaInvalidGpuAddress;
  } else if (!has_cpu_) {
    return GUAError_t::guaInvalidCpuAddress;
  }
#endif
  return GUAError_t::guaSuccess;
}

void GPUUnifiedAddress::GUAToJSON() {}

GUAError_t GPUUnifiedAddress::getIpcHandle(cudaIpcMemHandle_t& handle) {
#ifdef ENABLE_GPU
  if (ipc_owner_) {
    // get cudaIpcMemHandle
    cudaError_t result = cudaIpcGetMemHandle(
        reinterpret_cast<cudaIpcMemHandle_t*>(&cuda_handle_.handle), gpu_ptr_);
    if (result != cudaSuccess) {
      handle = cuda_handle_.handle;
      return GUAError_t::guaGetHandleFailed;
    }
  }
#endif
  return GUAError_t::guaSuccess;
}

std::vector<int64_t> GPUUnifiedAddress::getIpcHandleVec() {
  if (ipc_owner_) {
    cudaIpcMemHandle_t handle;
    getIpcHandle(handle);
    return std::vector<int64_t>(std::begin(cuda_handle_.handle_vec),
                                std::end(cuda_handle_.handle_vec));
  }
  return std::vector<int64_t>();
}

void GPUUnifiedAddress::setIpcHandle(cudaIpcMemHandle_t handle) {
  cuda_handle_.handle = handle;
  has_gpu_ = true;
}

void GPUUnifiedAddress::setIpcHandleVec(std::vector<int64_t> handle_vec) {
  if (handle_vec.size() != 8) {
    return;
  }
  for (size_t i = 0; i < 8; i++) {
    cuda_handle_.handle_vec[i] = handle_vec[i];
  }
  has_cpu_ = true;
}

void GPUUnifiedAddress::setGPUMemPtr(void* ptr) { gpu_ptr_ = ptr; }

void* GPUUnifiedAddress::getGPUMemPtr() { return gpu_ptr_; }

void GPUUnifiedAddress::setCPUMemPtr(void* ptr) { cpu_ptr_ = ptr; }

void* GPUUnifiedAddress::getCPUMemPtr() { return cpu_ptr_; }

int64_t GPUUnifiedAddress::getSize() { return data_size_; }

void GPUUnifiedAddress::setSize(int64_t data_size) { data_size_ = data_size; }

}  // namespace vineyard
