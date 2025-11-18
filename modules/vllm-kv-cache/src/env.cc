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

#include <string>

#include "vllm-kv-cache/src/env.h"

namespace vineyard {

std::string VLLMKVCacheEnv::GetKVStorageConcurrency() {
  static std::string kv_storage_concurrency =
      read_env("VLLM_KV_STORAGE_CONCURRENCY",
               std::to_string(std::thread::hardware_concurrency()));
  return kv_storage_concurrency;
}

std::string VLLMKVCacheEnv::GetVLLMBlockPrefix() {
  static std::string vllm_block_prefix =
      read_env("VLLM_BLOCK_PREFIX", "block.hash.key.");
  return vllm_block_prefix;
}

std::string VLLMKVCacheEnv::GetVineyardVLLMKVCacheIOTimeoutMilliseconds() {
  static std::string io_timeout_milliseconds =
      read_env("VINEYARD_VLLM_KV_CACHE_IO_TIMEOUT_MILLISECONDS", "5000");
  return io_timeout_milliseconds;
}

std::string VLLMKVCacheEnv::GetDirectIOAlign() {
  static std::string direct_io_align = read_env("DIRECT_IO_ALIGN", "4096");
  return direct_io_align;
}

std::string VLLMKVCacheEnv::AIORetryWaitMicroseconds() {
  static std::string aio_retry_wait_microseconds =
      read_env("VINEYARD_AIO_RETRY_WAIT_MICROSECONDS", "1000");
  return aio_retry_wait_microseconds;
}

std::string VLLMKVCacheEnv::AIOGCWaitTimeMicroseconds() {
  static std::string aio_gc_wait_time_microseconds =
      read_env("VINEYARD_AIO_GC_WAIT_TIME_MICROSECONDS", "10000");
  return aio_gc_wait_time_microseconds;
}

std::string VLLMKVCacheEnv::GetVineyardAIOSubmitConcurrency() {
  static std::string aio_submit_concurrency =
      read_env("VINEYARD_AIO_SUBMIT_CONCURRENCY", "4");
  return aio_submit_concurrency;
}

std::string VLLMKVCacheEnv::VineyardEnableVLLMKVCacheMemCopy() {
  std::string enable_mem_copy =
      read_env("VINEYARD_ENABLE_VLLM_KV_CACHE_MEM_COPY", "1");
  return enable_mem_copy;
}

std::string VLLMKVCacheEnv::VineyardEnableVLLMKVCacheDirectIO() {
  std::string enable_direct_io =
      read_env("VINEYARD_ENABLE_VLLM_KV_CACHE_DIRECT_IO", "1");
  return enable_direct_io;
}

std::string VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskPath() {
  static std::string disk_path =
      read_env("VINEYARD_VLLM_KV_CACHE_DISK_PATH", "");
  return disk_path;
}

std::string VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskType() {
  static std::string disk_type =
      read_env("VINEYARD_VLLM_KV_CACHE_DISK_TYPE", "");
  return disk_type;
}

std::string VLLMKVCacheEnv::LocalVineyardVLLMKVCache() {
  static std::string local_vineyard_vllm_cache =
      read_env("LOCAL_VINEYARD_VLLM_KVCACHE", "1");
  return local_vineyard_vllm_cache;
}

std::string VLLMKVCacheEnv::GetVineyardVLLMBlockMetaMagicSize() {
  static std::string meta_magic_size =
      read_env("VINEYARD_VLLM_BLOCK_META_MAGIC_SIZE", "4096");
  return meta_magic_size;
}

std::string VLLMKVCacheEnv::GetVineyardVLLMMaxBlockNum() {
  static std::string max_block_num =
      read_env("VINEYARD_VLLM_MAX_BLOCK_SIZE", "8192");
  return max_block_num;
}

std::string VLLMKVCacheEnv::GetAIOPullResultInterval() {
  static std::string aio_pull_result_interval =
      read_env("VINEYARD_AIO_PULL_RESULT_INTERVAL", "10");
  return aio_pull_result_interval;
}

};  // namespace vineyard
