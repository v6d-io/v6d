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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/util/env.h"

#include "common/util/logging.h"
#include "vllm-kv-cache/src/env.h"
#include "vllm-kv-cache/src/vllm_kv_cache_util.h"

namespace vineyard {

extern std::shared_ptr<ThreadPool> KVCacheHelper::construct_helper_pool_;
extern int KVCacheHelper::log_level_;

Status KVCacheHelper::Init(size_t concurrency) {
  if (construct_helper_pool_ == nullptr) {
    construct_helper_pool_ = std::make_shared<ThreadPool>(concurrency);
  }
  std::string log_level_str = VineyardEnv::GetVineyardTraceLogLevel();
  try {
    log_level_ = std::stoi(log_level_str);
  } catch (...) { log_level_ = 0; }
  LOG(INFO) << "Set KVCacheHelper log level to " << log_level_;
  return Status::OK();
}

int KVCacheHelper::GetTraceLogLevel() { return log_level_; }

std::string KVCacheHelper::GetBLockNamePrefix() {
  return VLLMKVCacheEnv::GetVLLMBlockPrefix();
}

std::string KVCacheHelper::BuildBlockName(uint64_t hash) {
  return GetBLockNamePrefix() + std::to_string(hash);
}

uint64_t KVCacheHelper::GetLayer(std::vector<uint64_t>& shape,
                                 int layer_idnex) {
  if (shape.size() == 0 || layer_idnex < 0 ||
      static_cast<size_t>(layer_idnex) >= shape.size()) {
    return 0;
  }
  return shape[layer_idnex];
}

uint64_t KVCacheHelper::GetBlobNumsPerLayer(std::vector<uint64_t>& shape,
                                            int layer_index) {
  if (shape.size() < 2) {
    return 0;
  }
  uint64_t blob_nums = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    blob_nums *= shape[i];
  }
  return blob_nums / shape[layer_index];
}

uint64_t KVCacheHelper::GetContinuousBlobNums(std::vector<uint64_t>& shape,
                                              int layer_index) {
  if (layer_index < 0 || static_cast<size_t>(layer_index) >= shape.size()) {
    return 0;
  }
  uint64_t continuous_block_nums = 1;
  for (size_t i = layer_index + 1; i < shape.size(); ++i) {
    continuous_block_nums *= shape[i];
  }
  return continuous_block_nums;
}

void KVCacheHelper::ShuffleBlockToLayer(
    std::vector<std::vector<ObjectID>>& local_layer_blobs,
    std::vector<std::vector<ObjectID>>& remote_layer_blobs,
    std::vector<std::vector<size_t>>& layer_sizes,
    std::vector<std::vector<ObjectID>>& local_blobs,
    std::vector<std::vector<ObjectID>>& remote_blobs,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index) {
  uint64_t layer_num = GetLayer(shape, layer_index);
  local_layer_blobs.resize(layer_num);
  remote_layer_blobs.resize(layer_num);
  layer_sizes.resize(layer_num);
  uint64_t layer_continuous_blob_num =
      GetContinuousBlobNums(shape, layer_index);
  /*
   * time complexity:
   * O(cycle)
   * = O(block_num * blobs_per_block / (layer_num * layer_continuous_blob_num) *
   * layer_index * layer_continuous_blob_num) = O(num_blobs)
   */
  for (size_t block_index = 0; block_index < local_blobs.size();
       block_index++) {
    for (size_t blob_index_base = 0; blob_index_base < local_blobs[0].size();
         blob_index_base += layer_continuous_blob_num * layer_num) {
      for (uint64_t layer_index = 0; layer_index < layer_num; layer_index++) {
        for (uint64_t blob_index = 0; blob_index < layer_continuous_blob_num;
             blob_index++) {
          local_layer_blobs[layer_index].push_back(
              local_blobs[block_index][blob_index_base +
                                       layer_index * layer_continuous_blob_num +
                                       blob_index]);
          remote_layer_blobs[layer_index].push_back(
              remote_blobs[block_index]
                          [blob_index_base +
                           layer_index * layer_continuous_blob_num +
                           blob_index]);
          layer_sizes[layer_index].push_back(
              sizes_vec[block_index]
                       [blob_index_base +
                        layer_index * layer_continuous_blob_num + blob_index]);
        }
      }
    }
  }
}

std::string KVCacheHelper::MicrosecondToTimestamp(int64_t microsecond) {
  const uint64_t microseconds_per_second = 1000000;
  std::time_t seconds_part = microsecond / microseconds_per_second;
  uint64_t microseconds_part = microsecond % microseconds_per_second;

  struct tm time_info;
  localtime_r(&seconds_part, &time_info);

  char buffer[32];
  size_t written_len =
      strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &time_info);
  std::stringstream ss;
  ss << std::setw(6) << std::setfill('0') << microseconds_part;

  if (written_len > 0) {
    return std::string(buffer, written_len) + "." + ss.str();
  }

  return "";
}

}  // namespace vineyard
