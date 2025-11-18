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

#ifndef MODULES_VLLM_KV_CACHE_SRC_VLLM_KV_CACHE_UTIL_H_
#define MODULES_VLLM_KV_CACHE_SRC_VLLM_KV_CACHE_UTIL_H_

#include <iomanip>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "client/ds/object_meta.h"
#include "common/util/logging.h"
#include "common/util/uuid.h"

#include "thread-pool/thread_pool.h"

namespace vineyard {

class KVCacheHelper {
 public:
  static Status Init(size_t concurrency = std::thread::hardware_concurrency());

  static int GetTraceLogLevel();

  static std::string GetBLockNamePrefix();

  static std::string BuildBlockName(uint64_t hash);

  static uint64_t GetLayer(std::vector<uint64_t>& shape, int layer_idnex);

  static uint64_t GetBlobNumsPerLayer(std::vector<uint64_t>& shape,
                                      int layer_index);

  static uint64_t GetContinuousBlobNums(std::vector<uint64_t>& shape,
                                        int layer_index);

  static void ShuffleBlockToLayer(
      std::vector<std::vector<ObjectID>>& local_layer_offsets,
      std::vector<std::vector<ObjectID>>& remote_layer_blobs,
      std::vector<std::vector<size_t>>& layer_sizes,
      std::vector<std::vector<ObjectID>>& local_offsets,
      std::vector<std::vector<ObjectID>>& remote_blobs,
      std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
      int layer_index);

  static std::shared_ptr<ThreadPool> GetConstructThreadPool() {
    return construct_helper_pool_;
  }

  static std::string MicrosecondToTimestamp(int64_t microsecond);

 private:
  static std::shared_ptr<ThreadPool> construct_helper_pool_;
  static int log_level_;
};

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_VLLM_KV_CACHE_UTIL_H_
