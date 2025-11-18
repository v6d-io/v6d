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

#ifndef MODULES_VLLM_KV_CACHE_SRC_ENV_H_
#define MODULES_VLLM_KV_CACHE_SRC_ENV_H_

#include <string>

#include "common/util/env.h"

namespace vineyard {

class VLLMKVCacheEnv : public VineyardEnv {
 public:
  static std::string GetKVStorageConcurrency();

  static std::string GetVLLMBlockPrefix();

  static std::string GetVineyardVLLMKVCacheIOTimeoutMilliseconds();

  static std::string GetDirectIOAlign();

  static std::string AIORetryWaitMicroseconds();

  static std::string AIOGCWaitTimeMicroseconds();

  static std::string GetVineyardAIOSubmitConcurrency();

  static std::string VineyardEnableVLLMKVCacheMemCopy();

  static std::string VineyardEnableVLLMKVCacheDirectIO();

  static std::string GetVineyardVLLMKVCacheDiskPath();

  static std::string GetVineyardVLLMKVCacheDiskType();

  static std::string LocalVineyardVLLMKVCache();

  static std::string GetVineyardVLLMBlockMetaMagicSize();

  static std::string GetVineyardVLLMMaxBlockNum();

  static std::string GetAIOPullResultInterval();
};

};  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_ENV_H_
