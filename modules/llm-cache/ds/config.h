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

#ifndef MODULES_LLM_CACHE_DS_CONFIG_H_
#define MODULES_LLM_CACHE_DS_CONFIG_H_

#include <string>

#include "llm-cache/storage/file_storage.h"

namespace vineyard {

struct KVCacheConfig {
  int tensorByte;
  int cacheCapacity;
  int layer;
};

struct VineyardCacheConfig : public KVCacheConfig {
  int blockSize;
  int syncInterval;
  std::string llmCacheSyncLock;
  std::string llmCacheObjectName;
  std::string llmRefcntObjectName;

  VineyardCacheConfig(int tensorByte = 10, int cacheCapacity = 10,
                      int layer = 1, int blockSize = 5, int syncInterval = 3,
                      std::string llmCacheSyncLock = "llmCacheSyncLock",
                      std::string llmCacheObjectName = "llm_cache_object",
                      std::string llmRefcntObjectName = "llm_refcnt_object")
      : KVCacheConfig{tensorByte, cacheCapacity, layer},
        blockSize(blockSize),
        syncInterval(syncInterval),
        llmCacheSyncLock(llmCacheSyncLock),
        llmCacheObjectName(llmCacheObjectName),
        llmRefcntObjectName(llmRefcntObjectName) {}
};

struct FileCacheConfig : public KVCacheConfig {
  int batchSize;
  int splitNumber;
  std::string root;
  FilesystemType filesystemType;
  int clientGCInterval;  // second
  int ttl;               // second
  bool enbaleGlobalGC;
  int globalGCInterval;  // second
  int globalTTL;         // second

  // Default gc interval is 30 minutes and default global gc interval is 3
  // hours.
  FileCacheConfig(int tensorByte = 10, int cacheCapacity = 10, int layer = 1,
                  int batchSize = 4, int splitNumber = 2,
                  std::string root = "/tmp/llm_cache/",
                  FilesystemType filesystemType = LOCAL,
                  int clientGCInterval = 30 * 60, int ttl = 30 * 60,
                  bool enbaleGlobalGC = false,
                  int globalGCInterval = 3 * 60 * 60,
                  int globalTTL = 3 * 60 * 60)
      : KVCacheConfig{tensorByte, cacheCapacity, layer} {
    this->root = root;
    this->batchSize = batchSize;
    this->splitNumber = splitNumber;
    this->filesystemType = filesystemType;
    this->clientGCInterval = clientGCInterval;
    this->ttl = ttl;
    this->enbaleGlobalGC = enbaleGlobalGC;
    this->globalGCInterval = globalGCInterval;
    this->globalTTL = globalTTL;
  }
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_CONFIG_H_
