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

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "client/client.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "llm-cache/ds/kv_state_cache.h"
#include "llm-cache/ds/kv_state_cache_manager.h"
#include "llm-cache/storage/blob_storage.h"
#include "llm-cache/storage/local_file_storage.h"

namespace vineyard {

KVStateCacheManager::KVStateCacheManager(
    std::shared_ptr<IStorage> storageImpl) {
  storage = storageImpl;
}

// use the memory storage for manager
Status KVStateCacheManager::Make(Client& client,
                                 std::shared_ptr<KVStateCacheManager>& manager,
                                 VineyardCacheConfig& config) {
  std::shared_ptr<BlobStorage> blob_storage;
  VINEYARD_CHECK_OK(blob_storage->Make(
      client, blob_storage, config.tensorByte, config.cacheCapacity,
      config.layer, config.blockSize, config.syncInterval,
      config.llmCacheSyncLock, config.llmCacheObjectName,
      config.llmRefcntObjectName));
  manager = std::make_shared<KVStateCacheManager>(blob_storage);
  return Status::OK();
}

// use the file storage for manager
Status KVStateCacheManager::Make(std::shared_ptr<KVStateCacheManager>& manager,
                                 FileCacheConfig& config) {
  std::shared_ptr<FileStorage> file_storage;
  if (config.filesystemType == FilesystemType::LOCAL) {
    file_storage =
        std::make_shared<LocalFileStorage>(config.batchSize, config.root);
  } else {
    return Status::Invalid("Unsupported filesystem type");
  }
  manager = std::make_shared<KVStateCacheManager>(file_storage);
  return Status::OK();
}

Status KVStateCacheManager::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  return storage->Update(tokenList, kvStateList);
}

Status KVStateCacheManager::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  return storage->Update(tokenList, nextToken, kvState);
}

Status KVStateCacheManager::Query(
    const std::vector<int>& tokenList, int nextToken,
    std::map<int, std::pair<LLMKV, LLMKV>>& kvState) {
  return storage->Query(tokenList, nextToken, kvState);
}

Status KVStateCacheManager::Query(
    const std::vector<int>& tokenList,
    std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  return storage->Query(tokenList, kvStateList);
}

Status KVStateCacheManager::ClearGlobalCache(Client& client,
                                             VineyardCacheConfig& config) {
  std::shared_ptr<BlobStorage> blob_storage =
      std::dynamic_pointer_cast<BlobStorage>(storage);
  return blob_storage->ClearGlobalCache(client, config.llmCacheSyncLock,
                                        config.llmCacheObjectName,
                                        config.llmRefcntObjectName);
}

Status ClearGlobalCache(Client& client, FileCacheConfig& config) {
  // TBD
  return Status::OK();
}

void KVStateCacheManager::Close() { storage->CloseCache(); }

KVStateCacheManager::~KVStateCacheManager() {}

}  // namespace vineyard
