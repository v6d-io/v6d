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
  if (config.tensorByte <= 0 || config.cacheCapacity <= 0 ||
      config.layer <= 0) {
    return Status::Invalid("Invalid tensor byte, cache capacity or layer.");
  }
  if (config.blockSize <= 0 || config.syncInterval <= 0) {
    return Status::Invalid("Invalid block size or sync interval.");
  }
  if (config.llmCacheObjectName.size() == 0 ||
      config.llmRefcntObjectName.size() == 0 ||
      config.llmCacheSyncLock.size() == 0) {
    return Status::Invalid(
        "Invalid object name, refcnt object name or sync lock name.");
  }

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
  if (config.batchSize <= 0 || config.splitNumber <= 0) {
    return Status::Invalid("Invalid batch size or split number.");
  }
  if (config.tensorByte <= 0 || config.cacheCapacity <= 0 ||
      config.layer <= 0) {
    return Status::Invalid("Invalid tensor byte, cache capacity or layer.");
  }

  std::shared_ptr<FileStorage> file_storage;
  if (config.filesystemType == FilesystemType::LOCAL) {
    file_storage = std::make_shared<LocalFileStorage>(
        config.tensorByte, config.cacheCapacity, config.layer, config.batchSize,
        config.splitNumber, config.root);
  } else {
    return Status::Invalid("Unsupported filesystem type");
  }
  manager = std::make_shared<KVStateCacheManager>(file_storage);
  return Status::OK();
}

Status KVStateCacheManager::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  if (kvStateList.size() != tokenList.size()) {
    return Status::Invalid("Token list size not match kv state list size");
  }
  return storage->Update(tokenList, kvStateList);
}

Status KVStateCacheManager::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>& kvStateList) {
  if (kvStateList.size() != tokenList.size()) {
    return Status::Invalid("Token list size not match kv state list size");
  }
  return storage->Update(prefix, tokenList, kvStateList);
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
  return BlobStorage::ClearGlobalCache(client, config.llmCacheSyncLock,
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
