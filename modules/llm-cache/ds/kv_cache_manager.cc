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
#include "llm-cache/ds/kv_cache.h"
#include "llm-cache/ds/kv_cache_manager.h"
#include "llm-cache/storage/blob_storage.h"
#include "llm-cache/storage/local_file_storage.h"
#include "llm-cache/storage/vineyard_file_storage.h"

namespace vineyard {

KVCacheManager::KVCacheManager(std::shared_ptr<IStorage> storageImpl) {
  storage = storageImpl;
}

// use the memory storage for manager
Status KVCacheManager::Make(Client& client,
                            std::shared_ptr<KVCacheManager>& manager,
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
  manager = std::make_shared<KVCacheManager>(blob_storage);
  manager->config = std::make_shared<VineyardCacheConfig>(config);
  return Status::OK();
}

// use the file storage for manager
Status KVCacheManager::Make(std::shared_ptr<KVCacheManager>& manager,
                            FileCacheConfig& config) {
  if (config.chunkSize <= 0 || config.hashChunkSize <= 0) {
    return Status::Invalid("Invalid batch size or split number.");
  }
  if (config.tensorByte <= 0 || config.cacheCapacity <= 0 ||
      config.layer <= 0) {
    return Status::Invalid("Invalid tensor byte, cache capacity or layer.");
  }

  std::shared_ptr<FileStorage> file_storage;
  if (config.filesystemType == FilesystemType::LOCAL) {
    file_storage = std::make_shared<LocalFileStorage>(
        config.tensorByte, config.cacheCapacity, config.layer, config.chunkSize,
        config.hashChunkSize, config.root, config.gcInterval, config.ttl,
        config.enbaleGlobalGC, config.globalGCInterval, config.globalTTL);
  } else {
    return Status::Invalid("Unsupported filesystem type");
  }
  manager = std::make_shared<KVCacheManager>(file_storage);
  RETURN_ON_ERROR(file_storage->Init());
  manager->config = std::make_shared<FileCacheConfig>(config);
  return Status::OK();
}

Status KVCacheManager::Make(RPCClient& client,
                            std::shared_ptr<KVCacheManager>& manager,
                            FileCacheConfig& config) {
  if (config.chunkSize <= 0 || config.hashChunkSize <= 0) {
    return Status::Invalid("Invalid batch size or split number.");
  }
  if (config.tensorByte <= 0 || config.cacheCapacity <= 0 ||
      config.layer <= 0) {
    return Status::Invalid("Invalid tensor byte, cache capacity or layer.");
  }

  std::shared_ptr<FileStorage> file_storage;
  if (config.filesystemType == FilesystemType::VINEYARD) {
    file_storage = std::make_shared<VineyardFileStorage>(
        client, config.tensorByte, config.cacheCapacity, config.layer,
        config.chunkSize, config.hashChunkSize, config.root, config.gcInterval,
        config.ttl, config.enbaleGlobalGC, config.globalGCInterval,
        config.globalTTL);
  } else {
    return Status::Invalid("Unsupported filesystem type");
  }
  manager = std::make_shared<KVCacheManager>(file_storage);
  RETURN_ON_ERROR(file_storage->Init());
  manager->config = std::make_shared<FileCacheConfig>(config);
  return Status::OK();
}

/**
 * @brief Update the kv state with the given token list in the kv state cache
 * manager.
 *
 * @param tokenList The token list to be updated.
 * @param kvCacheList The kv state list of the token list.
 *                    It's a 2D vector, the first dimension is the token index,
 *                    and the second dimension is the layer index.
 *                    The kv state is a pair of LLMKV, the first is the K tensor
 *                    and the second is the V tensor. It contains two fields:
 *                    data and length. The data is the pointer to the tensor
 *                    , and the length is the size of the tensor.
 * @param updated The number of tokens that have been updated successfully. It's
 *                a return value.
 *
 *           *****************************************************************
 *           * Whether the underlying storage is blob or file, the kv state  *
 *           * list must be initialized(pre-allocated) and released by the   *
 *           * caller.                                                       *
 *           *                                                               *
 *           * Assume the layer is 2, and the token list is [1,2] you should *
 *           * allocate the memory for the kv state like this:               *
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvCacheList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorNBytes);                     *
 *           *     value_state.data = malloc(tensorNBytes)                    *
 *           *     // Copy the k_state of LLM KV Cache to key_state.data     *
 *           *     // Copy the v_state of LLM KV Cache to value_state.data   *
 *           *     key_state.length = tensorNBytes;                           *
 *           *     value_state.length = tensorNBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvCacheList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, you must release(free) the       *
 *           * kv buffer of the kvCacheList manually                         *
 *           *                                                               *
 *           *****************************************************************
 *
 *
 * @note The length of the token list should be as same as the length of the
 * kvCacheList.
 *
 *
 * @return Status
 */
Status KVCacheManager::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& updated) {
  if (kvCacheList.size() != tokenList.size()) {
    return Status::Invalid("Token list size not match kv state list size");
  }
  return storage->Update(tokenList, kvCacheList, updated);
}

/**
 * @brief Update the kv state with the given token list and its prefix in the kv
 * state cache manager.
 *
 * @param prefix The prefix of the token list.
 * @param tokenList The token list to be updated.
 * @param kvCacheList The kv state list of the token list.
 *                    It's a 2D vector, the first dimension is the token index,
 *                    and the second dimension is the layer index.
 *                    The kv state is a pair of LLMKV, the first is the K tensor
 *                    and the second is the V tensor. It contains two fields:
 *                    data and length. The data is the pointer to the tensor
 *                    , and the length is the size of the tensor.
 * @param updated It's a return value to indicate the number of tokens that have
 *                been updated successfully.
 *
 *           *****************************************************************
 *           * Whether the underlying storage is blob or file, the kv state  *
 *           * list must be initialized(pre-allocated) and released by the   *
 *           * caller.                                                       *
 *           *                                                               *
 *           * Assume the layer is 2, and the token list is [1,2] you should *
 *           * allocate the memory for the kv state like this:               *
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvCacheList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorNBytes);                     *
 *           *     value_state.data = malloc(tensorNBytes)                    *
 *           *     // Copy the k_state of LLM KV Cache to key_state.data     *
 *           *     // Copy the v_state of LLM KV Cache to value_state.data   *
 *           *     key_state.length = tensorNBytes;                           *
 *           *     value_state.length = tensorNBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvCacheList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, you must release(free) the       *
 *           * kv buffer of the kvCacheList manually                         *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The length of the token list should be as same as the length of the
 * kvCacheList.
 *
 * @return Status
 */
Status KVCacheManager::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& updated) {
  if (kvCacheList.size() != tokenList.size()) {
    return Status::Invalid("Token list size not match kv state list size");
  }
  return storage->Update(prefix, tokenList, kvCacheList, updated);
}

/**
 * @brief Update the kv state with the given token in the kv state cache
 * manager.
 *
 * @param tokenList The token list as the prefix of the next token.
 * @param nextToken The next token to be updated.
 * @param kvState The kv state of the next token.
 *
 *           *****************************************************************
 *           * Now only support for blob storage, the kv state must be       *
 *           * initialized(pre-allocated) and released by the caller.        *
 *           *                                                               *
 *           * Assume the layer is 2, you should allocate the memory for the *
 *           * kv state like this:                                           *
 *           * std::vector<std::pair<LLMKV, LLMKV>> kvState;                 *
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   LLMKV key_state;                                            *
 *           *   LLMKV value_state;                                          *
 *           *   key_state.data = malloc(tensorNBytes);                       *
 *           *   value_state.data = malloc(tensorNBytes)                      *
 *           *   // Copy the k_state of LLM KV Cache to key_state.data       *
 *           *   // Copy the v_state of LLM KV Cache to value_state.data     *
 *           *   key_state.length = tensorNBytes;                             *
 *           *   value_state.length = tensorNBytes;                           *
 *           *   kvState.emplace_back(key_state, value_state);               *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, you must release(free) the       *
 *           * key_state buffer manually.                                    *
 *           *                                                               *
 *           *****************************************************************
 *
 * @return Status to indicate whether the kv state has been updated
 * successfully.
 */
Status KVCacheManager::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  return storage->Update(tokenList, nextToken, kvState);
}

/**
 * @brief Query the kv state with the given token list in the kv state cache
 * manager.
 *
 * @param tokenList The token list to be queried.
 * @param kvCacheList The kv state list of the token list.
 *                    It must be initialized before calling this function,
 *                    including the data and length of the kv tensor.
 *                    It's a 2D vector, the first dimension is the token index,
 *                    and the second dimension is the layer index.
 *                    The kv state is a pair of LLMKV, the first is the K tensor
 *                    and the second is the V tensor. It contains two fields:
 *                    data and length. The data is the pointer to the tensor
 *                    , and the length is the size of the tensor.
 * @param matched It's a return value to indicate the number of tokens that have
 *                been matched successfully.
 *
 *           *****************************************************************
 *           *                        Blob storage                           *
 *           *****************************************************************
 *           * Important, the kv state is managed by the kv state cache      *
 *           * manager, the caller does not need to malloc and free the      *
 *           * memory of the kv state. Besides, the data pointer should be   *
 *           * nullptr and the length should be 0.                           *
 *           *                                                               *
 *           * Assume the layer is 2, and the token list is [1,2] you should *
 *           * allocate the memory for the kv state like this:               *
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvCacheList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = nullptr                                  *
 *           *     value_state.data = nullptr                                *
 *           *     key_state.length = 0;                                     *
 *           *     value_state.length = 0;                                   *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvCacheList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, the key_state's data is pointing *
 *           * to the K tensor data stored in vineyard blob, and the         *
 *           * value_state's data is pointing to the V tensor data stored in *
 *           * vineyard blob. All the length of the kv state is the size of  *
 *           * the tensor data. Then you can copy the kv state to the LLM KV *
 *           * Cache. The memory of the kv state will be freed when calling  *
 *           * the close function of the kv state cache manager.             *
 *           *                                                               *
 *           *****************************************************************
 *
 *           *****************************************************************
 *           *                        File storage                           *
 *           *****************************************************************
 *           * Important, the kv state is managed by the caller itself, the  *
 *           * caller need to malloc and free the memory of the kv state.    *
 *           *                                                               *
 *           * Assume the layer is 2, and the token list is [1,2] you should *
 *           * allocate the memory for the kv state like this:               *
 *           * std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kvCacheList;*
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   std::vector<std::pair<LLMKV, LLMKV>> kvState;               *
 *           *   for (int j = 0; j < 2; j++) {                               *
 *           *     LLMKV key_state;                                          *
 *           *     LLMKV value_state;                                        *
 *           *     key_state.data = malloc(tensorNBytes);                     *
 *           *     value_state.data = malloc(tensorNBytes)                    *
 *           *     key_state.length = tensorNBytes;                           *
 *           *     value_state.length = tensorNBytes;                         *
 *           *     kvState.emplace_back(key_state, value_state);             *
 *           *   }                                                           *
 *           *   kvCacheList.push_back(kvState);                             *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, the key_state and value_state    *
 *           * will be fulfilled with the kv state of the token, then you    *
 *           * can copy the kv state to the LLM KV Cache. At last, you       *
 *           * must free the memory of the kv state manually.                *
 *           *                                                               *
 *           *****************************************************************
 *
 * @note The length of the token list should be as same as the length of the
 * kvCacheList. and the second dimension of the kvCacheList should be as same as
 * the layer of the kv state.
 *
 * @return Status
 */
Status KVCacheManager::Query(
    const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& matched) {
  return storage->Query(tokenList, kvCacheList, matched);
}

/**
 * @brief Query the kv state with the given token and its prefix in the kv state
 * cache manager.
 *
 * @param tokenList The token list as the prefix of the next token.
 * @param nextToken The next token to be queried.
 * @param kvState The kv state of the next token. It must be initialized before
 *                calling this function, including the data and length of the kv
 *                tensor. Also, the length of the kvState should be as same as
 *                the layer of the kv state.
 *
 *           *****************************************************************
 *           * Only support for blob storage, the kv state is managed by the *
 *           * kv state cache manager, the caller does not need to malloc    *
 *           * and free the memory of the kv state. Besides, the data        *
 *           * pointer should be nullptr and the length should be 0.         *
 *           *                                                               *
 *           * Assume the layer is 2, you should allocate the memory for the *
 *           * kv state like this:                                           *
 *           * std::vector<std::pair<LLMKV, LLMKV>> kvState;                 *
 *           * for (int i = 0; i < 2; i++) {                                 *
 *           *   LLMKV key_state;                                            *
 *           *   LLMKV value_state;                                          *
 *           *   key_state.data = nullptr                                    *
 *           *   value_state.data = nullptr                                  *
 *           *   key_state.length = 0;                                       *
 *           *   value_state.length = 0;                                     *
 *           *   kvState.emplace_back(key_state, value_state);               *
 *           *}                                                              *
 *           *                                                               *
 *           * After calling this function, the key_state's data is pointing *
 *           * to the K tensor data stored in vineyard blob, and the         *
 *           * value_state's data is pointing to the V tensor data stored in *
 *           * vineyard blob. All the length of the kv state is the size of  *
 *           * the tensor data. Then you can copy the kv state to the LLM KV *
 *           * Cache. The memory of the kv state will be freed when calling  *
 *           * the close function of the kv state cache manager.             *
 *           *                                                               *
 *           *****************************************************************
 *
 * @return Status to indicate whether the kv state has been queried
 * successfully.
 */
Status KVCacheManager::Query(const std::vector<int>& prefix, int nextToken,
                             std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  return storage->Query(prefix, nextToken, kvState);
}

Status KVCacheManager::Query(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& matched) {
  return storage->Query(prefix, tokenList, kvCacheList, matched);
}

Status KVCacheManager::ClearGlobalCache(Client& client,
                                        VineyardCacheConfig& config) {
  return BlobStorage::ClearGlobalCache(client, config.llmCacheSyncLock,
                                       config.llmCacheObjectName,
                                       config.llmRefcntObjectName);
}

void KVCacheManager::Close() { storage->CloseCache(); }

void KVCacheManager::StopGlobalGCThread() { storage->StopGlobalGCThread(); }

void KVCacheManager::StartGlobalGCThread() { storage->StartGlobalGCThread(); }

KVCacheManager::~KVCacheManager() {}

}  // namespace vineyard
