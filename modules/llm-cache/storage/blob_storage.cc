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

#include "llm-cache/storage/blob_storage.h"

namespace vineyard {

BlobStorage::BlobStorage(Client& client, std::shared_ptr<KVCacheBuilder>& cache,
                         int syncInterval, std::string& llmCacheSyncLock,
                         std::string& llmCacheObjectName,
                         std::string& llmRefcntObjectName)
    : client(client) {
  this->syncInterval = syncInterval;
  this->kvCacheBuilder = cache;
  this->llmCacheSyncLock = llmCacheSyncLock;
  this->llmCacheObjectName = llmCacheObjectName;
  this->llmRefcntObjectName = llmRefcntObjectName;
  this->syncThread = std::thread(SyncThreadFunc, this);
}

Status BlobStorage::Make(Client& client, std::shared_ptr<BlobStorage>& storage,
                         int tensorNBytes, int cacheCapacity, int layer,
                         int blockSize, int syncInterval,
                         std::string llmCacheSyncLock,
                         std::string llmCacheObjectName,
                         std::string llmRefcntObjectName) {
  RETURN_ON_ASSERT(client.Connected(), "The client is not connected.");
  // TBD
  // try to get cache object
  std::string actualKey;
  AcquireServerLock(client, llmCacheSyncLock, actualKey);

  // sync global cache object with vineyard
  ObjectID globalKVCacheID;
  std::set<ObjectID> blockIDSetToAdd;
  std::set<ObjectID> blockIDSetToDelete;
  Status status = client.GetName(llmCacheObjectName, globalKVCacheID);
  std::shared_ptr<KVCacheBuilder> kvCacheBuilder;
  if (status.ok()) {
    // if success, pull the cache object
    std::shared_ptr<KVCache> globalKVCache = std::dynamic_pointer_cast<KVCache>(
        client.FetchAndGetObject(globalKVCacheID));
    Status status = KVCacheBuilder::Make(client, kvCacheBuilder, globalKVCache);
    if (!status.ok()) {
      ReleaseServerLock(client, actualKey);
      return Status::Invalid(
          "Failed to make the cache object from global cache object.");
    }
    if (globalKVCache->id() != globalKVCacheID) {
      VLOG(100) << "Del migrate object";
      Status status = client.DelData(globalKVCache->id());
      if (!status.ok()) {
        LOG(ERROR) << "Delete object failed: " << status.ToString()
                   << " It may cause memory leak.";
      }
    }

    kvCacheBuilder->GetCurrentBlockIDSet(blockIDSetToAdd);
    blockIDSetToDelete = kvCacheBuilder->GetBlockIDSetToDelete();
  } else {
    // if failed, create a new cache object
    LOG(INFO) << "failed to get the cache object, create a new one.";
    Status status = KVCacheBuilder::Make(client, kvCacheBuilder, tensorNBytes,
                                         cacheCapacity, layer, blockSize);
    if (!status.ok()) {
      ReleaseServerLock(client, actualKey);
      return Status::Invalid("Failed to make new cache object.");
    }
  }

  // TBD
  // use lease to prevent the deadlock if the client is down
  storage = std::make_shared<BlobStorage>(client, kvCacheBuilder, syncInterval,
                                          llmCacheSyncLock, llmCacheObjectName,
                                          llmRefcntObjectName);
  VINEYARD_CHECK_OK(storage->SetRefcntMap(blockIDSetToDelete, blockIDSetToAdd));
  // release the lock
  ReleaseServerLock(client, actualKey);
  return Status::OK();
}

Status BlobStorage::UpdateInternal(
    const std::vector<int>& tokenList, int nextToken,
    const std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  return kvCacheBuilder->Update(tokenList, nextToken, kvState);
}

Status BlobStorage::QueryInternal(
    const std::vector<int>& tokenList, int token,
    std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  return kvCacheBuilder->Query(tokenList, token, kvState);
}

/**
 * @brief Update the kv state with the given token and its prefix in the kv
 * state cache manager.
 *
 * @param tokenList The token list as the prefix of the updated token.
 * @param nextToken The next token to be updated.
 * @param kvState The kv state of the token. The length of the kv state should
 *                be as same as the layer of the kv state cache manager.
 *
 *           *****************************************************************
 *           * Important, the kv state must be initialized(pre-allocated)    *
 *           * and released by the caller.                                   *
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
 * @return Status
 */
Status BlobStorage::Update(
    const std::vector<int>& tokenList, int nextToken,
    const std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  std::unique_lock<std::mutex> lock(cacheAccessMutex, std::defer_lock);
  if (!lock.try_lock()) {
    // If failed to gain the lock, return OK and wait for next time
    return Status::OK();
  }

  if (isClosed) {
    return Status::Invalid("The memory storage is closed.");
  }

  return UpdateInternal(tokenList, nextToken, kvState);
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
 * @param updated It's a return value to indicate the number of tokens that have
 *                been updated successfully.
 *
 *           *****************************************************************
 *           * Important, the kv state List must be                          *
 *           * initialized(pre-allocated) and released by the caller.        *
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
 * kvCacheList. and the second dimension of the kvCacheList should be as same as
 * the layer of the kv state.
 *
 * @return Status
 */
Status BlobStorage::Update(
    const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& updated) {
  std::unique_lock<std::mutex> lock(cacheAccessMutex, std::defer_lock);
  if (!lock.try_lock()) {
    return Status::OK();
  }
  if (isClosed) {
    return Status::Invalid("The memory storage is closed.");
  }
  std::vector<int> tokenListCopy;
  for (size_t i = 0; i < tokenList.size(); i++) {
    Status result = UpdateInternal(tokenListCopy, tokenList[i], kvCacheList[i]);
    if (!result.ok()) {
      break;
    }
    tokenListCopy.push_back(tokenList[i]);
    updated++;
  }

  return Status::OK();
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
 *
 *           *****************************************************************
 *           * Important, the kv state List must be                          *
 *           * initialized(pre-allocated) and released by the caller.        *
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
 *           * kv buffer of the kvCacheList                                  *
 *           *                                                               *
 *           *****************************************************************
 *
 * @param updated It's a return value to indicate the number of tokens that have
 *                been updated successfully.
 *
 * @return Status
 */
Status BlobStorage::Update(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& updated) {
  std::unique_lock<std::mutex> lock(cacheAccessMutex, std::defer_lock);
  if (!lock.try_lock()) {
    return Status::OK();
  }
  if (isClosed) {
    return Status::Invalid("The memory storage is closed.");
  }
  std::vector<int> tokenListCopy(prefix.begin(), prefix.end());
  for (size_t i = 0; i < tokenList.size(); i++) {
    Status result = UpdateInternal(tokenListCopy, tokenList[i], kvCacheList[i]);
    if (!result.ok()) {
      break;
    }
    tokenListCopy.push_back(tokenList[i]);
    updated++;
  }

  return Status::OK();
}

/**
 * @brief Query the kv state with the given token list and its prefix in the kv
 * state cache manager.
 *
 * @param tokenList The token list as the prefix of the updated token.
 * @param kvCacheList The kv state list of the token list. It must be
 *                    initialized before calling this function, including the
 *                    data and length of the kv tensor.
 *                    The kv state list is a 2D vector, the first dimension is
 *                    the token index, and the second dimension is the layer
 *                    index. The kv state is a pair of LLMKV, the first is
 *                    the K tensor and the second is the V tensor.
 *                    It contains two fields: data and length. The data is
 *                    the pointer to the tensor, and the length is the size
 *                    of the tensor.
 * @param matched It's a return value to indicate the number of tokens that have
 *                been matched successfully.
 *
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
 * @return Status
 */
Status BlobStorage::Query(
    const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& matched) {
  std::unique_lock<std::mutex> lock(cacheAccessMutex, std::defer_lock);
  if (!lock.try_lock()) {
    return Status::Invalid("Query cache failed: can not gain the cache lock.");
  }
  if (isClosed) {
    return Status::Invalid("The memory storage is closed.");
  }

  // support partial match of the token list
  // copy the token list and query the cache one token by one token
  matched = 0;
  std::vector<int> tokenListPrefix;
  for (size_t i = 0; i < tokenList.size() && i < kvCacheList.size(); i++) {
    Status result =
        QueryInternal(tokenListPrefix, tokenList[i], kvCacheList[i]);
    if (!result.ok()) {
      return Status::OK();
    }
    matched += 1;
    tokenListPrefix.push_back(tokenList[i]);
  }

  return Status::OK();
}

/**
 * @brief Query the kv state with the given token and its prefix in the kv state
 * cache manager.
 *
 * @param prefix The token list as the prefix of the updated token.
 * @param token The token to be queried.
 * @param kvState The kv state of the token. It must be initialized(allocated)
 *                before calling this function, including the data and length
 *                of the kv state. The length of the kv state should be as same
 *                as the layer of the kv state cache manager.
 *
 *           *****************************************************************
 *           * Important, the kv state is managed by the kv state cache      *
 *           * manager, the caller does not need to malloc and free the      *
 *           * memory of the kv state. Besides, the data pointer should be   *
 *           * nullptr and the length should be 0.                           *
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
 * @return Status
 */
Status BlobStorage::Query(const std::vector<int>& prefix, int token,
                          std::vector<std::pair<LLMKV, LLMKV>>& kvState) {
  std::unique_lock<std::mutex> lock(cacheAccessMutex, std::defer_lock);
  if (!lock.try_lock()) {
    // If failed to gain the lock, return OK and wait for next time
    return Status::OK();
  }
  if (isClosed) {
    return Status::Invalid("The memory storage is closed.");
  }
  return QueryInternal(prefix, token, kvState);
}

Status BlobStorage::Query(
    const std::vector<int>& prefix, const std::vector<int>& tokenList,
    std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kvCacheList,
    size_t& matched) {
  return Status::NotImplemented();
}

BlobStorage::~BlobStorage() {
  StopSync();
  LOG(INFO) << "BlobStorage exit.";
}

// This function is used for testing
void BlobStorage::Delete(std::vector<int>& token) {
  std::shared_ptr<NodeData> evictedNode;
  kvCacheBuilder->GetRootTree()->Delete(token, evictedNode);
  kvCacheBuilder->Delete(evictedNode);
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(kvCacheBuilder->GetRootTree()->tree);
  }
}

Status BlobStorage::Sync() {
  Status status;
  std::set<ObjectID> blockIDSetToAdd;
  std::set<ObjectID> blockIDSetToDelete;
  std::set<ObjectID> globalBlockIDSet;
  // 1. pull the cache object
  ObjectID globalKVCacheID;
  std::vector<ObjectID> deleteList;

  std::shared_ptr<KVCache> globalKVCache = nullptr;
  status = client.GetName(llmCacheObjectName, globalKVCacheID);
  if (status.ok()) {
    deleteList.push_back(globalKVCacheID);
    globalKVCache = std::dynamic_pointer_cast<KVCache>(
        client.FetchAndGetObject(globalKVCacheID));
    globalKVCache->GetCurrentBlockIDSet(globalBlockIDSet);
  } else {
    // Not an error.
    VLOG(100) << "There is no cache object in the meta server.";
  }

  // 2. merge the cache object
  // only the global cache object with higher version will be merged
  VLOG(100) << "Current builder version:" << kvCacheBuilder->GetVersion()
            << " global version:"
            << (globalKVCache == nullptr
                    ? "null"
                    : std::to_string(globalKVCache->GetVersion()));
  if (globalKVCache != nullptr &&
      kvCacheBuilder->GetVersion() < globalKVCache->GetVersion()) {
    status = kvCacheBuilder->Merge(globalKVCache);
    RETURN_ON_ERROR(status);
    if (globalKVCache->id() != globalKVCacheID) {
      VLOG(100) << "Del migrate object";
      Status status = client.DelData(globalKVCache->id());
      if (!status.ok()) {
        LOG(ERROR) << "Delete object failed: " << status.ToString()
                   << " It may cause memory leak.";
      }
    }
  }
  kvCacheBuilder->UpdateVersion();

  /**
   * 3. get the current block id set, which stores the block id(instead of block
   * ptr) and the block id set to delete.
   */
  std::set<ObjectID> currentObjectIDSet;
  kvCacheBuilder->GetCurrentBlockIDSet(currentObjectIDSet);
  blockIDSetToDelete = kvCacheBuilder->GetBlockIDSetToDelete();

  // 4. push the cache object to the vineyardd
  kvCache = std::dynamic_pointer_cast<KVCache>(kvCacheBuilder->_Seal(client));

  std::set<ObjectID> currentGlobalBlockIDSet;
  kvCacheBuilder->GetCurrentBlockIDSet(currentGlobalBlockIDSet);

  status = client.Persist(kvCache->id());
  RETURN_ON_ERROR(status);

  // 5. put the name of the new cache object to the meta server
  status = client.DropName(llmCacheObjectName);
  RETURN_ON_ERROR(status);
  status = client.PutName(kvCache->id(), llmCacheObjectName);
  RETURN_ON_ERROR(status);

  // 6. delete old cache object
  status = client.DelData(deleteList, false, true);
  if (!status.ok()) {
    LOG(ERROR) << "Delete old cache object failed: " << status.ToString()
               << " It may cause memory leak.";
  }

  // 7. create a global cache object replica
  kvCache->Resolve();
  RETURN_ON_ERROR(KVCacheBuilder::Make(client, kvCacheBuilder, kvCache));

  kvCacheBuilder->GetCurrentBlockIDSet(blockIDSetToAdd);

  /**
   * 8. get the add set, which contains the block id in the new cache object
   * but not in the current cache object.
   * CurrentObjectIDSet must be the subset of blockIDSetToAdd.
   */

  std::set<ObjectID> differenceSet;
  std::set_difference(blockIDSetToAdd.begin(), blockIDSetToAdd.end(),
                      currentObjectIDSet.begin(), currentObjectIDSet.end(),
                      std::inserter(differenceSet, differenceSet.begin()));

  std::set<ObjectID> globalBlockIDToDelete;
  std::set<ObjectID> globalBlockIDToAdd;
  std::set_difference(
      globalBlockIDSet.begin(), globalBlockIDSet.end(),
      currentGlobalBlockIDSet.begin(), currentGlobalBlockIDSet.end(),
      std::inserter(globalBlockIDToDelete, globalBlockIDToDelete.begin()));
  std::set_difference(
      currentGlobalBlockIDSet.begin(), currentGlobalBlockIDSet.end(),
      globalBlockIDSet.begin(), globalBlockIDSet.end(),
      std::inserter(globalBlockIDToAdd, globalBlockIDToAdd.begin()));

  // 9. update the global refcnt map
  RETURN_ON_ERROR(SetRefcntMap(blockIDSetToDelete, differenceSet));
  RETURN_ON_ERROR(SetRefcntMap(globalBlockIDToDelete, globalBlockIDToAdd));

  return Status::OK();

  // TBD
  // use lease to prevent the deadlock if the client is down
}

void BlobStorage::SyncThreadFunc(BlobStorage* storage) {
  uint64_t last_time = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  while (true) {
    std::unique_lock<std::mutex> lock(storage->exitMutex);
    if (storage->cv.wait_for(
            lock, std::chrono::seconds(storage->syncInterval),
            [&storage, &last_time] {
              uint64_t current_time =
                  std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
              return storage->exitFlag ||
                     static_cast<int>(current_time - last_time) >=
                         storage->syncInterval;
            })) {
      if (storage->exitFlag) {
        break;
      }
      std::lock_guard<std::mutex> lock(storage->cacheAccessMutex);
      std::string actualKey;

      AcquireServerLock(storage->client, storage->llmCacheSyncLock, actualKey);
      Status status = storage->Sync();
      if (!status.ok()) {
        while (!storage->AfterSyncFailed().ok()) {
          VLOG(100) << "Recover from sync failed failed. Retry later.";
          sleep(1);
        }
      }

      ReleaseServerLock(storage->client, actualKey);

      last_time = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    }
  }
  LOG(INFO) << "Sync thread exit.";
}

Status BlobStorage::AfterSyncFailed() {
  std::vector<ObjectID> deleteList;
  /**
   * If there is no global cache object, the local cache object will be
   * can be used directly. And Sync will be tried again later.
   * If there exists a global cache object, recover from the global object
   * and delete the cache object if the builder is sealed.
   */
  ObjectID globalKVCacheID;
  std::shared_ptr<KVCache> globalKVCache = nullptr;
  Status status = client.GetName(llmCacheObjectName, globalKVCacheID);
  if (status.ok()) {
    globalKVCache = std::dynamic_pointer_cast<KVCache>(
        client.FetchAndGetObject(globalKVCacheID));
  } else {
    VLOG(100) << "There is no cache object in the meta server.";
    return Status::OK();
  }

  status = KVCacheBuilder::Make(client, kvCacheBuilder, globalKVCache);
  RETURN_ON_ERROR(status);
  if (kvCache != nullptr && kvCache->id() != globalKVCacheID) {
    // It means the builder is sealed but not pushed to the vineyardd
    deleteList.push_back(kvCache->id());
    deleteList.push_back(globalKVCache->id());
  }
  status = client.DelData(deleteList, false, true);
  if (!status.ok()) {
    LOG(ERROR) << "Delete object failed: " << status.ToString()
               << " It may cause memory leak.";
  }
  kvCache = nullptr;

  return Status::OK();
}

void BlobStorage::AcquireServerLock(Client& client, std::string& lockKey,
                                    std::string& actualKey) {
  bool result = false;
  while ((!(client.TryAcquireLock(lockKey, result, actualKey).ok())) ||
         !result) {
    VLOG(100) << "Failed to gain the lock, wait for next time.";
    sleep(1);
  }
}

void BlobStorage::ReleaseServerLock(Client& client, std::string& actualKey) {
  bool result = false;
  while ((!(client.TryReleaseLock(actualKey, result).ok())) || !result) {
    VLOG(100) << "Failed to release the lock, wait for next time.";
    sleep(1);
  }
}

void BlobStorage::StopSync() {
  LOG(INFO) << "Wait for sync thread to exit.";
  std::lock_guard<std::mutex> exitLock(exitMutex);
  if (!exitFlag) {
    exitFlag = true;
    exitMutex.unlock();
    cv.notify_one();
    syncThread.join();
  }
}

Status BlobStorage::ClearGlobalCache(Client& client,
                                     std::string& llmCacheSyncLock,
                                     std::string& llmCacheObjectName,
                                     std::string& llmRefcntObjectName) {
  RETURN_ON_ASSERT(client.Connected(), "The client is not connected.");

  ObjectID globalCacheObjectID;
  ObjectID globalRefcntMapId;
  RETURN_ON_ERROR(client.GetName(llmCacheObjectName, globalCacheObjectID));
  RETURN_ON_ERROR(client.DropName(llmCacheObjectName));
  RETURN_ON_ERROR(client.GetName(llmRefcntObjectName, globalRefcntMapId));
  RETURN_ON_ERROR(client.DropName(llmRefcntObjectName));

  std::shared_ptr<KVCache> globalCacheObject =
      std::dynamic_pointer_cast<KVCache>(
          client.FetchAndGetObject(globalCacheObjectID));
  std::set<ObjectID> blockIDSetToDelete;
  globalCacheObject->GetCurrentBlockIDSet(blockIDSetToDelete);
  std::vector<ObjectID> deleteList(blockIDSetToDelete.begin(),
                                   blockIDSetToDelete.end());
  if (globalCacheObjectID != globalCacheObject->id()) {
    deleteList.push_back(globalCacheObject->id());
  }
  deleteList.push_back(globalCacheObjectID);
  deleteList.push_back(globalRefcntMapId);

  RETURN_ON_ERROR(client.DelData(deleteList));
  return Status::OK();
}

void BlobStorage::CloseCache() {
  // recycle blob
  StopSync();

  LOG(INFO) << "Clear block set and recycle blob.";
  std::lock_guard<std::mutex> cacheLock(cacheAccessMutex);
  this->kvCacheBuilder->Close();
  this->isClosed = true;
  RefreshRefcnt();
}

Status BlobStorage::SetRefcntMap(std::set<ObjectID>& blockIDSetToDelete,
                                 std::set<ObjectID>& blockIDSetToAdd) {
  VLOG(100) << "SetRefcntMap:"
            << " add size:" << blockIDSetToAdd.size()
            << " delete size:" << blockIDSetToDelete.size();
  ObjectID globalRefcntMapObjectID;
  Status status = client.GetName(llmRefcntObjectName, globalRefcntMapObjectID);
  if (status.ok()) {
    std::shared_ptr<RefcntMapObject> globalRefcntMapObject =
        std::dynamic_pointer_cast<RefcntMapObject>(
            client.FetchAndGetObject(globalRefcntMapObjectID));
    std::shared_ptr<RefcntMapObjectBuilder> refcntMapObjectBuilder =
        std::make_shared<RefcntMapObjectBuilder>(client, globalRefcntMapObject);
    if (globalRefcntMapObject->id() != globalRefcntMapObjectID) {
      // if the global object is migrated, delete the old object
      VLOG(100) << "Del migrate object";
      Status status = client.DelData(globalRefcntMapObject->id());
      if (!status.ok()) {
        LOG(ERROR) << "Delete object failed: " << status.ToString()
                   << " It may cause memory leak.";
      }
    }

    refcntMapObjectBuilder->IncSetRefcnt(blockIDSetToAdd);
    refcntMapObjectBuilder->DecSetRefcnt(blockIDSetToDelete);
    if (VLOG_IS_ON(100)) {
      refcntMapObjectBuilder->PrintRefcntMap();
    }

    std::shared_ptr<Object> newRefcntMapObject =
        refcntMapObjectBuilder->_Seal(client);
    RETURN_ON_ERROR(client.Persist(newRefcntMapObject->id()));
    RETURN_ON_ERROR(client.DropName(llmRefcntObjectName));
    RETURN_ON_ERROR(
        client.PutName(newRefcntMapObject->id(), llmRefcntObjectName));
    // Delete old refcnt map object.
    Status status = client.DelData(globalRefcntMapObjectID);
    if (!status.ok()) {
      LOG(ERROR) << "Delete object failed: " << status.ToString()
                 << " It may cause memory leak.";
    }
  } else {
    std::shared_ptr<RefcntMapObjectBuilder> refcntMapObjectBuilder =
        std::make_shared<RefcntMapObjectBuilder>(client);
    refcntMapObjectBuilder->IncSetRefcnt(blockIDSetToAdd);
    refcntMapObjectBuilder->DecSetRefcnt(blockIDSetToDelete);
    if (VLOG_IS_ON(100)) {
      refcntMapObjectBuilder->PrintRefcntMap();
    }

    std::shared_ptr<Object> newRefcntMapObject =
        refcntMapObjectBuilder->_Seal(client);
    RETURN_ON_ERROR(client.Persist(newRefcntMapObject->id()));
    RETURN_ON_ERROR(
        client.PutName(newRefcntMapObject->id(), llmRefcntObjectName));
  }
  return Status::OK();
}

void BlobStorage::RefreshRefcnt() {
  std::set<ObjectID> blockIDSetToDelete =
      this->kvCacheBuilder->GetBlockIDSetToDelete();
  std::set<ObjectID> blockIDSetToAdd;
  std::string actualKey;
  AcquireServerLock(client, llmCacheSyncLock, actualKey);
  Status status = SetRefcntMap(blockIDSetToDelete, blockIDSetToAdd);
  if (!status.ok()) {
    LOG(ERROR) << "Update refcnt failed: " << status.ToString()
               << " It may cause memory leak.";
  }
  ReleaseServerLock(client, actualKey);
}

}  // namespace vineyard
