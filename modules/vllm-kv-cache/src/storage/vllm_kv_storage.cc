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

#include <sys/mman.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/memory/memcpy.h"
#include "common/util/get_tid.h"
#include "common/util/logging.h"
#include "common/util/monitor.h"
#include "vllm-kv-cache/ds/vllm_block.h"
#include "vllm-kv-cache/src/env.h"
#include "vllm-kv-cache/src/io/aio_adaptor.h"
#include "vllm-kv-cache/src/io/mock_io_adapter.h"
#include "vllm-kv-cache/src/storage/vllm_kv_storage.h"
#include "vllm-kv-cache/src/vllm_kv_cache_util.h"

namespace vineyard {

extern std::atomic<uint64_t> VLLMKVStorage::req_count_;
extern uint64_t VLLMKVStorage::storage_base_pointer_;
extern uint64_t VLLMKVStorage::threads_;

extern std::vector<std::shared_ptr<Client>> VLLMKVStorage::vineyard_clients_;
extern std::vector<std::shared_ptr<ThreadPool>> VLLMKVStorage::req_thread_vec_;
extern std::shared_ptr<ThreadPool> VLLMKVStorage::io_thread_pool_;
extern std::shared_ptr<ThreadPool> VLLMKVStorage::copy_thread_pool_;
extern std::shared_ptr<ThreadPool> VLLMKVStorage::fast_opt_thread_pool_;
extern std::shared_ptr<ThreadPool> VLLMKVStorage::block_opt_thread_pool_;
extern bool VLLMKVStorage::use_copy_;
extern bool VLLMKVStorage::direct_io_;
extern monitor::Monitor VLLMKVStorage::load_from_disk_io_monitor_;
extern monitor::Monitor VLLMKVStorage::load_memory_copy_monitor_;
extern monitor::Monitor VLLMKVStorage::load_from_disk_monitor_;
extern monitor::Monitor VLLMKVStorage::save_to_disk_io_monitor_;
extern monitor::Monitor VLLMKVStorage::save_to_disk_monitor_;
extern monitor::Monitor VLLMKVStorage::save_memory_copy_monitor_;
extern vllm_kv_cache::io::IOAdaptorFactory VLLMKVStorage::io_adaptor_factory_;

Status VLLMKVStorage::InitStorage(uint64_t base_pointer, std::string ipc_socket,
                                  std::string io_type, bool enable_mem_copy,
                                  bool direct_io) {
  storage_base_pointer_ = base_pointer;
  threads_ = std::stoull(VLLMKVCacheEnv::GetKVStorageConcurrency()) / 2;

  RETURN_ON_ERROR(KVCacheHelper::Init(threads_));
  size_t vllm_max_block_num =
      std::stoull(VLLMKVCacheEnv::GetVineyardVLLMMaxBlockNum());
  size_t vllm_block_meta_magic_size =
      std::stoull(VLLMKVCacheEnv::GetVineyardVLLMBlockMetaMagicSize());
  for (size_t i = 0; i < threads_; i++) {
    req_thread_vec_.push_back(std::make_shared<ThreadPool>(1));
    vineyard_clients_.emplace_back(std::make_shared<Client>());
    RETURN_ON_ERROR(vineyard_clients_.back()->Connect(ipc_socket));
    RETURN_ON_ERROR(vineyard_clients_.back()->RequireExtraRequestMemory(
        vllm_block_meta_magic_size * vllm_max_block_num));
  }
  LOG(INFO) << "VLLMKVStorage::InitStorage: vineyard_clients_ size = "
            << vineyard_clients_.size();
  io_thread_pool_ = std::make_shared<ThreadPool>(threads_);
  copy_thread_pool_ = std::make_shared<ThreadPool>(threads_);
  fast_opt_thread_pool_ = std::make_shared<ThreadPool>(threads_);
  block_opt_thread_pool_ = std::make_shared<ThreadPool>(threads_);
  LOG(INFO) << "VLLMKVStorage::InitStorage: req_thread_pool_ threads = "
            << threads_ << ", io_thread_pool_ threads = " << threads_
            << ", copy_thread_pool_ threads = " << threads_
            << ", fast_opt_thread_pool_ threads = " << threads_
            << ", block_opt_thread_pool_ threads = " << threads_;

  InitMonitor();
  use_copy_ = enable_mem_copy;
  direct_io_ = direct_io;
  LOG(INFO) << "VLLMKVStorage::InitStorage: use_copy_ = " << use_copy_;
  LOG(INFO) << "VLLMKVStorage::InitStorage: direct_io_ = " << direct_io_;

  // Initialize io_adaptor_factory_ to use AIOAdaptor by default
  if (io_type == "aio") {
    io_adaptor_factory_ = vineyard::vllm_kv_cache::io::GetAIOAdaptorFactory();
  } else if (io_type == "mock") {
    io_adaptor_factory_ =
        vineyard::vllm_kv_cache::io::GetMockIOAdaptorFactory();
  } else {
    return Status::Invalid("Invalid io_type: " + io_type);
  }
  return Status::OK();
}

Status VLLMKVStorage::GetBlockLocation(
    std::vector<uint64_t>& block_hash,
    std::vector<std::set<std::string>>& locations, std::string req_flag) {
  if (block_hash.size() == 0) {
    return Status::OK();
  }
  size_t index = req_count_.fetch_add(1) % threads_;
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << " arrived at GetBlockLocation, "
      << "assigned to thread: " << index;
  return req_thread_vec_[index]
      ->enqueue([&]() {
        uint64_t start =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        Status status = GetBlockLocation(*vineyard_clients_[index], block_hash,
                                         locations, req_flag);
        VINEYARD_ASSERT(vineyard_clients_[index]->Connected(),
                        "vineyard client is not connected");
        uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        VLOG(KVCacheHelper::GetTraceLogLevel())
            << "Request : " << req_flag
            << ". GetBlockLocation cost:" << (end - start) << " us.";
        return status;
      })
      .get();
}

Status VLLMKVStorage::GetBlockLocation(
    Client& client, std::vector<uint64_t>& block_hash,
    std::vector<std::set<std::string>>& locations, std::string& req_flag) {
  VLOG(2) << "GetBlockLocation for request: " << req_flag;
  std::vector<std::string> block_names;
  for (auto hash : block_hash) {
    block_names.push_back(KVCacheHelper::BuildBlockName(hash));
  }
  return client.GetObjectLocation(block_names, locations, req_flag);
}

Status VLLMKVStorage::GetBlockKVCacheLayerwise(
    std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::string rpc_endpoint,
    std::shared_ptr<VLLMLayers>& layers, std::string req_flag) {
  RETURN_ON_ASSERT(
      block_hash_vec.size() == offsets_vec.size(),
      "block_hash and offsets size not match for request: " + req_flag);
  RETURN_ON_ASSERT(
      block_hash_vec.size() == sizes_vec.size(),
      "block_hash and sizes size not match for request: " + req_flag);
  RETURN_ON_ASSERT(layer_index >= 0,
                   "layer_index must be >= 0 for request: " + req_flag);
  RETURN_ON_ASSERT(layer_index < static_cast<int>(shape.size()),
                   "layer_index must be < shape.size()");
  if (block_hash_vec.size() == 0) {
    return Status::OK();
  }

  size_t index = req_count_.fetch_add(1) % threads_;
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << " arrived at GetBlockKVCacheLayerwise, "
      << "assigned to thread: " << index;
  return req_thread_vec_[index]
      ->enqueue([&]() {
        uint64_t start =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        Status status = GetBlockKVCacheLayerwise(
            *vineyard_clients_[index], block_hash_vec, offsets_vec, sizes_vec,
            shape, layer_index, rpc_endpoint, layers, req_flag);
        VINEYARD_ASSERT(vineyard_clients_[index]->Connected(),
                        "vineyard client is not connected");
        uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        VLOG(KVCacheHelper::GetTraceLogLevel())
            << "Request : " << req_flag
            << ". GetBlockKVCacheLayerwise cost:" << (end - start) << " us.";
        return status;
      })
      .get();
}

Status VLLMKVStorage::GetBlockKVCacheLayerwise(
    Client& client, std::vector<uint64_t>& block_hash,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::string rpc_endpoint,
    std::shared_ptr<VLLMLayers>& layers, std::string req_flag) {
  // from name get meta
  // 1. send request to remote to get meta
  // 2. shuffle buffer to layer wise
  // 3. get remote buffer layer by layer
  VLOG(2) << "GetBlockKVCacheLayerwise: "
          << "block_hash size: " << block_hash.size()
          << ", offsets_vec size: " << offsets_vec.size()
          << ", sizes_vec size: " << sizes_vec.size()
          << ", shape size: " << shape.size()
          << ", layer_index: " << layer_index << ", request id: " << req_flag;
  uint64_t start = 0, end = 0;
  RETURN_ON_ASSERT(
      block_hash.size() == offsets_vec.size(),
      "block_hash and offsets size not match for request: " + req_flag);
  RETURN_ON_ASSERT(
      block_hash.size() == sizes_vec.size(),
      "block_hash and sizes size not match for request: " + req_flag);
  RETURN_ON_ASSERT(layer_index >= 0,
                   "layer_index must be >= 0 for request: " + req_flag);
  RETURN_ON_ASSERT(
      layer_index < static_cast<int>(shape.size()),
      "layer_index must be < shape.size() for request: " + req_flag);
  std::vector<std::string> block_names;
  for (auto hash : block_hash) {
    block_names.push_back(KVCacheHelper::BuildBlockName(hash));
  }
  std::vector<ObjectMeta> meta_vec;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  RETURN_ON_ERROR(client.VineyardGetMetasByNames(block_names, rpc_endpoint,
                                                 meta_vec, req_flag));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". Get remote meta cost: " << (end - start)
      << " us";
  RETURN_ON_ASSERT(
      meta_vec.size() == block_hash.size(),
      "meta_vec and block_hash size not match for request: " + req_flag);
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  Status status = VLLMLayers::FromBlocks(
      client, block_hash, offsets_vec, sizes_vec, shape, layer_index, meta_vec,
      rpc_endpoint, layers, req_flag);
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Construct layers from blocks cost:" << (end - start) << " us";
  return status;
}

Status VLLMKVStorage::PutBlockKVCache(
    std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& statuses, std::string req_flag) {
  RETURN_ON_ASSERT(
      block_hash_vec.size() == offsets_vec.size(),
      "block_hash and offsets size not match for request: " + req_flag);
  RETURN_ON_ASSERT(
      block_hash_vec.size() == sizes_vec.size(),
      "block_hash and sizes size not match for request: " + req_flag);
  RETURN_ON_ASSERT(layer_index >= 0,
                   "layer_index must be >= 0 for request: " + req_flag);
  RETURN_ON_ASSERT(
      layer_index < static_cast<int>(shape.size()),
      "layer_index must be < shape.size() for request: " + req_flag);
  if (block_hash_vec.size() == 0) {
    return Status::OK();
  }

  size_t index = req_count_.fetch_add(1) % threads_;
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << " arrived at PutBlockKVCache, "
      << "assigned to thread: " << index;
  return req_thread_vec_[index]
      ->enqueue([&]() {
        uint64_t start =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        Status status = PutBlockKVCache(*vineyard_clients_[index],
                                        block_hash_vec, offsets_vec, sizes_vec,
                                        shape, layer_index, statuses, req_flag);
        VINEYARD_ASSERT(vineyard_clients_[index]->Connected(),
                        "vineyard client is not connected");
        uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        VLOG(KVCacheHelper::GetTraceLogLevel())
            << "Request : " << req_flag
            << ". PutBlockKVCache cost:" << (end - start) << " us.";
        return status;
      })
      .get();
}

Status VLLMKVStorage::PutBlockKVCache(
    Client& client, std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& statuses, std::string& req_flag) {
  std::vector<std::shared_ptr<VLLMBlock>> blocks;
  return PutBlockKVCache(client, block_hash_vec, offsets_vec, sizes_vec, shape,
                         layer_index, blocks, statuses, req_flag);
}

Status VLLMKVStorage::PutBlockKVCache(
    Client& client, std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec,
    std::vector<uint64_t>& shape_vec, int layer_index,
    std::vector<std::shared_ptr<VLLMBlock>>& blocks,
    std::vector<Status>& statuses, std::string& req_flag) {
  VLOG(2) << "PutBlockKVCache: "
          << "block_hash_vec size: " << block_hash_vec.size()
          << ", offsets_vec size: " << offsets_vec.size()
          << ", sizes_vec size: " << sizes_vec.size()
          << ", shape_vec size: " << shape_vec.size()
          << ", layer_index: " << layer_index << ", request id: " << req_flag;
  uint64_t start = 0, end = 0;

  std::vector<std::shared_ptr<VLLMBlockBuilder>> block_builders_to_delete;
  std::vector<std::shared_ptr<VLLMBlock>> blocks_to_delete;
  statuses.resize(block_hash_vec.size(), Status::OK());
  std::vector<std::shared_ptr<VLLMBlockBuilder>> block_builders;
  std::vector<std::string> block_names;
  std::vector<ObjectID> ids;
  for (auto hash : block_hash_vec) {
    block_names.push_back(KVCacheHelper::BuildBlockName(hash));
  }
  Status status =
      VLLMBlockBuilder::Make(client, offsets_vec, sizes_vec, shape_vec,
                             layer_index, block_builders, req_flag);
  if (!status.ok()) {
    statuses.assign(block_hash_vec.size(), status);
    return Status::OK();
  }

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  status = VLLMBlockBuilder::BatchSeal(client, block_builders, blocks, ids,
                                       req_flag);
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << " BatchSeal cost:" << (end - start)
      << " us";
  if (!status.ok()) {
    LOG(ERROR) << "Failed to seal blocks: " << status.ToString()
               << ", request id: " << req_flag;
    for (auto& builder : block_builders) {
      block_builders_to_delete.push_back(builder);
    }
    statuses.assign(block_hash_vec.size(), status);
  } else {
    start = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
    Status status = client.PutNames(ids, block_names, req_flag, false);
    end = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
    VLOG(KVCacheHelper::GetTraceLogLevel())
        << "Request: " << req_flag << " PutNames cost:" << (end - start)
        << " us";
    if (!status.ok()) {
      LOG(ERROR) << "Failed to put block names: " << status.ToString()
                 << ", request id: " << req_flag;
      for (auto& block : blocks) {
        blocks_to_delete.push_back(block);
      }
      statuses.assign(block_hash_vec.size(), status);
    }
  }

  /*
   * Here we can discard the result of DeleteBlocks with no fatal effect because
   * this error occurs when put block name to meta service. If it is failed,
   * other node can not get the block by name, so it is safe to notify user
   * that the block is not put successfully. It just cause a memory leak
   * if the block is not deleted.
   */
  if (!DeleteBlocks(client, blocks_to_delete, req_flag).ok()) {
    LOG(ERROR) << "Failed to delete blocks, may cause memory leak. Request id: "
               << req_flag;
  }

  if (!DeleteBlockBuilders(client, block_builders_to_delete, req_flag).ok()) {
    LOG(ERROR) << "Failed to delete block builders, may cause memory leak. "
                  "Request id: "
               << req_flag;
  }

  return Status::OK();
}

Status VLLMKVStorage::PutBlockKVCache(
    std::vector<uint64_t>& block_hash_vec,
    std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
    std::vector<Status>& statuses, std::string& req_flag) {
  size_t index = req_count_.fetch_add(1) % threads_;
  return req_thread_vec_[index]
      ->enqueue([&]() {
        Status status =
            PutBlockKVCache(*vineyard_clients_[index], block_hash_vec,
                            block_builders, statuses, req_flag);
        VINEYARD_ASSERT(vineyard_clients_[index]->Connected(),
                        "vineyard client is not connected");
        return status;
      })
      .get();
}

Status VLLMKVStorage::PutBlockKVCache(
    Client& client, std::vector<uint64_t>& block_hash_vec,
    std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
    std::vector<Status>& statuses, std::string& req_flag) {
  std::vector<std::shared_ptr<VLLMBlock>> blocks_to_delete;
  std::vector<std::shared_ptr<VLLMBlockBuilder>> block_builders_to_delete;
  RETURN_ON_ASSERT(
      block_hash_vec.size() == block_builders.size(),
      "block_hash and block_builders size not match for request: " + req_flag);
  statuses.resize(block_hash_vec.size());
  for (size_t i = 0; i < block_hash_vec.size(); ++i) {
    std::shared_ptr<Object> block;
    Status status = block_builders[i]->Seal(client, block);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to seal block: " << status.ToString()
                 << ", request id: " << req_flag;
      block_builders_to_delete.push_back(block_builders[i]);
      statuses[i] = status;
      continue;
    }
    status = client.PutName(
        block->id(), KVCacheHelper::BuildBlockName(block_hash_vec[i]), false);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to put block name: " << status.ToString()
                 << ", request id: " << req_flag;
      blocks_to_delete.push_back(std::dynamic_pointer_cast<VLLMBlock>(block));
      statuses[i] = status;
      continue;
    }
    statuses[i] = status;
  }
  if (!DeleteBlockBuilders(client, block_builders_to_delete, req_flag).ok()) {
    LOG(ERROR) << "Failed to delete block builders, may cause memory leak. "
                  "Request id: "
               << req_flag;
  }

  if (!DeleteBlocks(client, blocks_to_delete, req_flag).ok()) {
    LOG(ERROR) << "Failed to delete blocks, may cause memory leak. Request id: "
               << req_flag;
  }
  return Status::OK();
}

Status VLLMKVStorage::DeleteBlocks(std::vector<uint64_t> block_hash_vec,
                                   std::string req_flag) {
  if (block_hash_vec.size() == 0) {
    return Status::OK();
  }
  size_t index = req_count_.fetch_add(1) % threads_;
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << " arrived at DeleteBlocks, "
      << "assigned to thread: " << index;
  return req_thread_vec_[index]
      ->enqueue([&]() {
        uint64_t start =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        Status status = block_opt_thread_pool_
                            ->enqueue([&]() {
                              return DeleteBlocks(*vineyard_clients_[index],
                                                  block_hash_vec, req_flag);
                            })
                            .get();

        VINEYARD_ASSERT(vineyard_clients_[index]->Connected(),
                        "vineyard client is not connected");
        uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        VLOG(KVCacheHelper::GetTraceLogLevel())
            << "Request : " << req_flag
            << ". DeleteBlocks cost:" << (end - start) << " us.";
        return status;
      })
      .get();
}

Status VLLMKVStorage::DeleteBlocks(Client& client,
                                   std::vector<uint64_t> block_hash_vec,
                                   std::string& req_flag) {
  uint64_t start = 0, end = 0;
  std::vector<std::string> block_name_vec;
  size_t block_num = block_hash_vec.size();

  block_name_vec.reserve(block_num);
  for (auto block_hash : block_hash_vec) {
    std::string block_name = KVCacheHelper::BuildBlockName(block_hash);
    block_name_vec.push_back(block_name);
  }

  std::vector<ObjectID> id_vec;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  VINEYARD_CHECK_OK(client.GetNames(block_name_vec, id_vec, req_flag));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Get block ids from names cost: " << (end - start) << " us";

  std::vector<ObjectID> valid_id_vec;
  for (auto id : id_vec) {
    if (id != InvalidObjectID()) {
      valid_id_vec.push_back(id);
    }
  }
  if (valid_id_vec.size() == 0) {
    return Status::OK();
  }

  std::vector<ObjectMeta> meta_vec;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  VINEYARD_CHECK_OK(
      client.GetHugeMetaData(valid_id_vec, meta_vec, req_flag, false, true));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". Get block metas cost: " << (end - start)
      << " us";

  // Drop name will triggler rpc, so we discard the error result
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  Status status = client.DropNames(block_name_vec, req_flag);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to drop block names: " << status.ToString()
                 << ", may cause inconsistency. Request id: " << req_flag;
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". Drop block names cost: " << (end - start)
      << " us";

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  VINEYARD_CHECK_OK(
      client.DelHugeData(valid_id_vec, false, false, true, req_flag));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Delete block objects cost: " << (end - start) << " us";

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  VINEYARD_CHECK_OK(CleanBlockBlobs(client, meta_vec, req_flag));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Clean block blobs cost: " << (end - start) << " us";

  return Status::OK();
}

Status VLLMKVStorage::DeleteBlocks(
    Client& client, std::vector<std::shared_ptr<VLLMBlock>>& blocks,
    std::string& req_flag) {
  std::vector<ObjectMeta> meta_vec;
  std::vector<ObjectID> id_vec;
  meta_vec.reserve(blocks.size());
  id_vec.reserve(blocks.size());

  for (auto block : blocks) {
    id_vec.push_back(block->id());
    meta_vec.push_back(block->meta());
  }

  VINEYARD_CHECK_OK(client.DelData(id_vec));
  VINEYARD_CHECK_OK(CleanBlockBlobs(client, meta_vec, req_flag));
  return Status::OK();
}

Status VLLMKVStorage::DeleteBlockBuilders(
    std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
    std::string& req_flag) {
  size_t index = req_count_.fetch_add(1) % threads_;
  return req_thread_vec_[index]
      ->enqueue([&]() {
        Status status = DeleteBlockBuilders(*vineyard_clients_[index],
                                            block_builders, req_flag);
        VINEYARD_ASSERT(vineyard_clients_[index]->Connected(),
                        "vineyard client is not connected");
        return status;
      })
      .get();
}

Status VLLMKVStorage::DeleteBlockBuilders(
    Client& client,
    std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
    std::string& req_flag) {
  Status status = Status::OK();
  for (auto block_builder : block_builders) {
    Status status_ = CleanBlockBuilderBlobs(client, block_builder, req_flag);
    if (!status_.ok()) {
      LOG(WARNING) << "Failed to clean block builder"
                   << ", may cause memory leak. Error: " << status_.ToString()
                   << ", request id: " << req_flag;
      status += status_;
    }
  }
  return status;
}

/**
 * FIXME: If the user 1 create a udc, and user 2 create the same udc, it
 * can occur that the user 2 rename file failed and then the udc is deleted
 * by ttl. Then the udc of user 2 will lose some blocks but the system think
 * the udc is valid.
 *
 * FIXME: refresh block ttl after save to disk.
 */
Status VLLMKVStorage::SaveToDisk(
    std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& statuses, uint64_t ttl, bool wait,
    std::string req_flag) {
  RETURN_ON_ASSERT(block_hash_vec.size() == offsets_vec.size(),
                   "block_hash and offsets size not match");
  RETURN_ON_ASSERT(block_hash_vec.size() == sizes_vec.size(),
                   "block_hash and sizes size not match");
  RETURN_ON_ASSERT(layer_index >= 0, "layer_index must be >= 0");
  RETURN_ON_ASSERT(layer_index < static_cast<int>(shape.size()),
                   "layer_index must be < shape.size()");
  if (block_hash_vec.size() == 0) {
    return Status::OK();
  }

  size_t index = req_count_.fetch_add(1) % threads_;
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << " arrived at SaveToDisk, "
      << "assigned to thread: " << index;
  return req_thread_vec_[index]
      ->enqueue([&]() {
        uint64_t start = 0, end = 0;
        start = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();

        statuses.resize(block_hash_vec.size(), Status::OK());
        std::vector<uint64_t> filtered_hash_vec;
        std::vector<uint64_t> filtered_hash_index;
        std::vector<uint64_t> exist_block_vec;
        std::vector<std::vector<uint64_t>> filtered_offsets_vec;
        std::vector<std::vector<size_t>> filtered_sizes_vec;
        if (!FilterFiles(block_hash_vec, exist_block_vec, filtered_hash_vec,
                         filtered_hash_index)
                 .ok()) {
          filtered_hash_vec = block_hash_vec;
          filtered_hash_index.resize(block_hash_vec.size());
          for (size_t i = 0; i < block_hash_vec.size(); ++i) {
            filtered_hash_index[i] = i;
          }
          filtered_offsets_vec = offsets_vec;
          filtered_sizes_vec = sizes_vec;
        }
        for (auto index : filtered_hash_index) {
          filtered_offsets_vec.push_back(offsets_vec[index]);
          filtered_sizes_vec.push_back(sizes_vec[index]);
        }
        RETURN_ON_ERROR(UpdateTTL(*vineyard_clients_[index], exist_block_vec,
                                  ttl, req_flag));

        Status status = Status::OK();
        if (use_copy_) {
          status = SaveToDiskWithCopy(*vineyard_clients_[index],
                                      filtered_hash_vec, filtered_offsets_vec,
                                      filtered_sizes_vec, shape, layer_index,
                                      statuses, ttl, wait, req_flag);
        } else {
          status = SaveToDiskWithoutCopy(
              *vineyard_clients_[index], filtered_hash_vec,
              filtered_offsets_vec, filtered_sizes_vec, shape, layer_index,
              statuses, ttl, req_flag);
        }
        VINEYARD_ASSERT(vineyard_clients_[index]->Connected(),
                        "vineyard client is not connected");

        end = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
        VLOG(KVCacheHelper::GetTraceLogLevel())
            << "Request: " << req_flag << ". SaveToDisk cost:" << (end - start)
            << " us.";
        return status;
      })
      .get();
}

Status VLLMKVStorage::FilterFiles(std::vector<uint64_t>& block_hash_vec,
                                  std::vector<uint64_t>& exist_block_vec,
                                  std::vector<uint64_t>& filtered_hash_vec,
                                  std::vector<uint64_t>& filtered_hash_index) {
  uint64_t start = 0, end = 0;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  std::vector<std::future<Status>> exist_statuses;
  exist_block_vec.reserve(block_hash_vec.size());
  exist_statuses.reserve(block_hash_vec.size());
  for (size_t i = 0; i < block_hash_vec.size(); ++i) {
    exist_statuses.push_back(fast_opt_thread_pool_->enqueue(
        [&](size_t i) {
          std::string prefix_dir, file_name;
          Hash2PrefixDirAndSuffixFile(block_hash_vec[i], prefix_dir, file_name);
          std::string file_path =
              VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskPath() + '/' +
              file_name;
          if (std::filesystem::exists(file_path)) {
            VLOG(100) << "Block file exists: " << file_path
                      << ", skip write to disk.";
            return Status::OK();
          }
          return Status::ObjectNotExists();
        },
        i));
  }

  for (size_t i = 0; i < exist_statuses.size(); ++i) {
    Status status = exist_statuses[i].get();
    if (status.IsObjectNotExists()) {
      filtered_hash_vec.push_back(block_hash_vec[i]);
      filtered_hash_index.push_back(i);
    } else if (!status.ok()) {
      // unexpected error
      return status;
    } else {
      exist_block_vec.push_back(block_hash_vec[i]);
    }
  }

  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "FilterFiles cost:" << (end - start) << " us";
  return Status::OK();
}

Status VLLMKVStorage::UpdateTTL(Client& client,
                                std::vector<uint64_t>& block_hash_vec,
                                uint64_t ttl, std::string& req_flag) {
  uint64_t start = 0, end = 0;
  if (block_hash_vec.empty()) {
    return Status::OK();
  }
  if (VLLMKVCacheEnv::LocalVineyardVLLMKVCache() == "1") {
    LOG(INFO) << "LocalVineyardVLLMKVCache is set, skip UpdateTTL.";
    return Status::OK();
  }

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  std::string block_location;
  std::vector<std::string> disk_type;
  std::vector<std::string> block_name;
  RETURN_ON_ERROR(GetIOTag(block_location));
  for (size_t i = 0; i < block_hash_vec.size(); ++i) {
    disk_type.push_back(block_location);
    block_name.push_back(KVCacheHelper::BuildBlockName(block_hash_vec[i]));
  }
  client.PutObjectLocation(block_name, disk_type, ttl, req_flag);
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". UpdateTTL cost: " << (end - start)
      << " us";
  return Status::OK();
}

Status VLLMKVStorage::SaveToDiskWithCopy(
    Client& client, std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& status_vec, uint64_t ttl, bool wait,
    std::string& req_flag) {
  uint64_t start = 0, end = 0;
  MONITOR_AUTO(save_to_disk_monitor_);

  size_t hash_num = block_hash_vec.size();
  if (hash_num == 0) {
    LOG(INFO) << "No new blocks to save to disk for request: " << req_flag;
    return Status::OK();
  }
  status_vec.resize(hash_num, Status::OK());

  std::vector<std::string> tmp_file_name_vec;
  std::vector<std::string> file_name_vec;
  std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>>
      tmp_io_adaptor_vec;
  std::vector<std::shared_ptr<std::string>> auto_delete_vec;
  tmp_io_adaptor_vec.resize(hash_num);
  tmp_file_name_vec.resize(hash_num);
  file_name_vec.resize(hash_num);

  std::string pidstr = std::to_string(getpid());
  Status status = Status::OK();

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  for (size_t i = 0; i < hash_num; ++i) {
    std::string prefix_dir;
    Hash2PrefixDirAndSuffixFile(block_hash_vec[i], prefix_dir,
                                file_name_vec[i]);
    status = CreateDirectoriesIfNotExists(
        VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskPath() + '/' + prefix_dir);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to create directory: " << prefix_dir
                 << ", request id: " << req_flag
                 << ", error: " << status.ToString();
      status_vec[i] = status;
      continue;
    }

    std::string tmp_file_name = file_name_vec[i] + "_" + pidstr + "_" +
                                std::to_string(gettid()) + "_tmp";
    // We do not to check if the file is exists.
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor;
    status = GetIOAdaptor(io_adaptor, tmp_file_name);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to get IOAdaptor for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString()
                 << ", request id: " << req_flag;
      status_vec[i] = status;
      continue;
    }

    status = io_adaptor->Open("w", direct_io_);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to open IOAdaptor for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString()
                 << ", request id: " << req_flag;
      status_vec[i] = status;
      continue;
    }

    tmp_io_adaptor_vec[i] = io_adaptor;
    tmp_file_name_vec[i] = tmp_file_name;
    auto_delete_vec.push_back(std::shared_ptr<std::string>(
        new std::string(tmp_file_name), [&](std::string* ptr) {
          if (!std::filesystem::remove(GetIOPathPrefix() + "/" + *ptr) &&
              std::filesystem::exists(GetIOPathPrefix() + "/" + *ptr)) {
            LOG(WARNING) << "Failed to remove temporary file: " << *ptr
                         << ", may cause resource leak.";
          }
          delete ptr;
        }));
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Create tmp file io adaptors cost: " << (end - start) << " us";

  start = end;
  MONITOR_START(save_memory_copy_monitor_)
  std::vector<std::shared_ptr<char[]>> data_ptr_vec;
  std::vector<size_t> file_size_vec;
  std::vector<std::future<Status>> copy_statuses;
  copy_statuses.resize(hash_num);
  data_ptr_vec.resize(hash_num);
  file_size_vec.resize(hash_num);
  auto copy_func = [&](size_t i) -> Status {
    std::shared_ptr<char[]> data_ptr;
    size_t file_size = 0;
    json meta_json;
    ConstructVLLMBlockFileMeta(offsets_vec[i], sizes_vec[i], shape, layer_index,
                               meta_json);
    std::string meta_str = meta_json.dump();
    Status status = CopyBlockToMemoryInternal(
        meta_str, offsets_vec[i], sizes_vec[i], data_ptr, file_size);
    if (status.ok()) {
      data_ptr_vec[i] = (data_ptr);
      file_size_vec[i] = (file_size);
      if (!tmp_io_adaptor_vec[i]->FileTruncate(file_size_vec[i]).ok()) {
        LOG(WARNING) << "Failed to truncate file for block: "
                     << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                     << ", may cause performance issue.";
      }
    }
    return status;
  };

  for (size_t i = 0; i < hash_num; ++i) {
    if (status_vec[i].ok()) {
      copy_statuses[i] = copy_thread_pool_->enqueue(copy_func, i);
    }
  }

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }

    if (!copy_statuses[i].valid()) {
      LOG(ERROR) << "Future is not valid for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i]);
      tmp_io_adaptor_vec[i]->Close();
      status_vec[i] = Status::IOError("Future is not valid");
      continue;
    }

    Status status = copy_statuses[i].get();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to copy block to memory: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString();
      tmp_io_adaptor_vec[i]->Close();
      status_vec[i] = status;
    }
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Copy blocks to memory cost: " << (end - start) << " us";

  MONITOR_END(save_memory_copy_monitor_);

  MONITOR_START(save_to_disk_io_monitor_);

  if (wait) {
    RETURN_ON_ERROR(SaveToDiskSubmitIO(client, tmp_io_adaptor_vec,
                                       file_size_vec, data_ptr_vec,
                                       block_hash_vec, status_vec, req_flag));

    RETURN_ON_ERROR(SaveToDiskMoveFile(
        client, block_hash_vec, tmp_io_adaptor_vec, file_name_vec,
        tmp_file_name_vec, status_vec, ttl, req_flag));
  } else {
    block_opt_thread_pool_->enqueue(
        [&client, ttl, req_flag, auto_delete_vec_ = std::move(auto_delete_vec),
         tmp_io_adaptor_vec_ = std::move(tmp_io_adaptor_vec),
         file_name_vec_ = std::move(file_name_vec),
         tmp_file_name_vec_ = std::move(tmp_file_name_vec),
         data_ptr_vec_ = std::move(data_ptr_vec),
         file_size_vec_ = std::move(file_size_vec),
         block_hash_vec_ = std::move(block_hash_vec),
         status_vec_ = status_vec]() {
          std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>>
              tmp_io_adaptor_vec__ = tmp_io_adaptor_vec_;
          std::vector<std::string> file_name_vec__ = file_name_vec_;
          std::vector<std::string> tmp_file_name_vec__ = tmp_file_name_vec_;
          std::vector<std::shared_ptr<char[]>> data_ptr_vec__ = data_ptr_vec_;
          std::vector<size_t> file_size_vec__ = file_size_vec_;
          std::vector<uint64_t> block_hash_vec__ = block_hash_vec_;
          std::vector<Status> status_vec__ = status_vec_;

          SaveToDiskSubmitIO(client, tmp_io_adaptor_vec__, file_size_vec__,
                             data_ptr_vec__, block_hash_vec__, status_vec__,
                             req_flag);
          SaveToDiskMoveFile(client, block_hash_vec__, tmp_io_adaptor_vec__,
                             file_name_vec__, tmp_file_name_vec__, status_vec__,
                             ttl, req_flag);
          return Status::OK();
        });
  }
  return Status::OK();
}

Status VLLMKVStorage::SaveToDiskSubmitIO(
    Client& client,
    std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>>& io_adaptor_vec,
    std::vector<size_t>& file_size_vec,
    std::vector<std::shared_ptr<char[]>>& data_ptr_vec,
    std::vector<uint64_t>& block_hash_vec, std::vector<Status>& status_vec,
    std::string req_flag) {
  uint64_t start = 0, end = 0;
  size_t hash_num = block_hash_vec.size();
  std::vector<std::future<Status>> submit_status_vecs;
  submit_status_vecs.resize(hash_num);
  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    submit_status_vecs[i] =
        io_adaptor_vec[i]->AsyncWrite(data_ptr_vec[i], file_size_vec[i], 0);
  }

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    if (!submit_status_vecs[i].valid()) {
      LOG(ERROR) << "Future is not valid for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", request id: " << req_flag;
      io_adaptor_vec[i]->Close();
      status_vec[i] =
          Status::IOError("Future is not valid. Request id: " + req_flag);
      continue;
    }
    Status status = submit_status_vecs[i].get();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to write block to disk: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString()
                 << ", request id: " << req_flag;
      io_adaptor_vec[i]->Close();
      status_vec[i] = status;
    }
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Write blocks to disk cost: " << (end - start) << " us.";

  return Status::OK();
}

Status VLLMKVStorage::SaveToDiskMoveFile(
    Client& client, std::vector<uint64_t>& block_hash_vec,
    std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>>& io_adaptor_vec,
    std::vector<std::string>& file_name_vec,
    std::vector<std::string>& tmp_file_name_vec,
    std::vector<Status>& status_vec, uint64_t ttl, std::string req_flag) {
  uint64_t start = 0, end = 0;
  uint64_t hash_num = block_hash_vec.size();

  std::string disk_location;
  Status status = GetIOTag(disk_location);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to get disk tag for blocks, error: "
               << status.ToString() << ", request id: " << req_flag;
    return status;
  }

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  std::vector<std::future<Status>> move_status_vec;
  move_status_vec.resize(hash_num);
  auto move_func = [&](size_t i) -> Status {
    RETURN_ON_ERROR(io_adaptor_vec[i]->Close());
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor;
    GetIOAdaptor(io_adaptor, file_name_vec[i]);
    RETURN_ON_ERROR(io_adaptor->Open("w", direct_io_));
    RETURN_ON_ERROR(io_adaptor->Close());

    try {
      std::filesystem::rename(
          (GetIOPathPrefix() + "/" + tmp_file_name_vec[i]).c_str(),
          (GetIOPathPrefix() + "/" + file_name_vec[i]).c_str());
    } catch (const std::filesystem::filesystem_error& e) {
      LOG(ERROR) << "Failed to rename file: "
                 << GetIOPathPrefix() + "/" + tmp_file_name_vec[i] << " to "
                 << GetIOPathPrefix() + "/" + file_name_vec[i]
                 << ", error: " << e.what();
      return Status::IOError("Failed to rename file: " + GetIOPathPrefix() +
                             "/" + tmp_file_name_vec[i] + " to " +
                             GetIOPathPrefix() + "/" + file_name_vec[i] +
                             ", error: " + e.what());
    }
    return Status::OK();
  };
  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    move_status_vec[i] = io_thread_pool_->enqueue(move_func, i);
  }

  std::vector<std::string> written_block_name_vec;
  std::vector<std::string> block_location_vec;
  written_block_name_vec.resize(hash_num);
  block_location_vec.resize(hash_num);

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    Status status = move_status_vec[i].get();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to move file for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString()
                 << ", request id: " << req_flag;
      status_vec[i] = status;
      continue;
    }
    written_block_name_vec[i] =
        KVCacheHelper::BuildBlockName(block_hash_vec[i]);
    block_location_vec[i] = disk_location;
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Move blocks to final path cost: " << (end - start) << " us";
  MONITOR_END(save_to_disk_io_monitor_);

  start = end;
  if (VLLMKVCacheEnv::LocalVineyardVLLMKVCache() == "1") {
    LOG(INFO) << "Test mode will skip putting object location.";
    return Status::OK();
  }

  status = client.PutObjectLocation(written_block_name_vec, block_location_vec,
                                    ttl, req_flag);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to put object location for blocks, error: "
                 << status.ToString() << ", request id: " << req_flag;
    for (auto path : file_name_vec) {
      std::filesystem::remove(GetIOPathPrefix() + "/" + path);
    }
    return status;
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Put object location cost: " << (end - start) << " us";

  return Status::OK();
}

Status VLLMKVStorage::SaveToDiskWithoutCopy(
    Client& client, std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& status_vec, uint64_t ttl,
    std::string& req_flag) {
  uint64_t start = 0, end = 0;
  MONITOR_AUTO(save_to_disk_monitor_);
  size_t hash_num = block_hash_vec.size();
  if (hash_num == 0) {
    return Status::OK();
  }
  status_vec.resize(hash_num, Status::OK());

  std::vector<std::string> tmp_file_name_vec;
  std::vector<std::string> file_name_vec;
  std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>>
      tmp_io_adaptor_vec;
  std::vector<std::shared_ptr<std::string>> auto_delete_vec;
  tmp_io_adaptor_vec.resize(hash_num);
  tmp_file_name_vec.resize(hash_num);
  file_name_vec.resize(hash_num);

  std::string disk_location;
  Status status = GetIOTag(disk_location);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to get disk tag for blocks, error: "
               << status.ToString() << ", request id: " << req_flag;
    return status;
  }

  std::string pidstr = std::to_string(getpid());

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  for (size_t i = 0; i < hash_num; ++i) {
    std::string prefix_dir;
    Hash2PrefixDirAndSuffixFile(block_hash_vec[i], prefix_dir,
                                file_name_vec[i]);
    status = CreateDirectoriesIfNotExists(
        VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskPath() + '/' + prefix_dir);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to create directory: " << prefix_dir;
      status_vec[i] = status;
      continue;
    }

    std::string tmp_file_name = file_name_vec[i] + "_" + pidstr + "_" +
                                std::to_string(gettid()) + "_tmp";
    // We do not to check if the file is exists.
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor;
    status = GetIOAdaptor(io_adaptor, tmp_file_name);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to get IOAdaptor for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString();
      status_vec[i] = status;
      continue;
    }

    status = io_adaptor->Open("w", direct_io_);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to open IOAdaptor for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString();
      status_vec[i] = status;
      continue;
    }

    tmp_io_adaptor_vec[i] = io_adaptor;
    tmp_file_name_vec[i] = tmp_file_name;
    auto_delete_vec.push_back(std::shared_ptr<std::string>(
        new std::string(tmp_file_name), [&](std::string* ptr) {
          if (!std::filesystem::remove(GetIOPathPrefix() + "/" + *ptr) &&
              std::filesystem::exists(GetIOPathPrefix() + "/" + *ptr)) {
            LOG(WARNING) << "Failed to remove temporary file: " << *ptr
                         << ", may cause resource leak.";
          }
          delete ptr;
        }));
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Create tmp file io adaptors cost: " << (end - start) << " us";

  start = end;
  std::vector<std::future<Status>> ret_status_future_vec;
  std::vector<std::vector<std::future<Status>>> write_status_future_vec;
  write_status_future_vec.resize(hash_num);
  ret_status_future_vec.resize(hash_num);
  auto write_func = [&](size_t i) -> Status {
    json meta_json;
    return WriteBlockToDisk(tmp_io_adaptor_vec[i], offsets_vec[i], sizes_vec[i],
                            shape, layer_index, write_status_future_vec[i]);
  };

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    ret_status_future_vec[i] = io_thread_pool_->enqueue(write_func, i);
  }

  // check write results
  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    if (!ret_status_future_vec[i].valid()) {
      LOG(ERROR) << "Future is not valid for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i]);
      tmp_io_adaptor_vec[i]->Close();
      status_vec[i] = Status::IOError("Future is not valid");
      continue;
    }
    Status status = ret_status_future_vec[i].get();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to write block to disk: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString();
      tmp_io_adaptor_vec[i]->Close();
      status_vec[i] = status;
      continue;
    }
    for (auto& write_status : write_status_future_vec[i]) {
      if (!write_status.valid()) {
        LOG(ERROR) << "Future is not valid for block: "
                   << KVCacheHelper::BuildBlockName(block_hash_vec[i]);
        tmp_io_adaptor_vec[i]->Close();
        status_vec[i] = Status::IOError("Future is not valid");
        break;
      }
      Status s = write_status.get();
      if (!s.ok()) {
        LOG(ERROR) << "Failed to write block to disk: "
                   << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                   << ", error: " << s.ToString();
        tmp_io_adaptor_vec[i]->Close();
        status_vec[i] = s;
        break;
      }
    }
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Write blocks to disk cost: " << (end - start) << " us.";

  start = end;
  // move tmp file to final location
  std::vector<std::future<Status>> move_statuses;
  move_statuses.resize(hash_num);
  auto move_func = [&](size_t i) -> Status {
    RETURN_ON_ERROR(tmp_io_adaptor_vec[i]->Close());
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor_tmp;
    GetIOAdaptor(io_adaptor_tmp, file_name_vec[i]);
    RETURN_ON_ERROR(io_adaptor_tmp->Open("w", direct_io_));
    RETURN_ON_ERROR(io_adaptor_tmp->Close());

    try {
      std::filesystem::rename(
          (GetIOPathPrefix() + "/" + tmp_file_name_vec[i]).c_str(),
          (GetIOPathPrefix() + "/" + file_name_vec[i]).c_str());
    } catch (const std::filesystem::filesystem_error& e) {
      LOG(ERROR) << "Failed to rename file: "
                 << GetIOPathPrefix() + "/" + tmp_file_name_vec[i] << " to "
                 << GetIOPathPrefix() + "/" + file_name_vec[i]
                 << ", error: " << e.what();
      return Status::IOError("Failed to rename file: " + GetIOPathPrefix() +
                             "/" + tmp_file_name_vec[i] + " to " +
                             GetIOPathPrefix() + "/" + file_name_vec[i] +
                             ", error: " + e.what());
    }
    return Status::OK();
  };
  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    move_statuses[i] = io_thread_pool_->enqueue(move_func, i);
  }

  std::vector<std::string> written_block_name_vec;
  std::vector<std::string> block_location_vec;
  written_block_name_vec.resize(hash_num);
  block_location_vec.resize(hash_num);

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    Status status = move_statuses[i].get();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to move file for block: "
                 << KVCacheHelper::BuildBlockName(block_hash_vec[i])
                 << ", error: " << status.ToString();
      status_vec[i] = status;
      continue;
    }
    written_block_name_vec[i] =
        KVCacheHelper::BuildBlockName(block_hash_vec[i]);
    block_location_vec[i] = disk_location;
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Move blocks to final path cost: " << (end - start) << " us.";

  start = end;
  if (VLLMKVCacheEnv::LocalVineyardVLLMKVCache() == "1") {
    LOG(INFO) << "Test mode will skip put object location for blocks.";
    return Status::OK();
  }
  status = client.PutObjectLocation(written_block_name_vec, block_location_vec,
                                    ttl, req_flag);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to put object location for blocks, error: "
                 << status.ToString();
    for (auto path : file_name_vec) {
      std::filesystem::remove(GetIOPathPrefix() + "/" + path);
    }
    return status;
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Put object location cost: " << (end - start) << " us";

  return Status::OK();
}

Status VLLMKVStorage::LoadFromDisk(
    std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& statuses, std::string req_flag) {
  RETURN_ON_ASSERT(block_hash_vec.size() == offsets_vec.size(),
                   "block_hash_vec.size() and offsets_vec.size() must be equal "
                   "for request: " +
                       req_flag);
  RETURN_ON_ASSERT(
      block_hash_vec.size() == sizes_vec.size(),
      "block_hash_vec.size() and sizes_vec.size() must be equal for request: " +
          req_flag);
  RETURN_ON_ASSERT(layer_index >= 0,
                   "layer_index must be >= 0 for request: " + req_flag);
  RETURN_ON_ASSERT(
      layer_index < static_cast<int>(shape.size()),
      "layer_index must be < shape.size() for request: " + req_flag);
  if (block_hash_vec.empty()) {
    return Status::OK();
  }

  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << " arrived at LoadFromDisk.";
  uint64_t start = 0, end = 0;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();

  Status status = Status::OK();
  if (use_copy_) {
    status = LoadFromDiskWithCopy(block_hash_vec, offsets_vec, sizes_vec, shape,
                                  layer_index, statuses, req_flag);
  } else {
    status = LoadFromDiskWithoutCopy(block_hash_vec, offsets_vec, sizes_vec,
                                     shape, layer_index, statuses, req_flag);
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". LoadFromDisk cost: " << (end - start)
      << " us.";
  return status;
}

Status VLLMKVStorage::LoadFromDiskWithCopy(
    std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& status_vec, std::string& req_flag) {
  uint64_t start = 0, end = 0;
  MONITOR_AUTO(load_from_disk_monitor_);

  size_t hash_num = block_hash_vec.size();
  if (hash_num == 0) {
    return Status::OK();
  }
  status_vec.resize(hash_num, Status::OK());

  std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>> io_adaptor_vec;
  std::vector<std::shared_ptr<char[]>> data_ptr_vec;
  std::vector<size_t> file_size_vec;
  std::vector<std::string> file_name_vec;
  io_adaptor_vec.resize(hash_num);
  data_ptr_vec.resize(hash_num);
  file_size_vec.resize(hash_num);
  file_name_vec.resize(hash_num);

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  for (size_t i = 0; i < hash_num; ++i) {
    std::string prefix_dir;
    Hash2PrefixDirAndSuffixFile(block_hash_vec[i], prefix_dir,
                                file_name_vec[i]);
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor;
    Status status = GetIOAdaptor(io_adaptor, file_name_vec[i]);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to get IOAdaptor for file: " << file_name_vec[i]
                 << ", error: " << status.ToString();
      status_vec[i] = status;
      continue;
    }
    status = io_adaptor->Open("r", direct_io_);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to open IOAdaptor for file: " << file_name_vec[i]
                 << ", error: " << status.ToString();
      status_vec[i] = status;
      if (!io_adaptor->Close().ok()) {
        LOG(ERROR) << "Failed to close IOAdaptor for file: " << file_name_vec[i]
                   << ", may cause resource leak.";
      }
      continue;
    }
    size_t file_size = 0;
    status = io_adaptor->GetFileSize(file_size);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to get file size for file: " << file_name_vec[i]
                 << ", error: " << status.ToString();
      status_vec[i] = status;
      if (!io_adaptor->Close().ok()) {
        LOG(ERROR) << "Failed to close IOAdaptor for file: " << file_name_vec[i]
                   << ", may cause resource leak.";
      }
      continue;
    }
    if (file_size == 0) {
      LOG(ERROR) << "File to load: " << file_name_vec[i] << " is empty!";
      status_vec[i] =
          Status::IOError("File to load: " + file_name_vec[i] + " is empty!");
      if (!io_adaptor->Close().ok()) {
        LOG(ERROR) << "Failed to close IOAdaptor for file: " << file_name_vec[i]
                   << ", may cause resource leak.";
      }
      continue;
    }
    // align to align
    size_t align = std::stoull(VLLMKVCacheEnv::GetDirectIOAlign());
    size_t mmap_size = ((file_size + align - 1) / align) * align;
    void* data_ptr = mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data_ptr == MAP_FAILED) {
      LOG(ERROR) << "Failed to mmap memory for file: " << file_name_vec[i]
                 << ", size: " << mmap_size << ", error: " << strerror(errno);
      status_vec[i] = Status::Invalid(
          "Failed to mmap memory for file: " + file_name_vec[i] + ", size: " +
          std::to_string(mmap_size) + ", error: " + strerror(errno));
    }
    std::shared_ptr<char[]> data_ptr_shared(
        reinterpret_cast<char*>(data_ptr), [mmap_size](char* p) {
          if (munmap(p, mmap_size) != 0) {
            LOG(ERROR) << "Failed to munmap memory, may cause memory leak.";
          }
        });
    data_ptr_vec[i] = data_ptr_shared;

    io_adaptor_vec[i] = io_adaptor;
    file_size_vec[i] = file_size;
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Create read file io adaptors cost: " << (end - start) << " us";

  start = end;
  std::vector<std::future<Status>> submit_status_vec;
  submit_status_vec.resize(hash_num);
  MONITOR_START(load_from_disk_io_monitor_);

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    submit_status_vec[i] =
        io_adaptor_vec[i]->AsyncRead(data_ptr_vec[i], file_size_vec[i], 0);
  }

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    if (!submit_status_vec[i].valid()) {
      // It means that the read operation not completed successfully.
      status_vec[i] = Status::Invalid(
          "Invalid future for reading block from disk, maybe the IOAdaptor "
          "is not valid or the read operation is not started. Request id: " +
          req_flag);
      LOG(ERROR) << "Invalid future for reading block from disk."
                 << " Request id: " << req_flag;
      io_adaptor_vec[i]->Close();
      continue;
    }
    Status status = submit_status_vec[i].get();
    if (!status.ok()) {
      // It means that the read operation not completed successfully.
      LOG(ERROR) << "Failed to read block from disk, error: "
                 << status.ToString();
      status_vec[i] = status;
      io_adaptor_vec[i]->Close();
      continue;
    }
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Read blocks from disk cost: " << (end - start) << " us.";

  start = end;
  MONITOR_START(load_memory_copy_monitor_);
  std::vector<std::future<Status>> memcpy_status_future_vec;
  memcpy_status_future_vec.resize(hash_num);
  auto memory_copy_func = [&](size_t i) -> Status {
    return ReadBlockFromMemory(data_ptr_vec[i], file_size_vec[i],
                               offsets_vec[i], sizes_vec[i], shape,
                               layer_index);
  };

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    memcpy_status_future_vec[i] =
        copy_thread_pool_->enqueue(memory_copy_func, i);
  }

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    if (!memcpy_status_future_vec[i].valid()) {
      // It means that the read operation not completed successfully.
      LOG(ERROR) << "Invalid future for reading block from disk."
                 << " Request id: " << req_flag;
      status_vec[i] = Status::Invalid(
          "Invalid future for reading block from disk, maybe the data_ptr "
          "is not valid or the read operation is not started. Request id: " +
          req_flag);
      io_adaptor_vec[i]->Close();
      continue;
    }
    Status status = memcpy_status_future_vec[i].get();
    if (!status.ok()) {
      // It means that the read operation not completed successfully.
      LOG(ERROR) << "Failed to read block from disk, error: "
                 << status.ToString();
      io_adaptor_vec[i]->Close();
      status_vec[i] = status;
    }
  }

  MONITOR_END(load_memory_copy_monitor_);
  MONITOR_END(load_from_disk_io_monitor_);

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    if (!io_adaptor_vec[i]->Close().ok()) {
      // Data is already loaded, so we can ignore the error.
      // But we should log the error to report resource leak.
      LOG(WARNING) << "Failed to close IOAdaptor, may cause resource leak."
                   << " Request id: " << req_flag;
    }
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Copy blocks to memory cost: " << (end - start) << " us";

  return Status::OK();
}

Status VLLMKVStorage::LoadFromDiskWithoutCopy(
    std::vector<uint64_t>& block_hash_vec,
    std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index, std::vector<Status>& status_vec, std::string& req_flag) {
  uint64_t start = 0, end = 0;
  MONITOR_AUTO(load_from_disk_monitor_);

  size_t hash_num = block_hash_vec.size();
  if (hash_num == 0) {
    return Status::OK();
  }
  std::vector<std::string> file_name_vec;
  std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>> io_adaptor_vec;
  file_name_vec.resize(hash_num);
  io_adaptor_vec.resize(hash_num);
  status_vec.resize(hash_num, Status::OK());

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  for (size_t i = 0; i < block_hash_vec.size(); ++i) {
    std::string prefix_dir;
    Hash2PrefixDirAndSuffixFile(block_hash_vec[i], prefix_dir,
                                file_name_vec[i]);
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor;
    Status status = GetIOAdaptor(io_adaptor, file_name_vec[i]);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to get IOAdaptor for file: " << file_name_vec[i]
                 << ", error: " << status.ToString()
                 << ", request id: " << req_flag;
      status_vec[i] = status;
      continue;
    }
    status = io_adaptor->Open("r", direct_io_);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to open IOAdaptor for file: " << file_name_vec[i]
                 << ", error: " << status.ToString()
                 << ", request id: " << req_flag;
      status_vec[i] = status;
      if (!io_adaptor->Close().ok()) {
        LOG(ERROR) << "Failed to close IOAdaptor for file: " << file_name_vec[i]
                   << ", may cause resource leak."
                   << ", request id: " << req_flag;
      }
      continue;
    }

    io_adaptor_vec[i] = io_adaptor;
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Create read file io adaptors cost: " << (end - start) << " us";

  start = end;
  std::vector<std::future<Status>> ret_status_future_vec;
  std::vector<std::vector<std::future<Status>>> read_status_future_vec;
  ret_status_future_vec.resize(hash_num);
  read_status_future_vec.resize(hash_num);
  MONITOR_START(load_from_disk_io_monitor_);
  auto read_func = [&](size_t i) -> Status {
    return ReadBlockFromDisk(io_adaptor_vec[i], offsets_vec[i], sizes_vec[i],
                             shape, layer_index, read_status_future_vec[i]);
  };

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    ret_status_future_vec[i] = io_thread_pool_->enqueue(read_func, i);
  }

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    if (!ret_status_future_vec[i].valid()) {
      // It means that the read operation not completed successfully.
      status_vec[i] = Status::Invalid(
          "Invalid future for reading block from disk, maybe the IOAdaptor "
          "is not valid or the read operation is not started.");
      LOG(ERROR) << "Invalid future for reading block from disk."
                 << " Request id: " << req_flag;
      io_adaptor_vec[i]->Close();
      continue;
    }
    Status status = ret_status_future_vec[i].get();
    if (!status.ok()) {
      // It means that the read operation not completed successfully.
      LOG(ERROR) << "Failed to read block from disk, error: "
                 << status.ToString() << ", request id: " << req_flag;
      status_vec[i] = status;
      io_adaptor_vec[i]->Close();
      continue;
    }
    for (auto& read_status : read_status_future_vec[i]) {
      if (!read_status.valid()) {
        LOG(ERROR) << "Invalid future for reading block from disk.";
        status_vec[i] = Status::Invalid(
            "Invalid future for reading block from disk, maybe the read "
            "operation is not started. Request id:" +
            req_flag);
        break;
      }
      Status s = read_status.get();
      if (!s.ok()) {
        LOG(ERROR) << "Failed to read block from disk, error: " << s.ToString()
                   << ", request id: " << req_flag;
        status_vec[i] = s;
        break;
      }
    }
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Read blocks from disk cost: " << (end - start) << " us.";

  MONITOR_START(load_memory_copy_monitor_);
  MONITOR_END(load_memory_copy_monitor_);
  MONITOR_END(load_from_disk_io_monitor_);

  for (size_t i = 0; i < hash_num; ++i) {
    if (!status_vec[i].ok()) {
      continue;
    }
    if (!io_adaptor_vec[i]->Close().ok()) {
      // Data is already loaded, so we can ignore the error.
      // But we should log the error to report resource leak.
      LOG(WARNING) << "Failed to close IOAdaptor, may cause resource leak."
                   << " Request id: " << req_flag;
    }
  }

  return Status::OK();
}

Status VLLMKVStorage::CleanBlockBlobs(Client& client,
                                      std::vector<ObjectMeta> block_meta_vec,
                                      std::string& req_flag) {
  uint64_t start = 0, end = 0;
  std::vector<ObjectID> blob_id_vec;
  uint64_t total_blob = 0;
  std::vector<uint64_t> offsets;
  std::vector<uint64_t> nums_vec;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  for (const auto& block_meta : block_meta_vec) {
    VINEYARD_ASSERT(block_meta.GetTypeName() == type_name<VLLMBlock>(),
                    "Invalid block meta type: " + block_meta.GetTypeName() +
                        " for request: " + req_flag);
    uint64_t nums = block_meta.GetKeyValue<uint64_t>("nums");
    offsets.push_back(total_blob);
    nums_vec.push_back(nums);
    total_blob += nums;
  }
  blob_id_vec.resize(total_blob, InvalidObjectID());

  std::vector<std::future<Status>> decode_statuses;
  for (size_t i = 0; i < block_meta_vec.size(); ++i) {
    decode_statuses.emplace_back(io_thread_pool_->enqueue(
        [&](size_t index) {
          const auto& block_meta = block_meta_vec[index];
          std::string ids_str_encoder =
              block_meta.GetKeyValue<std::string>("blob_ids");
          std::string ids_str = base64_decode(ids_str_encoder);
          if (ids_str.size() != sizeof(ObjectID) * nums_vec[index]) {
            LOG(WARNING) << "Invalid blob ids size: " << ids_str.size()
                         << ", expected: " << sizeof(ObjectID) * nums_vec[index]
                         << ", which means meta has been corrupted."
                         << ", request id: " << req_flag;
          } else {
            memcpy(blob_id_vec.data() + offsets[index], ids_str.data(),
                   sizeof(ObjectID) * nums_vec[index]);
          }
          return Status::OK();
        },
        i));
  }
  for (auto& status : decode_statuses) {
    if (!status.get().ok()) {
      LOG(WARNING) << "Failed to decode blob ids for cleaning block blobs."
                   << ", request id: " << req_flag;
    }
  }

  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". Decode blob ids cost: " << (end - start)
      << " us, total blobs: " << total_blob;
  start = end;
  VINEYARD_CHECK_OK(client.DeleteUserBlobs(blob_id_vec, req_flag));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". Delete blobs cost: " << (end - start)
      << " us, total blobs: " << total_blob;
  return Status::OK();
}

Status VLLMKVStorage::CleanBlockBuilderBlobs(
    Client& client, std::shared_ptr<VLLMBlockBuilder> block_builder,
    std::string& req_flag) {
  std::vector<std::unique_ptr<UserBlobBuilder>>& blobs =
      block_builder->GetBlobs();
  std::vector<ObjectID> blob_ids;
  for (auto& blob : blobs) {
    blob_ids.push_back(blob->id());
  }
  return client.DeleteUserBlobs(blob_ids, req_flag);
}

Status VLLMKVStorage::GetIOAdaptor(
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>& io_adaptor,
    std::string file_name) {
  std::string path_prefix = GetIOPathPrefix();
  if (path_prefix.empty()) {
    return Status::Invalid("VINEYARD_VLLM_KV_CACHE_DISK_PATH is not set");
  }

  io_adaptor = io_adaptor_factory_(path_prefix + "/" + file_name);
  return Status::OK();
}

std::string VLLMKVStorage::GetIOPathPrefix() {
  static std::string path_prefix =
      VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskPath();
  return path_prefix;
}

Status VLLMKVStorage::GetIOTag(std::string& tag) {
  std::string disk_type = VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskType();
  if (disk_type == "cpfs") {
    tag = "cpfs";
  } else {
    return Status::Invalid("Invalid VINEYARD_VLLM_KV_CACHE_DISK_TYPE: " +
                           disk_type);
  }
  return Status::OK();
}

Status VLLMKVStorage::WriteBlockToDisk(
    Client& client, std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor,
    ObjectMeta& meta, std::vector<std::future<Status>>& statuses) {
  json file_meta_json;
  RETURN_ON_ERROR(ConstructVLLMBlockFileMeta(meta, file_meta_json));
  std::string meta_str = file_meta_json.dump();
  size_t meta_size = meta_str.size();
  size_t meta_magic_size =
      std::stoull(VLLMKVCacheEnv::GetVineyardVLLMBlockMetaMagicSize());

  // write meta
  void* meta_ptr = mmap(nullptr, meta_magic_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (meta_ptr == MAP_FAILED) {
    return Status::IOError("Failed to mmap block meta, error: " +
                           std::string(strerror(errno)));
  }
  std::shared_ptr<char[]> meta_ptr_shared(
      reinterpret_cast<char*>(meta_ptr),
      [meta_magic_size](char* p) { munmap(p, meta_magic_size); });
  memcpy(meta_ptr_shared.get(), &meta_size, sizeof(size_t));
  memcpy(meta_ptr_shared.get() + sizeof(size_t), meta_str.c_str(), meta_size);
  RETURN_ON_ERROR(
      io_adaptor->AsyncWrite(meta_ptr_shared, meta_magic_size, 0).get());

  // write data
  uint64_t nums = meta.GetKeyValue<uint64_t>("nums");
  std::string ids_str_encoder = meta.GetKeyValue<std::string>("blob_ids");
  std::string ids_str = base64_decode(ids_str_encoder);
  if (ids_str.size() != sizeof(ObjectID) * nums) {
    return Status::Invalid(
        "Invalid blob ids size: " + std::to_string(ids_str.size()) +
        ", expected: " + std::to_string(sizeof(ObjectID) * nums) +
        ", which means meta has been corrupted.");
  }
  std::vector<ObjectID> blob_ids;
  blob_ids.resize(nums);
  memcpy(blob_ids.data(), ids_str.data(), sizeof(ObjectID) * nums);

  std::vector<std::shared_ptr<UserBlob>> blobs;
  std::vector<void*> write_ptr_vec;
  std::vector<size_t> write_size_vec;
  std::vector<size_t> offset_vec;

  RETURN_ON_ERROR(client.GetUserBlobs(blob_ids, blobs));
  uint64_t offset = meta_magic_size;
  for (size_t i = 0; i < blobs.size(); ++i) {
    std::shared_ptr<UserBlob> blob = blobs[i];
    write_ptr_vec.push_back(
        reinterpret_cast<void*>(blob->offset() + storage_base_pointer_));
    write_size_vec.push_back(blob->size());
    offset_vec.push_back(offset);  // offset is not used in this case
    offset += blob->size();
  }

  if (!io_adaptor->FileTruncate(offset).ok()) {
    LOG(WARNING) << "Failed to truncate file to size: " << offset
                 << ", may cause performance issue.";
  }

  RETURN_ON_ERROR(io_adaptor->BatchAsyncWrite(write_ptr_vec, write_size_vec,
                                              offset_vec, statuses));

  return Status::OK();
}

Status VLLMKVStorage::WriteBlockToDisk(
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor,
    std::vector<uint64_t>& offset_vec, std::vector<size_t>& sizes_vec,
    std::vector<uint64_t>& shape, int layer_index,
    std::vector<std::future<Status>>& statuses) {
  json file_meta_json;
  RETURN_ON_ERROR(ConstructVLLMBlockFileMeta(offset_vec, sizes_vec, shape,
                                             layer_index, file_meta_json));
  std::string meta_str = file_meta_json.dump();
  size_t meta_size = meta_str.size();
  size_t meta_magic_size =
      std::stoull(VLLMKVCacheEnv::GetVineyardVLLMBlockMetaMagicSize());

  // write meta
  void* meta_ptr = mmap(nullptr, meta_magic_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (meta_ptr == MAP_FAILED) {
    return Status::IOError("Failed to mmap block meta, error: " +
                           std::string(strerror(errno)));
  }
  std::shared_ptr<char[]> meta_ptr_shared(
      reinterpret_cast<char*>(meta_ptr),
      [meta_magic_size](char* p) { munmap(p, meta_magic_size); });
  memcpy(meta_ptr_shared.get(), &meta_size, sizeof(size_t));
  memcpy(meta_ptr_shared.get() + sizeof(size_t), meta_str.c_str(), meta_size);
  RETURN_ON_ERROR(
      io_adaptor->AsyncWrite(meta_ptr_shared, meta_magic_size, 0).get());

  // write data
  std::vector<void*> write_ptr_vec;
  std::vector<size_t> write_size_vec;
  std::vector<size_t> file_offset_vec;

  uint64_t offset = meta_magic_size;
  for (size_t i = 0; i < offset_vec.size(); ++i) {
    write_ptr_vec.push_back(
        reinterpret_cast<void*>(offset_vec[i] + storage_base_pointer_));
    write_size_vec.push_back(sizes_vec[i]);
    file_offset_vec.push_back(offset);  // offset is not used in this case
    offset += sizes_vec[i];
  }

  if (!io_adaptor->FileTruncate(offset).ok()) {
    LOG(WARNING) << "Failed to truncate file to size: " << offset
                 << ", may cause performance issue.";
  }

  RETURN_ON_ERROR(io_adaptor->BatchAsyncWrite(write_ptr_vec, write_size_vec,
                                              file_offset_vec, statuses));

  return Status::OK();
}

Status VLLMKVStorage::CopyBlockToMemory(Client& client, ObjectMeta& meta,
                                        std::shared_ptr<char[]>& data_ptr,
                                        size_t& file_size) {
  std::vector<size_t> offsets_vec;
  std::vector<size_t> sizes_vec;
  json file_meta_json;
  RETURN_ON_ERROR(ConstructVLLMBlockFileMeta(meta, file_meta_json));
  std::string meta_str = file_meta_json.dump();

  uint64_t nums = meta.GetKeyValue<uint64_t>("nums");
  std::string ids_str_encoder = meta.GetKeyValue<std::string>("blob_ids");
  std::string ids_str = base64_decode(ids_str_encoder);
  if (ids_str.size() != sizeof(ObjectID) * nums) {
    return Status::Invalid(
        "Invalid blob ids size: " + std::to_string(ids_str.size()) +
        ", expected: " + std::to_string(sizeof(ObjectID) * nums) +
        ", which means meta has been corrupted.");
  }
  std::vector<ObjectID> blob_ids;
  blob_ids.resize(nums);
  memcpy(blob_ids.data(), ids_str.data(), sizeof(ObjectID) * nums);

  std::vector<std::shared_ptr<UserBlob>> blobs;
  RETURN_ON_ERROR(client.GetUserBlobs(blob_ids, blobs));
  for (size_t i = 0; i < blobs.size(); ++i) {
    std::shared_ptr<UserBlob> blob = blobs[i];
    offsets_vec.push_back(blob->offset());
    sizes_vec.push_back(blob->size());
  }

  return CopyBlockToMemoryInternal(meta_str, offsets_vec, sizes_vec, data_ptr,
                                   file_size);
}

Status VLLMKVStorage::CopyBlockToMemoryInternal(
    std::string& meta_str, std::vector<size_t>& offsets_vec,
    std::vector<size_t>& size_vec, std::shared_ptr<char[]>& data_ptr,
    size_t& file_size) {
  size_t meta_size = meta_str.size();
  size_t meta_magic_size =
      std::stoull(VLLMKVCacheEnv::GetVineyardVLLMBlockMetaMagicSize());
  file_size = meta_magic_size;

  for (size_t i = 0; i < size_vec.size(); ++i) {
    file_size += size_vec[i];
  }

  size_t align = std::stoull(VLLMKVCacheEnv::GetDirectIOAlign());
  if (file_size % align != 0) {
    file_size += (align - (file_size % align));
  }
  void* data = mmap(nullptr, file_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (data == MAP_FAILED) {
    return Status::IOError("Failed to mmap block data, error: " +
                           std::string(strerror(errno)));
  }

  data_ptr = std::shared_ptr<char[]>(
      reinterpret_cast<char*>(data), [file_size](char* p) {
        if (munmap(p, file_size) != 0) {
          LOG(ERROR) << "Failed to munmap memory, may cause memory leak.";
        }
      });

  if (data_ptr == nullptr) {
    return Status::IOError("Failed to allocate memory for block data.");
  }

  size_t offset = 0;
  memcpy(data_ptr.get() + offset, &meta_size, sizeof(size_t));
  offset += sizeof(size_t);
  memory::concurrent_memcpy(data_ptr.get() + offset, meta_str.c_str(),
                            meta_size);
  offset += meta_magic_size - sizeof(size_t);

  for (size_t i = 0; i < offsets_vec.size(); ++i) {
    if (offset + size_vec[i] > file_size) {
      return Status::IOError(
          "Write size exceeds file size, invalid block file.");
    }
    memory::concurrent_memcpy(
        data_ptr.get() + offset,
        reinterpret_cast<const void*>(offsets_vec[i] + storage_base_pointer_),
        size_vec[i]);
    offset += size_vec[i];
  }
  return Status::OK();
}

Status VLLMKVStorage::ReadBlockFromDisk(
    std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor,
    std::vector<uint64_t>& offsets_vec, std::vector<size_t>& sizes_vec,
    std::vector<uint64_t>& shape, int layer_index,
    std::vector<std::future<Status>>& statuses) {
  if (sizes_vec.size() == 0) {
    return Status::Invalid("Sizes vector is empty, invalid block file.");
  }
  size_t file_size = 0;
  RETURN_ON_ERROR(io_adaptor->GetFileSize(file_size));
  if (file_size == 0) {
    return Status::Invalid("File size is zero, invalid block file.");
  }

  uint64_t meta_magic_size =
      std::stoull(VLLMKVCacheEnv::GetVineyardVLLMBlockMetaMagicSize());
  void* meta_data = mmap(nullptr, meta_magic_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (meta_data == MAP_FAILED) {
    return Status::IOError("Failed to mmap block meta, error: " +
                           std::string(strerror(errno)));
  }
  std::shared_ptr<char[]> meta_ptr(
      reinterpret_cast<char*>(meta_data),
      [meta_magic_size](char* p) { munmap(p, meta_magic_size); });
  RETURN_ON_ERROR(io_adaptor->AsyncRead(meta_ptr, meta_magic_size, 0).get());
  size_t meta_size = reinterpret_cast<size_t*>(meta_ptr.get())[0];

  std::string meta_str;
  try {
    meta_str = std::string(meta_ptr.get() + sizeof(meta_size), meta_size);
  } catch (const std::exception& e) {
    return Status::IOError("Failed to allocate memory for meta string.");
  }

  json file_meta_json;
  try {
    file_meta_json = json::parse(meta_str);
  } catch (const json::parse_error& e) {
    LOG(ERROR) << "Failed to parse meta json: " << e.what()
               << ", meta_str: " << meta_str;
    return Status::Invalid("Failed to parse meta json: " +
                           std::string(e.what()) + ", meta_str: " + meta_str);
  }
  VLOG(100) << "file_meta_json: " << file_meta_json;

  std::vector<uint64_t> file_sizes_vec;
  std::vector<uint64_t> file_shape;
  size_t blob_nums = 0;
  int file_layer_index = 0;
  RETURN_ON_ERROR(ParseVLLMBlockFileJson(
      file_meta_json, blob_nums, file_sizes_vec, file_shape, file_layer_index));
  if (file_sizes_vec.size() == 0) {
    return Status::Invalid("Blob sizes vector is empty, invalid block file.");
  }

  if (!CheckVLLMBlockEqual(blob_nums, offsets_vec.size(), file_sizes_vec,
                           sizes_vec, file_shape, shape, file_layer_index,
                           layer_index)) {
    std::string error_msg =
        "Block file meta does not match the expected "
        "block structure. "
        "Expected blob nums: " +
        std::to_string(offsets_vec.size()) +
        ", file blob nums: " + std::to_string(blob_nums) +
        ", expected layer index: " + std::to_string(layer_index) +
        ", file layer index: " + std::to_string(file_layer_index);
    error_msg += ", expected sizes: [";
    for (const auto& size : sizes_vec) {
      error_msg += std::to_string(size) + ", ";
    }
    error_msg += "], file sizes: [";
    for (const auto& size : file_sizes_vec) {
      error_msg += std::to_string(size) + ", ";
    }
    error_msg += "]";
    error_msg += ", expected shape: [";
    for (const auto& dim : shape) {
      error_msg += std::to_string(dim) + ", ";
    }
    error_msg += "], file shape: [";
    for (const auto& dim : file_shape) {
      error_msg += std::to_string(dim) + ", ";
    }
    error_msg += "]";
    return Status::Invalid(error_msg);
  }
  std::vector<void*> read_ptr_vec;
  std::vector<size_t> file_offset_vec;
  size_t offset = meta_magic_size;
  for (size_t i = 0; i < blob_nums; i++) {
    read_ptr_vec.push_back(
        reinterpret_cast<void*>(offsets_vec[i] + storage_base_pointer_));
    file_offset_vec.push_back(offset);
    offset += sizes_vec[i];
  }
  // std::vector<std::future<Status>> read_futures;
  Status status = io_adaptor->BatchAsyncRead(read_ptr_vec, sizes_vec,
                                             file_offset_vec, statuses);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to read block from disk, error: "
               << status.ToString();
    return status;
  }

  return Status::OK();
}

Status VLLMKVStorage::ReadBlockFromMemory(std::shared_ptr<char[]> data_ptr,
                                          size_t file_size,
                                          std::vector<uint64_t>& offsets_vec,
                                          std::vector<size_t>& sizes_vec,
                                          std::vector<uint64_t>& shape,
                                          int layer_index) {
  if (sizes_vec.size() == 0) {
    return Status::Invalid("Sizes vector is empty, invalid block file.");
  }
  if (file_size == 0) {
    return Status::Invalid("File size is zero, invalid block file.");
  }

  size_t offset = 0;
  size_t meta_size = 0;
  size_t meta_magic_size =
      std::stoull(VLLMKVCacheEnv::GetVineyardVLLMBlockMetaMagicSize());
  memcpy(&meta_size, data_ptr.get(), sizeof(size_t));
  offset += sizeof(size_t);
  if (meta_size == 0) {
    return Status::Invalid("Meta size is zero, invalid block file.");
  }

  std::shared_ptr<char[]> meta_buffer;
  try {
    meta_buffer = std::shared_ptr<char[]>(new char[meta_size]);
    if (meta_buffer == nullptr) {
      return Status::Invalid("Failed to allocate memory for meta buffer.");
    }
  } catch (std::bad_alloc& e) {
    return Status::Invalid("Failed to allocate memory for meta buffer.");
  }

  memory::concurrent_memcpy(meta_buffer.get(), data_ptr.get() + offset,
                            meta_size);
  offset += meta_magic_size - sizeof(size_t);
  std::string meta_str(meta_buffer.get(), meta_size);
  json file_meta_json;
  try {
    file_meta_json = json::parse(meta_str);
  } catch (const json::parse_error& e) {
    LOG(ERROR) << "Failed to parse meta json: " << e.what()
               << ", meta_str: " << meta_str;
    return Status::Invalid("Failed to parse meta json: " +
                           std::string(e.what()) + ", meta_str: " + meta_str);
  }
  VLOG(100) << "file_meta_json: " << file_meta_json;

  std::vector<uint64_t> file_sizes_vec;
  std::vector<uint64_t> file_shape;
  size_t blob_nums = 0;
  int file_layer_index = 0;
  RETURN_ON_ERROR(ParseVLLMBlockFileJson(
      file_meta_json, blob_nums, file_sizes_vec, file_shape, file_layer_index));
  if (file_sizes_vec.size() == 0) {
    return Status::Invalid("Blob sizes vector is empty, invalid block file.");
  }

  if (!CheckVLLMBlockEqual(blob_nums, offsets_vec.size(), file_sizes_vec,
                           sizes_vec, file_shape, shape, file_layer_index,
                           layer_index)) {
    std::string error_msg =
        "Block file meta does not match the expected "
        "block structure. "
        "Expected blob nums: " +
        std::to_string(offsets_vec.size()) +
        ", file blob nums: " + std::to_string(blob_nums) +
        ", expected layer index: " + std::to_string(layer_index) +
        ", file layer index: " + std::to_string(file_layer_index);
    error_msg += ", expected sizes: [";
    for (const auto& size : sizes_vec) {
      error_msg += std::to_string(size) + ", ";
    }
    error_msg += "], file sizes: [";
    for (const auto& size : file_sizes_vec) {
      error_msg += std::to_string(size) + ", ";
    }
    error_msg += "]";
    error_msg += ", expected shape: [";
    for (const auto& dim : shape) {
      error_msg += std::to_string(dim) + ", ";
    }
    error_msg += "], file shape: [";
    for (const auto& dim : file_shape) {
      error_msg += std::to_string(dim) + ", ";
    }
    error_msg += "]";
    return Status::Invalid(error_msg);
  }

  for (size_t i = 0; i < blob_nums; i++) {
    if (offset + sizes_vec[i] > file_size) {
      return Status::Invalid(
          "Read size exceeds file size, invalid block file.");
    }
    memory::concurrent_memcpy(
        reinterpret_cast<void*>(offsets_vec[i] + storage_base_pointer_),
        reinterpret_cast<void*>(data_ptr.get() + offset), sizes_vec[i]);
    offset += sizes_vec[i];
  }

  return Status::OK();
}

void VLLMKVStorage::Hash2PrefixDirAndSuffixFile(const uint64_t hash_num,
                                                std::string& prefix_dir,
                                                std::string& file_path) {
  std::stringstream ss;
  for (int i = 3; i >= 1; --i) {
    uint16_t part = (hash_num >> (i * 16)) & 0xFFFF;
    ss << std::hex << std::setfill('0') << std::setw(4) << part << "/";
  }
  prefix_dir = ss.str();
  ss << std::hex << std::setfill('0') << std::setw(4) << (hash_num & 0xFFFF);
  file_path = ss.str();
}

size_t VLLMKVStorage::GetIOProcessingRequestNums() {
  static std::shared_ptr<vllm_kv_cache::io::RealAIOOperations> aio_ops =
      std::make_shared<vllm_kv_cache::io::RealAIOOperations>();
  static std::shared_ptr<vllm_kv_cache::io::AIOContext> aio_context =
      vllm_kv_cache::io::AIOContext::GetSingleInstance(aio_ops);
  return aio_context->GetProcessingIORequest();
}

Status VLLMKVStorage::CreateDirectoriesIfNotExists(const std::string& path) {
  try {
    std::filesystem::create_directories(path);
  } catch (const std::filesystem::filesystem_error& e) {
    return Status::IOError("Failed to create directories for path: " + path +
                           ", error: " + e.what());
  }
  return Status::OK();
}

bool CheckVLLMBlockEqual(size_t nums_1, size_t nums_2,
                         std::vector<size_t>& sizes_1,
                         std::vector<size_t>& sizes_2,
                         std::vector<size_t>& shape_1,
                         std::vector<size_t>& shape_2, int layer_index_1,
                         int layer_index_2) {
  if (nums_1 != nums_2 || layer_index_1 != layer_index_2) {
    return false;
  }

  if (sizes_1.size() != sizes_2.size() || shape_1.size() != shape_2.size()) {
    return false;
  }

  for (size_t i = 0; i < sizes_1.size(); ++i) {
    if (sizes_1[i] != sizes_2[i]) {
      return false;
    }
  }

  for (size_t i = 0; i < shape_1.size(); ++i) {
    if (shape_1[i] != shape_2[i]) {
      return false;
    }
  }

  return true;
}

}  // namespace vineyard
