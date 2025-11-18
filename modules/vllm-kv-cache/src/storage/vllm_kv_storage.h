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

#ifndef MODULES_VLLM_KV_CACHE_SRC_STORAGE_VLLM_KV_STORAGE_H_
#define MODULES_VLLM_KV_CACHE_SRC_STORAGE_VLLM_KV_STORAGE_H_

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "client/client.h"
#include "common/util/monitor.h"
#include "common/util/status.h"
#include "vllm-kv-cache/ds/vllm_block.h"
#include "vllm-kv-cache/ds/vllm_layer.h"
#include "vllm-kv-cache/src/env.h"
#include "vllm-kv-cache/src/io/io_adaptor.h"

#include "thread-pool/thread_pool.h"

namespace vineyard {

class VLLMKVStorage {
 public:
  VLLMKVStorage() = default;

  ~VLLMKVStorage() = default;

  static Status InitStorage(
      uint64_t base_pointer, std::string ipc_socket,
      std::string io_type = "aio",
      bool enable_mem_copy =
          VLLMKVCacheEnv::VineyardEnableVLLMKVCacheMemCopy() == "1",
      bool direct_io = VLLMKVCacheEnv::VineyardEnableVLLMKVCacheDirectIO() ==
                       "1");

  static Status SetStorageBasePointer(uint64_t base_pointer) {
    storage_base_pointer_ = base_pointer;
    return Status::OK();
  }

  /**
   * @brief Get the block location from vineyard.
   *
   * @param client The vineyard client.
   * @param block_hash The vector of block hashes.
   * @param locations The map to store the locations of the blocks.
   * The key is the block name and the value is a set of locations (e.g., IPs).
   *
   * @return Status indicating success or failure of the operation.
   */
  static Status GetBlockLocation(std::vector<uint64_t>& block_hash,
                                 std::vector<std::set<std::string>>& locations,
                                 std::string req_flag = "");

  /**
   * @brief Get the block kv cache object from vineyard with layerwise transfer.
   *
   * @param client The vineyard client.
   * @param block_hash_vec The vector of block hashes.
   * @param offsets_vec The vector of offsets for each block buffer.
   * @param sizes_vec The vector of sizes for each block buffer.
   * @param shape The shape of the block.
   * @param layer_index The index of the layer in the shape.
   * @param block_builders The vector of shared pointers to VLLMBlockBuilder
   * objects (Created by vineyard).
   * @param layers The shared pointer to VLLMLayers object.
   * @param rpc_endpoint The RPC endpoint to fetch the blocks.
   * @param block_nums The number of blocks fetched.
   * @return Status indicating success or failure of the operation.
   */
  static Status GetBlockKVCacheLayerwise(
      std::vector<uint64_t>& block_hash_vec,
      std::vector<std::vector<uint64_t>>& offsets_vec,
      std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
      int layer_index, std::string rpc_endpoint,
      std::shared_ptr<VLLMLayers>& layers, std::string req_flag = "");

  /**
   * @brief Put the block kv cache object to vineyard.
   *
   * @param client The vineyard client.
   * @param block_hash_vec The vector of block hashes.
   * @param offsets_vec The vector of offsets for each block buffer.
   * @param sizes_vec The vector of sizes for each block buffer.
   * @param shape The shape of the block.
   * @param layer_index The index of the layer in the shape.
   * @param statuses A map to store the status of each block put operation.
   *
   * @return Status indicating success or failure of the operation.
   */
  static Status PutBlockKVCache(std::vector<uint64_t>& block_hash_vec,
                                std::vector<std::vector<uint64_t>>& offsets_vec,
                                std::vector<std::vector<size_t>>& sizes_vec,
                                std::vector<uint64_t>& shape, int layer_index,
                                std::vector<Status>& statuses,
                                std::string req_flag = "");

  /**
   * @brief Delete blocks from vineyard.
   *
   * @param client The vineyard client.
   * @param block_hash_vec The vector of block hashes to
   * delete.
   *
   * @return Status indicating success or failure of the operation.
   */
  static Status DeleteBlocks(std::vector<uint64_t> block_hash_vec,
                             std::string req_flag = "");

  /**
   * @brief Save blocks to disk.
   *
   * @param client The vineyard client.
   * @param block_hash_vec The vector of block hashes to save.
   *
   * @return Status indicating success or failure of the operation.
   */
  static Status SaveToDisk(std::vector<uint64_t>& block_hash_vec,
                           std::vector<std::vector<uint64_t>>& offsets_vec,
                           std::vector<std::vector<size_t>>& sizes_vec,
                           std::vector<uint64_t>& shape, int layer_index,
                           std::vector<Status>& statuses, uint64_t ttl,
                           bool wait = true, std::string req_flag = "");

  static Status LoadFromDisk(std::vector<uint64_t>& block_hash_vec,
                             std::vector<std::vector<uint64_t>>& offsets_vec,
                             std::vector<std::vector<size_t>>& sizes_vec,
                             std::vector<uint64_t>& shape, int layer_index,
                             std::vector<Status>& statuses,
                             std::string req_flag = "");

  static void DumpMonitor() {
    DUMP_MONITOR_HEADER();
    DUMP_MONITOR(load_from_disk_io_monitor_);
    DUMP_MONITOR(load_memory_copy_monitor_);
    DUMP_MONITOR(load_from_disk_monitor_);
    DUMP_MONITOR(save_to_disk_io_monitor_);
    DUMP_MONITOR(save_to_disk_monitor_);
    DUMP_MONITOR(save_memory_copy_monitor_);
  }

  static void InitMonitor() {
    MONITOR_CLEAR(load_from_disk_io_monitor_, "LoadFromDisk_IO",
                  monitor::MILLISECONDS);
    MONITOR_CLEAR(load_memory_copy_monitor_, "LoadMemoryCopy",
                  monitor::MILLISECONDS);
    MONITOR_CLEAR(load_from_disk_monitor_, "LoadFromDisk",
                  monitor::MILLISECONDS);
    MONITOR_CLEAR(save_to_disk_io_monitor_, "SaveToDisk_IO",
                  monitor::MILLISECONDS);
    MONITOR_CLEAR(save_to_disk_monitor_, "SaveToDisk", monitor::MILLISECONDS);
    MONITOR_CLEAR(save_memory_copy_monitor_, "SaveMemoryCopy",
                  monitor::MILLISECONDS);
  }

  static void Hash2PrefixDirAndSuffixFile(const uint64_t hash_num,
                                          std::string& prefix_dir,
                                          std::string& file_path);

  static size_t GetIOProcessingRequestNums();

 public:
  // for mock test
  static Status SetClientVec(std::vector<std::shared_ptr<Client>> clients,
                             uint64_t threads) {
    threads_ = threads / 2;
    if (io_thread_pool_ == nullptr || copy_thread_pool_ == nullptr ||
        req_thread_vec_.size() == 0) {
      io_thread_pool_ = std::make_shared<ThreadPool>(threads_);
      copy_thread_pool_ = std::make_shared<ThreadPool>(threads_);
      fast_opt_thread_pool_ = std::make_shared<ThreadPool>(threads_);
      block_opt_thread_pool_ = std::make_shared<ThreadPool>(threads_);
      for (uint64_t i = 0; i < threads_; ++i) {
        req_thread_vec_.emplace_back(
            std::make_shared<ThreadPool>(1));  // single thread pool
      }
    }
    vineyard_clients_ = clients;
    return Status::OK();
  }

 private:
  static Status PutBlockKVCache(Client& client,
                                std::vector<uint64_t>& block_hash_vec,
                                std::vector<std::vector<uint64_t>>& offsets_vec,
                                std::vector<std::vector<size_t>>& sizes_vec,
                                std::vector<uint64_t>& shape, int layer_index,
                                std::vector<Status>& statuses,
                                std::string& req_flag);

  static Status PutBlockKVCache(Client& client,
                                std::vector<uint64_t>& block_hash_vec,
                                std::vector<std::vector<uint64_t>>& offsets_vec,
                                std::vector<std::vector<size_t>>& sizes_vec,
                                std::vector<uint64_t>& shape, int layer_index,
                                std::vector<std::shared_ptr<VLLMBlock>>& blocks,
                                std::vector<Status>& statuses,
                                std::string& req_flag);

  static Status GetBlockKVCacheLayerwise(
      Client& client, std::vector<uint64_t>& block_hash_vec,
      std::vector<std::vector<uint64_t>>& offsets_vec,
      std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
      int layer_index, std::string rpc_endpoint,
      std::shared_ptr<VLLMLayers>& layers, std::string req_flag);

  static Status GetBlockLocation(Client& client,
                                 std::vector<uint64_t>& block_hash,
                                 std::vector<std::set<std::string>>& locations,
                                 std::string& req_flag);

  static Status DeleteBlocks(Client& client,
                             std::vector<uint64_t> block_hash_vec,
                             std::string& req_flag);
  /**
   * @brief Put the block kv cache object to vineyard.
   *
   * @param client The vineyard client.
   * @param block_hash_vec The vector of block hashes.
   * @param block_builders The vector of shared pointers to VLLMBlockBuilder
   * objects (Provided by the user).
   * @param statuses A map to store the status of each block put operation.
   *
   * @return Status indicating success or failure of the operation.
   */
  static Status PutBlockKVCache(
      std::vector<uint64_t>& block_hash_vec,
      std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
      std::vector<Status>& statuses, std::string& req_flag);

  static Status PutBlockKVCache(
      Client& client, std::vector<uint64_t>& block_hash_vec,
      std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
      std::vector<Status>& statuses, std::string& req_flag);

  static Status DeleteBlocks(Client& client,
                             std::vector<std::shared_ptr<VLLMBlock>>& blocks,
                             std::string& req_flag);

  static Status DeleteBlockBuilders(
      std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
      std::string& req_flag);

  static Status DeleteBlockBuilders(
      Client& client,
      std::vector<std::shared_ptr<VLLMBlockBuilder>>& block_builders,
      std::string& req_flag);

  static Status CleanBlockBlobs(Client& client,
                                std::vector<ObjectMeta> block_meta_vec,
                                std::string& req_flag);

  static Status CleanBlockBuilderBlobs(
      Client& client, std::shared_ptr<VLLMBlockBuilder> block_builder,
      std::string& req_flag);

  static Status GetIOAdaptor(
      std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>& io_adaptor,
      std::string file_name);

  static std::string GetIOPathPrefix();

  static Status GetIOTag(std::string& tag);

  static Status FilterFiles(std::vector<uint64_t>& block_hash_vec,
                            std::vector<uint64_t>& exist_block_vec,
                            std::vector<uint64_t>& filtered_hash_vec,
                            std::vector<uint64_t>& filtered_hash_index);

  static Status UpdateTTL(Client& client, std::vector<uint64_t>& block_hash_vec,
                          uint64_t ttl, std::string& req_flag);

  static Status SaveToDiskWithCopy(
      Client& client, std::vector<uint64_t>& block_hash_vec,
      std::vector<std::vector<uint64_t>>& offsets_vec,
      std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
      int layer_index, std::vector<Status>& statuses, uint64_t ttl, bool wait,
      std::string& req_flag);

  // without get meta from v6d
  static Status SaveToDiskWithoutCopy(
      Client& client, std::vector<uint64_t>& block_hash_vec,
      std::vector<std::vector<uint64_t>>& offsets_vec,
      std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
      int layer_index, std::vector<Status>& statuses, uint64_t ttl,
      std::string& req_flag);

  static Status SaveToDiskSubmitIO(
      Client& client,
      std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>>&
          io_adaptor_vec,
      std::vector<size_t>& file_size_vec,
      std::vector<std::shared_ptr<char[]>>& data_ptr_vec,
      std::vector<uint64_t>& block_hash_vec, std::vector<Status>& statuses,
      std::string req_flag);

  static Status SaveToDiskMoveFile(
      Client& client, std::vector<uint64_t>& block_hash_vec,
      std::vector<std::shared_ptr<vllm_kv_cache::io::IIOAdaptor>>&
          io_adaptor_vec,
      std::vector<std::string>& file_name_vec,
      std::vector<std::string>& tmp_file_name_vec,
      std::vector<Status>& status_vec, uint64_t ttl, std::string req_flag);

  // without put to v6d
  static Status LoadFromDiskWithCopy(
      std::vector<uint64_t>& block_hash_vec,
      std::vector<std::vector<uint64_t>>& offsets_vec,
      std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
      int layer_index, std::vector<Status>& statuses, std::string& req_flag);

  // without put to v6d
  static Status LoadFromDiskWithoutCopy(
      std::vector<uint64_t>& block_hash_vec,
      std::vector<std::vector<uint64_t>>& offsets_vec,
      std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
      int layer_index, std::vector<Status>& statuses, std::string& req_flag);

  static Status CopyBlockToMemory(Client& client, ObjectMeta& meta,
                                  std::shared_ptr<char[]>& data_ptr,
                                  size_t& file_size);

  static Status CopyBlockToMemoryInternal(std::string& meta_str,
                                          std::vector<size_t>& offsets_vec,
                                          std::vector<size_t>& sizes_vec,
                                          std::shared_ptr<char[]>& data_ptr,
                                          size_t& file_size);

  static Status WriteBlockToDisk(
      Client& client, std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor,
      ObjectMeta& meta, std::vector<std::future<Status>>& statuses);

  static Status WriteBlockToDisk(
      std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor,
      std::vector<uint64_t>& offset_vec, std::vector<size_t>& sizes_vec,
      std::vector<uint64_t>& shape, int layer_index,
      std::vector<std::future<Status>>& statuses);

  static Status ReadBlockFromDisk(
      std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> io_adaptor,
      std::vector<uint64_t>& offset_vec, std::vector<size_t>& size_vec,
      std::vector<uint64_t>& shape, int layer_index,
      std::vector<std::future<Status>>& statuses);

  // for memcpy
  static Status ReadBlockFromMemory(std::shared_ptr<char[]> data_ptr,
                                    size_t file_size,
                                    std::vector<uint64_t>& offsets_vec,
                                    std::vector<size_t>& sizes_vec,
                                    std::vector<uint64_t>& shape,
                                    int layer_index);

  static Status CreateDirectoriesIfNotExists(const std::string& path);

  static std::atomic<uint64_t> req_count_;
  static uint64_t threads_;
  static uint64_t storage_base_pointer_;

  static std::vector<std::shared_ptr<Client>> vineyard_clients_;
  static std::vector<std::shared_ptr<ThreadPool>> req_thread_vec_;
  static std::shared_ptr<ThreadPool> io_thread_pool_;
  static std::shared_ptr<ThreadPool> copy_thread_pool_;
  static std::shared_ptr<ThreadPool> fast_opt_thread_pool_;
  static std::shared_ptr<ThreadPool> block_opt_thread_pool_;
  static bool use_copy_;
  static bool direct_io_;

  static monitor::Monitor load_from_disk_io_monitor_;
  static monitor::Monitor load_memory_copy_monitor_;
  static monitor::Monitor load_from_disk_monitor_;
  static monitor::Monitor save_to_disk_io_monitor_;
  static monitor::Monitor save_to_disk_monitor_;
  static monitor::Monitor save_memory_copy_monitor_;

  static vllm_kv_cache::io::IOAdaptorFactory io_adaptor_factory_;

  friend class VLLMLayers;
};

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_STORAGE_VLLM_KV_STORAGE_H_
