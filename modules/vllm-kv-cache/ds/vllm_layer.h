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

#ifndef MODULES_VLLM_KV_CACHE_DS_VLLM_LAYER_H_
#define MODULES_VLLM_KV_CACHE_DS_VLLM_LAYER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/status.h"
#include "vllm-kv-cache/ds/vllm_block.h"

namespace vineyard {

class VLLMLayers {
 public:
  VLLMLayers() = default;

  ~VLLMLayers();

  static Status FromBlocks(Client& client, std::vector<uint64_t>& block_hash,
                           std::vector<std::vector<uint64_t>>& offsets_vec,
                           std::vector<std::vector<size_t>>& sizes_vec,
                           std::vector<uint64_t>& shape_vec, int layer_index,
                           std::vector<ObjectMeta>& metas,
                           std::string rpc_endpoint,
                           std::shared_ptr<VLLMLayers>& layers,
                           std::string& req_flag);

  Status IsReceived(int index, bool& received);

  int GetLayerNum() const { return layer_nums_; }

  size_t GetBlockNum() const { return block_nums_; }

  Status PutBlocks(std::vector<Status>& statuses);

  // Put blocks with filter, and delete not saved blocks.
  // If status is not OK, the block will not be deleted.
  Status PutAndCleanWithFilter(std::vector<size_t>& filter,
                               std::vector<Status>& statuses);

  Status DeleteNotSavedBlocks();

  void Dump();

  const std::vector<uint64_t>& GetBlockHashes() const { return block_hash_; }

 private:
  static Status FromBlocksInternal(
      Client& client, std::vector<uint64_t> shape, int layer_index,
      std::vector<std::vector<size_t>>& local_offset,
      std::vector<std::vector<ObjectID>>& remote_blobs,
      std::vector<std::vector<size_t>>& sizes_vec, std::string rpc_endpoint,
      std::vector<uint64_t>& block_hash, std::shared_ptr<VLLMLayers>& layers,
      std::string& req_flag);

  static Status Make(std::vector<std::vector<size_t>> local_offsets,
                     std::vector<std::vector<ObjectID>> remote_blobs,
                     std::vector<std::vector<size_t>> sizes_vec,
                     std::vector<uint64_t> shape, int layer_index,
                     std::string rpc_endpoint,
                     std::shared_ptr<VLLMLayers>& layer, std::string& req_flag);

  Status Transfer(Client& client);

  std::vector<uint64_t> block_hash_;
  uint64_t block_nums_ = 0;

  bool is_transferring_ = false;
  bool is_finished_ = false;
  bool has_put_ = false;
  std::vector<std::vector<size_t>> local_offset_;
  std::vector<std::vector<ObjectID>> remote_id_;
  std::vector<std::vector<size_t>> sizes_;
  int layer_nums_ = 0;
  std::string rpc_endpoint_;
  int fd_ = -1;
  void* recv_flag_mem_ = nullptr;
  std::string req_flag_ = "";
};

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_DS_VLLM_LAYER_H_
