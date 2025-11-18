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

#include <bitset>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "client/client.h"
#include "common/util/logging.h"
#include "common/util/sidecar.h"
#include "vllm-kv-cache/ds/vllm_layer.h"
#include "vllm-kv-cache/src/storage/vllm_kv_storage.h"
#include "vllm-kv-cache/src/vllm_kv_cache_util.h"

namespace vineyard {
Status VLLMLayers::FromBlocks(Client& client, std::vector<uint64_t>& block_hash,
                              std::vector<std::vector<uint64_t>>& offsets_vec,
                              std::vector<std::vector<size_t>>& sizes_vec,
                              std::vector<uint64_t>& shape_vec, int layer_index,
                              std::vector<ObjectMeta>& metas,
                              std::string rpc_endpoint,
                              std::shared_ptr<VLLMLayers>& layers,
                              std::string& req_flag) {
  VLOG(2) << "Creating VLLMLayer from blocks, block_hash size: "
          << block_hash.size() << ", offsets_vec size: " << offsets_vec.size()
          << ", sizes_vec size: " << sizes_vec.size()
          << ", request: " << req_flag;
  uint64_t start = 0, end = 0;
  if (block_hash.size() == 0) {
    return Status::OK();
  }
  RETURN_ON_ASSERT(offsets_vec.size() == block_hash.size(),
                   "offsets_vec size not match");
  RETURN_ON_ASSERT(sizes_vec.size() == block_hash.size(),
                   "sizes_vec size not match");
  RETURN_ON_ASSERT(metas.size() == block_hash.size(), "meta size not match");

  // fetch meta data
  std::vector<std::vector<ObjectID>> remote_blobs;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  for (uint64_t i = 0; i < metas.size(); i++) {
    int remote_layer_index = metas[i].GetKeyValue<int>("layer_index");
    RETURN_ON_ASSERT(remote_layer_index == layer_index,
                     "layer index not match");
    std::vector<uint64_t> remote_shape;
    metas[i].GetKeyValue("shape", remote_shape);
    RETURN_ON_ASSERT(remote_shape.size() == shape_vec.size(),
                     "remote shape size not match");
    RETURN_ON_ASSERT(
        std::equal(remote_shape.begin(), remote_shape.end(), shape_vec.begin()),
        "remote shape not match");

    std::vector<uint64_t> remote_size_vec;
    metas[i].GetKeyValue("sizes", remote_size_vec);
    RETURN_ON_ASSERT(remote_size_vec.size() == sizes_vec[i].size(),
                     "remote size of size vector not match");
    RETURN_ON_ASSERT(std::equal(remote_size_vec.begin(), remote_size_vec.end(),
                                sizes_vec[i].begin()),
                     "remote blob size not match");

    std::vector<uint64_t> remote_offsets_vec;
    metas[i].GetKeyValue("offsets", remote_offsets_vec);
    RETURN_ON_ASSERT(remote_offsets_vec.size() == offsets_vec[i].size(),
                     "remote offsets size not match");

    std::vector<ObjectID> remote_blob;
    uint64_t nums = metas[i].GetKeyValue<uint64_t>("nums");
    RETURN_ON_ASSERT(nums == offsets_vec[i].size(),
                     "remote blob nums not match");

    std::string ids_str_encoder = metas[i].GetKeyValue<std::string>("blob_ids");
    std::string ids_str = base64_decode(ids_str_encoder);
    if (ids_str.size() != sizeof(ObjectID) * nums) {
      return Status::Invalid(
          "Invalid blob ids size: " + std::to_string(ids_str.size()) +
          ", expected: " + std::to_string(sizeof(ObjectID) * nums) +
          ", which means meta has been corrupted." + " Request: " + req_flag);
    }
    std::vector<ObjectID> ids;
    ids.resize(nums);
    memcpy(ids.data(), ids_str.data(), ids_str.size());
    for (uint64_t j = 0; j < nums; j++) {
      ObjectID blob_id = ids[j];
      remote_blob.push_back(blob_id);
      VLOG(100) << ObjectIDToString(remote_blob.back());
    }
    remote_blobs.push_back(remote_blob);
  }

  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Fetch remote blob ids cost: " << (end - start) << " us";

  // create layer object
  start = end;
  Status ret = FromBlocksInternal(client, shape_vec, layer_index, offsets_vec,
                                  remote_blobs, sizes_vec, rpc_endpoint,
                                  block_hash, layers, req_flag);
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Create VLLMLayers from blocks cost: " << (end - start) << " us";
  return ret;
}

VLLMLayers::~VLLMLayers() {
  if (recv_flag_mem_ != nullptr) {
    munmap(recv_flag_mem_, GET_BLOB_RECV_MEM_SIZE);
    recv_flag_mem_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
  return;
}

Status VLLMLayers::FromBlocksInternal(
    Client& client, std::vector<uint64_t> shape, int layer_index,
    std::vector<std::vector<size_t>>& local_offset,
    std::vector<std::vector<ObjectID>>& remote_blobs,
    std::vector<std::vector<size_t>>& sizes_vec, std::string rpc_endpoint,
    std::vector<uint64_t>& block_hash, std::shared_ptr<VLLMLayers>& layers,
    std::string& req_flag) {
  VLOG(2) << "Creating VLLMLayer from blocks, local_offset size: "
          << local_offset.size()
          << ", remote_blobs size: " << remote_blobs.size()
          << "hash num:" << block_hash.size() << ", request: " << req_flag;
  uint64_t start = 0, end = 0;
  if (local_offset.size() == 0 || remote_blobs.size() == 0) {
    return Status::OK();
  }
  RETURN_ON_ASSERT(
      local_offset.size() == remote_blobs.size(),
      "local and remote blobs size not match, request:" + req_flag);

  std::vector<std::vector<size_t>> local_layer_offsets;
  std::vector<std::vector<ObjectID>> remote_layer_blobs;
  std::vector<std::vector<size_t>> layer_sizes;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  KVCacheHelper::ShuffleBlockToLayer(local_layer_offsets, remote_layer_blobs,
                                     layer_sizes, local_offset, remote_blobs,
                                     sizes_vec, shape, layer_index);
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Shuffle blocks to layer cost: " << (end - start) << " us";

  RETURN_ON_ERROR(Make(local_layer_offsets, remote_layer_blobs, layer_sizes,
                       shape, layer_index, rpc_endpoint, layers, req_flag));

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  RETURN_ON_ERROR(layers->Transfer(client));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Start transfer VLLMLayers cost: " << (end - start) << " us.";

  layers->block_nums_ = block_hash.size();
  return Status::OK();
}

Status VLLMLayers::IsReceived(int index, bool& received) {
  if (recv_flag_mem_ == nullptr) {
    return Status::IOError(
        "recv_flag_mem_ is not initialized. Please identify if the transfer is "
        "beginning. Request id: " +
        req_flag_);
  }

  unsigned char error_code = reinterpret_cast<unsigned char*>(
      recv_flag_mem_)[GET_BLOB_RECV_MEM_SIZE - sizeof(unsigned char)];
  std::string error_msg(reinterpret_cast<char*>(recv_flag_mem_) +
                            GET_BLOB_RECV_MEM_SIZE - ERROR_MSG_LENGTH -
                            sizeof(unsigned char),
                        ERROR_MSG_LENGTH);
  if (error_code != 0) {
    std::cerr << "Error code: " << static_cast<int>(error_code)
              << ", error message: " << error_msg
              << ", request id: " << req_flag_ << std::endl;
    Status status =
        Status(StatusCode(error_code),
               "Check block received failed. Request id: " + req_flag_ +
                   ", error message: " + error_msg);
    is_transferring_ = false;
    return status;
  }
  if (index == -1) {
    for (int i = 0; i < layer_nums_; ++i) {
      received = true;
      if (reinterpret_cast<char*>(recv_flag_mem_)[i] == 0) {
        received = false;
        break;
      }
    }
    return Status::OK();
  } else if (index >= 0 && index < layer_nums_) {
    received = (reinterpret_cast<char*>(recv_flag_mem_))[index] == 1;
    if (received && index == layer_nums_ - 1) {
      is_finished_ = true;
      is_transferring_ = false;
    }
    return Status::OK();
  } else {
    return Status::Invalid("Index out of range. Request id: " + req_flag_);
  }
  return Status::OK();
}

Status VLLMLayers::PutBlocks(std::vector<Status>& statuses) {
  if (!is_finished_) {
    return Status::Invalid("VLLMLayers transfer is not finished.");
  }
  if (has_put_) {
    return Status::Invalid("Blocks have already been put.");
  }

  // RETURN_ON_ERROR(VLLMKVStorage::PutBlockKVCache(block_hash_,
  //                                                block_builders_, statuses,
  //                                                req_flag_));
  // has_put_ = true;
  // need_to_delete_builder_.clear();
  return Status::OK();
}

Status VLLMLayers::PutAndCleanWithFilter(std::vector<size_t>& filter,
                                         std::vector<Status>& statuses) {
  if (!is_finished_) {
    return Status::Invalid("VLLMLayers transfer is not finished.");
  }
  if (has_put_) {
    return Status::Invalid("Blocks have already been put.");
  }

  // std::set<size_t> filter_set;
  // for (size_t i = 0; i < filter.size(); ++i) {
  //   if (filter[i] > block_builders_.size()) {
  //     return Status::Invalid("Filter index out of range.");
  //   }
  //   if (filter_set.find(filter[i]) != filter_set.end()) {
  //     return Status::Invalid("Filter index duplicated.");
  //   }
  //   auto it = need_to_delete_builder_.find(block_builders_[filter[i]]);
  //   if (it == need_to_delete_builder_.end()) {
  //     return Status::Invalid("Block builder not found.");
  //   }
  //   filter_set.insert(filter[i]);
  // }

  // std::vector<std::shared_ptr<VLLMBlockBuilder>> need_to_put_builders;
  // std::vector<uint64_t> need_to_put_block_hash;
  // for (size_t i = 0; i < filter.size(); ++i) {
  //   need_to_put_builders.push_back(block_builders_[filter[i]]);
  //   need_to_put_block_hash.push_back(block_hash_[filter[i]]);
  // }

  // RETURN_ON_ERROR(VLLMKVStorage::PutBlockKVCache(need_to_put_block_hash,
  //                                                need_to_put_builders,
  //                                                statuses, req_flag_));

  // has_put_ = true;
  // for (size_t i = 0; i < need_to_put_builders.size(); ++i) {
  //   auto it = need_to_delete_builder_.find(need_to_put_builders[i]);
  //   if (it != need_to_delete_builder_.end()) {
  //     need_to_delete_builder_.erase(it);
  //   }
  // }
  // if (!DeleteNotSavedBlocks().ok()) {
  //   LOG(ERROR) << "Failed to delete not saved blocks.";
  // }

  return Status::OK();
}

Status VLLMLayers::DeleteNotSavedBlocks() {
  if (is_transferring_) {
    LOG(WARNING) << "Transfer is still in progress, cannot delete blocks.";
    return Status::Invalid("Transfer is still in progress.");
  }

  // std::vector<std::shared_ptr<VLLMBlockBuilder>>
  // not_sealed_block_builder_vec(
  //     need_to_delete_builder_.begin(), need_to_delete_builder_.end());
  // return VLLMKVStorage::DeleteBlockBuilders(not_sealed_block_builder_vec,
  // req_flag_);
  return Status::OK();
}

Status VLLMLayers::Make(std::vector<std::vector<ObjectID>> local_offsets,
                        std::vector<std::vector<ObjectID>> remote_blobs,
                        std::vector<std::vector<size_t>> sizes_vec,
                        std::vector<uint64_t> shape, int layer_index,
                        std::string rpc_endpoint,
                        std::shared_ptr<VLLMLayers>& layers,
                        std::string& req_flag) {
  layers = std::make_shared<VLLMLayers>();
  layers->local_offset_ = std::move(local_offsets);
  layers->remote_id_ = std::move(remote_blobs);
  layers->rpc_endpoint_ = rpc_endpoint;
  layers->layer_nums_ = layers->local_offset_.size();
  layers->req_flag_ = req_flag;
  layers->sizes_ = std::move(sizes_vec);
  return Status::OK();
}

Status VLLMLayers::Transfer(Client& client) {
  if (is_transferring_) {
    return Status::OK();
  }

  RETURN_ON_ERROR(client.VineyardGetRemoteBlobsWithOffset(
      local_offset_, remote_id_, sizes_, rpc_endpoint_, fd_, req_flag_));
  recv_flag_mem_ = mmap(nullptr, GET_BLOB_RECV_MEM_SIZE, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd_, 0);
  if (recv_flag_mem_ == MAP_FAILED) {
    return Status::IOError("Failed to mmap recv_flag_mem. Request id: " +
                           req_flag_);
  }

  is_transferring_ = true;
  VLOG(2) << "Transfer is beginning, address of recv_flag_mem_: "
          << static_cast<void*>(recv_flag_mem_)
          << ", size: " << GET_BLOB_RECV_MEM_SIZE;
  return Status::OK();
}

void VLLMLayers::Dump() {
  std::cout << "VLLMLayer dump:" << std::endl;
  std::cout << "blob map:" << std::endl;
  for (size_t i = 0; i < local_offset_.size(); i++) {
    std::cout << "layer " << i << ": " << std::endl;
    for (size_t j = 0; j < local_offset_[i].size(); j++) {
      std::cout << "local offsets: " << local_offset_[i][j] << ", "
                << "remote: " << ObjectIDToString(remote_id_[i][j])
                << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "rpc endpoint: " << rpc_endpoint_ << std::endl;
}

}  // namespace vineyard
