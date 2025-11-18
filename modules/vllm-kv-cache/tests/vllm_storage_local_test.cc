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

#include <filesystem>
#include <queue>

#include "client/client.h"
#include "common/util/logging.h"
#include "vllm-kv-cache/ds/vllm_block.h"
#include "vllm-kv-cache/ds/vllm_layer.h"
#include "vllm-kv-cache/src/storage/vllm_kv_storage.h"

using namespace vineyard;  // NOLINT(build/namespaces)

std::string ipc_socket;

int fd = -1;
uint64_t base = 0;
size_t map_size = 0;
size_t memory_size = 0;

// Model config, here just use a fake config for testing
uint64_t layer_num = 1;
uint64_t kv_num = 2;
uint64_t buffer_num = 1;
uint64_t buffer_size = 128 * 1024;  // 128KB
std::vector<uint64_t> shape = {layer_num, kv_num,
                               buffer_num};  // layer, kv, buffers
int layer_index = 0;  // means that the layer index in shape

Client client;

struct ModelConfig {
  uint64_t layer_num;
  uint64_t kv_num;
  uint64_t buffer_num;
  uint64_t buffer_size;
  std::vector<uint64_t> shape;
  int layer_index;
};

struct VLLMBLock_ {
  uint64_t block_hash_;
  std::vector<uint64_t> offsets_;
  std::vector<size_t> sizes_;
  std::vector<uint64_t> shape_ = shape;
  int layer_index_ = layer_index;
};

// Simulate a VLLMBLockAllocator for testing.
class FakeVLLMBLockAllocator {
 public:
  FakeVLLMBLockAllocator() = default;

  // Init the whole memory region and separate into blocks.
  // For testing, we just create not more than 1000 blocks.
  void Init(void* memory_addr, size_t size, ModelConfig model_config) {
    LOG(INFO) << "FakeVLLMBLockAllocator Init called with config:"
              << " layer_num=" << model_config.layer_num
              << ", kv_num=" << model_config.kv_num
              << ", buffer_num=" << model_config.buffer_num
              << ", buffer_size=" << model_config.buffer_size;
    memory_addr_ = memory_addr;
    size_ = size;
    model_config_ = model_config;
    uint64_t block_size = model_config_.layer_num * model_config_.kv_num *
                          model_config_.buffer_num * model_config_.buffer_size;
    num_blocks_ = size_ / block_size;
    num_blocks_ =
        std::min(num_blocks_,
                 static_cast<uint64_t>(1000));  // limit to 1k blocks for fake

    LOG(INFO) << "Calculating " << num_blocks_ << " blocks for memory size "
              << size_ << " with block size " << block_size << ".";
    LOG(INFO) << "Buffer num per blocks:"
              << model_config_.buffer_num * model_config_.kv_num *
                     model_config_.layer_num
              << ", each buffer size: " << model_config_.buffer_size;

    uint64_t per_layer_size = model_config_.kv_num * model_config_.buffer_num *
                              model_config_.buffer_size;
    uint64_t per_kv_size = model_config_.buffer_num * model_config_.buffer_size;
    LOG(INFO) << "Per layer size: " << per_layer_size
              << ", per kv size: " << per_kv_size;
    for (uint64_t i = 0; i < num_blocks_; ++i) {
      VLLMBLock_ block;
      // calculate offsets and sizes
      // whole memory laout is layer * kv * block * buffer
      for (uint64_t l = 0; l < model_config_.layer_num; ++l) {
        for (uint64_t k = 0; k < model_config_.kv_num; ++k) {
          for (uint64_t b = 0; b < model_config_.buffer_num; ++b) {
            uint64_t offset = i * block_size + l * per_layer_size +
                              k * per_kv_size + b * model_config_.buffer_size;
            block.offsets_.push_back(offset);
            block.sizes_.push_back(model_config_.buffer_size);
          }
        }
      }
      free_block_queue_.push(block);
    }
    LOG(INFO) << "FakeVLLMBLockAllocator initialized with " << num_blocks_
              << " blocks.";
  }

  // Release all resources.
  void Release() {
    LOG(INFO) << "FakeVLLMBLockAllocator Release called.";
    memory_addr_ = nullptr;
    size_ = 0;
    model_config_ = ModelConfig();
    num_blocks_ = 0;
    while (!free_block_queue_.empty()) {
      free_block_queue_.pop();
    }
    LOG(INFO) << "FakeVLLMBLockAllocator released.";
  }

  // Allocate blocks from the free queue.
  Status AllocateBlocks(uint64_t block_nums, std::vector<VLLMBLock_>& blocks) {
    LOG(INFO) << "AllocateBlocks called for " << block_nums << " blocks.";
    if (block_nums > free_block_queue_.size()) {
      LOG(ERROR) << "Not enough blocks to allocate, requested: " << block_nums
                 << ", available: " << (num_blocks_ - free_block_queue_.size());
      return Status::Invalid("Not enough blocks to allocate");
    }

    while (block_nums > 0) {
      VLLMBLock_ block;
      if (!free_block_queue_.empty()) {
        block = free_block_queue_.front();
        block.block_hash_ = block_nums;  // just a fake hash
        free_block_queue_.pop();
      } else {
        return Status::Invalid("No more free blocks available");
      }
      blocks.push_back(block);
      --block_nums;
    }
    LOG(INFO) << "Allocated " << blocks.size() << " blocks.";
    return Status::OK();
  }

  // Free blocks back to the free queue.
  Status FreeBlocks(std::vector<VLLMBLock_>& blocks) {
    LOG(INFO) << "FreeBlocks called for " << blocks.size() << " blocks.";
    while (!blocks.empty()) {
      free_block_queue_.push(blocks.back());
      blocks.pop_back();
    }
    LOG(INFO) << "Freed blocks.";
    return Status::OK();
  }

 private:
  void* memory_addr_;
  size_t size_;
  ModelConfig model_config_;
  uint64_t num_blocks_;
  std::queue<VLLMBLock_> free_block_queue_;

  std::vector<uint64_t> layer_offsets;  // size equals to layer_num, means the
                                        // offsets of each layer in the buffer
  std::vector<uint64_t> kv_offsets;     // size equals to kv_num
};

std::shared_ptr<FakeVLLMBLockAllocator> fake_block_allocator;

/*
 * Initialize the VLLMKVStorage with mmaped memory. Then initialize the
 * FakeVLLMBLockAllocator. Allocator will manage the mmap memory region
 * so that when data swap to host memory, it can be seened by vineyardd,
 * which means that vineyardd can send the data to other vineyardd with
 * out extra copy.
 */
void init() {
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;
  uint64_t offset;
  VINEYARD_CHECK_OK(client.GetVineyardMmapFd(fd, map_size, offset));
  LOG(INFO) << "Mmaped fd: " << fd << ", length: " << map_size
            << ", offset: " << offset;
  base = reinterpret_cast<uint64_t>(
      mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (base == reinterpret_cast<uint64_t>(MAP_FAILED)) {
    LOG(ERROR) << "Failed to mmap received fd as a writable buffer: "
               << strerror(errno);
    throw std::runtime_error("Failed to mmap memory");
  }

  base = reinterpret_cast<uint64_t>(base) + offset;  // Adjust base address
  Status status =
      VLLMKVStorage::InitStorage(base, ipc_socket, "aio", true, false);
  if (!status.ok()) {
    LOG(ERROR) << "InitStorage failed: " << status;
    exit(-1);
  }
  memory_size = map_size - offset;
  LOG(INFO) << "Mmaped fd: " << fd << ", length: " << map_size
            << ", free size: " << memory_size << ", offset: " << offset
            << ", base address: " << std::hex << base;

  fake_block_allocator = std::make_shared<FakeVLLMBLockAllocator>();
  fake_block_allocator->Init(reinterpret_cast<void*>(base), memory_size,
                             ModelConfig{layer_num, kv_num, buffer_num,
                                         buffer_size, shape, layer_index});

  VLLMKVStorage::InitStorage(base, ipc_socket);
  LOG(INFO) << "VLLMKVStorage initialized.";
}

/*
 * This function simulates swapping data from GPU to host memory.
 * It fills the allocated blocks with fake data for testing.
 */
Status fake_swap_from_gpu(std::vector<VLLMBLock_>& blocks) {
  LOG(INFO) << "Fake swap from GPU for " << blocks.size() << " blocks.";
  for (auto& block : blocks) {
    for (auto offset : block.offsets_) {
      LOG(INFO) << "Filling fake data at offset: " << offset
                << ", size: " << buffer_size;
      uint8_t* ptr = reinterpret_cast<uint8_t*>(base + offset);
      for (size_t i = 0; i < buffer_size; ++i) {
        ptr[i] = static_cast<uint8_t>(i);  // fake data
      }
    }
  }
  return Status::OK();
}

/*
 * This function show how to save blocks to vineyardd via VLLMKVStorage.
 * When a put operation is returned ok, it means that vineyardd can access
 * the data in the mmaped memory region directly.
 */
Status save_to_v6d(std::vector<VLLMBLock_>& blocks) {
  LOG(INFO) << "Save to v6d for " << blocks.size() << " blocks.";
  std::vector<uint64_t> block_hash_vec;
  std::vector<std::vector<uint64_t>> offsets_vec;
  std::vector<std::vector<size_t>> sizes_vec;

  for (auto& block : blocks) {
    block_hash_vec.push_back(block.block_hash_);
    offsets_vec.push_back(block.offsets_);
    sizes_vec.push_back(block.sizes_);
  }

  std::vector<Status> statuses;
  VINEYARD_CHECK_OK(VLLMKVStorage::PutBlockKVCache(
      block_hash_vec, offsets_vec, sizes_vec, shape, layer_index, statuses,
      "test-request"));
  for (const auto& status : statuses) {
    if (!status.ok()) {
      LOG(ERROR) << "Failed to put block to v6d, error: " << status.ToString();
      return status;
    }
  }

  return Status::OK();
}

/*
 * This function show how to delete blocks from vineyardd via VLLMKVStorage.
 * Because the memory management is at client side, this API just detele
 * the metadata in vineyardd. After delete, vineyardd do not have the block
 * info so that this block is unvisible to other vineyardd.
 */
Status delete_from_v6d(std::vector<uint64_t>& block_hashes) {
  return VLLMKVStorage::DeleteBlocks(block_hashes, "test-delete-request");
}

// This function show how to save blocks to disk via VLLMKVStorage.
Status save_to_disk(std::vector<VLLMBLock_>& blocks) {
  LOG(INFO) << "Save to disk for " << blocks.size() << " blocks.";
  std::vector<uint64_t> block_hash_vec;
  std::vector<std::vector<uint64_t>> offsets_vec;
  std::vector<std::vector<size_t>> sizes_vec;

  for (auto& block : blocks) {
    block_hash_vec.push_back(block.block_hash_);
    offsets_vec.push_back(block.offsets_);
    sizes_vec.push_back(block.sizes_);
  }

  std::vector<Status> statuses;
  VINEYARD_CHECK_OK(VLLMKVStorage::SaveToDisk(
      block_hash_vec, offsets_vec, sizes_vec, shape, layer_index, statuses, 5,
      true, "test-disk-request"));
  for (const auto& status : statuses) {
    if (!status.ok()) {
      LOG(ERROR) << "Failed to put block to disk, error: " << status.ToString();
      return status;
    }
  }

  return Status::OK();
}

// This function show how to load blocks from disk via VLLMKVStorage.
Status load_from_disk(std::vector<uint64_t>& block_hashes,
                      std::vector<VLLMBLock_>& blocks) {
  LOG(INFO) << "Load from disk for " << block_hashes.size() << " blocks.";
  RETURN_ON_ASSERT(block_hashes.size() == blocks.size(),
                   "block_hashes.size() and blocks.size() must be equal");
  std::vector<std::vector<uint64_t>> offsets_vec;
  std::vector<std::vector<size_t>> sizes_vec;

  for (size_t i = 0; i < block_hashes.size(); ++i) {
    std::vector<uint64_t> offsets = blocks[i].offsets_;
    std::vector<size_t> sizes = blocks[i].sizes_;
    offsets_vec.push_back(offsets);
    sizes_vec.push_back(sizes);
  }
  std::vector<Status> statuses;
  VINEYARD_CHECK_OK(VLLMKVStorage::LoadFromDisk(
      block_hashes, offsets_vec, sizes_vec, shape, layer_index, statuses,
      "test-load-disk-request"));
  for (size_t i = 0; i < statuses.size(); ++i) {
    if (!statuses[i].ok()) {
      LOG(ERROR) << "Failed to load block from disk, error: "
                 << statuses[i].ToString();
      return statuses[i];
    }
  }

  return Status::OK();
}

Status check_blocks_equal(std::vector<VLLMBLock_>& blocks1,
                          std::vector<VLLMBLock_>& blocks2) {
  RETURN_ON_ASSERT(blocks1.size() == blocks2.size(),
                   "blocks1.size() and blocks2.size() must be equal");
  for (size_t i = 0; i < blocks1.size(); ++i) {
    auto& block1 = blocks1[i];
    auto& block2 = blocks2[i];
    RETURN_ON_ASSERT(block1.block_hash_ == block2.block_hash_,
                     "block hashes are not equal");
    RETURN_ON_ASSERT(block1.offsets_.size() == block2.offsets_.size(),
                     "offsets sizes are not equal");
    RETURN_ON_ASSERT(block1.sizes_.size() == block2.sizes_.size(),
                     "sizes sizes are not equal");
    for (size_t j = 0; j < block1.offsets_.size(); ++j) {
      uint8_t* ptr1 = reinterpret_cast<uint8_t*>(base + block1.offsets_[j]);
      uint8_t* ptr2 = reinterpret_cast<uint8_t*>(base + block2.offsets_[j]);
      for (size_t k = 0; k < block1.sizes_[j]; ++k) {
        RETURN_ON_ASSERT(
            ptr1[k] == ptr2[k],
            "block data are not equal at block " + std::to_string(i) +
                ", buffer " + std::to_string(j) + ", byte " +
                std::to_string(k) + ". Block 1 data is " +
                std::to_string(static_cast<int>(ptr1[k])) +
                ", Block 2 data is " +
                std::to_string(static_cast<int>(ptr2[k])) +
                ". Block 1 offset is " + std::to_string(block1.offsets_[j]) +
                ", Block 2 offset is " + std::to_string(block2.offsets_[j]));
      }
    }
  }
  return Status::OK();
}

int main(int argc, char** argv) {
  ipc_socket = std::string("/tmp/vineyard1.sock");
  std::string disk_file_path = "/tmp/vllm_kv_cache_disk_dir";
  if (setenv("VINEYARD_VLLM_KV_CACHE_DISK_PATH", disk_file_path.c_str(), 1) !=
      0) {
    LOG(ERROR) << "Failed to set VINEYARD_VLLM_KV_CACHE_DISK_PATH environment "
                  "variable";
    return -1;
  } else {
    LOG(INFO) << "Set VINEYARD_VLLM_KV_CACHE_DISK_PATH to " << disk_file_path
              << ", read env:"
              << VLLMKVCacheEnv::GetVineyardVLLMKVCacheDiskPath();
    if (!std::filesystem::exists(disk_file_path)) {
      std::filesystem::create_directories(disk_file_path);
    }
  }

  uint64_t num_blocks = 4;
  init();

  std::vector<VLLMBLock_> blocks;
  fake_block_allocator->AllocateBlocks(num_blocks, blocks);

  VINEYARD_CHECK_OK(fake_swap_from_gpu(blocks));
  save_to_v6d(blocks);

  std::vector<uint64_t> block_hashes;
  for (auto& block : blocks) {
    block_hashes.push_back(block.block_hash_);
  }
  VINEYARD_CHECK_OK(delete_from_v6d(block_hashes));

  VINEYARD_CHECK_OK(save_to_disk(blocks));

  std::vector<VLLMBLock_> loaded_blocks;
  VINEYARD_CHECK_OK(
      fake_block_allocator->AllocateBlocks(num_blocks, loaded_blocks));
  VINEYARD_CHECK_OK(load_from_disk(block_hashes, loaded_blocks));

  VINEYARD_CHECK_OK(check_blocks_equal(blocks, loaded_blocks));

  LOG(INFO) << "All tests passed!";

  fake_block_allocator->FreeBlocks(blocks);
  fake_block_allocator->FreeBlocks(loaded_blocks);

  fake_block_allocator->Release();

  if (munmap(reinterpret_cast<void*>(base), map_size) != 0) {
    LOG(ERROR) << "Failed to munmap memory: " << strerror(errno);
    return -1;
  }

  client.Disconnect();

  std::filesystem::remove_all(disk_file_path);

  return 0;
}
