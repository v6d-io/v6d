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

#ifndef MODULES_VLLM_KV_CACHE_DS_VLLM_BLOCK_H_
#define MODULES_VLLM_KV_CACHE_DS_VLLM_BLOCK_H_

#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/remote_blob.h"

namespace vineyard {

class VLLMBlockBuilder;

class VLLMBlock : public vineyard::Registered<VLLMBlock> {
 public:
  VLLMBlock() = default;

  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::unique_ptr<Object>(new VLLMBlock());
  }

  void Construct(const ObjectMeta& meta) override;

  void Dump();

  std::vector<ObjectID> GetBlobs() { return blobs_; }

  std::vector<uint64_t> GetOffsets() { return offsets_; }

  std::vector<uint64_t> GetSizes() { return sizes_; }

  std::vector<uint64_t> GetShape() { return shape_; }

  int GetLayerIndex() { return layer_index_; }

  Status FromBuilder(VLLMBlockBuilder& builder);

 private:
  // FIXME: assume the shape is layer * kv * block_id
  std::vector<ObjectID> blobs_;
  std::vector<uint64_t> offsets_;
  std::vector<uint64_t> sizes_;
  std::vector<uint64_t> shape_;
  int layer_index_ = -1;

  friend class VLLMBlockBuilder;
};

class VLLMBlockBuilder : public vineyard::ObjectBuilder {
 public:
  VLLMBlockBuilder() {}

  ~VLLMBlockBuilder() {}

  static Status Make(Client& client, std::vector<uint64_t> offsets,
                     std::vector<size_t> sizes, std::vector<uint64_t> shape,
                     int layer_index,
                     std::shared_ptr<VLLMBlockBuilder>& builder,
                     std::string& req_flag);

  static Status Make(Client& client,
                     std::vector<std::vector<uint64_t>>& offsets,
                     std::vector<std::vector<size_t>>& sizes,
                     std::vector<uint64_t>& shape, int layer_index,
                     std::vector<std::shared_ptr<VLLMBlockBuilder>>& builder,
                     std::string& req_flag);

  static Status BatchSeal(
      Client& client, std::vector<std::shared_ptr<VLLMBlockBuilder>>& builders,
      std::vector<std::shared_ptr<VLLMBlock>>& objects,
      std::vector<ObjectID>& ids, std::string& req_flag);

  Status Build(Client& client) override;

  Status _Seal(Client& client, std::shared_ptr<Object>& object) override;

  const std::vector<ObjectID>& GetBlobIDs() { return blob_ids_; }

  std::vector<std::unique_ptr<UserBlobBuilder>>& GetBlobs() { return blobs_; }

  std::vector<uint64_t> GetOffsets() { return offsets_; }

  std::vector<uint64_t> GetSizes() { return sizes_; }

  std::vector<uint64_t> GetShape() { return shape_; }

  int GetLayerIndex() { return layer_index_; }

  void Dump();

 private:
  std::vector<std::unique_ptr<UserBlobBuilder>> blobs_;
  std::vector<ObjectID> blob_ids_;
  std::vector<uint64_t> offsets_;
  std::vector<uint64_t> sizes_;
  // FIXME: assume the shape is layer * kv * block_id
  std::vector<uint64_t> shape_;
  int layer_index_ = -1;

  friend class VLLMLayer;
};

void ConstructVLLMBlockMeta(std::vector<ObjectID>& blob_ids, size_t nums,
                            std::vector<size_t>& shape,
                            std::vector<uint64_t>& offsets,
                            std::vector<size_t>& sizes, int layer_index,
                            ObjectMeta& meta);

Status ConstructVLLMBlockFileMeta(ObjectMeta& meta, json& file_meta_json);

Status ConstructVLLMBlockFileMeta(std::vector<size_t>& offsets,
                                  std::vector<size_t>& sizes,
                                  std::vector<uint64_t>& shape, int layer_index,
                                  json& file_meta_json);

Status ParseVLLMBlockFileJson(json& block_meta_json, size_t& nums,
                              std::vector<size_t>& sizes,
                              std::vector<size_t>& shape, int& layer_index);

bool CheckVLLMBlockEqual(size_t nums_1, size_t nums_2,
                         std::vector<size_t>& sizes_1,
                         std::vector<size_t>& sizes_2,
                         std::vector<size_t>& shape_1,
                         std::vector<size_t>& shape_2, int layer_index_1,
                         int layer_index_2);
}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_DS_VLLM_BLOCK_H_
