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

#include <utility>

#include "client/client.h"
#include "client/ds/blob.h"
#include "common/memory/memcpy.h"
#include "common/util/logging.h"
#include "vllm-kv-cache/ds/vllm_block.h"
#include "vllm-kv-cache/src/vllm_kv_cache_util.h"

namespace vineyard {

void VLLMBlock::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  if (meta_.GetTypeName() != type_name<VLLMBlock>()) {
    return;
  }
  meta_.GetKeyValue("shape", this->shape_);
  uint64_t nums = meta_.GetKeyValue<uint64_t>("nums");
  meta_.GetKeyValue("offsets", this->offsets_);
  meta_.GetKeyValue("sizes", this->sizes_);
  meta_.GetKeyValue<int>("layer_index", this->layer_index_);
  std::string ids_str_encode = meta_.GetKeyValue<std::string>("blob_ids");
  std::string ids_str = base64_decode(ids_str_encode);
  if (ids_str.size() != sizeof(ObjectID) * nums) {
    LOG(WARNING) << "Invalid blob ids size: " << ids_str.size()
                 << ", expected: " << sizeof(ObjectID) * nums
                 << ", which means meta has been corrupted.";
    return;
  }
  blobs_.resize(nums);
  memory::concurrent_memcpy(blobs_.data(), ids_str.data(), ids_str.size());
}

Status VLLMBlock::FromBuilder(VLLMBlockBuilder& builder) {
  this->shape_ = builder.GetShape();
  this->offsets_ = builder.GetOffsets();
  this->sizes_ = builder.GetSizes();
  this->layer_index_ = builder.GetLayerIndex();
  const std::vector<ObjectID> blob_ids = builder.GetBlobIDs();
  this->blobs_.resize(blob_ids.size());
  memory::concurrent_memcpy(this->blobs_.data(), blob_ids.data(),
                            sizeof(ObjectID) * blob_ids.size());
  return Status::OK();
}

void VLLMBlock::Dump() {
  std::cout << "VLLMBlock dump:" << std::endl;
  std::cout << "id:" << ObjectIDToString(id_) << std::endl;
  std::cout << "shape:"
            << "[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i];
    if (i != shape_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "offsets:"
            << "[";
  for (size_t i = 0; i < offsets_.size(); ++i) {
    std::cout << offsets_[i];
    if (i != offsets_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "sizes:"
            << "[";
  for (size_t i = 0; i < sizes_.size(); ++i) {
    std::cout << sizes_[i];
    if (i != sizes_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "blobs:"
            << "[";
  for (size_t i = 0; i < blobs_.size(); ++i) {
    std::cout << ObjectIDToString(blobs_[i]);
    if (i != blobs_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
}

// TBD: batch create
Status VLLMBlockBuilder::Make(Client& client, std::vector<uint64_t> offsets,
                              std::vector<size_t> sizes,
                              std::vector<uint64_t> shape, int layer_index,
                              std::shared_ptr<VLLMBlockBuilder>& builder,
                              std::string& req_flag) {
  builder = std::make_shared<VLLMBlockBuilder>();
  builder->shape_ = shape;
  builder->offsets_ = offsets;
  builder->sizes_ = sizes;
  builder->layer_index_ = layer_index;

  RETURN_ON_ERROR(
      client.CreateUserBlobs(offsets, sizes, builder->blobs_, req_flag));
  builder->blob_ids_.resize(builder->blobs_.size());
  for (size_t i = 0; i < builder->blobs_.size(); ++i) {
    builder->blob_ids_[i] = builder->blobs_[i]->id();
  }
  return Status::OK();
}

Status VLLMBlockBuilder::Make(
    Client& client, std::vector<std::vector<uint64_t>>& offsets_vec,
    std::vector<std::vector<size_t>>& sizes_vec, std::vector<uint64_t>& shape,
    int layer_index,
    std::vector<std::shared_ptr<VLLMBlockBuilder>>& builder_vec,
    std::string& req_flag) {
  size_t num = offsets_vec.size();
  builder_vec.resize(num);
  std::vector<uint64_t> offsets;
  std::vector<size_t> sizes;
  std::vector<std::unique_ptr<UserBlobBuilder>> blob_writers;
  uint64_t start = 0, end = 0;
  for (size_t i = 0; i < num; ++i) {
    builder_vec[i] = std::make_shared<VLLMBlockBuilder>();
    builder_vec[i]->shape_ = shape;
    builder_vec[i]->offsets_ = offsets_vec[i];
    builder_vec[i]->sizes_ = sizes_vec[i];
    builder_vec[i]->layer_index_ = layer_index;

    for (size_t j = 0; j < offsets_vec[i].size(); ++j) {
      offsets.push_back(offsets_vec[i][j]);
      sizes.push_back(sizes_vec[i][j]);
    }
  }
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  RETURN_ON_ERROR(
      client.CreateUserBlobs(offsets, sizes, blob_writers, req_flag));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". Create " << blob_writers.size()
      << " user blobs cost: " << (end - start) << " us";

  for (size_t i = 0; i < num; ++i) {
    builder_vec[i]->blobs_.resize(blob_writers.size() / num);
    builder_vec[i]->blob_ids_.resize(blob_writers.size() / num);
    for (size_t j = 0; j < blob_writers.size() / num; ++j) {
      builder_vec[i]->blobs_[j] =
          std::move(blob_writers[i * (blob_writers.size() / num) + j]);
      builder_vec[i]->blob_ids_[j] = builder_vec[i]->blobs_[j]->id();
    }
  }

  return Status::OK();
}

Status VLLMBlockBuilder::Build(Client& client) { return Status::OK(); }

Status VLLMBlockBuilder::_Seal(Client& client,
                               std::shared_ptr<Object>& object) {
  RETURN_ON_ERROR(Build(client));
  std::shared_ptr<VLLMBlock> block = std::make_shared<VLLMBlock>();
  std::vector<ObjectID> blob_ids;
  blob_ids.reserve(blobs_.size());
  for (size_t i = 0; i < blobs_.size(); ++i) {
    std::shared_ptr<Object> blob;
    RETURN_ON_ERROR(blobs_[i]->Seal(client, blob));
    blob_ids.push_back(blob->id());
  }

  ObjectMeta meta;
  ConstructVLLMBlockMeta(blob_ids, blobs_.size(), shape_, offsets_, sizes_,
                         layer_index_, meta);
  meta.SetId(InvalidObjectID());
  block->Object::Construct(meta);
  block->Construct(meta);

  RETURN_ON_ERROR(client.CreateMetaData(block->meta_, block->id_));
  Status status = client.Persist(block->id_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to persist block: " << status.ToString();
    client.DelData(block->id_);
    return status;
  }
  object = std::dynamic_pointer_cast<Object>(block);

  return Status::OK();
}

Status VLLMBlockBuilder::BatchSeal(
    Client& client, std::vector<std::shared_ptr<VLLMBlockBuilder>>& builders,
    std::vector<std::shared_ptr<VLLMBlock>>& objects,
    std::vector<ObjectID>& ids, std::string& req_flag) {
  std::vector<std::vector<ObjectID>> blob_ids;
  std::vector<ObjectMeta> metas;

  uint64_t start = 0, end = 0;
  metas.resize(builders.size());
  std::vector<std::future<Status>> results;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  for (size_t i = 0; i < builders.size(); ++i) {
    results.emplace_back(KVCacheHelper::GetConstructThreadPool()->enqueue(
        [&](size_t index) {
          ConstructVLLMBlockMeta(
              builders[index]->blob_ids_, builders[index]->blobs_.size(),
              builders[index]->shape_, builders[index]->offsets_,
              builders[index]->sizes_, builders[index]->layer_index_,
              metas[index]);
          return Status::OK();
        },
        i));
  }
  for (auto& result : results) {
    uint64_t index = 0;
    if (!result.get().ok()) {
      LOG(WARNING) << "Failed to construct VLLMBlock meta, block index: "
                   << index << ", request: " << req_flag;
    }
    index++;
  }

  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Batch construct VLLMBlock builder meta use:" << (end - start)
      << " us";

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  RETURN_ON_ERROR(client.CreateHugeMetaData(metas, ids, req_flag));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag << ". CreateHugeMetaData use:" << (end - start)
      << " us";

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  objects.reserve(builders.size());
  for (auto& builder : builders) {
    std::shared_ptr<VLLMBlock> block = std::make_shared<VLLMBlock>();
    block->FromBuilder(*builder);
    objects.push_back(block);
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(KVCacheHelper::GetTraceLogLevel())
      << "Request: " << req_flag
      << ". Create VLLMBlock meta use:" << (end - start) << " us";

  return Status::OK();
}

void VLLMBlockBuilder::Dump() {
  std::cout << "VLLMBlockBuilder dump:" << std::endl;
  std::cout << "shape:"
            << "[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i];
    if (i != shape_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "offsets:"
            << "[";
  for (size_t i = 0; i < offsets_.size(); ++i) {
    std::cout << offsets_[i];
    if (i != offsets_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "sizes:"
            << "[";
  for (size_t i = 0; i < sizes_.size(); ++i) {
    std::cout << sizes_[i];
    if (i != sizes_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "blobs:"
            << "[";
  for (size_t i = 0; i < blob_ids_.size(); ++i) {
    std::cout << ObjectIDToString(blob_ids_[i]);
    if (i != blob_ids_.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]" << std::endl;
}

void ConstructVLLMBlockMeta(std::vector<ObjectID>& blob_ids, size_t nums,
                            std::vector<size_t>& shape,
                            std::vector<uint64_t>& offsets,
                            std::vector<size_t>& sizes, int layer_index,
                            ObjectMeta& meta) {
  std::string ids_str = std::string(blob_ids.size() * sizeof(ObjectID), 0);
  memcpy(ids_str.data(), blob_ids.data(), blob_ids.size() * sizeof(ObjectID));
  std::string ids_str_encode = base64_encode(ids_str);
  meta.AddKeyValue("nums", nums);
  meta.AddKeyValue("shape", shape);
  meta.AddKeyValue("offsets", offsets);
  meta.AddKeyValue("sizes", sizes);
  meta.AddKeyValue("layer_index", layer_index);
  meta.AddKeyValue("blob_ids", ids_str_encode);
  meta.SetTypeName(type_name<VLLMBlock>());
}

Status ConstructVLLMBlockFileMeta(ObjectMeta& meta, json& file_meta_json) {
  std::string type;
  std::vector<size_t> shape;
  std::vector<size_t> sizes;
  int layer_index = -1;
  uint64_t nums = 0;
  type = meta.GetTypeName();
  nums = meta.GetKeyValue<uint64_t>("nums");
  meta.GetKeyValue("shape", shape);
  meta.GetKeyValue("sizes", sizes);
  meta.GetKeyValue<int>("layer_index", layer_index);
  meta.GetKeyValue<uint64_t>("nums", nums);

  file_meta_json["type"] = type;
  file_meta_json["shape"] = shape;
  file_meta_json["sizes"] = sizes;
  file_meta_json["layer_index"] = layer_index;
  file_meta_json["nums"] = nums;

  return Status::OK();
}

Status ConstructVLLMBlockFileMeta(std::vector<size_t>& offsets,
                                  std::vector<size_t>& sizes,
                                  std::vector<uint64_t>& shape, int layer_index,
                                  json& file_meta_json) {
  std::string type;
  type = type_name<VLLMBlock>();
  uint64_t nums = offsets.size();

  file_meta_json["type"] = type;
  file_meta_json["shape"] = shape;
  file_meta_json["sizes"] = sizes;
  file_meta_json["layer_index"] = layer_index;
  file_meta_json["nums"] = nums;

  return Status::OK();
}

Status ParseVLLMBlockFileJson(json& block_meta_json, size_t& nums,
                              std::vector<size_t>& sizes,
                              std::vector<size_t>& shape, int& layer_index) {
  if (unlikely(!block_meta_json.contains("type") ||
               !block_meta_json.contains("shape") ||
               !block_meta_json.contains("sizes") ||
               !block_meta_json.contains("layer_index") ||
               !block_meta_json.contains("nums"))) {
    return Status::Invalid(
        "Invalid VLLMBlock metadata, missing required fields.");
  }

  if (block_meta_json["type"] != type_name<VLLMBlock>()) {
    return Status::Invalid("Invalid VLLMBlock metadata, type mismatch.");
  }

  try {
    nums = block_meta_json["nums"].get<size_t>();
    shape = block_meta_json["shape"].get<std::vector<size_t>>();
    sizes = block_meta_json["sizes"].get<std::vector<size_t>>();
    layer_index = block_meta_json["layer_index"].get<int>();
  } catch (const std::exception& e) {
    return Status::Invalid("Failed to parse VLLMBlock metadata: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

}  // namespace vineyard
