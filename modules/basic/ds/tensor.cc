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

#include "basic/ds/tensor.h"  // NOLINT(build/include)

namespace vineyard {

void GlobalTensor::PostConstruct(const ObjectMeta& meta) {
  if (meta.HasKey("shape_")) {
    meta.GetKeyValue("shape_", this->shape_);
  }
  if (meta.HasKey("partition_shape_")) {
    meta.GetKeyValue("partition_shape_", this->partition_shape_);
  }
}

std::vector<int64_t> const& GlobalTensor::shape() const { return shape_; }

std::vector<int64_t> const& GlobalTensor::partition_shape() const {
  return partition_shape_;
}

const std::vector<std::shared_ptr<ITensor>> GlobalTensor::LocalPartitions(
    Client& client) const {
  std::vector<std::shared_ptr<ITensor>> local_chunks;
  for (auto iter = LocalBegin(); iter != LocalEnd(); iter.NextLocal()) {
    local_chunks.emplace_back(*iter);
  }
  return local_chunks;
}

std::vector<int64_t> const& GlobalTensorBuilder::partition_shape() const {
  return this->partition_shape_;
}

void GlobalTensorBuilder::set_partition_shape(
    std::vector<int64_t> const& partition_shape) {
  this->partition_shape_ = partition_shape;
  this->AddKeyValue("partition_shape_", partition_shape);
}

std::vector<int64_t> const& GlobalTensorBuilder::shape() const {
  return this->shape_;
}

void GlobalTensorBuilder::set_shape(std::vector<int64_t> const& shape) {
  this->shape_ = shape;
  this->AddKeyValue("shape_", shape);
}

void GlobalTensorBuilder::AddPartition(const ObjectID partition_id) {
  this->AddMember(partition_id);
}

void GlobalTensorBuilder::AddPartitions(
    const std::vector<ObjectID>& partition_ids) {
  this->AddMembers(partition_ids);
}

}  // namespace vineyard
