/** Copyright 2020 Alibaba Group Holding Limited.

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

#include "basic/ds/tensor.h"

namespace vineyard {

std::vector<int64_t> const& GlobalTensor::shape() const { return shape_; }

std::vector<int64_t> const& GlobalTensor::partition_shape() const {
  return partition_shape_;
}

const std::vector<std::shared_ptr<Object>>& GlobalTensor::LocalPartitions(
    Client& client) const {
  return partitions_.ObjectsAt(client.instance_id());
}

const std::vector<std::shared_ptr<Object>>& GlobalTensor::LocalPartitions(
    const InstanceID instance_id) const {
  return partitions_.ObjectsAt(instance_id);
}

std::vector<int64_t> const& GlobalTensorBuilder::partition_shape() const {
  return this->partition_shape_;
}

void GlobalTensorBuilder::set_partition_shape(
    std::vector<int64_t> const& partition_shape) {
  this->set_partition_shape_(partition_shape);
}

std::vector<int64_t> const& GlobalTensorBuilder::shape() const {
  return this->shape_;
}

void GlobalTensorBuilder::set_shape(std::vector<int64_t> const& shape) {
  this->set_shape_(shape);
}

void GlobalTensorBuilder::AddPartition(const InstanceID instance_id,
                                       const ObjectID partition_id) {
  partitions_builder_.AddObject(instance_id, partition_id);
}

void GlobalTensorBuilder::AddPartitions(
    const InstanceID instance_id, const std::vector<ObjectID>& partition_ids) {
  partitions_builder_.AddObjects(instance_id, partition_ids);
}

std::shared_ptr<Object> GlobalTensorBuilder::_Seal(Client& client) {
  auto object = GlobalTensorBaseBuilder::_Seal(client);
  // Global object will be persist automatically.
  VINEYARD_CHECK_OK(client.Persist(object->id()));
  return object;
}

Status GlobalTensorBuilder::Build(Client& client) {
  this->set_partitions_(partitions_builder_.Seal(client));
  return Status::OK();
}

}  // namespace vineyard
