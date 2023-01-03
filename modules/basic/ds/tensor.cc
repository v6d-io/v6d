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

#include "basic/ds/tensor.h"

namespace vineyard {

std::vector<int64_t> const& GlobalTensor::shape() const { return shape_; }

std::vector<int64_t> const& GlobalTensor::partition_shape() const {
  return partition_shape_;
}

void GlobalTensor::Construct(const ObjectMeta& meta) {
  std::string __type_name = type_name<GlobalTensor>();
  VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                  "Expect typename '" + __type_name + "', but got '" +
                      meta.GetTypeName() + "'");
  this->meta_ = meta;
  this->id_ = meta.GetId();

  if (meta.Haskey("shape_")) {
    meta.GetKeyValue("shape_", this->shape_);
  }
  if (meta.Haskey("partition_shape_")) {
    meta.GetKeyValue("partition_shape_", this->partition_shape_);
  }
  for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("partitions_-size");
       ++__idx) {
    auto chunk = std::dynamic_pointer_cast<ITensor>(
        meta.GetMember("partitions_-" + std::to_string(__idx)));
    this->partitions_[chunk->meta().GetInstanceId()].emplace_back(chunk);
  }

  if (meta.IsLocal()) {
    this->PostConstruct(meta);
  }
}

const std::vector<std::shared_ptr<ITensor>>& GlobalTensor::LocalPartitions(
    Client& client) const {
  return partitions_[client.instance_id()];
}

const std::vector<std::shared_ptr<ITensor>>& GlobalTensor::LocalPartitions(
    const InstanceID instance_id) const {
  return partitions_[instance_id];
}

std::shared_ptr<Object> GlobalTensorBaseBuilder::_Seal(Client& client) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  VINEYARD_DISCARD(client.SyncMetaData());

  VINEYARD_CHECK_OK(this->Build(client));
  auto __value = std::make_shared<GlobalTensor>();

  size_t __value_nbytes = 0;

  __value->meta_.SetTypeName(type_name<GlobalTensor>());
  if (std::is_base_of<GlobalObject, GlobalTensor>::value) {
    __value->meta_.SetGlobal(true);
  }

  __value->shape_ = shape_;
  __value->meta_.AddKeyValue("shape_", __value->shape_);

  __value->partition_shape_ = partition_shape_;
  __value->meta_.AddKeyValue("partition_shape_", __value->partition_shape_);

  size_t __partitions__idx = 0;
  for (auto& __partitions__value : partitions_) {
    auto __value_partitions_ = client.GetObject<ITensor>(__partitions__value);
    __value->partitions_[__value_partitions_->meta().GetInstanceId()]
        .emplace_back(__value_partitions_);
    __value->meta_.AddMember("partitions_-" + std::to_string(__partitions__idx),
                             __partitions__value);
    __value_nbytes += __value_partitions_->nbytes();
    __partitions__idx += 1;
  }
  __value->meta_.AddKeyValue("partitions_-size", partitions_.size());

  __value->meta_.SetNBytes(__value_nbytes);

  VINEYARD_CHECK_OK(client.CreateMetaData(__value->meta_, __value->id_));

  // mark the builder as sealed
  this->set_sealed(true);

  // run `PostConstruct` to return a valid object
  __value->PostConstruct(__value->meta_);

  return std::static_pointer_cast<Object>(__value);
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

void GlobalTensorBuilder::AddPartition(const ObjectID partition_id) {
  this->add_partitions_(partition_id);
}

void GlobalTensorBuilder::AddPartitions(
    const std::vector<ObjectID>& partition_ids) {
  for (auto const& partition_id : partition_ids) {
    this->add_partitions_(partition_id);
  }
}

std::shared_ptr<Object> GlobalTensorBuilder::_Seal(Client& client) {
  auto object = GlobalTensorBaseBuilder::_Seal(client);
  // Global object will be persist automatically.
  VINEYARD_CHECK_OK(client.Persist(object->id()));
  return object;
}

Status GlobalTensorBuilder::Build(Client& client) { return Status::OK(); }

}  // namespace vineyard
