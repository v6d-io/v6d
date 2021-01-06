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

#include "basic/ds/dataframe.h"

namespace vineyard {

class DataFrameBuilder;

const std::vector<json>& DataFrame::Columns() const { return this->columns_; }

std::shared_ptr<ITensor> DataFrame::Index() const {
  return values_.at(INDEX_COL_NAME);
}

std::shared_ptr<ITensor> DataFrame::Column(json const& column) const {
  return values_.at(column);
}

const std::pair<size_t, size_t> DataFrame::partition_index() const {
  return std::make_pair(this->partition_index_row_,
                        this->partition_index_column_);
}

const std::pair<size_t, size_t> DataFrame::shape() const {
  if (values_.empty()) {
    return std::make_pair(0, 0);
  } else {
    return std::make_pair(values_.begin()->second->shape()[0], columns_.size());
  }
}

const std::pair<size_t, size_t> DataFrameBuilder::partition_index() const {
  return std::make_pair(this->partition_index_row_,
                        this->partition_index_column_);
}

void DataFrameBuilder::set_partition_index(size_t partition_index_row,
                                           size_t partition_index_column) {
  this->set_partition_index_row_(partition_index_row);
  this->set_partition_index_column_(partition_index_column);
}

void DataFrameBuilder::set_row_batch_index(size_t row_batch_index) {
  this->set_row_batch_index_(row_batch_index);
}

void DataFrameBuilder::set_index(std::shared_ptr<ITensorBuilder> builder) {
  this->values_.emplace(INDEX_COL_NAME, builder);
}

std::shared_ptr<ITensorBuilder> DataFrameBuilder::Column(
    json const& column) const {
  return values_.at(column);
}

void DataFrameBuilder::AddColumn(json const& column,
                                 std::shared_ptr<ITensorBuilder> builder) {
  this->columns_.emplace_back(column);
  this->values_.emplace(column, builder);
}

void DataFrameBuilder::DropColumn(json const& column) {
  // FIXME: how to ensure the removed builder got destroyed/aborted.
}

Status DataFrameBuilder::Build(Client& client) {
  this->set_columns_(columns_);
  for (auto const& kv : values_) {
    this->set_values_(
        kv.first,
        std::dynamic_pointer_cast<ObjectBuilder>(kv.second)->Seal(client));
  }
  return Status::OK();
}

const std::pair<size_t, size_t> GlobalDataFrame::partition_shape() const {
  return std::make_pair(this->partition_shape_row_,
                        this->partition_shape_column_);
}

const std::vector<std::shared_ptr<Object>>& GlobalDataFrame::LocalPartitions(
    Client& client) const {
  return objects_.ObjectsAt(client.instance_id());
}

const std::vector<std::shared_ptr<Object>>& GlobalDataFrame::LocalPartitions(
    const InstanceID instance_id) const {
  return objects_.ObjectsAt(instance_id);
}

const std::pair<size_t, size_t> GlobalDataFrameBuilder::partition_shape()
    const {
  return std::make_pair(this->partition_shape_row_,
                        this->partition_shape_column_);
}

void GlobalDataFrameBuilder::set_partition_shape(
    size_t partition_shape_row, size_t partition_shape_column) {
  this->set_partition_shape_row_(partition_shape_row);
  this->set_partition_shape_column_(partition_shape_column);
}

void GlobalDataFrameBuilder::AddPartition(const InstanceID instance_id,
                                          ObjectID const partition_id) {
  object_set_builder_.AddObject(instance_id, partition_id);
}

void GlobalDataFrameBuilder::AddPartitions(
    const InstanceID instance_id, const std::vector<ObjectID>& partition_ids) {
  object_set_builder_.AddObjects(instance_id, partition_ids);
}

std::shared_ptr<Object> GlobalDataFrameBuilder::_Seal(Client& client) {
  auto object = GlobalDataFrameBaseBuilder::_Seal(client);
  // Global object will be persist automatically.
  VINEYARD_CHECK_OK(client.Persist(object->id()));
  return object;
}

Status GlobalDataFrameBuilder::Build(Client& client) {
  this->set_objects_(object_set_builder_.Seal(client));
  return Status::OK();
}

}  // namespace vineyard
