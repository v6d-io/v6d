/** Copyright 2020-2021 Alibaba Group Holding Limited.

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
  return values_.at("index_");
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

const std::shared_ptr<arrow::RecordBatch> DataFrame::AsBatch(bool copy) const {
  size_t num_columns = this->Columns().size();
  int64_t num_rows = 0;
  std::vector<std::shared_ptr<arrow::Array>> columns(num_columns);
  std::vector<std::shared_ptr<arrow::Field>> fields(num_columns);
  for (size_t i = 0; i < num_columns; ++i) {
    // cast to arrow::Array
    auto cname = this->Columns()[i];
    std::string field_name;
    if (cname.is_string()) {
      field_name = cname.get_ref<std::string const&>();
    } else {
      field_name = json_to_string(cname);
    }
    auto df_col = this->Column(cname);
    num_rows = df_col->shape()[0];

    if (auto tensor = std::dynamic_pointer_cast<Tensor<int32_t>>(df_col)) {
      num_rows = tensor->shape()[0];
    } else if (auto tensor =
                   std::dynamic_pointer_cast<Tensor<uint32_t>>(df_col)) {
      num_rows = tensor->shape()[0];
    } else if (auto tensor =
                   std::dynamic_pointer_cast<Tensor<int64_t>>(df_col)) {
      num_rows = tensor->shape()[0];
    } else if (auto tensor =
                   std::dynamic_pointer_cast<Tensor<uint64_t>>(df_col)) {
      num_rows = tensor->shape()[0];
    } else if (auto tensor = std::dynamic_pointer_cast<Tensor<float>>(df_col)) {
      num_rows = tensor->shape()[0];
    } else if (auto tensor =
                   std::dynamic_pointer_cast<Tensor<double>>(df_col)) {
      num_rows = tensor->shape()[0];
    }

    std::shared_ptr<arrow::Buffer> copied_buffer;
    if (copy) {
      CHECK_ARROW_ERROR_AND_ASSIGN(
          copied_buffer,
          df_col->buffer()->CopySlice(0, df_col->buffer()->size()));
    } else {
      copied_buffer = df_col->buffer();
    }

    columns[i] = arrow::MakeArray(arrow::ArrayData::Make(
        FromAnyType(df_col->value_type()), num_rows, {nullptr, copied_buffer}));
    fields[i] = std::make_shared<arrow::Field>(
        field_name, FromAnyType(df_col->value_type()));
  }
  return arrow::RecordBatch::Make(arrow::schema(fields), num_rows, columns);
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
  this->values_.emplace("index_", builder);
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

void GlobalDataFrame::Construct(const ObjectMeta& meta) {
  std::string __type_name = type_name<GlobalDataFrame>();
  VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                  "Expect typename '" + __type_name + "', but got '" +
                      meta.GetTypeName() + "'");
  this->meta_ = meta;
  this->id_ = meta.GetId();

  meta.GetKeyValue("partition_shape_row_", this->partition_shape_row_);
  meta.GetKeyValue("partition_shape_column_", this->partition_shape_column_);

  for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("partitions_-size");
       ++__idx) {
    auto chunk = std::dynamic_pointer_cast<DataFrame>(
        meta.GetMember("partitions_-" + std::to_string(__idx)));
    this->partitions_[chunk->meta().GetInstanceId()].emplace_back(chunk);
  }

  if (meta.IsLocal()) {
    this->PostConstruct(meta);
  }
}

const std::pair<size_t, size_t> GlobalDataFrame::partition_shape() const {
  return std::make_pair(this->partition_shape_row_,
                        this->partition_shape_column_);
}

const std::vector<std::shared_ptr<DataFrame>>& GlobalDataFrame::LocalPartitions(
    Client& client) const {
  return partitions_[client.instance_id()];
}

const std::vector<std::shared_ptr<DataFrame>>& GlobalDataFrame::LocalPartitions(
    const InstanceID instance_id) const {
  return partitions_[instance_id];
}

std::shared_ptr<Object> GlobalDataFrameBaseBuilder::_Seal(Client& client) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  VINEYARD_DISCARD(client.SyncMetaData());

  VINEYARD_CHECK_OK(this->Build(client));
  auto __value = std::make_shared<GlobalDataFrame>();

  size_t __value_nbytes = 0;

  __value->meta_.SetTypeName(type_name<GlobalDataFrame>());
  if (std::is_base_of<GlobalObject, GlobalDataFrame>::value) {
    __value->meta_.SetGlobal(true);
  }

  __value->partition_shape_row_ = partition_shape_row_;
  __value->meta_.AddKeyValue("partition_shape_row_",
                             __value->partition_shape_row_);

  __value->partition_shape_column_ = partition_shape_column_;
  __value->meta_.AddKeyValue("partition_shape_column_",
                             __value->partition_shape_column_);

  size_t __partitions__idx = 0;
  for (auto& __partitions__value : partitions_) {
    auto __value_partitions_ = client.GetObject<DataFrame>(__partitions__value);
    __value->partitions_[__value_partitions_->meta().GetInstanceId()]
        .emplace_back(__value_partitions_);
    __value->meta_.AddMember("partitions_-" + std::to_string(__partitions__idx),
                             __partitions__value);
    __value_nbytes += __value_partitions_->nbytes();
    __partitions__idx += 1;
  }
  __value->meta_.AddKeyValue("partitions_-size", __value->partitions_.size());

  __value->meta_.SetNBytes(__value_nbytes);

  VINEYARD_CHECK_OK(client.CreateMetaData(__value->meta_, __value->id_));

  // mark the builder as sealed
  this->set_sealed(true);

  return std::static_pointer_cast<Object>(__value);
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

void GlobalDataFrameBuilder::AddPartition(ObjectID const partition_id) {
  this->add_partitions_(partition_id);
}

void GlobalDataFrameBuilder::AddPartitions(
    const std::vector<ObjectID>& partition_ids) {
  for (auto const& partition_id : partition_ids) {
    this->add_partitions_(partition_id);
  }
}

std::shared_ptr<Object> GlobalDataFrameBuilder::_Seal(Client& client) {
  auto object = GlobalDataFrameBaseBuilder::_Seal(client);
  // Global object will be persist automatically.
  VINEYARD_CHECK_OK(client.Persist(object->id()));
  return object;
}

Status GlobalDataFrameBuilder::Build(Client& client) { return Status::OK(); }

}  // namespace vineyard
