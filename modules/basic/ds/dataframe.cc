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

#include "basic/ds/dataframe.h"  // NOLINT(build/include)

#include <string>

#include "common/util/logging.h"  // IWYU pragma: keep

namespace vineyard {

class DataFrameBuilder;

DataFrameBuilder::DataFrameBuilder(Client& client)
    : DataFrameBaseBuilder(client) {}

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
    } else if (auto tensor =
                   std::dynamic_pointer_cast<Tensor<std::string>>(df_col)) {
      num_rows = tensor->shape()[0];
    } else if (auto tensor =
                   std::dynamic_pointer_cast<Tensor<uint8_t>>(df_col)) {
      num_rows = tensor->shape()[0];
    } else if (auto tensor =
                   std::dynamic_pointer_cast<Tensor<int8_t>>(df_col)) {
      num_rows = tensor->shape()[0];
    }

    std::vector<std::shared_ptr<arrow::Buffer>> buffer{
        nullptr /* null bitmap */};

    // process the second buffer for std::string type
    if (auto tensor = std::dynamic_pointer_cast<Tensor<std::string>>(df_col)) {
      std::shared_ptr<arrow::Buffer> copied_buffer;
      if (copy) {
        CHECK_ARROW_ERROR_AND_ASSIGN(
            copied_buffer,
            df_col->buffer()->CopySlice(0, df_col->auxiliary_buffer()->size()));
      } else {
        copied_buffer = df_col->buffer();
      }
      buffer.push_back(copied_buffer);
    }

    // process buffer
    {
      std::shared_ptr<arrow::Buffer> copied_buffer;
      if (copy) {
        CHECK_ARROW_ERROR_AND_ASSIGN(
            copied_buffer,
            df_col->buffer()->CopySlice(0, df_col->buffer()->size()));
      } else {
        copied_buffer = df_col->buffer();
      }
      buffer.push_back(copied_buffer);
    }

    columns[i] = arrow::MakeArray(arrow::ArrayData::Make(
        FromAnyType(df_col->value_type()), num_rows, buffer));

    std::shared_ptr<arrow::Scalar> sca;
    CHECK_ARROW_ERROR_AND_ASSIGN(sca, columns[i]->GetScalar(0));

    DLOG(INFO) << "at column" << i << " start element : " << sca->ToString()
               << " value type: " << df_col->value_type()
               << " meta data type name:" << df_col->meta().GetTypeName()
               << std::endl;

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

void GlobalDataFrame::PostConstruct(const ObjectMeta& meta) {
  if (meta.HasKey("partition_shape_row_")) {
    meta.GetKeyValue("partition_shape_row_", this->partition_shape_row_);
  }
  if (meta.HasKey("partition_shape_column_")) {
    meta.GetKeyValue("partition_shape_column_", this->partition_shape_column_);
  }
}

const std::vector<std::shared_ptr<DataFrame>> GlobalDataFrame::LocalPartitions(
    Client& client) const {
  std::vector<std::shared_ptr<DataFrame>> local_chunks;
  for (auto iter = LocalBegin(); iter != LocalEnd(); iter.NextLocal()) {
    local_chunks.emplace_back(*iter);
  }
  return local_chunks;
}

const std::pair<size_t, size_t> GlobalDataFrame::partition_shape() const {
  return std::make_pair(this->partition_shape_row_,
                        this->partition_shape_column_);
}

const std::pair<size_t, size_t> GlobalDataFrameBuilder::partition_shape()
    const {
  return std::make_pair(this->partition_shape_row_,
                        this->partition_shape_column_);
}

void GlobalDataFrameBuilder::set_partition_shape(
    const size_t partition_shape_row, const size_t partition_shape_column) {
  this->partition_shape_row_ = partition_shape_row;
  this->partition_shape_column_ = partition_shape_column;
  this->AddKeyValue("partition_shape_row_", partition_shape_row_);
  this->AddKeyValue("partition_shape_column_", partition_shape_column_);
}

void GlobalDataFrameBuilder::AddPartition(const ObjectID partition_id) {
  this->AddMember(partition_id);
}

void GlobalDataFrameBuilder::AddPartitions(
    const std::vector<ObjectID>& partition_ids) {
  this->AddMembers(partition_ids);
}

}  // namespace vineyard
