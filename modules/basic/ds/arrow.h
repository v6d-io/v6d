/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#ifndef MODULES_BASIC_DS_ARROW_H_
#define MODULES_BASIC_DS_ARROW_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "basic/ds/arrow.vineyard.h"
#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "client/ds/blob.h"

namespace vineyard {

namespace detail {
std::shared_ptr<ObjectBuilder> BuildSimpleArray(
    Client& client, std::shared_ptr<arrow::Array> array);

std::shared_ptr<ObjectBuilder> BuildArray(Client& client,
                                          std::shared_ptr<arrow::Array> array);
}  // namespace detail

#ifndef BUILD_NULL_BITMAP
#define BUILD_NULL_BITMAP(builder, array)                                \
  {                                                                      \
    if (array->null_bitmap() && array->null_count() > 0) {               \
      std::unique_ptr<BlobWriter> bitmap_buffer_writer;                  \
      RETURN_ON_ERROR(client.CreateBlob(array->null_bitmap()->size(),    \
                                        bitmap_buffer_writer));          \
      memcpy(bitmap_buffer_writer->data(), array->null_bitmap()->data(), \
             array->null_bitmap()->size());                              \
      builder->set_null_bitmap_(                                         \
          std::shared_ptr<BlobWriter>(std::move(bitmap_buffer_writer))); \
    } else {                                                             \
      builder->set_null_bitmap_(Blob::MakeEmpty(client));                \
    }                                                                    \
  }
#endif

/**
 * @brief NumericArrayBuilder is designed for building Arrow numeric arrays
 *
 * @tparam T
 */
template <typename T>
class NumericArrayBuilder : public NumericArrayBaseBuilder<T> {
 public:
  using ArrayType = typename ConvertToArrowType<T>::ArrayType;

  explicit NumericArrayBuilder(Client& client)
      : NumericArrayBaseBuilder<T>(client) {
    CHECK_ARROW_ERROR(
        typename ConvertToArrowType<T>::BuilderType{}.Finish(&array_));
  }

  NumericArrayBuilder(Client& client, std::shared_ptr<ArrayType> array)
      : NumericArrayBaseBuilder<T>(client), array_(array) {}

  std::shared_ptr<ArrayType> GetArray() { return array_; }

  Status Build(Client& client) override {
    std::unique_ptr<BlobWriter> buffer_writer;
    RETURN_ON_ERROR(client.CreateBlob(array_->values()->size(), buffer_writer));
    memcpy(buffer_writer->data(), array_->values()->data(),
           array_->values()->size());

    this->set_length_(array_->length());
    this->set_null_count_(array_->null_count());
    this->set_offset_(array_->offset());
    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
    BUILD_NULL_BITMAP(this, array_);
    return Status::OK();
  }

 private:
  std::shared_ptr<ArrayType> array_;
};

using Int8Builder = NumericArrayBuilder<int8_t>;
using Int16Builder = NumericArrayBuilder<int16_t>;
using Int32Builder = NumericArrayBuilder<int32_t>;
using Int64Builder = NumericArrayBuilder<int64_t>;
using UInt8Builder = NumericArrayBuilder<uint8_t>;
using UInt16Builder = NumericArrayBuilder<uint16_t>;
using UInt32Builder = NumericArrayBuilder<uint32_t>;
using UInt64Builder = NumericArrayBuilder<uint64_t>;
using FloatBuilder = NumericArrayBuilder<float>;
using DoubleBuilder = NumericArrayBuilder<double>;

/**
 * @brief BooleanArrayBuilder is designed for constructing  Arrow arrays of
 * boolean data type
 *
 */
class BooleanArrayBuilder : public BooleanArrayBaseBuilder {
 public:
  using ArrayType = typename ConvertToArrowType<bool>::ArrayType;

  // build an empty array
  explicit BooleanArrayBuilder(Client& client)
      : BooleanArrayBaseBuilder(client) {
    CHECK_ARROW_ERROR(
        typename ConvertToArrowType<bool>::BuilderType{}.Finish(&array_));
  }

  BooleanArrayBuilder(Client& client, std::shared_ptr<ArrayType> array)
      : BooleanArrayBaseBuilder(client), array_(array) {}

  std::shared_ptr<ArrayType> GetArray() { return array_; }

  Status Build(Client& client) override {
    std::unique_ptr<BlobWriter> buffer_writer;
    RETURN_ON_ERROR(client.CreateBlob(array_->values()->size(), buffer_writer));
    memcpy(buffer_writer->data(), array_->values()->data(),
           array_->values()->size());

    this->set_length_(array_->length());
    this->set_null_count_(array_->null_count());
    this->set_offset_(array_->offset());
    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
    BUILD_NULL_BITMAP(this, array_);
    return Status::OK();
  }

 private:
  std::shared_ptr<ArrayType> array_;
};

/**
 * @brief BaseBinaryArray is designed for constructing  Arrow arrays of
 * binary data type
 *
 */
template <typename ArrayType, typename BuilderType>
class BaseBinaryArrayBuilder : public BaseBinaryArrayBaseBuilder<ArrayType> {
 public:
  explicit BaseBinaryArrayBuilder(Client& client)
      : BaseBinaryArrayBaseBuilder<ArrayType>(client) {
    CHECK_ARROW_ERROR(BuilderType{}.Finish(&array_));
  }

  BaseBinaryArrayBuilder(Client& client, std::shared_ptr<ArrayType> array)
      : BaseBinaryArrayBaseBuilder<ArrayType>(client), array_(array) {}

  std::shared_ptr<ArrayType> GetArray() { return array_; }

  Status Build(Client& client) override {
    {
      std::unique_ptr<BlobWriter> buffer_writer;
      RETURN_ON_ERROR(
          client.CreateBlob(array_->value_offsets()->size(), buffer_writer));
      memcpy(buffer_writer->data(), array_->value_offsets()->data(),
             array_->value_offsets()->size());

      this->set_buffer_offsets_(
          std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
    }
    {
      std::unique_ptr<BlobWriter> buffer_writer;
      RETURN_ON_ERROR(
          client.CreateBlob(array_->value_data()->size(), buffer_writer));
      memcpy(buffer_writer->data(), array_->value_data()->data(),
             array_->value_data()->size());

      this->set_buffer_data_(
          std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
    }
    this->set_length_(array_->length());
    this->set_null_count_(array_->null_count());
    this->set_offset_(array_->offset());
    BUILD_NULL_BITMAP(this, array_);
    return Status::OK();
  }

 private:
  std::shared_ptr<ArrayType> array_;
};

using BinaryArrayBuilder =
    BaseBinaryArrayBuilder<arrow::BinaryArray, arrow::BinaryBuilder>;
using LargeBinaryArrayBuilder =
    BaseBinaryArrayBuilder<arrow::LargeBinaryArray, arrow::LargeBinaryBuilder>;
using StringArrayBuilder =
    BaseBinaryArrayBuilder<arrow::StringArray, arrow::StringBuilder>;
using LargeStringArrayBuilder =
    BaseBinaryArrayBuilder<arrow::LargeStringArray, arrow::LargeStringBuilder>;

/**
 * @brief FixedSizeBinaryArrayBuilder is designed for constructing Arrow arrays
 * of a fixed-size binary data type
 *
 */
class FixedSizeBinaryArrayBuilder : public FixedSizeBinaryArrayBaseBuilder {
 public:
  FixedSizeBinaryArrayBuilder(Client& client,
                              const std::shared_ptr<arrow::DataType>& type)
      : FixedSizeBinaryArrayBaseBuilder(client) {
    CHECK_ARROW_ERROR(arrow::FixedSizeBinaryBuilder{type}.Finish(&array_));
  }

  FixedSizeBinaryArrayBuilder(
      Client& client, std::shared_ptr<arrow::FixedSizeBinaryArray> array)
      : FixedSizeBinaryArrayBaseBuilder(client), array_(array) {}

  std::shared_ptr<arrow::FixedSizeBinaryArray> GetArray() { return array_; }

  Status Build(Client& client) override {
    VINEYARD_ASSERT(array_->length() == 0 || array_->values()->size() != 0,
                    "Invalid array values");

    std::unique_ptr<BlobWriter> buffer_writer;
    RETURN_ON_ERROR(client.CreateBlob(array_->values()->size(), buffer_writer));
    memcpy(buffer_writer->data(), array_->values()->data(),
           array_->values()->size());

    this->set_byte_width_(array_->byte_width());
    this->set_length_(array_->length());
    this->set_null_count_(array_->null_count());
    this->set_offset_(array_->offset());
    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
    BUILD_NULL_BITMAP(this, array_);
    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::FixedSizeBinaryArray> array_;
};

/**
 * @brief NullArrayBuilder is used for generating Arrow arrays of null data type
 *
 */
class NullArrayBuilder : public NullArrayBaseBuilder {
 public:
  explicit NullArrayBuilder(Client& client) : NullArrayBaseBuilder(client) {
    CHECK_ARROW_ERROR(arrow::NullBuilder{}.Finish(&array_));
  }

  NullArrayBuilder(Client& client, std::shared_ptr<arrow::NullArray> array)
      : NullArrayBaseBuilder(client), array_(array) {}

  std::shared_ptr<arrow::NullArray> GetArray() { return array_; }

  Status Build(Client& client) override {
    this->set_length_(array_->length());
    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::NullArray> array_;
};

/**
 * @brief BaseListArrayBuilder is designed for constructing  Arrow arrays of
 * list data type
 *
 */
template <typename ArrayType>
class BaseListArrayBuilder : public BaseListArrayBaseBuilder<ArrayType> {
 public:
  BaseListArrayBuilder(Client& client, std::shared_ptr<ArrayType> array)
      : BaseListArrayBaseBuilder<ArrayType>(client), array_(array) {}

  std::shared_ptr<ArrayType> GetArray() { return array_; }

  Status Build(Client& client) override {
    {
      std::unique_ptr<BlobWriter> buffer_writer;
      RETURN_ON_ERROR(
          client.CreateBlob(array_->value_offsets()->size(), buffer_writer));
      memcpy(buffer_writer->data(), array_->value_offsets()->data(),
             array_->value_offsets()->size());

      this->set_buffer_offsets_(
          std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
    }
    {
      // Assuming the list is not nested.
      // We need to split the definition to .cc if someday we need to consider
      // nested list in list case.
      this->set_values_(detail::BuildSimpleArray(client, array_->values()));
    }
    this->set_length_(array_->length());
    this->set_null_count_(array_->null_count());
    this->set_offset_(array_->offset());
    BUILD_NULL_BITMAP(this, array_);
    return Status::OK();
  }

 private:
  std::shared_ptr<ArrayType> array_;
};

using ListArrayBuilder = BaseListArrayBuilder<arrow::ListArray>;
using LargeListArrayBuilder = BaseListArrayBuilder<arrow::LargeListArray>;

/**
 * @brief FixedSizeListArrayBuilder is designed for constructing  Arrow arrays
 * of fixed-size list data type
 *
 */
class FixedSizeListArrayBuilder : public FixedSizeListArrayBaseBuilder {
 public:
  FixedSizeListArrayBuilder(Client& client,
                            std::shared_ptr<arrow::FixedSizeListArray> array)
      : FixedSizeListArrayBaseBuilder(client), array_(array) {}

  std::shared_ptr<arrow::FixedSizeListArray> GetArray() { return array_; }

  Status Build(Client& client) override {
    {
      // Assuming the list is not nested.
      // We need to split the definition to .cc if someday we need to consider
      // nested list in list case.
      this->set_values_(detail::BuildSimpleArray(client, array_->values()));
    }
    this->set_length_(array_->length());
    this->set_list_size_(array_->list_type()->list_size());
    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::FixedSizeListArray> array_;
};

#undef BUILD_NULL_BITMAP

/**
 * @brief SchemaProxyBuilder is used for initiating proxies for the schemas
 *
 */
class SchemaProxyBuilder : public SchemaProxyBaseBuilder {
 public:
  SchemaProxyBuilder(Client& client, std::shared_ptr<arrow::Schema> schema)
      : SchemaProxyBaseBuilder(client), schema_(schema) {}

 public:
  Status Build(Client& client) override {
    std::shared_ptr<arrow::Buffer> schema_buffer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 2000000
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        schema_buffer, arrow::ipc::SerializeSchema(
                           *schema_, nullptr, arrow::default_memory_pool()));
#else
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        schema_buffer,
        arrow::ipc::SerializeSchema(*schema_, arrow::default_memory_pool()));
#endif
    std::unique_ptr<BlobWriter> schema_writer;
    RETURN_ON_ERROR(client.CreateBlob(schema_buffer->size(), schema_writer));
    memcpy(schema_writer->data(), schema_buffer->data(), schema_buffer->size());

    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(schema_writer)));
    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::Schema> schema_;
};

/**
 * @brief RecordBatchBuilder is used for generating the batch of rows of columns
 * of equal length
 *
 */
class RecordBatchBuilder : public RecordBatchBaseBuilder {
 public:
  RecordBatchBuilder(Client& client, std::shared_ptr<arrow::RecordBatch> batch)
      : RecordBatchBaseBuilder(client), batch_(batch) {}

  Status Build(Client& client) override {
    this->set_column_num_(batch_->num_columns());
    this->set_row_num_(batch_->num_rows());
    this->set_schema_(
        std::make_shared<SchemaProxyBuilder>(client, batch_->schema()));
    for (int64_t idx = 0; idx < batch_->num_columns(); ++idx) {
      this->add_columns_(detail::BuildArray(client, batch_->column(idx)));
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::RecordBatch> batch_;
};

/**
 * @brief RecordBatchExtender supports extending the batch of rows of columns of
 * equal length
 *
 */
class RecordBatchExtender : public RecordBatchBaseBuilder {
 public:
  RecordBatchExtender(Client& client, std::shared_ptr<RecordBatch> batch)
      : RecordBatchBaseBuilder(client) {
    row_num_ = batch->num_rows();
    column_num_ = batch->num_columns();
    schema_ = batch->schema();
    for (auto const& column : batch->columns()) {
      this->add_columns_(column);
    }
  }

  size_t num_rows() const { return row_num_; }

  Status AddColumn(Client& client, const std::string& field_name,
                   std::shared_ptr<arrow::Array> column) {
    // validate input
    if (static_cast<size_t>(column->length()) != row_num_) {
      return Status::Invalid(
          "The newly added columns doesn't have a matched shape");
    }
    // extend schema
    auto field = ::arrow::field(std::move(field_name), column->type());
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        schema_, schema_->AddField(schema_->num_fields(), field));
    // extend columns
    arrow_columns_.push_back(column);
    column_num_ += 1;
    return Status::OK();
  }

 public:
  Status Build(Client& client) override {
    this->set_row_num_(row_num_);
    this->set_column_num_(column_num_);
    this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
    for (size_t idx = 0; idx < arrow_columns_.size(); ++idx) {
      this->add_columns_(detail::BuildArray(client, arrow_columns_[idx]));
    }
    return Status::OK();
  }

 private:
  size_t row_num_ = 0, column_num_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::Array>> arrow_columns_;
};

class RecordBatchConsolidator : public RecordBatchBaseBuilder {
 public:
  RecordBatchConsolidator(Client& client, std::shared_ptr<RecordBatch> batch)
      : RecordBatchBaseBuilder(client) {
    row_num_ = batch->num_rows();
    column_num_ = batch->num_columns();
    schema_ = batch->schema();
    for (auto const& column : batch->columns()) {
      this->add_columns_(column);
    }
    for (auto const& column : batch->arrow_columns()) {
      arrow_columns_.push_back(column);
    }
  }

  size_t num_rows() const { return row_num_; }

  std::shared_ptr<arrow::Schema> schema() const { return schema_; }

  Status ConsolidateColumns(Client& client,
                            std::vector<std::string> const& columns,
                            std::string const& consolidate_name) {
    std::vector<int64_t> column_indexes;
    for (auto const& column : columns) {
      auto index = schema_->GetFieldIndex(column);
      if (index < 0) {
        return Status::Invalid("The column name '" + column +
                               "' is not found in the schema");
      }
      column_indexes.push_back(index);
    }
    return ConsolidateColumns(client, column_indexes, consolidate_name);
  }

  Status ConsolidateColumns(Client& client, std::vector<int64_t> const& columns,
                            std::string const& consolidate_name) {
    std::vector<std::shared_ptr<arrow::Array>> columns_to_consolidate;
    for (int64_t const& column : columns) {
      columns_to_consolidate.push_back(this->arrow_columns_[column]);
    }
    std::shared_ptr<arrow::Array> consolidated_column;
    RETURN_ON_ERROR(vineyard::ConsolidateColumns(columns_to_consolidate,
                                                 consolidated_column));

    this->column_num_ -= (columns.size() - 1);
    std::vector<int64_t> sorted_column_indexes(columns);
    std::sort(sorted_column_indexes.begin(), sorted_column_indexes.end());
    for (size_t index = 0; index < sorted_column_indexes.size(); ++index) {
      size_t index_to_remove =
          sorted_column_indexes[sorted_column_indexes.size() - 1 - index];
      this->remove_columns_(index_to_remove);
      this->arrow_columns_.erase(this->arrow_columns_.begin() +
                                 index_to_remove);
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(schema_,
                                       schema_->RemoveField(index_to_remove));
    }
    this->arrow_columns_.emplace_back(consolidated_column);
    this->add_columns_(detail::BuildArray(client, consolidated_column));
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        schema_,
        schema_->AddField(
            schema_->num_fields(),
            ::arrow::field(consolidate_name, consolidated_column->type())));
    return Status::OK();
  }

 public:
  Status Build(Client& client) override {
    this->set_row_num_(row_num_);
    this->set_column_num_(column_num_);
    this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
    return Status::OK();
  }

 private:
  size_t row_num_ = 0, column_num_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::Array>> arrow_columns_;
};

/**
 * @brief TableBuilder is designed for generating tables, which are collections
 * of top-level named, equal length Arrow arrays
 *
 */
class TableBuilder : public TableBaseBuilder {
 public:
  TableBuilder(Client& client, std::shared_ptr<arrow::Table> table)
      : TableBaseBuilder(client), table_(table) {}

 public:
  Status Build(Client& client) override {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    RETURN_ON_ERROR(TableToRecordBatches(table_, &batches));
    this->set_batch_num_(batches.size());
    this->set_num_rows_(table_->num_rows());
    this->set_num_columns_(table_->num_columns());
    for (auto const& batch : batches) {
      this->add_batches_(std::make_shared<RecordBatchBuilder>(client, batch));
    }
    this->set_schema_(
        std::make_shared<SchemaProxyBuilder>(client, table_->schema()));
    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::Table> table_;
};

/**
 * @brief TableExtender is used for extending tables
 *
 */
class TableExtender : public TableBaseBuilder {
 public:
  TableExtender(Client& client, std::shared_ptr<Table> table)
      : TableBaseBuilder(client) {
    row_num_ = table->num_rows();
    column_num_ = table->num_columns();
    schema_ = table->schema();
    for (auto const& batch : table->batches()) {
      record_batch_extenders_.push_back(
          std::make_shared<RecordBatchExtender>(client, batch));
    }
  }

  Status AddColumn(Client& client, const std::string& field_name,
                   std::shared_ptr<arrow::Array> column) {
    // validate input
    if (static_cast<size_t>(column->length()) != row_num_) {
      return Status::Invalid(
          "The newly added columns doesn't have a matched shape");
    }
    // extend schema
    auto field = ::arrow::field(field_name, column->type());
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        schema_, schema_->AddField(schema_->num_fields(), field));

    // extend columns on every batch
    size_t offset = 0;
    for (auto& extender : record_batch_extenders_) {
      RETURN_ON_ERROR(extender->AddColumn(
          client, field_name, column->Slice(offset, extender->num_rows())));
      offset += extender->num_rows();
    }
    column_num_ += 1;
    return Status::OK();
  }

  /**
   * NOTE: `column` is aligned with `table`.
   */
  Status AddColumn(Client& client, const std::string& field_name,
                   std::shared_ptr<arrow::ChunkedArray> column) {
    // validate input
    if (static_cast<size_t>(column->length()) != row_num_) {
      return Status::Invalid(
          "The newly added columns doesn't have a matched shape");
    }
    // extend schema
    auto field = ::arrow::field(field_name, column->type());
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        schema_, schema_->AddField(schema_->num_fields(), field));

    // extend columns on every batch
    size_t chunk_index = 0;
    for (auto& extender : record_batch_extenders_) {
      RETURN_ON_ERROR(
          extender->AddColumn(client, field_name, column->chunk(chunk_index)));
      chunk_index += 1;
    }
    column_num_ += 1;
    return Status::OK();
  }

  Status Build(Client& client) override {
    this->set_batch_num_(record_batch_extenders_.size());
    this->set_num_rows_(row_num_);
    this->set_num_columns_(column_num_);
    for (auto const& extender : record_batch_extenders_) {
      this->add_batches_(extender);
    }
    this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
    return Status::OK();
  }

 private:
  size_t row_num_ = 0, column_num_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<RecordBatchExtender>> record_batch_extenders_;
};

class TableConsolidator : public TableBaseBuilder {
 public:
  TableConsolidator(Client& client, std::shared_ptr<Table> table)
      : TableBaseBuilder(client) {
    row_num_ = table->num_rows();
    column_num_ = table->num_columns();
    schema_ = table->schema();
    for (auto const& batch : table->batches()) {
      record_batch_consolidators_.push_back(
          std::make_shared<RecordBatchConsolidator>(client, batch));
    }
  }

  Status ConsolidateColumns(Client& client,
                            std::vector<std::string> const& columns,
                            std::string const& consolidate_name) {
    std::vector<int64_t> column_indexes;
    for (auto const& column : columns) {
      auto index = schema_->GetFieldIndex(column);
      if (index < 0) {
        return Status::Invalid("The column name '" + column +
                               "' is not found in the schema");
      }
      column_indexes.push_back(index);
    }
    return ConsolidateColumns(client, column_indexes, consolidate_name);
  }

  Status ConsolidateColumns(Client& client, std::vector<int64_t> const& columns,
                            std::string const& consolidate_name) {
    for (auto& consolidator : record_batch_consolidators_) {
      RETURN_ON_ERROR(
          consolidator->ConsolidateColumns(client, columns, consolidate_name));
    }
    column_num_ -= (columns.size() - 1);
    return Status::OK();
  }

  Status Build(Client& client) override {
    this->set_batch_num_(record_batch_consolidators_.size());
    this->set_num_rows_(row_num_);
    this->set_num_columns_(column_num_);
    for (auto const& extender : record_batch_consolidators_) {
      this->add_batches_(extender);
    }
    if (record_batch_consolidators_.empty()) {
      this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
    } else {
      this->set_schema_(std::make_shared<SchemaProxyBuilder>(
          client, record_batch_consolidators_[0]->schema()));
    }
    return Status::OK();
  }

 private:
  size_t row_num_ = 0, column_num_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<RecordBatchConsolidator>>
      record_batch_consolidators_;
};

}  // namespace vineyard
#endif  // MODULES_BASIC_DS_ARROW_H_
