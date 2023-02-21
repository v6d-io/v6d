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

Status BuildArray(Client& client, const std::shared_ptr<arrow::Array> array,
                  std::shared_ptr<ObjectBuilder>& builder);

Status BuildArray(Client& client,
                  const std::shared_ptr<arrow::ChunkedArray> array,
                  std::shared_ptr<ObjectBuilder>& builder);

std::shared_ptr<ObjectBuilder> BuildArray(Client& client,
                                          std::shared_ptr<arrow::Array> array);

std::shared_ptr<ObjectBuilder> BuildArray(
    Client& client, std::shared_ptr<arrow::ChunkedArray> array);

}  // namespace detail

/**
 * @brief NumericArrayBuilder is designed for building Arrow numeric arrays
 *
 * @tparam T
 */
template <typename T>
class NumericArrayBuilder : public NumericArrayBaseBuilder<T> {
 public:
  using ArrayType = ArrowArrayType<T>;

  explicit NumericArrayBuilder(Client& client);

  NumericArrayBuilder(Client& client, const std::shared_ptr<ArrayType> array);

  NumericArrayBuilder(Client& client,
                      const std::vector<std::shared_ptr<ArrayType>>& arrays);

  NumericArrayBuilder(Client& client,
                      const std::shared_ptr<arrow::ChunkedArray> array);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

/**
 * @brief FixedNumericArrayBuilder is designed for building Arrow numeric
 * arrays with known size. It is useful for allocating buffers directly
 * on the vineyard's shared memory.
 *
 * @tparam T
 */
template <typename T>
class FixedNumericArrayBuilder : public NumericArrayBaseBuilder<T> {
 public:
  using ArrayType = ArrowArrayType<T>;

  FixedNumericArrayBuilder(Client& client, const size_t size);

  size_t size() const;

  T* MutablePointer(int64_t i) const;

  T* data() const;

  Status Build(Client& client) override;

 private:
  size_t size_ = 0;
  std::unique_ptr<BlobWriter> writer_ = nullptr;
  T* data_ = nullptr;
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

using FixedInt8Builder = FixedNumericArrayBuilder<int8_t>;
using FixedInt16Builder = FixedNumericArrayBuilder<int16_t>;
using FixedInt32Builder = FixedNumericArrayBuilder<int32_t>;
using FixedInt64Builder = FixedNumericArrayBuilder<int64_t>;
using FixedUInt8Builder = FixedNumericArrayBuilder<uint8_t>;
using FixedUInt16Builder = FixedNumericArrayBuilder<uint16_t>;
using FixedUInt32Builder = FixedNumericArrayBuilder<uint32_t>;
using FixedUInt64Builder = FixedNumericArrayBuilder<uint64_t>;
using FixedFloatBuilder = FixedNumericArrayBuilder<float>;
using FixedDoubleBuilder = FixedNumericArrayBuilder<double>;

/**
 * @brief BooleanArrayBuilder is designed for constructing  Arrow arrays of
 * boolean data type
 *
 */
class BooleanArrayBuilder : public BooleanArrayBaseBuilder {
 public:
  using ArrayType = ArrowArrayType<bool>;

  // build an empty array
  explicit BooleanArrayBuilder(Client& client);

  BooleanArrayBuilder(Client& client, const std::shared_ptr<ArrayType> array);

  BooleanArrayBuilder(Client& client,
                      const std::vector<std::shared_ptr<ArrayType>>& arrays);

  BooleanArrayBuilder(Client& client,
                      const std::shared_ptr<arrow::ChunkedArray> array);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

/**
 * @brief BaseBinaryArray is designed for constructing  Arrow arrays of
 * binary data type
 *
 */
template <typename ArrayType, typename BuilderType>
class GenericBinaryArrayBuilder : public BaseBinaryArrayBaseBuilder<ArrayType> {
 public:
  explicit GenericBinaryArrayBuilder(Client& client);

  GenericBinaryArrayBuilder(Client& client,
                            const std::shared_ptr<ArrayType> array);

  GenericBinaryArrayBuilder(
      Client& client, const std::vector<std::shared_ptr<ArrayType>>& array);

  GenericBinaryArrayBuilder(Client& client,
                            const std::shared_ptr<arrow::ChunkedArray> array);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

using BinaryArrayBuilder =
    GenericBinaryArrayBuilder<arrow::BinaryArray, arrow::BinaryBuilder>;
using LargeBinaryArrayBuilder =
    GenericBinaryArrayBuilder<arrow::LargeBinaryArray,
                              arrow::LargeBinaryBuilder>;
using StringArrayBuilder =
    GenericBinaryArrayBuilder<arrow::StringArray, arrow::StringBuilder>;
using LargeStringArrayBuilder =
    GenericBinaryArrayBuilder<arrow::LargeStringArray,
                              arrow::LargeStringBuilder>;

/**
 * @brief FixedSizeBinaryArrayBuilder is designed for constructing Arrow arrays
 * of a fixed-size binary data type
 *
 */
class FixedSizeBinaryArrayBuilder : public FixedSizeBinaryArrayBaseBuilder {
 public:
  using ArrayType = arrow::FixedSizeBinaryArray;

  FixedSizeBinaryArrayBuilder(Client& client,
                              const std::shared_ptr<arrow::DataType>& type);

  FixedSizeBinaryArrayBuilder(Client& client,
                              const std::shared_ptr<ArrayType> array);

  FixedSizeBinaryArrayBuilder(
      Client& client, const std::vector<std::shared_ptr<ArrayType>>& array);

  FixedSizeBinaryArrayBuilder(Client& client,
                              const std::shared_ptr<arrow::ChunkedArray> array);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

/**
 * @brief PodArrayBuilder is designed for constructing Arrow arrays of POD data
 * type
 *
 * @tparam T
 */
template <typename T>
class PodArrayBuilder : public FixedSizeBinaryArrayBaseBuilder {
 public:
  explicit PodArrayBuilder(Client& client, size_t size)
      : FixedSizeBinaryArrayBaseBuilder(client), size_(size) {
    if (size != 0) {
      VINEYARD_CHECK_OK(client.CreateBlob(size * sizeof(T), buffer_));
      data_ = reinterpret_cast<T*>(buffer_->Buffer()->mutable_data());
    }
  }

  T* MutablePointer(int64_t i) const {
    if (data_) {
      return data_ + i;
    }
    return nullptr;
  }

  T* data() const { return data_; }

  size_t size() const { return size_; }

  Status Build(Client& client) override {
    this->set_byte_width_(sizeof(T));
    this->set_null_count_(0);
    this->set_offset_(0);
    if (buffer_) {
      this->set_length_(buffer_->Buffer()->size() / sizeof(T));
      this->set_buffer_(std::move(buffer_));
    } else {
      this->set_length_(0);
      this->set_buffer_(Blob::MakeEmpty(client));
    }
    this->set_null_bitmap_(Blob::MakeEmpty(client));
    return Status::OK();
  }

 private:
  size_t size_;
  std::unique_ptr<BlobWriter> buffer_;
  T* data_ = nullptr;
};  // namespace vineyard

/**
 * @brief NullArrayBuilder is used for generating Arrow arrays of null data type
 *
 */
class NullArrayBuilder : public NullArrayBaseBuilder {
 public:
  using ArrayType = arrow::NullArray;

  explicit NullArrayBuilder(Client& client);

  NullArrayBuilder(Client& client, const std::shared_ptr<ArrayType> array);

  NullArrayBuilder(Client& client,
                   const std::vector<std::shared_ptr<ArrayType>>& array);

  NullArrayBuilder(Client& client,
                   const std::shared_ptr<arrow::ChunkedArray> array);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

/**
 * @brief BaseListArrayBuilder is designed for constructing  Arrow arrays of
 * list data type
 *
 */
template <typename ArrayType>
class BaseListArrayBuilder : public BaseListArrayBaseBuilder<ArrayType> {
 public:
  BaseListArrayBuilder(Client& client, const std::shared_ptr<ArrayType> array);

  BaseListArrayBuilder(Client& client,
                       const std::vector<std::shared_ptr<ArrayType>>& array);

  BaseListArrayBuilder(Client& client,
                       const std::shared_ptr<arrow::ChunkedArray> array);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
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
  using ArrayType = arrow::FixedSizeListArray;

  FixedSizeListArrayBuilder(Client& client,
                            const std::shared_ptr<ArrayType> array);

  FixedSizeListArrayBuilder(
      Client& client, const std::vector<std::shared_ptr<ArrayType>>& array);

  FixedSizeListArrayBuilder(Client& client,
                            const std::shared_ptr<arrow::ChunkedArray> array);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

#undef BUILD_NULL_BITMAP

/**
 * @brief SchemaProxyBuilder is used for initiating proxies for the schemas
 *
 */
class SchemaProxyBuilder : public SchemaProxyBaseBuilder {
 public:
  SchemaProxyBuilder(Client& client,
                     const std::shared_ptr<arrow::Schema> schema);

 public:
  Status Build(Client& client) override;

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
  RecordBatchBuilder(Client& client,
                     const std::shared_ptr<arrow::RecordBatch> batch);

  RecordBatchBuilder(
      Client& client,
      const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches);

  Status Build(Client& client) override;

 private:
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
};

/**
 * @brief RecordBatchExtender supports extending the batch of rows of columns of
 * equal length
 *
 */
class RecordBatchExtender : public RecordBatchBaseBuilder {
 public:
  RecordBatchExtender(Client& client, const std::shared_ptr<RecordBatch> batch);

  size_t num_rows() const { return row_num_; }

  Status AddColumn(Client& client, const std::string& field_name,
                   const std::shared_ptr<arrow::Array> column);

  Status Build(Client& client) override;

 private:
  size_t row_num_ = 0, column_num_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::Array>> arrow_columns_;
};

class RecordBatchConsolidator : public RecordBatchBaseBuilder {
 public:
  RecordBatchConsolidator(Client& client,
                          const std::shared_ptr<RecordBatch> batch);

  size_t num_rows() const { return row_num_; }

  std::shared_ptr<arrow::Schema> schema() const { return schema_; }

  Status ConsolidateColumns(Client& client,
                            std::vector<std::string> const& columns,
                            std::string const& consolidate_name);

  Status ConsolidateColumns(Client& client, std::vector<int64_t> const& columns,
                            std::string const& consolidate_name);

  Status Build(Client& client) override;

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
class TableBuilder : public CollectionBuilder<RecordBatch> {
 public:
  TableBuilder(Client& client, const std::shared_ptr<arrow::Table> table,
               const bool merge_chunks = false);

  TableBuilder(Client& client,
               const std::vector<std::shared_ptr<arrow::Table>>& table,
               const bool merge_chunks = false);

  Status Build(Client& client) override;

  void set_num_rows(const size_t num_rows);
  // for backward compatibility
  void set_num_rows_(const size_t num_rows);

  void set_num_columns(const size_t num_columns);
  // for backward compatibility
  void set_num_columns_(const size_t num_columns);

  void set_batch_num(const size_t batch_num);
  // for backward compatibility
  void set_batch_num_(const size_t batch_num);

  Status set_schema(const std::shared_ptr<arrow::Schema>& schema);

  Status set_schema(const std::shared_ptr<ObjectBuilder>& schema);
  // for backward compatibility
  Status set_schema_(const std::shared_ptr<ObjectBuilder>& schema);

 private:
  std::vector<std::shared_ptr<arrow::Table>> tables_;
  bool merge_chunks_ = false;
};

/**
 * @brief TableExtender is used for extending tables
 *
 */
class TableExtender : public TableBuilder {
 public:
  TableExtender(Client& client, std::shared_ptr<Table> table);

  Status AddColumn(Client& client, const std::string& field_name,
                   const std::shared_ptr<arrow::Array> column);

  /**
   * NOTE: `column` is aligned with `table`.
   */
  Status AddColumn(Client& client, const std::string& field_name,
                   const std::shared_ptr<arrow::ChunkedArray> column);

  Status Build(Client& client) override;

 private:
  size_t row_num_ = 0, column_num_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<RecordBatchExtender>> record_batch_extenders_;
};

class TableConsolidator : public TableBuilder {
 public:
  TableConsolidator(Client& client, std::shared_ptr<Table> table);

  Status ConsolidateColumns(Client& client,
                            std::vector<std::string> const& columns,
                            std::string const& consolidate_name);

  Status ConsolidateColumns(Client& client, std::vector<int64_t> const& columns,
                            std::string const& consolidate_name);

  Status Build(Client& client) override;

 private:
  size_t row_num_ = 0, column_num_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<RecordBatchConsolidator>>
      record_batch_consolidators_;
};

}  // namespace vineyard
#endif  // MODULES_BASIC_DS_ARROW_H_
