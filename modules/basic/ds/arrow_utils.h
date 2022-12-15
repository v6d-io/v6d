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

#ifndef MODULES_BASIC_DS_ARROW_UTILS_H_
#define MODULES_BASIC_DS_ARROW_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/types.h"
#include "common/util/arrow.h"
#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

namespace arrow_types {
using OIDT = std::string;
using VIDT = uint64_t;
using EIDT = uint64_t;
}  // namespace arrow_types

struct RefString;

template <typename T>
struct ConvertToArrowType {};

template <typename T>
class NumericArray;

template <typename T>
class NumericArrayBuilder;

#define CONVERT_TO_ARROW_TYPE(type, data_type, array_type, builder_type,       \
                              type_value)                                      \
  template <>                                                                  \
  struct ConvertToArrowType<type> {                                            \
    using DataType = data_type;                                                \
    using ArrayType = array_type;                                              \
    using VineyardArrayType = NumericArray<type>;                              \
    using BuilderType = builder_type;                                          \
    using VineyardBuilderType = NumericArrayBuilder<type>;                     \
    static std::shared_ptr<arrow::DataType> TypeValue() { return type_value; } \
  };

template <typename T>
using ArrowDataType = typename ConvertToArrowType<T>::DataType;

template <typename T>
using ArrowArrayType = typename ConvertToArrowType<T>::ArrayType;

template <typename T>
using ArrowVineyardArrayType =
    typename ConvertToArrowType<T>::VineyardArrayType;

template <typename T>
using ArrowBuilderType = typename ConvertToArrowType<T>::BuilderType;

template <typename T>
using ArrowVineyardBuilderType =
    typename ConvertToArrowType<T>::VineyardBuilderType;

CONVERT_TO_ARROW_TYPE(void, arrow::NullType, arrow::NullArray,
                      arrow::NullBuilder, arrow::null())
CONVERT_TO_ARROW_TYPE(bool, arrow::BooleanType, arrow::BooleanArray,
                      arrow::BooleanBuilder, arrow::boolean())
CONVERT_TO_ARROW_TYPE(int8_t, arrow::Int8Type, arrow::Int8Array,
                      arrow::Int8Builder, arrow::int8())
CONVERT_TO_ARROW_TYPE(uint8_t, arrow::UInt8Type, arrow::UInt8Array,
                      arrow::UInt8Builder, arrow::uint8())
CONVERT_TO_ARROW_TYPE(int16_t, arrow::Int16Type, arrow::Int16Array,
                      arrow::Int16Builder, arrow::int16())
CONVERT_TO_ARROW_TYPE(uint16_t, arrow::UInt16Type, arrow::UInt16Array,
                      arrow::UInt16Builder, arrow::uint16())
CONVERT_TO_ARROW_TYPE(int32_t, arrow::Int32Type, arrow::Int32Array,
                      arrow::Int32Builder, arrow::int32())
CONVERT_TO_ARROW_TYPE(uint32_t, arrow::UInt32Type, arrow::UInt32Array,
                      arrow::UInt32Builder, arrow::uint32())
CONVERT_TO_ARROW_TYPE(int64_t, arrow::Int64Type, arrow::Int64Array,
                      arrow::Int64Builder, arrow::int64())
CONVERT_TO_ARROW_TYPE(uint64_t, arrow::UInt64Type, arrow::UInt64Array,
                      arrow::UInt64Builder, arrow::uint64())
CONVERT_TO_ARROW_TYPE(float, arrow::FloatType, arrow::FloatArray,
                      arrow::FloatBuilder, arrow::float32())
CONVERT_TO_ARROW_TYPE(double, arrow::DoubleType, arrow::DoubleArray,
                      arrow::DoubleBuilder, arrow::float64())
CONVERT_TO_ARROW_TYPE(RefString, arrow::LargeStringType,
                      arrow::LargeStringArray, arrow::LargeStringBuilder,
                      arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(std::string, arrow::LargeStringType,
                      arrow::LargeStringArray, arrow::LargeStringBuilder,
                      arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(arrow_string_view, arrow::LargeStringType,
                      arrow::LargeStringArray, arrow::LargeStringBuilder,
                      arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(arrow::TimestampType, arrow::TimestampType,
                      arrow::TimestampArray, arrow::TimestampBuilder,
                      arrow::timestamp(arrow::TimeUnit::MILLI))
CONVERT_TO_ARROW_TYPE(arrow::Date32Type, arrow::Date32Type, arrow::Date32Array,
                      arrow::Date32Builder, arrow::date32())
CONVERT_TO_ARROW_TYPE(arrow::Date64Type, arrow::Date64Type, arrow::Date64Array,
                      arrow::Date64Builder, arrow::date64())

std::shared_ptr<arrow::DataType> FromAnyType(AnyType type);

namespace detail {
/**
 * @brief Make a copy for a arrow ArrayData.
 *
 * arrow::ArrayData::Copy() is a shallow thus is not suitable in many
 * cases.
 */
Status Copy(std::shared_ptr<arrow::ArrayData> const& array,
            std::shared_ptr<arrow::ArrayData>& out,

            bool shallow = true,
            arrow::MemoryPool* pool = arrow::default_memory_pool());

Status Copy(std::shared_ptr<arrow::Array> const& array,
            std::shared_ptr<arrow::Array>& out, bool shallow = true,
            arrow::MemoryPool* pool = arrow::default_memory_pool());

Status Copy(std::shared_ptr<arrow::ChunkedArray> const& array,
            std::shared_ptr<arrow::ChunkedArray>& out, bool shallow = true,
            arrow::MemoryPool* pool = arrow::default_memory_pool());

/**
 * @brief Make a copy for a arrow recordbatch.
 */
Status Copy(std::shared_ptr<arrow::RecordBatch> const& batch,
            std::shared_ptr<arrow::RecordBatch>& out, bool shallow = true,
            arrow::MemoryPool* pool = arrow::default_memory_pool());

Status Copy(std::shared_ptr<arrow::Table> const& batch,
            std::shared_ptr<arrow::Table>& out, bool shallow = true,
            arrow::MemoryPool* pool = arrow::default_memory_pool());

}  // namespace detail

struct EmptyTableBuilder {
  static Status Build(const std::shared_ptr<arrow::Schema>& schema,
                      std::shared_ptr<arrow::Table>& table);

  static std::shared_ptr<arrow::Schema> EmptySchema();
};

/**
 * Similar to arrow's `GetRecordBatchSize`, but considering the schema.
 *
 * Used for pre-allocate buffer for NewStreamWriter's `WriteRecordBatch`.
 */
Status GetRecordBatchStreamSize(const arrow::RecordBatch& batch, size_t* size);

Status SerializeDataType(const std::shared_ptr<arrow::DataType>& type,
                         std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeDataType(const std::shared_ptr<arrow::Buffer>& buffer,
                           std::shared_ptr<arrow::DataType>* type);

Status SerializeSchema(const arrow::Schema& schema,
                       std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeSchema(const std::shared_ptr<arrow::Buffer>& buffer,
                         std::shared_ptr<arrow::Schema>* schema);

Status SerializeRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch,
                            std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeRecordBatch(const std::shared_ptr<arrow::Buffer>& buffer,
                              std::shared_ptr<arrow::RecordBatch>* batch);

Status SerializeRecordBatchesToAllocatedBuffer(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer);

Status SerializeRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeRecordBatches(
    const std::shared_ptr<arrow::Buffer>& buffer,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches);

Status RecordBatchesToTable(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table);

Status RecordBatchesToTable(
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table);

Status CombineRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::RecordBatch>* batch);

Status TableToRecordBatches(
    const std::shared_ptr<arrow::Table> table,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches);

std::shared_ptr<arrow::ChunkedArray> ConcatenateChunkedArrays(
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& arrays);

std::shared_ptr<arrow::ChunkedArray> ConcatenateChunkedArrays(
    const std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
        arrays);

Status SerializeTableToAllocatedBuffer(
    const std::shared_ptr<arrow::Table> table,
    std::shared_ptr<arrow::Buffer>* buffer);

Status SerializeTable(const std::shared_ptr<arrow::Table> table,
                      std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeTable(const std::shared_ptr<arrow::Buffer> buffer,
                        std::shared_ptr<arrow::Table>* table);

/**
 * @brief Concatenate multiple arrow tables into one.
 *
 * Like `arrow::ConcatenateTables`, but unify the column names as well.
 */
Status ConcatenateTables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables,
    std::shared_ptr<arrow::Table>& table);

/**
 * @brief Add extra metadata mapping to existing recordbatch.
 */
std::shared_ptr<arrow::RecordBatch> AddMetadataToRecordBatch(
    std::shared_ptr<arrow::RecordBatch> const& batch,
    std::map<std::string, std::string> const& meta);

/**
 * @brief Add extra metadata mapping to existing recordbatch, the
 * std::unordered_map variant.
 */
std::shared_ptr<arrow::RecordBatch> AddMetadataToRecordBatch(
    std::shared_ptr<arrow::RecordBatch> const& batch,
    std::unordered_map<std::string, std::string> const& meta);

/**
 * @brief Convert type name in string to arrow type.
 *
 */
std::shared_ptr<arrow::DataType> type_name_to_arrow_type(
    const std::string& name);

std::string type_name_from_arrow_type(
    const std::shared_ptr<arrow::DataType>& type);

const void* get_arrow_array_data(std::shared_ptr<arrow::Array> const& array);

Status TypeLoosen(const std::vector<std::shared_ptr<arrow::Schema>>& schemas,
                  std::shared_ptr<arrow::Schema>& schema);

Status CastStringToBigString(const std::shared_ptr<arrow::Array>& in,
                             const std::shared_ptr<arrow::DataType>& to_type,
                             std::shared_ptr<arrow::Array>& out);

Status CastNullToOthers(const std::shared_ptr<arrow::Array>& in,
                        const std::shared_ptr<arrow::DataType>& to_type,
                        std::shared_ptr<arrow::Array>& out);

Status GeneralCast(const std::shared_ptr<arrow::Array>& in,
                   const std::shared_ptr<arrow::DataType>& to_type,
                   std::shared_ptr<arrow::Array>& out);

Status CastTableToSchema(const std::shared_ptr<arrow::Table>& table,
                         const std::shared_ptr<arrow::Schema>& schema,
                         std::shared_ptr<arrow::Table>& out);

Status ConsolidateColumns(
    const std::vector<std::shared_ptr<arrow::Array>>& columns,
    std::shared_ptr<arrow::Array>& out);

Status ConsolidateColumns(
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& columns,
    std::shared_ptr<arrow::ChunkedArray>& out);

/**
 * @brief Consolidate columns in an arrow table into one column
 * (FixedSizeListArray).
 *
 * Note that the bitmap in the given columns will be discard.
 *
 * @param table
 * @return boost::leaf::result<std::shared_ptr<arrow::Table>>
 */
Status ConsolidateColumns(const std::shared_ptr<arrow::Table>& table,
                          std::vector<std::string> const& column_names,
                          std::string const& consolidated_column_name,
                          std::shared_ptr<arrow::Table>& out);

Status ConsolidateColumns(const std::shared_ptr<arrow::Table>& table,
                          std::shared_ptr<arrow::Table>& out);

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_ARROW_UTILS_H_
