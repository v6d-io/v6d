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
#include "client/ds/blob.h"
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

template <typename T>
class FixedNumericArrayBuilder;

class NullArray;
class NullArrayBuilder;
class BooleanArray;
class BooleanArrayBuilder;

#define CONVERT_TO_ARROW_TYPE(                                                 \
    type, data_type, array_type, vineyard_array_type, builder_type,            \
    vineyard_builder_type, fixed_vineyard_builder_type, type_value)            \
  template <>                                                                  \
  struct ConvertToArrowType<type> {                                            \
    using DataType = data_type;                                                \
    using ArrayType = array_type;                                              \
    using VineyardArrayType = vineyard_array_type;                             \
    using BuilderType = builder_type;                                          \
    using VineyardBuilderType = vineyard_builder_type;                         \
    using FixedVineyardBuilderType = fixed_vineyard_builder_type;              \
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

template <typename T>
using FixedArrowVineyardBuilderType =
    typename ConvertToArrowType<T>::FixedVineyardBuilderType;

template <typename ArrayType>
class BaseBinaryArray;

template <typename ArrayType>
class BaseBinaryArrayBuilder;

template <typename ArrayType, typename BuilderType>
class GenericBinaryArrayBuilder;

using LargeStringArrayBuilder =
    GenericBinaryArrayBuilder<arrow::LargeStringArray,
                              arrow::LargeStringBuilder>;

CONVERT_TO_ARROW_TYPE(void, arrow::NullType, arrow::NullArray, NullArray,
                      arrow::NullBuilder, NullArrayBuilder, void, arrow::null())
CONVERT_TO_ARROW_TYPE(bool, arrow::BooleanType, arrow::BooleanArray,
                      BooleanArray, arrow::BooleanBuilder, BooleanArrayBuilder,
                      void, arrow::boolean())
CONVERT_TO_ARROW_TYPE(int8_t, arrow::Int8Type, arrow::Int8Array,
                      NumericArray<int8_t>, arrow::Int8Builder,
                      NumericArrayBuilder<int8_t>,
                      FixedNumericArrayBuilder<int8_t>, arrow::int8())
CONVERT_TO_ARROW_TYPE(uint8_t, arrow::UInt8Type, arrow::UInt8Array,
                      NumericArray<uint8_t>, arrow::UInt8Builder,
                      NumericArrayBuilder<uint8_t>,
                      FixedNumericArrayBuilder<uint8_t>, arrow::uint8())
CONVERT_TO_ARROW_TYPE(int16_t, arrow::Int16Type, arrow::Int16Array,
                      NumericArray<int16_t>, arrow::Int16Builder,
                      NumericArrayBuilder<int16_t>,
                      FixedNumericArrayBuilder<int16_t>, arrow::int16())
CONVERT_TO_ARROW_TYPE(uint16_t, arrow::UInt16Type, arrow::UInt16Array,
                      NumericArray<uint16_t>, arrow::UInt16Builder,
                      NumericArrayBuilder<uint16_t>,
                      FixedNumericArrayBuilder<uint16_t>, arrow::uint16())
CONVERT_TO_ARROW_TYPE(int32_t, arrow::Int32Type, arrow::Int32Array,
                      NumericArray<int32_t>, arrow::Int32Builder,
                      NumericArrayBuilder<int32_t>,
                      FixedNumericArrayBuilder<int32_t>, arrow::int32())
CONVERT_TO_ARROW_TYPE(uint32_t, arrow::UInt32Type, arrow::UInt32Array,
                      NumericArray<uint32_t>, arrow::UInt32Builder,
                      NumericArrayBuilder<uint32_t>,
                      FixedNumericArrayBuilder<uint32_t>, arrow::uint32())
CONVERT_TO_ARROW_TYPE(int64_t, arrow::Int64Type, arrow::Int64Array,
                      NumericArray<int64_t>, arrow::Int64Builder,
                      NumericArrayBuilder<int64_t>,
                      FixedNumericArrayBuilder<int64_t>, arrow::int64())
CONVERT_TO_ARROW_TYPE(uint64_t, arrow::UInt64Type, arrow::UInt64Array,
                      NumericArray<uint64_t>, arrow::UInt64Builder,
                      NumericArrayBuilder<uint64_t>,
                      FixedNumericArrayBuilder<uint64_t>, arrow::uint64())
CONVERT_TO_ARROW_TYPE(float, arrow::FloatType, arrow::FloatArray,
                      NumericArray<float>, arrow::FloatBuilder,
                      NumericArrayBuilder<float>,
                      FixedNumericArrayBuilder<float>, arrow::float32())
CONVERT_TO_ARROW_TYPE(double, arrow::DoubleType, arrow::DoubleArray,
                      NumericArray<double>, arrow::DoubleBuilder,
                      NumericArrayBuilder<double>,
                      FixedNumericArrayBuilder<double>, arrow::float64())
CONVERT_TO_ARROW_TYPE(RefString, arrow::LargeStringType,
                      arrow::LargeStringArray,
                      BaseBinaryArray<arrow::LargeStringArray>,
                      arrow::LargeStringBuilder, LargeStringArrayBuilder, void,
                      arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(std::string, arrow::LargeStringType,
                      arrow::LargeStringArray,
                      BaseBinaryArray<arrow::LargeStringArray>,
                      arrow::LargeStringBuilder, LargeStringArrayBuilder, void,
                      arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(arrow_string_view, arrow::LargeStringType,
                      arrow::LargeStringArray,
                      BaseBinaryArray<arrow::LargeStringArray>,
                      arrow::LargeStringBuilder, LargeStringArrayBuilder, void,
                      arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(arrow::TimestampType, arrow::TimestampType,
                      arrow::TimestampArray, void, arrow::TimestampBuilder,
                      void, void, arrow::timestamp(arrow::TimeUnit::MILLI))
CONVERT_TO_ARROW_TYPE(arrow::Date32Type, arrow::Date32Type, arrow::Date32Array,
                      void, arrow::Date32Builder, void, void, arrow::date32())
CONVERT_TO_ARROW_TYPE(arrow::Date64Type, arrow::Date64Type, arrow::Date64Array,
                      void, arrow::Date64Builder, void, void, arrow::date64())

std::shared_ptr<arrow::DataType> FromAnyType(AnyType type);

std::shared_ptr<arrow::Buffer> ToArrowBuffer(
    const std::shared_ptr<Buffer>& buffer);

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

Status CombineRecordBatches(
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::RecordBatch>* batch);

Status CombineRecordBatches(
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table);

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
 * @brief Concatenate multiple arrow tables into one in column wise.
 */
Status ConcatenateTablesColumnWise(
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

namespace arrow_shim {

/**
 * @brief Textual schema representation.
 *
 * @see
 * https://github.com/apache/arrow-rs/blob/27f4762c8794ef1c5d042933562185980eb85ae5/arrow/src/datatypes/datatype.rs#L536
 */
Status SchemaToJSON(const std::shared_ptr<arrow::Schema>& schema, json& object);

Status SchemaFromJSON(const json& object,
                      std::shared_ptr<arrow::Schema>& schema);

Status DataTypeToJSON(const std::shared_ptr<arrow::DataType>& datatype,
                      json& object);

Status DataTypeFromJSON(const json& object,
                        std::shared_ptr<arrow::DataType>& datatype);

Status FieldToJSON(const std::shared_ptr<arrow::Field>& field, json& object);

Status FieldFromJSON(const json& object, std::shared_ptr<arrow::Field>& field);

}  // namespace arrow_shim

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_ARROW_UTILS_H_
