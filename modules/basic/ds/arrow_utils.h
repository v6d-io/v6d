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

#ifndef MODULES_BASIC_DS_ARROW_UTILS_H_
#define MODULES_BASIC_DS_ARROW_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/buffer.h"
#include "arrow/table.h"
#include "arrow/table_builder.h"
#include "arrow/util/config.h"
#include "glog/logging.h"

#include "basic/ds/types.h"
#include "common/util/status.h"

namespace vineyard {

namespace arrow_types {
using OIDT = std::string;
using VIDT = uint64_t;
using EIDT = uint64_t;
}  // namespace arrow_types

struct RefString;

#define CHECK_ARROW_ERROR(expr) \
  VINEYARD_CHECK_OK(::vineyard::Status::ArrowError(expr))

#define CHECK_ARROW_ERROR_AND_ASSIGN(lhs, expr) \
  do {                                          \
    auto status = (expr);                       \
    CHECK_ARROW_ERROR(status.status());         \
    lhs = std::move(status).ValueOrDie();       \
  } while (0)

#define RETURN_ON_ARROW_ERROR(expr)                  \
  do {                                               \
    auto status = (expr);                            \
    if (!status.ok()) {                              \
      return ::vineyard::Status::ArrowError(status); \
    }                                                \
  } while (0)

#define RETURN_ON_ARROW_ERROR_AND_ASSIGN(lhs, expr)           \
  do {                                                        \
    auto result = (expr);                                     \
    if (!result.status().ok()) {                              \
      return ::vineyard::Status::ArrowError(result.status()); \
    }                                                         \
    lhs = std::move(result).ValueOrDie();                     \
  } while (0)

template <typename T>
struct ConvertToArrowType {};

#define CONVERT_TO_ARROW_TYPE(type, array_type, builder_type, type_value)      \
  template <>                                                                  \
  struct ConvertToArrowType<type> {                                            \
    using ArrayType = array_type;                                              \
    using BuilderType = builder_type;                                          \
    static std::shared_ptr<arrow::DataType> TypeValue() { return type_value; } \
  };

CONVERT_TO_ARROW_TYPE(bool, arrow::BooleanArray, arrow::BooleanBuilder,
                      arrow::boolean())
CONVERT_TO_ARROW_TYPE(int8_t, arrow::Int8Array, arrow::Int8Builder,
                      arrow::int8())
CONVERT_TO_ARROW_TYPE(uint8_t, arrow::UInt8Array, arrow::UInt8Builder,
                      arrow::uint8())
CONVERT_TO_ARROW_TYPE(int16_t, arrow::Int16Array, arrow::Int16Builder,
                      arrow::int16())
CONVERT_TO_ARROW_TYPE(uint16_t, arrow::UInt16Array, arrow::UInt16Builder,
                      arrow::uint16())
CONVERT_TO_ARROW_TYPE(int32_t, arrow::Int32Array, arrow::Int32Builder,
                      arrow::int32())
CONVERT_TO_ARROW_TYPE(uint32_t, arrow::UInt32Array, arrow::UInt32Builder,
                      arrow::uint32())
CONVERT_TO_ARROW_TYPE(int64_t, arrow::Int64Array, arrow::Int64Builder,
                      arrow::int64())
CONVERT_TO_ARROW_TYPE(uint64_t, arrow::UInt64Array, arrow::UInt64Builder,
                      arrow::uint64())
CONVERT_TO_ARROW_TYPE(float, arrow::FloatArray, arrow::FloatBuilder,
                      arrow::float32())
CONVERT_TO_ARROW_TYPE(double, arrow::DoubleArray, arrow::DoubleBuilder,
                      arrow::float64())
CONVERT_TO_ARROW_TYPE(RefString, arrow::LargeStringArray,
                      arrow::LargeStringBuilder, arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(std::string, arrow::LargeStringArray,
                      arrow::LargeStringBuilder, arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(arrow::TimestampType, arrow::TimestampArray,
                      arrow::TimestampBuilder,
                      arrow::timestamp(arrow::TimeUnit::MILLI))

std::shared_ptr<arrow::DataType> FromAnyType(AnyType type);

/**
 * @brief PodArrayBuilder is designed for constructing Arrow arrays of POD data
 * type
 *
 * @tparam T
 */
template <typename T>
class PodArrayBuilder : public arrow::FixedSizeBinaryBuilder {
 public:
  explicit PodArrayBuilder(
      arrow::MemoryPool* pool = arrow::default_memory_pool())
      : arrow::FixedSizeBinaryBuilder(arrow::fixed_size_binary(sizeof(T)),
                                      pool) {}

  T* MutablePointer(int64_t i) {
    return reinterpret_cast<T*>(
        arrow::FixedSizeBinaryBuilder::GetMutableValue(i));
  }
};

bool SameShape(std::shared_ptr<arrow::ChunkedArray> ca1,
               std::shared_ptr<arrow::ChunkedArray> ca2);

template <typename ARRAY_TYPE, typename FUNC_TYPE>
Status IterateDualChunkedArray(std::shared_ptr<arrow::ChunkedArray> ca1,
                               std::shared_ptr<arrow::ChunkedArray> ca2,
                               const FUNC_TYPE& func) {
  VINEYARD_ASSERT(SameShape(ca1, ca2),
                  "Two chunked arrays have different shapes");
  size_t chunk_num = ca1->num_chunks();
  size_t index = 0;
  for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
    std::shared_ptr<ARRAY_TYPE> a1 =
        std::dynamic_pointer_cast<ARRAY_TYPE>(ca1->chunk(chunk_i));
    std::shared_ptr<ARRAY_TYPE> a2 =
        std::dynamic_pointer_cast<ARRAY_TYPE>(ca2->chunk(chunk_i));
    size_t len = a1->length();
    for (size_t i = 0; i != len; ++i) {
      func(a1->GetView(i), a2->GetView(i), index++);
    }
  }
  return Status::OK();
}

template <typename ARRAY_TYPE, typename FUNC_TYPE>
inline Status IterateChunkedArray(std::shared_ptr<arrow::ChunkedArray> ca,
                                  const FUNC_TYPE& func) {
  size_t chunk_num = ca->num_chunks();
  size_t index = 0;
  for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
    std::shared_ptr<ARRAY_TYPE> a =
        std::dynamic_pointer_cast<ARRAY_TYPE>(ca->chunk(chunk_i));
    size_t len = a->length();
    for (size_t i = 0; i != len; ++i) {
      func(a->GetView(i), index++);
    }
  }
  return Status::OK();
}

/**
 * @brief The base representation of columns in vineyard
 *
 */
class Column {
 public:
  Column(std::shared_ptr<arrow::ChunkedArray> chunked_array, size_t chunk_size)
      : chunked_array_(chunked_array), chunk_size_(chunk_size) {}
  virtual ~Column() {}

  std::shared_ptr<arrow::DataType> Type() const {
    return chunked_array_->type();
  }

  size_t Length() const { return chunked_array_->length(); }

 protected:
  std::shared_ptr<arrow::ChunkedArray> chunked_array_;
  size_t chunk_size_;
};

/**
 * @brief The representation of concrete columns in vineyard
 *
 * @tparam T
 */
template <typename T>
class ConcreteColumn : public Column {
  using array_type = typename ConvertToArrowType<T>::ArrayType;

 public:
  ConcreteColumn(std::shared_ptr<arrow::ChunkedArray> chunked_array,
                 size_t chunk_size)
      : Column(chunked_array, chunk_size) {}

  T GetView(size_t index) {
    std::shared_ptr<array_type> chunk = std::dynamic_pointer_cast<array_type>(
        Column::chunked_array_->chunk(index / Column::chunk_size_));
    return chunk->GetView(index % Column::chunk_size_);
  }
};

using Int64Column = ConcreteColumn<int64_t>;
using Int32Column = ConcreteColumn<int32_t>;
using UInt64Column = ConcreteColumn<uint64_t>;
using UInt32Column = ConcreteColumn<uint32_t>;
using FloatColumn = ConcreteColumn<float>;
using DoubleColumn = ConcreteColumn<double>;
using StringColumn = ConcreteColumn<RefString>;
using TimestampColumn = ConcreteColumn<arrow::TimestampType>;

std::shared_ptr<Column> CreateColumn(
    std::shared_ptr<arrow::ChunkedArray> chunked_array, size_t chunk_size);

/**
 * Similar to arrow's `GetRecordBatchSize`, but considering the schema.
 *
 * Used for pre-allocate buffer for NewStreamWriter's `WriteRecordBatch`.
 */
Status GetRecordBatchStreamSize(const arrow::RecordBatch& batch, size_t* size);

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

Status CombineRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::RecordBatch>* batch);

Status TableToRecordBatches(
    std::shared_ptr<arrow::Table> table,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches);

Status SerializeTableToAllocatedBuffer(std::shared_ptr<arrow::Table> table,
                                       std::shared_ptr<arrow::Buffer>* buffer);

Status SerializeTable(std::shared_ptr<arrow::Table> table,
                      std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeTable(std::shared_ptr<arrow::Buffer> buffer,
                        std::shared_ptr<arrow::Table>* table);

struct EmptyTableBuilder {
  static Status Build(const std::shared_ptr<arrow::Schema>& schema,
                      std::shared_ptr<arrow::Table>& table) {
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

    for (int i = 0; i < schema->num_fields(); i++) {
      std::shared_ptr<arrow::Array> dummy;
      auto type = schema->field(i)->type();

      if (type == arrow::uint64()) {
        arrow::UInt64Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::int64()) {
        arrow::Int64Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::uint32()) {
        arrow::UInt32Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::int32()) {
        arrow::Int32Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::float32()) {
        arrow::FloatBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::float64()) {
        arrow::DoubleBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::large_utf8()) {
        arrow::LargeStringBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::null()) {
        arrow::NullBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else {
        return Status::NotImplemented("Unsupported type: " + type->ToString());
      }
      columns.push_back(std::make_shared<arrow::ChunkedArray>(dummy));
    }
    table = arrow::Table::Make(schema, columns);
    return Status::OK();
  }
};

template <typename T>
struct AppendHelper {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    return Status::NotImplemented("Unimplemented for type: " +
                                  array->type()->ToString());
  }
};

template <>
struct AppendHelper<uint64_t> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::UInt64Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::UInt64Array>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<int64_t> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::Int64Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::Int64Array>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<uint32_t> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::UInt32Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::UInt32Array>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<int32_t> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::Int32Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::Int32Array>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<float> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::FloatBuilder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::FloatArray>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<double> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::DoubleBuilder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::DoubleArray>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<std::string> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::BinaryBuilder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::BinaryArray>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<arrow::TimestampType> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(
        dynamic_cast<arrow::TimestampBuilder*>(builder)->Append(
            std::dynamic_pointer_cast<arrow::TimestampArray>(array)->GetView(
                offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<void> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(
        dynamic_cast<arrow::NullBuilder*>(builder)->Append(nullptr));
    return Status::OK();
  }
};

typedef Status (*appender_func)(arrow::ArrayBuilder*,
                                std::shared_ptr<arrow::Array>, size_t);

/**
 * @brief TableAppender supports the append operation for tables in vineyard
 *
 */
class TableAppender {
 public:
  explicit TableAppender(std::shared_ptr<arrow::Schema> schema);

  Status Apply(std::unique_ptr<arrow::RecordBatchBuilder>& builder,
               std::shared_ptr<arrow::RecordBatch> batch, size_t offset,
               std::vector<std::shared_ptr<arrow::RecordBatch>>& batches_out);

  Status Flush(std::unique_ptr<arrow::RecordBatchBuilder>& builder,
               std::vector<std::shared_ptr<arrow::RecordBatch>>& batches_out);

 private:
  std::vector<appender_func> funcs_;
  size_t col_num_;
};

/**
 * @brief Concatenate multiple arrow tables into one.
 */
std::shared_ptr<arrow::Table> ConcatenateTables(
    std::vector<std::shared_ptr<arrow::Table>>& tables);

/**
 * @brief Convert type name in string to arrow type.
 *
 */
inline std::shared_ptr<arrow::DataType> type_name_to_arrow_type(
    const std::string& name) {
  if (name == "bool") {
    return vineyard::ConvertToArrowType<bool>::TypeValue();
  } else if (name == "int8_t" || name == "int8" || name == "byte") {
    return vineyard::ConvertToArrowType<int8_t>::TypeValue();
  } else if (name == "uint8_t" || name == "uint8" || name == "char") {
    return vineyard::ConvertToArrowType<uint8_t>::TypeValue();
  } else if (name == "int16_t" || name == "int16" || name == "half") {
    return vineyard::ConvertToArrowType<int16_t>::TypeValue();
  } else if (name == "uint16_t" || name == "uint16") {
    return vineyard::ConvertToArrowType<uint16_t>::TypeValue();
  } else if (name == "int32_t" || name == "int32" || name == "int") {
    return vineyard::ConvertToArrowType<int32_t>::TypeValue();
  } else if (name == "uint32_t" || name == "uint32") {
    return vineyard::ConvertToArrowType<uint32_t>::TypeValue();
  } else if (name == "int64_t" || name == "int64" || name == "long") {
    return vineyard::ConvertToArrowType<int64_t>::TypeValue();
  } else if (name == "uint64_t" || name == "uint64") {
    return vineyard::ConvertToArrowType<uint64_t>::TypeValue();
  } else if (name == "float") {
    return vineyard::ConvertToArrowType<float>::TypeValue();
  } else if (name == "double") {
    return vineyard::ConvertToArrowType<double>::TypeValue();
  } else if (name == "string" || name == "std::string" || name == "str" ||
             name == "std::__1::string") {
    return vineyard::ConvertToArrowType<std::string>::TypeValue();
  } else if (name == "null" || name == "NULL") {
    return arrow::null();
  } else {
    LOG(ERROR) << "Unsupported data type: " << name;
    return nullptr;
  }
}

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_ARROW_UTILS_H_
