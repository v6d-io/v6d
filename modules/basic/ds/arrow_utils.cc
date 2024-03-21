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

#include "basic/ds/arrow_utils.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>

#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "arrow/ipc/api.h"  // IWYU pragma: keep
#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/join.hpp"
#include "boost/algorithm/string/split.hpp"

#include "client/ds/blob.h"
#include "client/ds/remote_blob.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "common/util/typename.h"

namespace vineyard {

std::shared_ptr<arrow::DataType> FromAnyType(AnyType type) {
  switch (type) {
  case AnyType::Undefined:
    return arrow::null();
  case AnyType::Int32:
    return arrow::int32();
  case AnyType::UInt32:
    return arrow::uint32();
  case AnyType::Int64:
    return arrow::int64();
  case AnyType::UInt64:
    return arrow::uint64();
  case AnyType::Float:
    return arrow::float32();
  case AnyType::Double:
    return arrow::float64();
  case AnyType::String:
    return arrow::large_utf8();
  case AnyType::Date32:
    return arrow::date32();
  case AnyType::Date64:
    return arrow::date64();
  case AnyType::Time32:
    return arrow::time32(DefaultTimeUnit);
  case AnyType::Time64:
    return arrow::time64(DefaultTimeUnit);
  case AnyType::Timestamp:
    return arrow::timestamp(DefaultTimeUnit);
  default:
    return arrow::null();
  }
}

std::shared_ptr<arrow::Buffer> ToArrowBuffer(
    const std::shared_ptr<Buffer>& buffer) {
  if (buffer == nullptr) {
    return nullptr;
  }
  return arrow::Buffer::Wrap(buffer->data(), buffer->size());
}

const std::shared_ptr<arrow::Buffer> Blob::ArrowBuffer() const {
  return ToArrowBuffer(this->Buffer());
}

const std::shared_ptr<arrow::Buffer> Blob::ArrowBufferOrEmpty() const {
  return ToArrowBuffer(this->BufferOrEmpty());
}

const std::shared_ptr<arrow::Buffer> RemoteBlob::ArrowBuffer() const {
  return ToArrowBuffer(this->Buffer());
}

const std::shared_ptr<arrow::Buffer> RemoteBlob::ArrowBufferOrEmpty() const {
  return ToArrowBuffer(this->BufferOrEmpty());
}

namespace detail {

Status Copy(std::shared_ptr<arrow::ArrayData> const& array,
            std::shared_ptr<arrow::ArrayData>& out, bool shallow,
            arrow::MemoryPool* pool) {
  if (array == nullptr) {
    out = array;
    return Status::OK();
  }
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  for (auto const& buffer : array->buffers) {
    if (buffer == nullptr || buffer->size() == 0) {
      buffers.push_back(buffer);
    } else {
      if (shallow) {
        buffers.push_back(buffer);
      } else {
        std::shared_ptr<arrow::Buffer> buf;
        RETURN_ON_ARROW_ERROR_AND_ASSIGN(
            buf, buffer->CopySlice(0, buffer->size(), pool));
        buffers.push_back(buf);
      }
    }
  }
  std::vector<std::shared_ptr<arrow::ArrayData>> child_data;
  for (auto const& child : array->child_data) {
    std::shared_ptr<arrow::ArrayData> data;
    RETURN_ON_ERROR(Copy(child, data, shallow, pool));
    child_data.push_back(data);
  }
  std::shared_ptr<arrow::ArrayData> directory;
  RETURN_ON_ERROR(Copy(array->dictionary, directory, shallow, pool));
  out = arrow::ArrayData::Make(array->type, array->length, buffers, child_data,
                               directory, array->null_count, array->offset);
  return Status::OK();
}

Status Copy(std::shared_ptr<arrow::Array> const& array,
            std::shared_ptr<arrow::Array>& out, bool shallow,
            arrow::MemoryPool* pool) {
  if (array == nullptr) {
    out = array;
    return Status::OK();
  }
  std::shared_ptr<arrow::ArrayData> data;
  RETURN_ON_ERROR(Copy(array->data(), data, shallow, pool));
  out = arrow::MakeArray(data);
  return Status::OK();
}

Status Copy(std::shared_ptr<arrow::ChunkedArray> const& array,
            std::shared_ptr<arrow::ChunkedArray>& out, bool shallow,
            arrow::MemoryPool* pool) {
  if (array == nullptr) {
    out = array;
    return Status::OK();
  }
  std::vector<std::shared_ptr<arrow::Array>> chunks;
  for (auto const& chunk : array->chunks()) {
    std::shared_ptr<arrow::Array> data;
    RETURN_ON_ERROR(Copy(chunk, data, shallow, pool));
    chunks.push_back(data);
  }
  out = std::make_shared<arrow::ChunkedArray>(chunks, array->type());
  return Status::OK();
}

Status Copy(std::shared_ptr<arrow::RecordBatch> const& batch,
            std::shared_ptr<arrow::RecordBatch>& out, bool shallow,
            arrow::MemoryPool* pool) {
  if (batch == nullptr) {
    out = batch;
    return Status::OK();
  }
  arrow::ArrayDataVector columns_data;
  for (auto const& column : batch->column_data()) {
    std::shared_ptr<arrow::ArrayData> data;
    RETURN_ON_ERROR(Copy(column, data, shallow, pool));
    columns_data.push_back(data);
  }
  out = arrow::RecordBatch::Make(batch->schema(), batch->num_rows(),
                                 columns_data);
  return Status::OK();
}

Status Copy(std::shared_ptr<arrow::Table> const& table,
            std::shared_ptr<arrow::Table>& out, bool shallow,
            arrow::MemoryPool* pool) {
  if (table == nullptr) {
    out = table;
    return Status::OK();
  }
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  for (auto const& column : table->columns()) {
    std::shared_ptr<arrow::ChunkedArray> data;
    RETURN_ON_ERROR(Copy(column, data, shallow, pool));
    columns.push_back(data);
  }
  out = arrow::Table::Make(table->schema(), columns);
  return Status::OK();
}

}  // namespace detail

Status EmptyTableBuilder::Build(const std::shared_ptr<arrow::Schema>& schema,
                                std::shared_ptr<arrow::Table>& table) {
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

  for (int i = 0; i < schema->num_fields(); i++) {
    std::shared_ptr<arrow::Array> dummy;
    auto type = schema->field(i)->type();

    if (arrow::boolean()->Equals(type)) {
      arrow::BooleanBuilder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::uint64()->Equals(type)) {
      arrow::UInt64Builder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::int64()->Equals(type)) {
      arrow::Int64Builder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::uint32()->Equals(type)) {
      arrow::UInt32Builder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::int32()->Equals(type)) {
      arrow::Int32Builder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::float32()->Equals(type)) {
      arrow::FloatBuilder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::float64()->Equals(type)) {
      arrow::DoubleBuilder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::utf8()->Equals(type)) {
      arrow::StringBuilder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::large_utf8()->Equals(type)) {
      arrow::LargeStringBuilder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::date32()->Equals(type)) {
      arrow::Date32Builder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::date64()->Equals(type)) {
      arrow::Date64Builder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (type->id() == arrow::Type::TIME32) {
      arrow::Time32Builder builder(type, arrow::default_memory_pool());
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (type->id() == arrow::Type::TIME64) {
      arrow::Time64Builder builder(type, arrow::default_memory_pool());
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (type->id() == arrow::Type::TIMESTAMP) {
      arrow::TimestampBuilder builder(type, arrow::default_memory_pool());
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (arrow::list(arrow::uint64())->Equals(type)) {
      auto builder = std::make_shared<arrow::UInt64Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (arrow::list(arrow::int64())->Equals(type)) {
      auto builder = std::make_shared<arrow::Int64Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (arrow::list(arrow::uint32())->Equals(type)) {
      auto builder = std::make_shared<arrow::UInt32Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (arrow::list(arrow::int32())->Equals(type)) {
      auto builder = std::make_shared<arrow::Int32Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (arrow::list(arrow::float64())->Equals(type)) {
      auto builder = std::make_shared<arrow::DoubleBuilder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (arrow::list(arrow::int64())->Equals(type)) {
      auto builder = std::make_shared<arrow::FloatBuilder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (arrow::null()->Equals(type)) {
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

std::shared_ptr<arrow::Schema> EmptyTableBuilder::EmptySchema() {
#if defined(ARROW_VERSION) && ARROW_VERSION >= 4000000
  return std::shared_ptr<arrow::Schema>(
      new arrow::Schema({}, arrow::Endianness::Native));
#else
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({}));
#endif
}

Status GetRecordBatchStreamSize(const arrow::RecordBatch& batch, size_t* size) {
  // emulates the behavior of Write without actually writing
  arrow::io::MockOutputStream dst;

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 2000000
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::NewStreamWriter(&dst, batch.schema()));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::MakeStreamWriter(&dst, batch.schema()));
#endif
  RETURN_ON_ARROW_ERROR(writer->WriteRecordBatch(batch));
  RETURN_ON_ARROW_ERROR(writer->Close());
  *size = dst.GetExtentBytesWritten();
  return Status::OK();
}

Status SerializeDataType(const std::shared_ptr<arrow::DataType>& type,
                         std::shared_ptr<arrow::Buffer>* buffer) {
  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{
          std::make_shared<arrow::Field>("_", type)});
  return SerializeSchema(*schema, buffer);
}

Status DeserializeDataType(const std::shared_ptr<arrow::Buffer>& buffer,
                           std::shared_ptr<arrow::DataType>* type) {
  std::shared_ptr<arrow::Schema> schema;
  RETURN_ON_ERROR(DeserializeSchema(buffer, &schema));
  *type = schema->field(0)->type();
  return Status::OK();
}

Status SerializeSchema(const arrow::Schema& schema,
                       std::shared_ptr<arrow::Buffer>* buffer) {
#if defined(ARROW_VERSION) && ARROW_VERSION < 2000000
  arrow::ipc::DictionaryMemo out_memo;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      *buffer, arrow::ipc::SerializeSchema(schema, out_memo,
                                           arrow::default_memory_pool()));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      *buffer,
      arrow::ipc::SerializeSchema(schema, arrow::default_memory_pool()));
#endif
  return Status::OK();
}

Status DeserializeSchema(const std::shared_ptr<arrow::Buffer>& buffer,
                         std::shared_ptr<arrow::Schema>* schema) {
  arrow::ipc::DictionaryMemo memo;
  arrow::io::BufferReader reader(buffer);
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(*schema,
                                   arrow::ipc::ReadSchema(&reader, &memo));
  return Status::OK();
}

Status SerializeRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch,
                            std::shared_ptr<arrow::Buffer>* buffer) {
  std::shared_ptr<arrow::io::BufferOutputStream> out_stream;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(out_stream,
                                   arrow::io::BufferOutputStream::Create(1024));
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      {batch}, arrow::ipc::IpcOptions::Defaults(), out_stream.get()));
#else
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      {batch}, arrow::ipc::IpcWriteOptions::Defaults(), out_stream.get()));
#endif
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(*buffer, out_stream->Finish());
  return Status::OK();
}

Status DeserializeRecordBatch(const std::shared_ptr<arrow::Buffer>& buffer,
                              std::shared_ptr<arrow::RecordBatch>* batch) {
  if (buffer == nullptr || buffer->size() == 0) {
    return Status::Invalid(
        "Unable to deserialize to recordbatch: buffer is empty");
  }
  arrow::io::BufferReader reader(buffer);
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      batch_reader, arrow::ipc::RecordBatchStreamReader::Open(&reader));
  RETURN_ON_ARROW_ERROR(batch_reader->ReadNext(batch));
  return Status::OK();
}

Status SerializeRecordBatchesToAllocatedBuffer(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer) {
  arrow::io::FixedSizeBufferWriter stream(*buffer);
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      batches, arrow::ipc::IpcOptions::Defaults(), &stream));
#else
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      batches, arrow::ipc::IpcWriteOptions::Defaults(), &stream));
#endif
  return Status::OK();
}

Status SerializeRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer) {
  std::shared_ptr<arrow::io::BufferOutputStream> out_stream;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(out_stream,
                                   arrow::io::BufferOutputStream::Create(1024));
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      batches, arrow::ipc::IpcOptions::Defaults(), out_stream.get()));
#else
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      batches, arrow::ipc::IpcWriteOptions::Defaults(), out_stream.get()));
#endif
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(*buffer, out_stream->Finish());
  return Status::OK();
}

Status DeserializeRecordBatches(
    const std::shared_ptr<arrow::Buffer>& buffer,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches) {
  arrow::io::BufferReader reader(buffer);
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      batch_reader, arrow::ipc::RecordBatchStreamReader::Open(&reader));
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  RETURN_ON_ARROW_ERROR(batch_reader->ReadAll(batches));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(*batches, batch_reader->ToRecordBatches());
#endif
  return Status::OK();
}

Status RecordBatchesToTable(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table) {
  return RecordBatchesToTable(nullptr, batches, table);
}

Status RecordBatchesToTable(
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table) {
  if (batches.size() > 0) {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(*table,
                                     arrow::Table::FromRecordBatches(batches));
    return Status::OK();
  } else {
    if (schema != nullptr) {
      return EmptyTableBuilder::Build(schema, *table);
    } else {
      return Status::Invalid("Unable to create empty table without schema");
    }
  }
}

Status RecordBatchesToTableWithCast(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table) {
  std::shared_ptr<arrow::Schema> out;
  RETURN_ON_ERROR(TypeLoosen(batches, out));
  return RecordBatchesToTableWithCast(out, batches, table);
}

Status RecordBatchesToTableWithCast(
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> outs;
  for (auto const& batch : batches) {
    std::shared_ptr<arrow::RecordBatch> out;
    RETURN_ON_ERROR(CastBatchToSchema(batch, schema, out));
    outs.push_back(out);
  }
  return RecordBatchesToTable(schema, outs, table);
}

Status CombineRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::RecordBatch>* batch) {
  return CombineRecordBatches(nullptr, batches, batch);
}

Status CombineRecordBatches(
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::Table> table, combined_table;
  RETURN_ON_ERROR(RecordBatchesToTable(schema, batches, &table));
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      combined_table, table->CombineChunks(arrow::default_memory_pool()));
  arrow::TableBatchReader reader(*combined_table);
  RETURN_ON_ARROW_ERROR(reader.ReadNext(batch));
  std::shared_ptr<arrow::RecordBatch> test_batch;
  RETURN_ON_ARROW_ERROR(reader.ReadNext(&test_batch));
  RETURN_ON_ASSERT(test_batch == nullptr);
  return Status::OK();
}

Status CombineRecordBatches(
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table) {
  std::shared_ptr<arrow::Table> chunked_table;
  RETURN_ON_ERROR(RecordBatchesToTable(schema, batches, &chunked_table));
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      *table, chunked_table->CombineChunks(arrow::default_memory_pool()));
  return Status::OK();
}

Status TableToRecordBatches(
    const std::shared_ptr<arrow::Table> table,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches) {
  std::shared_ptr<arrow::Table> fixed_table = table;
  if (fixed_table->num_rows() == 0) {
    // ensure there are chunks, even empty chunks
    RETURN_ON_ERROR(
        EmptyTableBuilder::Build(fixed_table->schema(), fixed_table));
  }
  if (fixed_table->num_columns() == 0) {
    auto batch =
        arrow::RecordBatch::Make(fixed_table->schema(), fixed_table->num_rows(),
                                 std::vector<std::shared_ptr<arrow::Array>>{});
    batches->push_back(batch);
    return Status::OK();
  }

  // chunks[row_index][column_index]
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> chunks;
  for (int64_t column_index = 0; column_index < fixed_table->num_columns();
       ++column_index) {
    for (int64_t row_index = 0;
         row_index < fixed_table->column(column_index)->num_chunks();
         ++row_index) {
      chunks.resize(fixed_table->column(column_index)->num_chunks());
      chunks[row_index].push_back(
          fixed_table->column(column_index)->chunk(row_index));
    }
  }
  batches->clear();
  for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
    auto batch = arrow::RecordBatch::Make(fixed_table->schema(),
                                          chunks[chunk_index][0]->length(),
                                          chunks[chunk_index]);
    batches->push_back(batch);
  }
  return Status::OK();
}

std::shared_ptr<arrow::ChunkedArray> ConcatenateChunkedArrays(
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& arrays) {
  std::shared_ptr<arrow::DataType> dtype;
  std::vector<std::shared_ptr<arrow::Array>> chunks;
  for (auto const& array : arrays) {
    if (array) {
      dtype = array->type();
      for (int64_t i = 0; i < array->num_chunks(); ++i) {
        chunks.push_back(array->chunk(i));
      }
    }
  }
  if (chunks.empty()) {
    return nullptr;
  } else {
    return std::make_shared<arrow::ChunkedArray>(chunks, dtype);
  }
}

std::shared_ptr<arrow::ChunkedArray> ConcatenateChunkedArrays(
    const std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
        arrays) {
  std::vector<std::shared_ptr<arrow::ChunkedArray>> chunked_arrays;
  for (auto const& array : arrays) {
    chunked_arrays.insert(chunked_arrays.begin(), array.begin(), array.end());
  }
  return ConcatenateChunkedArrays(chunked_arrays);
}

Status SerializeTableToAllocatedBuffer(
    const std::shared_ptr<arrow::Table> table,
    std::shared_ptr<arrow::Buffer>* buffer) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(TableToRecordBatches(table, &batches));
  RETURN_ON_ERROR(SerializeRecordBatchesToAllocatedBuffer(batches, buffer));
  return Status::OK();
}

Status SerializeTable(const std::shared_ptr<arrow::Table> table,
                      std::shared_ptr<arrow::Buffer>* buffer) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(TableToRecordBatches(table, &batches));
  RETURN_ON_ERROR(SerializeRecordBatches(batches, buffer));
  return Status::OK();
}

Status DeserializeTable(const std::shared_ptr<arrow::Buffer> buffer,
                        std::shared_ptr<arrow::Table>* table) {
  arrow::io::BufferReader reader(buffer);
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      batch_reader, arrow::ipc::RecordBatchStreamReader::Open(&reader));
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  RETURN_ON_ARROW_ERROR(batch_reader->ReadAll(table));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(*table, batch_reader->ToTable());
#endif
  return Status::OK();
}

Status ConcatenateTables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables,
    std::shared_ptr<arrow::Table>& table) {
  if (tables.size() == 1) {
    table = tables[0];
    return Status::OK();
  }
  std::vector<std::shared_ptr<arrow::Table>> out_tables(tables.size());
  out_tables[0] = tables[0];
  auto col_names = tables[0]->ColumnNames();
  for (size_t i = 1; i < tables.size(); ++i) {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(out_tables[i],
                                     tables[i]->RenameColumns(col_names));
  }
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(table, arrow::ConcatenateTables(out_tables));
  return Status::OK();
}

Status ConcatenateTablesColumnWise(
    const std::vector<std::shared_ptr<arrow::Table>>& tables,
    std::shared_ptr<arrow::Table>& table) {
  if (tables.size() == 1) {
    table = tables[0];
    return Status::OK();
  }
  table = tables[0];
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns = table->columns();
  std::vector<std::shared_ptr<arrow::Field>> fields = table->fields();
  for (size_t i = 1; i < tables.size(); ++i) {
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& right_columns =
        tables[i]->columns();
    columns.insert(columns.end(), right_columns.begin(), right_columns.end());

    const std::vector<std::shared_ptr<arrow::Field>>& right_fields =
        tables[i]->fields();
    fields.insert(fields.end(), right_fields.begin(), right_fields.end());
  }
  table =
      arrow::Table::Make(arrow::schema(std::move(fields)), std::move(columns));
  return Status::OK();
}

std::shared_ptr<arrow::RecordBatch> AddMetadataToRecordBatch(
    std::shared_ptr<arrow::RecordBatch> const& batch,
    std::map<std::string, std::string> const& meta) {
  if (batch == nullptr || meta.empty()) {
    return batch;
  }
  std::shared_ptr<arrow::KeyValueMetadata> metadata;
  if (batch->schema()->metadata() != nullptr) {
    metadata = batch->schema()->metadata()->Copy();
  } else {
    metadata.reset(new arrow::KeyValueMetadata());
  }

  for (auto const& kv : meta) {
    CHECK_ARROW_ERROR(metadata->Set(kv.first, kv.second));
  }
  return batch->ReplaceSchemaMetadata(metadata);
}

std::shared_ptr<arrow::RecordBatch> AddMetadataToRecordBatch(
    std::shared_ptr<arrow::RecordBatch> const& batch,
    std::unordered_map<std::string, std::string> const& meta) {
  if (batch == nullptr || meta.empty()) {
    return batch;
  }
  std::shared_ptr<arrow::KeyValueMetadata> metadata;
  if (batch->schema()->metadata() != nullptr) {
    metadata = batch->schema()->metadata()->Copy();
  } else {
    metadata.reset(new arrow::KeyValueMetadata());
  }

  for (auto const& kv : meta) {
    CHECK_ARROW_ERROR(metadata->Set(kv.first, kv.second));
  }
  return batch->ReplaceSchemaMetadata(metadata);
}

namespace detail {

inline std::string type_name_from_arrow_date_unit(
    arrow::TimeUnit::type const& unit) {
  switch (unit) {
  case arrow::TimeUnit::SECOND:
    return "[S]";
  case arrow::TimeUnit::MILLI:
    return "[MS]";
  case arrow::TimeUnit::MICRO:
    return "[US]";
  case arrow::TimeUnit::NANO:
    return "[NS]";
  default:
    return "Unsupported time unit: '" + std::to_string(static_cast<int>(unit)) +
           "'";
  }
}

inline arrow::TimeUnit::type type_name_to_arrow_date_unit(const char* unit) {
  if (std::strncmp(unit, "[S]", 3) == 0) {
    return arrow::TimeUnit::SECOND;
  } else if (std::strncmp(unit, "[MS]", 4) == 0) {
    return arrow::TimeUnit::MILLI;
  } else if (std::strncmp(unit, "[US]", 4) == 0) {
    return arrow::TimeUnit::MICRO;
  } else if (std::strncmp(unit, "[NS]", 4) == 0) {
    return arrow::TimeUnit::NANO;
  } else {
    LOG(ERROR) << "Unsupported time unit: '" << unit << "'";
    return arrow::TimeUnit::SECOND;
  }
}

}  // namespace detail

std::shared_ptr<arrow::DataType> type_name_to_arrow_type(
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
  } else if (name == "uint32_t" || name == "uint32" || name == "uint") {
    return vineyard::ConvertToArrowType<uint32_t>::TypeValue();
  } else if (name == "int64_t" || name == "int64" || name == "long") {
    return vineyard::ConvertToArrowType<int64_t>::TypeValue();
  } else if (name == "uint64_t" || name == "uint64" || name == "ulong") {
    return vineyard::ConvertToArrowType<uint64_t>::TypeValue();
  } else if (name == "float") {
    return vineyard::ConvertToArrowType<float>::TypeValue();
  } else if (name == "double") {
    return vineyard::ConvertToArrowType<double>::TypeValue();
  } else if (name == "string" || name == "std::string" || name == "str" ||
             name == "std::__1::string" || name == "std::__cxx11::string") {
    return vineyard::ConvertToArrowType<std::string>::TypeValue();
  } else if (name == "date32[day]") {
    return vineyard::ConvertToArrowType<arrow::Date32Type>::TypeValue();
  } else if (name == "date64[ms]") {
    return vineyard::ConvertToArrowType<arrow::Date64Type>::TypeValue();
  } else if (name.substr(0, std::string("time[32]").length()) ==
             std::string("time[32]")) {
    const std::string unit_content =
        name.substr(std::string("time[32]").length());
    arrow::TimeUnit::type unit = DefaultTimeUnit;
    if (unit_content.length() >= 3) {
      unit = detail::type_name_to_arrow_date_unit(unit_content.c_str());
    }
    return arrow::time32(unit);
  } else if (name.substr(0, std::string("time[64]").length()) ==
             std::string("time[64]")) {
    const std::string unit_content =
        name.substr(std::string("time[64]").length());
    arrow::TimeUnit::type unit = DefaultTimeUnit;
    if (unit_content.length() >= 3) {
      unit = detail::type_name_to_arrow_date_unit(unit_content.c_str());
    }
    return arrow::time64(unit);
  } else if (name.substr(0, std::string("timestamp").length()) ==
             std::string("timestamp")) {
    const std::string unit_content =
        name.substr(std::string("timestamp").length());
    arrow::TimeUnit::type unit = DefaultTimeUnit;
    if (unit_content.length() >= 3) {
      unit = detail::type_name_to_arrow_date_unit(unit_content.c_str());
      size_t timezone_start =
          std::string("timestamp").length() +
          detail::type_name_from_arrow_date_unit(unit).length() + 1;
      size_t timezone_end = name.length() - 1;
      if (timezone_end <= timezone_start) {
        return arrow::timestamp(unit);
      }
      std::string timezone =
          name.substr(timezone_start, timezone_end - timezone_start);
      return arrow::timestamp(unit, timezone);
    }
    return arrow::timestamp(unit);
  } else if (name.substr(0, std::string("list<item: ").length()) ==
             std::string("list<item: ")) {
    std::string inner_type_name =
        name.substr(std::string("list<item: ").length(),
                    name.length() - std::string("list<item: ").length() - 1);
    return arrow::list(type_name_to_arrow_type(inner_type_name));
  } else if (name.substr(0, std::string("large_list<item: ").length()) ==
             std::string("large_list<item: ")) {
    std::string inner_type_name = name.substr(
        std::string("large_list<item: ").length(),
        name.length() - std::string("large_list<item: ").length() - 1);
    return arrow::large_list(type_name_to_arrow_type(inner_type_name));
  } else if (name.substr(0, std::string("fixed_size_list<item: ").length()) ==
             std::string("fixed_size_list<item: ")) {
    auto pos = name.find_first_of('[');
    std::string inner_type_name =
        name.substr(std::string("fixed_size_list<item: ").length(),
                    pos - std::string("fixed_size_list<item: ").length() - 1);
    auto size = std::stoi(name.substr(pos + 1, name.length() - pos - 2));
    return arrow::fixed_size_list(type_name_to_arrow_type(inner_type_name),
                                  size);
  } else if (name == "null" || name == "NULL") {
    return arrow::null();
  } else {
    LOG(ERROR) << "Unsupported data type: '" << name << "'";
    return arrow::null();
  }
}

std::string type_name_from_arrow_type(
    std::shared_ptr<arrow::DataType> const& type) {
  if (arrow::null()->Equals(type)) {
    return "null";
  } else if (vineyard::ConvertToArrowType<bool>::TypeValue()->Equals(type)) {
    return type_name<bool>();
  } else if (vineyard::ConvertToArrowType<int8_t>::TypeValue()->Equals(type)) {
    return type_name<int8_t>();
  } else if (vineyard::ConvertToArrowType<uint8_t>::TypeValue()->Equals(type)) {
    return type_name<uint8_t>();
  } else if (vineyard::ConvertToArrowType<int16_t>::TypeValue()->Equals(type)) {
    return type_name<int16_t>();
  } else if (vineyard::ConvertToArrowType<uint16_t>::TypeValue()->Equals(
                 type)) {
    return type_name<uint16_t>();
  } else if (vineyard::ConvertToArrowType<int32_t>::TypeValue()->Equals(type)) {
    return type_name<int32_t>();
  } else if (vineyard::ConvertToArrowType<uint32_t>::TypeValue()->Equals(
                 type)) {
    return type_name<uint32_t>();
  } else if (vineyard::ConvertToArrowType<int64_t>::TypeValue()->Equals(type)) {
    return type_name<int64_t>();
  } else if (vineyard::ConvertToArrowType<uint64_t>::TypeValue()->Equals(
                 type)) {
    return type_name<uint64_t>();
  } else if (vineyard::ConvertToArrowType<float>::TypeValue()->Equals(type)) {
    return type_name<float>();
  } else if (vineyard::ConvertToArrowType<double>::TypeValue()->Equals(type)) {
    return type_name<double>();
  } else if (vineyard::ConvertToArrowType<std::string>::TypeValue()->Equals(
                 type)) {
    return type_name<std::string>();
  } else if (vineyard::ConvertToArrowType<arrow::Date32Type>::TypeValue()
                 ->Equals(type)) {
    return "date32[day]";
  } else if (vineyard::ConvertToArrowType<arrow::Date64Type>::TypeValue()
                 ->Equals(type)) {
    return "date64[ms]";
  } else if (type->id() == arrow::Type::TIME32) {
    auto time32_type = std::dynamic_pointer_cast<arrow::Time32Type>(type);
    const std::string unit =
        detail::type_name_from_arrow_date_unit(time32_type->unit());
    return "time[32]" + unit;
  } else if (type->id() == arrow::Type::TIME64) {
    auto time64_type = std::dynamic_pointer_cast<arrow::Time64Type>(type);
    const std::string unit =
        detail::type_name_from_arrow_date_unit(time64_type->unit());
    return "time[64]" + unit;
  } else if (type->id() == arrow::Type::TIMESTAMP) {
    auto timestamp_type = std::dynamic_pointer_cast<arrow::TimestampType>(type);
    const std::string unit =
        detail::type_name_from_arrow_date_unit(timestamp_type->unit());
    return "timestamp" + unit + "[" + timestamp_type->timezone() + "]";
  } else if (type != nullptr && type->id() == arrow::Type::LIST) {
    auto list_type = std::static_pointer_cast<arrow::ListType>(type);
    return "list<item: " + type_name_from_arrow_type(list_type->value_type()) +
           ">";
  } else if (type != nullptr && type->id() == arrow::Type::LARGE_LIST) {
    auto list_type = std::static_pointer_cast<arrow::LargeListType>(type);
    return "large_list<item: " +
           type_name_from_arrow_type(list_type->value_type()) + ">";
  } else if (type != nullptr && type->id() == arrow::Type::FIXED_SIZE_LIST) {
    auto list_type = std::static_pointer_cast<arrow::FixedSizeListType>(type);
    return "fixed_size_list<item: " +
           type_name_from_arrow_type(list_type->value_type()) + ">[" +
           std::to_string(list_type->list_size()) + "]";
  } else {
    LOG(ERROR) << "Unsupported arrow type '" << type->ToString()
               << "', type id: " << type->id();
    return "undefined";
  }
}

const void* get_arrow_array_data(std::shared_ptr<arrow::Array> const& array) {
  if (array->type()->Equals(arrow::int8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int8Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt8Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::int16())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int16Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint16())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt16Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::int32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int32Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt32Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::int64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int64Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt64Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::float32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::FloatArray>(array)->raw_values());
  } else if (array->type()->Equals(arrow::float64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::DoubleArray>(array)->raw_values());
  } else if (array->type()->Equals(arrow::utf8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::StringArray>(array).get());
  } else if (array->type()->Equals(arrow::large_utf8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::LargeStringArray>(array).get());
  } else if (array->type()->Equals(arrow::date32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Date32Array>(array).get());
  } else if (array->type()->Equals(arrow::date64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Date64Array>(array).get());
  } else if (array->type()->id() == arrow::Type::TIME32) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Time32Array>(array).get());
  } else if (array->type()->id() == arrow::Type::TIME64) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Time64Array>(array).get());
  } else if (array->type()->id() == arrow::Type::TIMESTAMP) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::TimestampArray>(array).get());
  } else if (array->type()->id() == arrow::Type::LIST) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::ListArray>(array).get());
  } else if (array->type()->id() == arrow::Type::LARGE_LIST) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::LargeListArray>(array).get());
  } else if (array->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::FixedSizeListArray>(array).get());
  } else if (array->type()->Equals(arrow::null())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::NullArray>(array).get());
  } else {
    LOG(ERROR) << "Unsupported arrow array type '" << array->type()->ToString()
               << "', type id: " << array->type()->id();
    return NULL;
  }
}

Status TypeLoosen(const std::vector<std::shared_ptr<arrow::Schema>>& schemas,
                  std::shared_ptr<arrow::Schema>& schema) {
  int field_num = -1;
  std::shared_ptr<arrow::KeyValueMetadata> metadata(
      new arrow::KeyValueMetadata());
  for (const auto& schema : schemas) {
    if (schema != nullptr) {
      RETURN_ON_ASSERT(
          field_num == -1 || field_num == schema->num_fields(),
          "Inconsistent field number in those schemas that will be unified");
      field_num = schema->num_fields();
      if (schema->metadata() != nullptr) {
        std::unordered_map<std::string, std::string> metakv;
        schema->metadata()->ToUnorderedMap(&metakv);
        for (auto const& kv : metakv) {
          metadata->Append(kv.first, kv.second);
        }
      }
    }
  }
  RETURN_ON_ASSERT(field_num > 0,
                   "Empty table list cannot be used for normalizing schema");

  // Perform type lossen.
  // int64 -> double -> utf8   binary (not supported)

  // Timestamp value are stored as as number of seconds, milliseconds,
  // microseconds or nanoseconds since UNIX epoch.
  // CSV reader can only produce timestamp in seconds.
  std::vector<std::vector<std::shared_ptr<arrow::Field>>> fields(field_num);
  for (int i = 0; i < field_num; ++i) {
    for (const auto& schema : schemas) {
      if (schema != nullptr) {
        fields[i].push_back(schema->field(i));
      }
    }
  }
  std::vector<std::shared_ptr<arrow::Field>> lossen_fields(field_num);

  for (int i = 0; i < field_num; ++i) {
    lossen_fields[i] = fields[i][0];
    auto res = fields[i][0]->type();
    if (res == arrow::null()) {
      continue;
    }
    if (res->Equals(arrow::boolean())) {
      res = arrow::int32();
    }
    if (res->Equals(arrow::int64())) {
      for (size_t j = 1; j < fields[i].size(); ++j) {
        if (fields[i][j]->type()->Equals(arrow::float64())) {
          res = arrow::float64();
        }
      }
    }
    if (res->Equals(arrow::float64())) {
      for (size_t j = 1; j < fields[i].size(); ++j) {
        if (fields[i][j]->type()->Equals(arrow::utf8())) {
          res = arrow::utf8();
        }
      }
    }
    if (res->Equals(arrow::utf8())) {
      res = arrow::large_utf8();
    }
    // Note [date, time, and timestamp conversion rules]
    //
    // GIE has specific own unit and timezone conversion for dates, times and
    // timestamps, see also:
    // https://github.com/alibaba/GraphScope/blob/main/interactive_engine/executor/ir/proto/common.proto#L58-L72
    //
    // More specifically,
    //
    // - Date32: for int32 days since 1970-01-01
    // - Time32: for int32 milliseconds past midnight
    // - Timestamp: int64 milliseconds since 1970-01-01 00:00:00.000000 (in an
    // unspecified timezone)
    //              the default timezone when parsing value is UTC in GIE.
    //
    // Thus we got the following conversion rules:
    //
    //  - Date32: no change
    //  - Date64 -> Timestamp
    //  - Time32_* -> Time32_MS
    //  - Time64_* -> Time32_MS
    //  - Timestamp_* -> Timestamp_MS_UTC
    if (res->Equals(arrow::date32())) {
      res = arrow::date32();
    }
    if (res->Equals(arrow::date64())) {
      res = arrow::timestamp(arrow::TimeUnit::MILLI, "UTC");
    }
    if (res->id() == arrow::Type::TIME32) {
      res = arrow::time32(arrow::TimeUnit::MILLI);
    }
    if (res->id() == arrow::Type::TIME64) {
      res = arrow::time32(arrow::TimeUnit::MILLI);
    }
    if (res->id() == arrow::Type::TIMESTAMP) {
      res = arrow::timestamp(arrow::TimeUnit::MILLI, "UTC");
    }
    lossen_fields[i] = lossen_fields[i]->WithType(res);
  }
  schema = std::make_shared<arrow::Schema>(lossen_fields, metadata);
  return Status::OK();
}

Status TypeLoosen(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Schema>& schema) {
  std::vector<std::shared_ptr<arrow::Schema>> schemas;
  schemas.reserve(batches.size());
  for (const auto& batch : batches) {
    if (batch != nullptr) {
      schemas.push_back(batch->schema());
    }
  }
  return TypeLoosen(schemas, schema);
}

Status TypeLoosen(const std::vector<std::shared_ptr<arrow::Table>>& tables,
                  std::shared_ptr<arrow::Schema>& schema) {
  std::vector<std::shared_ptr<arrow::Schema>> schemas;
  schemas.reserve(tables.size());
  for (const auto& table : tables) {
    if (table != nullptr) {
      schemas.push_back(table->schema());
    }
  }
  return TypeLoosen(schemas, schema);
}

Status CastStringToBigString(const std::shared_ptr<arrow::Array>& in,
                             const std::shared_ptr<arrow::DataType>& to_type,
                             std::shared_ptr<arrow::Array>& out) {
  auto array_data = in->data()->Copy();
  auto offset = array_data->buffers[1];
  using from_offset_type = typename arrow::StringArray::offset_type;
  using to_string_offset_type = typename arrow::LargeStringArray::offset_type;
  auto raw_value_offsets_ =
      offset == NULLPTR
          ? NULLPTR
          : reinterpret_cast<const from_offset_type*>(offset->data());
  std::vector<to_string_offset_type> to_offset(offset->size() /
                                               sizeof(from_offset_type));
  for (size_t i = 0; i < to_offset.size(); ++i) {
    to_offset[i] = raw_value_offsets_[i];
  }
  std::shared_ptr<arrow::Buffer> buffer;
  arrow::TypedBufferBuilder<to_string_offset_type> buffer_builder;
  RETURN_ON_ARROW_ERROR(
      buffer_builder.Append(to_offset.data(), to_offset.size()));
  RETURN_ON_ARROW_ERROR(buffer_builder.Finish(&buffer));
  array_data->type = to_type;
  array_data->buffers[1] = buffer;
  out = arrow::MakeArray(array_data);
  RETURN_ON_ARROW_ERROR(out->ValidateFull());
  return Status::OK();
}

Status CastNullToOthers(const std::shared_ptr<arrow::Array>& in,
                        const std::shared_ptr<arrow::DataType>& to_type,
                        std::shared_ptr<arrow::Array>& out) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  RETURN_ON_ARROW_ERROR(
      arrow::MakeBuilder(arrow::default_memory_pool(), to_type, &builder));
  RETURN_ON_ARROW_ERROR(builder->AppendNulls(in->length()));
  RETURN_ON_ARROW_ERROR(builder->Finish(&out));
  RETURN_ON_ARROW_ERROR(out->ValidateFull());
  return Status::OK();
}

Status GeneralCast(const std::shared_ptr<arrow::Array>& in,
                   const std::shared_ptr<arrow::DataType>& to_type,
                   std::shared_ptr<arrow::Array>& out) {
#if defined(ARROW_VERSION) && ARROW_VERSION < 1000000
  arrow::compute::FunctionContext ctx;
  ARROW_OK_OR_RAISE(arrow::compute::Cast(&ctx, *in, to_type, {}, &out));
#else
  CHECK_ARROW_ERROR_AND_ASSIGN(out, arrow::compute::Cast(*in, to_type));
#endif
  return Status::OK();
}

Status CastBatchToSchema(const std::shared_ptr<arrow::RecordBatch>& batch,
                         const std::shared_ptr<arrow::Schema>& schema,
                         std::shared_ptr<arrow::RecordBatch>& out) {
  if (batch->schema()->Equals(schema)) {
    out = batch;
    return Status::OK();
  }

  RETURN_ON_ASSERT(batch->num_columns() == schema->num_fields(),
                   "The schema of original recordbatch and expected schema is "
                   "not consistent");
  std::vector<std::shared_ptr<arrow::Array>> new_columns;
  for (int64_t i = 0; i < batch->num_columns(); ++i) {
    auto array = batch->column(i);
    auto from_type = batch->schema()->field(i)->type();
    auto to_type = schema->field(i)->type();
    if (from_type->Equals(to_type)) {
      new_columns.push_back(array);
      continue;
    }
    std::shared_ptr<arrow::Array> out;
    if (arrow::compute::CanCast(*from_type, *to_type)) {
      RETURN_ON_ERROR(GeneralCast(array, to_type, out));
    } else if (from_type->Equals(arrow::utf8()) &&
               to_type->Equals(arrow::large_utf8())) {
      RETURN_ON_ERROR(CastStringToBigString(array, to_type, out));
    } else if (from_type->Equals(arrow::null())) {
      RETURN_ON_ERROR(CastNullToOthers(array, to_type, out));
    } else {
      return Status::Invalid(
          "Unsupported cast: To type: " + to_type->ToString() +
          " vs. origin type: " + from_type->ToString());
    }
    new_columns.push_back(out);
  }
  out = arrow::RecordBatch::Make(schema, batch->num_rows(), new_columns);
  return Status::OK();
}

Status CastTableToSchema(const std::shared_ptr<arrow::Table>& table,
                         const std::shared_ptr<arrow::Schema>& schema,
                         std::shared_ptr<arrow::Table>& out) {
  if (table->schema()->Equals(schema)) {
    out = table;
    return Status::OK();
  }

  RETURN_ON_ASSERT(
      table->num_columns() == schema->num_fields(),
      "The schema of original table and expected schema is not consistent");
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
  for (int64_t i = 0; i < table->num_columns(); ++i) {
    auto chunked_column = table->column(i);
    if (table->field(i)->type()->Equals(schema->field(i)->type())) {
      new_columns.push_back(chunked_column);
      continue;
    }
    auto from_type = table->field(i)->type();
    auto to_type = schema->field(i)->type();
    std::vector<std::shared_ptr<arrow::Array>> chunks;
    for (int64_t j = 0; j < chunked_column->num_chunks(); ++j) {
      auto array = chunked_column->chunk(j);
      std::shared_ptr<arrow::Array> out;
      if (arrow::compute::CanCast(*from_type, *to_type)) {
        RETURN_ON_ERROR(GeneralCast(array, to_type, out));
        chunks.push_back(out);
      } else if (from_type->Equals(arrow::utf8()) &&
                 to_type->Equals(arrow::large_utf8())) {
        RETURN_ON_ERROR(CastStringToBigString(array, to_type, out));
        chunks.push_back(out);
      } else if (from_type->Equals(arrow::null())) {
        RETURN_ON_ERROR(CastNullToOthers(array, to_type, out));
        chunks.push_back(out);
      } else {
        return Status::Invalid(
            "Unsupported cast: To type: " + to_type->ToString() +
            " vs. origin type: " + from_type->ToString());
      }
    }
    new_columns.push_back(
        std::make_shared<arrow::ChunkedArray>(chunks, to_type));
  }
  out = arrow::Table::Make(schema, new_columns);
  return Status::OK();
}

inline bool IsDataTypeConsolidatable(std::shared_ptr<arrow::DataType> type) {
  if (type == nullptr) {
    return false;
  }
  switch (type->id()) {
  case arrow::Type::INT8:
  case arrow::Type::INT16:
  case arrow::Type::INT32:
  case arrow::Type::INT64:
  case arrow::Type::UINT8:
  case arrow::Type::UINT16:
  case arrow::Type::UINT32:
  case arrow::Type::UINT64:
  case arrow::Type::FLOAT:
  case arrow::Type::DOUBLE:
  case arrow::Type::DATE32:
  case arrow::Type::DATE64:
  case arrow::Type::TIME32:
  case arrow::Type::TIME64:
  case arrow::Type::TIMESTAMP: {
    return true;
  }
  default: {
    return false;
  }
  }
}

template <typename T>
inline void AssignArrayWithStride(std::shared_ptr<arrow::Buffer> array,
                                  std::shared_ptr<arrow::Buffer> target,
                                  int64_t length, int64_t stride,
                                  int64_t offset) {
  auto array_data = reinterpret_cast<const T*>(array->data());
  auto target_data = reinterpret_cast<T*>(target->mutable_data());
  for (int64_t i = 0; i < length; ++i) {
    target_data[i * stride + offset] = array_data[i];
  }
}

inline void AssignArrayWithStrideUntyped(std::shared_ptr<arrow::Array> array,
                                         std::shared_ptr<arrow::Buffer> target,
                                         int64_t length, int64_t stride,
                                         int64_t offset) {
  if (array->length() == 0) {
    return;
  }
  switch (array->type()->id()) {
  case arrow::Type::INT8: {
    AssignArrayWithStride<int8_t>(array->data()->buffers[1], target, length,
                                  stride, offset);
    return;
  }
  case arrow::Type::INT16: {
    AssignArrayWithStride<int16_t>(array->data()->buffers[1], target, length,
                                   stride, offset);
    return;
  }
  case arrow::Type::INT32: {
    AssignArrayWithStride<int32_t>(array->data()->buffers[1], target, length,
                                   stride, offset);
    return;
  }
  case arrow::Type::INT64: {
    AssignArrayWithStride<int64_t>(array->data()->buffers[1], target, length,
                                   stride, offset);
    return;
  }
  case arrow::Type::UINT8: {
    AssignArrayWithStride<uint8_t>(array->data()->buffers[1], target, length,
                                   stride, offset);
    return;
  }
  case arrow::Type::UINT16: {
    AssignArrayWithStride<uint16_t>(array->data()->buffers[1], target, length,
                                    stride, offset);
    return;
  }
  case arrow::Type::UINT32: {
    AssignArrayWithStride<uint32_t>(array->data()->buffers[1], target, length,
                                    stride, offset);
    return;
  }
  case arrow::Type::UINT64: {
    AssignArrayWithStride<uint64_t>(array->data()->buffers[1], target, length,
                                    stride, offset);
    return;
  }
  case arrow::Type::FLOAT: {
    AssignArrayWithStride<float>(array->data()->buffers[1], target, length,
                                 stride, offset);
    return;
  }
  case arrow::Type::DOUBLE: {
    AssignArrayWithStride<double>(array->data()->buffers[1], target, length,
                                  stride, offset);
    return;
  }
  case arrow::Type::DATE32: {
    AssignArrayWithStride<arrow::Date32Type::c_type>(
        array->data()->buffers[1], target, length, stride, offset);
    return;
  }
  case arrow::Type::DATE64: {
    AssignArrayWithStride<arrow::Date64Type::c_type>(
        array->data()->buffers[1], target, length, stride, offset);
    return;
  }
  case arrow::Type::TIME32: {
    AssignArrayWithStride<arrow::Time32Type::c_type>(
        array->data()->buffers[1], target, length, stride, offset);
    return;
  }
  case arrow::Type::TIME64: {
    AssignArrayWithStride<arrow::Time64Type::c_type>(
        array->data()->buffers[1], target, length, stride, offset);
    return;
  }
  case arrow::Type::TIMESTAMP: {
    AssignArrayWithStride<arrow::TimestampType::c_type>(
        array->data()->buffers[1], target, length, stride, offset);
    return;
  }
  default: {
  }
  }
}

Status ConsolidateColumns(
    const std::vector<std::shared_ptr<arrow::Array>>& columns,
    std::shared_ptr<arrow::Array>& out) {
  if (columns.size() == 0) {
    return Status::Invalid("No columns to consolidate");
  }
  // check the types of columns that will be consolidated
  std::shared_ptr<arrow::DataType> dtype = nullptr;
  for (auto const& column : columns) {
    auto column_dtype = column->type();
    if (!IsDataTypeConsolidatable(column_dtype)) {
      return Status::Invalid("column type '" + column->type()->ToString() +
                             "' is not a numeric type");
    }
    if (dtype == nullptr || dtype->Equals(column_dtype)) {
      dtype = column_dtype;
    } else {
      return Status::Invalid(
          "cannot consolidate columns"
          "', column type '" +
          column->type()->ToString() +
          "' has different type with other columns");
    }
  }

  // consolidate columns into one

  // build the data buffer
  std::shared_ptr<arrow::DataType> list_array_dtype =
      arrow::fixed_size_list(dtype, columns.size());

  std::shared_ptr<arrow::Buffer> data_buffer;
  CHECK_ARROW_ERROR_AND_ASSIGN(
      data_buffer,
      arrow::AllocateBuffer(
          columns[0]->length() * columns.size() *
          static_cast<arrow::FixedWidthType*>(dtype.get())->bit_width() / 8));

  for (size_t index = 0; index < columns.size(); ++index) {
    auto array = columns[index];
    AssignArrayWithStrideUntyped(array, data_buffer, array->length(),
                                 columns.size(), index);
  }

  // build the list array
  out = std::make_shared<arrow::FixedSizeListArray>(
      list_array_dtype, columns[0]->length(),
      std::make_shared<arrow::PrimitiveArray>(
          dtype, columns[0]->length() * columns.size(), data_buffer));
  return Status::OK();
}

Status ConsolidateColumns(
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& columns,
    std::shared_ptr<arrow::ChunkedArray>& out) {
  std::vector<std::shared_ptr<arrow::Array>> array_chunks;

  for (int64_t i = 0; i < columns[0]->num_chunks(); ++i) {
    std::vector<std::shared_ptr<arrow::Array>> columns_chunk;
    for (auto const& column : columns) {
      columns_chunk.push_back(column->chunk(i));
    }
    std::shared_ptr<arrow::Array> array_chunk;
    RETURN_ON_ERROR(ConsolidateColumns(columns_chunk, array_chunk));
    array_chunks.push_back(array_chunk);
  }

  CHECK_ARROW_ERROR_AND_ASSIGN(out, arrow::ChunkedArray::Make(array_chunks));
  return Status::OK();
}

/**
 * @brief Consolidate columns in an arrow table into one column
 * (FixedSizeListArray).
 *
 * Note that the bitmap in the given columns will be discard.
 *
 * @param table
 */
Status ConsolidateColumns(const std::shared_ptr<arrow::Table>& table,
                          std::vector<std::string> const& column_names,
                          std::string const& consolidated_column_name,
                          std::shared_ptr<arrow::Table>& out) {
  // check the types of columns that will be consolidated
  std::string column_names_joined = boost::algorithm::join(column_names, ",");
  auto schema = table->schema();
  std::shared_ptr<arrow::DataType> dtype = nullptr;
  std::vector<int> column_indexes;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  for (auto const& column_name : column_names) {
    auto column_index = schema->GetFieldIndex(column_name);
    if (column_index == -1) {
      return Status::Invalid("column name '" + column_name +
                             "' doesn't exist in the table");
    }
    auto column_dtype = schema->field(column_index)->type();
    if (!IsDataTypeConsolidatable(column_dtype)) {
      return Status::Invalid("column '" + column_name +
                             "' is not a numeric type");
    }
    if (dtype == nullptr || dtype->Equals(column_dtype)) {
      dtype = column_dtype;
      column_indexes.push_back(column_index);
      columns.push_back(table->column(column_index));
    } else {
      return Status::Invalid("cannot consolidate columns '" +
                             column_names_joined + "', column name '" +
                             column_name +
                             "' has different type with other columns");
    }
  }

  // consolidate columns into one
  std::shared_ptr<arrow::DataType> list_array_dtype =
      arrow::fixed_size_list(dtype, columns.size());
  std::shared_ptr<arrow::ChunkedArray> consolidated_array;
  RETURN_ON_ERROR(ConsolidateColumns(columns, consolidated_array));

  // replace those columns with the consolidated column
  std::vector<int> sorted_column_indexes(column_indexes);
  std::sort(sorted_column_indexes.begin(), sorted_column_indexes.end());
  std::shared_ptr<arrow::Table> result = table;
  for (size_t index = 0; index < sorted_column_indexes.size(); ++index) {
    CHECK_ARROW_ERROR_AND_ASSIGN(
        result,
        result->RemoveColumn(
            sorted_column_indexes[sorted_column_indexes.size() - 1 - index]));
  }
  std::shared_ptr<arrow::Field> field;
  if (consolidated_column_name.empty()) {
    field = arrow::field(column_names_joined, list_array_dtype);
  } else {
    field = arrow::field(consolidated_column_name, list_array_dtype);
  }
  CHECK_ARROW_ERROR_AND_ASSIGN(
      out, result->AddColumn(result->num_columns(), field, consolidated_array));
  return Status::OK();
}

Status ConsolidateColumns(const std::shared_ptr<arrow::Table>& table,
                          std::shared_ptr<arrow::Table>& out) {
  if (table == nullptr || table->schema() == nullptr ||
      table->schema()->metadata() == nullptr) {
    out = table;
    return Status::OK();
  }
  auto metadata = table->schema()->metadata();
  auto consolidate_columns_index = metadata->FindKey("consolidate");
  if (consolidate_columns_index == -1) {
    out = table;
    return Status::OK();
  }
  auto consolidate_columns = metadata->value(consolidate_columns_index);
  if (consolidate_columns.empty()) {
    out = table;
    return Status::OK();
  }

  std::vector<std::string> consolidate_columns_vec;
  boost::algorithm::split(consolidate_columns_vec, consolidate_columns,
                          boost::is_any_of(",;"));
  return ConsolidateColumns(table, consolidate_columns_vec, "", out);
}

namespace arrow_shim {

namespace detail {

inline Status TimeUnitToJSON(arrow::TimeUnit::type const& unit, json& object) {
  switch (unit) {
  case arrow::TimeUnit::SECOND:
    object = json{"s"};
    return Status::OK();
  case arrow::TimeUnit::MILLI:
    object = json{"ms"};
    return Status::OK();
  case arrow::TimeUnit::MICRO:
    object = json{"us"};
    return Status::OK();
  case arrow::TimeUnit::NANO:
    object = json{"ns"};
    return Status::OK();
  default:
    return Status::Invalid("invalid time unit: " + std::to_string(unit));
  }
}

inline Status TimeUnitFromJSON(json const& object,
                               arrow::TimeUnit::type& unit) {
  if (object.is_string()) {
    auto unit_str = object.get<std::string>();
    if (unit_str == "s") {
      unit = arrow::TimeUnit::SECOND;
      return Status::OK();
    } else if (unit_str == "ms") {
      unit = arrow::TimeUnit::MILLI;
      return Status::OK();
    } else if (unit_str == "us") {
      unit = arrow::TimeUnit::MICRO;
      return Status::OK();
    } else if (unit_str == "ns") {
      unit = arrow::TimeUnit::NANO;
      return Status::OK();
    } else {
      return Status::Invalid("invalid time unit: " + unit_str);
    }
  } else {
    return Status::Invalid("invalid time unit: " + object.dump());
  }
}

}  // namespace detail

/**
 * @brief Textual schema representation.
 *
 * @see
 * https://github.com/apache/arrow-rs/blob/27f4762c8794ef1c5d042933562185980eb85ae5/arrow/src/datatypes/datatype.rs#L536
 */
Status SchemaToJSON(const std::shared_ptr<arrow::Schema>& schema,
                    json& object) {
  if (schema == nullptr) {
    object = json{nullptr};
    return Status::OK();
  }
  json fields;
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto field = schema->field(i);
    json field_object;
    RETURN_ON_ERROR(FieldToJSON(field, field_object));
    fields.push_back(field_object);
  }
  json metadata_object;
  if (schema->metadata() != nullptr) {
    for (int64_t i = 0; i < schema->metadata()->size(); ++i) {
      metadata_object[schema->metadata()->key(i)] =
          schema->metadata()->value(i);
    }
  }
  object = json{{"fields", fields}, {"metadata", metadata_object}};
  return Status::OK();
}

Status SchemaFromJSON(const json& object,
                      std::shared_ptr<arrow::Schema>& schema) {
  if (object.is_null()) {
    schema = nullptr;
    return Status::OK();
  }
  if (!object.is_object()) {
    return Status::Invalid("invalid schema: " + object.dump());
  }
  auto fields_object = object.find("fields");
  if (fields_object == object.end()) {
    return Status::Invalid("invalid schema: " + object.dump());
  }
  if (!fields_object->is_array()) {
    return Status::Invalid("invalid schema: " + object.dump());
  }
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (auto& field_object : *fields_object) {
    std::shared_ptr<arrow::Field> field;
    RETURN_ON_ERROR(FieldFromJSON(field_object, field));
    fields.push_back(field);
  }
  auto metadata_object = object.find("metadata");
  if (metadata_object == object.end()) {
    return Status::Invalid("invalid schema: " + object.dump());
  }
  if (!metadata_object->is_object()) {
    return Status::Invalid("invalid schema: " + object.dump());
  }
  auto metadata = std::make_shared<arrow::KeyValueMetadata>();
  for (auto& pair : metadata_object->items()) {
    metadata->Append(pair.key(), pair.value().get<std::string>());
  }
  schema = arrow::schema(fields, metadata);
  return Status::OK();
}

Status DataTypeToJSON(const std::shared_ptr<arrow::DataType>& datatype,
                      json& object) {
  if (datatype == nullptr) {
    object = json{nullptr};
  } else if (datatype->id() == arrow::Type::NA) {
    object = json{{"name", "null"}};
  } else if (datatype->id() == arrow::Type::BOOL) {
    object = json{{"name", "bool"}};
  } else if (datatype->id() == arrow::Type::INT8) {
    object = json{{"name", "int"}, {"bit_width", 16}, {"signed", true}};
  } else if (datatype->id() == arrow::Type::INT16) {
    object = json{{"name", "int"}, {"bit_width", 16}, {"signed", true}};
  } else if (datatype->id() == arrow::Type::INT32) {
    object = json{{"name", "int"}, {"bit_width", 32}, {"signed", true}};
  } else if (datatype->id() == arrow::Type::INT64) {
    object = json{{"name", "int"}, {"bit_width", 64}, {"signed", true}};
  } else if (datatype->id() == arrow::Type::UINT8) {
    object = json{{"name", "int"}, {"bit_width", 16}, {"signed", false}};
  } else if (datatype->id() == arrow::Type::UINT16) {
    object = json{{"name", "int"}, {"bit_width", 16}, {"signed", false}};
  } else if (datatype->id() == arrow::Type::UINT32) {
    object = json{{"name", "int"}, {"bit_width", 32}, {"signed", false}};
  } else if (datatype->id() == arrow::Type::UINT64) {
    object = json{{"name", "int"}, {"bit_width", 64}, {"signed", false}};
  } else if (datatype->id() == arrow::Type::HALF_FLOAT) {
    object = json{{"name", "float"}, {"precision", "half"}};
  } else if (datatype->id() == arrow::Type::FLOAT) {
    object = json{{"name", "float"}, {"precision", "single"}};
  } else if (datatype->id() == arrow::Type::DOUBLE) {
    object = json{{"name", "float"}, {"precision", "double"}};
  } else if (datatype->id() == arrow::Type::STRING) {
    object = json{{"name", "utf8"}};
  } else if (datatype->id() == arrow::Type::LARGE_STRING) {
    object = json{{"name", "large_utf8"}};
  } else if (datatype->id() == arrow::Type::BINARY) {
    object = json{{"name", "binary"}};
  } else if (datatype->id() == arrow::Type::LARGE_BINARY) {
    object = json{{"name", "large_binary"}};
  } else if (datatype->id() == arrow::Type::FIXED_SIZE_BINARY) {
    auto fixed_size_binary_type =
        std::dynamic_pointer_cast<arrow::FixedSizeBinaryType>(datatype);
    object = json{{"name", "fixed_size_binary"},
                  {"byte_width", fixed_size_binary_type->byte_width()}};
  } else if (datatype->id() == arrow::Type::LIST) {
    auto list_type = std::dynamic_pointer_cast<arrow::ListType>(datatype);
    json list_type_object;
    RETURN_ON_ERROR(DataTypeToJSON(list_type->value_type(), list_type_object));
    object = json{{"name", "list"}, {"value_type", list_type_object}};
  } else if (datatype->id() == arrow::Type::LARGE_LIST) {
    auto list_type = std::dynamic_pointer_cast<arrow::LargeListType>(datatype);
    json list_type_object;
    RETURN_ON_ERROR(DataTypeToJSON(list_type->value_type(), list_type_object));
    object = json{{"name", "large_list"}, {"value_type", list_type_object}};
  } else if (datatype->id() == arrow::Type::FIXED_SIZE_LIST) {
    auto list_type =
        std::dynamic_pointer_cast<arrow::FixedSizeListType>(datatype);
    json list_type_object;
    RETURN_ON_ERROR(DataTypeToJSON(list_type->value_type(), list_type_object));
    object = json{{"name", "fixed_size_list"},
                  {"value_type", list_type_object},
                  {"list_size", list_type->list_size()}};
  } else if (datatype->id() == arrow::Type::TIME32) {
    auto time32_type = std::dynamic_pointer_cast<arrow::Time32Type>(datatype);
    json time_unit_object;
    RETURN_ON_ERROR(
        detail::TimeUnitToJSON(time32_type->unit(), time_unit_object));
    object =
        json{{"name", "time"}, {"unit", time_unit_object}, {"bit_width", 32}};
  } else if (datatype->id() == arrow::Type::TIME64) {
    auto time64_type = std::dynamic_pointer_cast<arrow::Time64Type>(datatype);
    json time_unit_object;
    RETURN_ON_ERROR(
        detail::TimeUnitToJSON(time64_type->unit(), time_unit_object));
    object =
        json{{"name", "time"}, {"unit", time_unit_object}, {"bit_width", 64}};
  } else if (datatype->id() == arrow::Type::DATE32) {
    object = json{{"name", "date"}, {"unit", "day"}};
  } else if (datatype->id() == arrow::Type::DATE64) {
    object = json{{"name", "date"}, {"unit", "millisecond"}};
  } else if (datatype->id() == arrow::Type::TIMESTAMP) {
    auto timestamp_type =
        std::dynamic_pointer_cast<arrow::TimestampType>(datatype);
    json time_unit_object;
    RETURN_ON_ERROR(
        detail::TimeUnitToJSON(timestamp_type->unit(), time_unit_object));
    object = json{{"name", "timestamp"},
                  {"unit", time_unit_object},
                  {"timezone", timestamp_type->timezone()}};
  } else if (datatype->id() == arrow::Type::INTERVAL_MONTHS) {
    object = json{{"name", "interval"}, {"unit", "month"}};
  } else if (datatype->id() == arrow::Type::INTERVAL_DAY_TIME) {
    object = json{{"name", "interval"}, {"unit", "day_time"}};
  } else if (datatype->id() == arrow::Type::INTERVAL_MONTH_DAY_NANO) {
    object = json{{"name", "interval"}, {"unit", "month_day_nano"}};
  } else if (datatype->id() == arrow::Type::DURATION) {
    auto duration_type =
        std::dynamic_pointer_cast<arrow::DurationType>(datatype);
    json time_unit_object;
    RETURN_ON_ERROR(
        detail::TimeUnitToJSON(duration_type->unit(), time_unit_object));
    object = json{{"name", "duration"}, {"unit", time_unit_object}};
  } else if (datatype->id() == arrow::Type::DECIMAL) {
    auto decimal_type = std::dynamic_pointer_cast<arrow::DecimalType>(datatype);
    object = json{{"name", "decimal"},
                  {"precision", decimal_type->precision()},
                  {"scale", decimal_type->scale()},
                  {"bit_width", decimal_type->bit_width()}};
  } else if (datatype->id() == arrow::Type::STRUCT) {
    auto struct_type = std::dynamic_pointer_cast<arrow::StructType>(datatype);
    json fields_object;
    for (auto const& field : struct_type->fields()) {
      json field_object;
      RETURN_ON_ERROR(FieldToJSON(field, field_object));
      fields_object.push_back(field_object);
    }
    object = json{{"name", "struct"}, {"fields", fields_object}};
  } else if (datatype->id() == arrow::Type::DENSE_UNION) {
    auto union_type = std::dynamic_pointer_cast<arrow::UnionType>(datatype);
    json fields_object;
    for (auto const& field : union_type->fields()) {
      json field_object;
      RETURN_ON_ERROR(FieldToJSON(field, field_object));
      fields_object.push_back(field_object);
    }
    object =
        json{{"name", "union"}, {"mode", "dense"}, {"fields", fields_object}};
  } else if (datatype->id() == arrow::Type::SPARSE_UNION) {
    auto union_type = std::dynamic_pointer_cast<arrow::UnionType>(datatype);
    json fields_object;
    for (auto const& field : union_type->fields()) {
      json field_object;
      RETURN_ON_ERROR(FieldToJSON(field, field_object));
      fields_object.push_back(field_object);
    }
    object =
        json{{"name", "union"}, {"mode", "sparse"}, {"fields", fields_object}};
  } else if (datatype->id() == arrow::Type::DICTIONARY) {
    auto dictionary_type =
        std::dynamic_pointer_cast<arrow::DictionaryType>(datatype);
    json index_type_object;
    RETURN_ON_ERROR(
        DataTypeToJSON(dictionary_type->index_type(), index_type_object));
    json value_type_object;
    RETURN_ON_ERROR(
        DataTypeToJSON(dictionary_type->value_type(), value_type_object));
    object = json{{"name", "dictionary"},
                  {"index_type", index_type_object},
                  {"value_type", value_type_object}};
  } else {
    return Status::Invalid("Not supported data type: '" + datatype->ToString() +
                           "'");
  }
  return Status::OK();
}

Status DataTypeFromJSON(const json& object,
                        std::shared_ptr<arrow::DataType>& datatype) {
  if (!object.is_null() && !object.object()) {
    return Status::Invalid("Invalid data type object: '" + object.dump() + "'");
  }
  if (object.is_null()) {
    datatype = nullptr;
    return Status::OK();
  }
  std::string name = object.value("name", "");
  if (name == "null") {
    datatype = arrow::null();
  } else if (name == "bool") {
    datatype = arrow::boolean();
  } else if (name == "int") {
    int bit_width = object.value("bit_width", -1);
    bool issigned = object.value("signed", true);
    if (bit_width == 8) {
      datatype = issigned ? arrow::int8() : arrow::uint8();
    } else if (bit_width == 16) {
      datatype = issigned ? arrow::int16() : arrow::uint16();
    } else if (bit_width == 32) {
      datatype = issigned ? arrow::int32() : arrow::uint32();
    } else if (bit_width == 64) {
      datatype = issigned ? arrow::int64() : arrow::uint64();
    } else {
      return Status::Invalid("Invalid bit width: '" +
                             std::to_string(bit_width) + "'");
    }
  } else if (name == "float") {
    std::string precision = object.value("precision", "");
    if (precision == "half") {
      datatype = arrow::float16();
    } else if (precision == "single") {
      datatype = arrow::float32();
    } else if (precision == "double") {
      datatype = arrow::float64();
    } else {
      return Status::Invalid("Invalid precision: '" + precision + "'");
    }
  } else if (name == "utf8") {
    datatype = arrow::utf8();
  } else if (name == "large_utf8") {
    datatype = arrow::large_utf8();
  } else if (name == "binary") {
    datatype = arrow::binary();
  } else if (name == "large_binary") {
    datatype = arrow::large_binary();
  } else if (name == "fixed_size_binary") {
    int byte_width = object.value("byte_width", -1);
    datatype = arrow::fixed_size_binary(byte_width);
  } else if (name == "list") {
    json value_type_object = object.value("value_type", json());
    std::shared_ptr<arrow::DataType> value_type;
    RETURN_ON_ERROR(DataTypeFromJSON(value_type_object, value_type));
    datatype = arrow::list(value_type);
  } else if (name == "large_list") {
    json value_type_object = object.value("value_type", json());
    std::shared_ptr<arrow::DataType> value_type;
    RETURN_ON_ERROR(DataTypeFromJSON(value_type_object, value_type));
    datatype = arrow::large_list(value_type);
  } else if (name == "fixed_size_list") {
    json value_type_object = object.value("value_type", json());
    std::shared_ptr<arrow::DataType> value_type;
    RETURN_ON_ERROR(DataTypeFromJSON(value_type_object, value_type));
    int list_size = object.value("list_size", -1);
    datatype = arrow::fixed_size_list(value_type, list_size);
  } else if (name == "time") {
    json time_unit_object = object.value("unit", json());
    arrow::TimeUnit::type time_unit;
    RETURN_ON_ERROR(detail::TimeUnitFromJSON(time_unit_object, time_unit));
    int bit_width = object.value("bit_width", -1);
    if (bit_width == 32) {
      datatype = arrow::time32(time_unit);
    } else if (bit_width == 64) {
      datatype = arrow::time64(time_unit);
    } else {
      return Status::Invalid("Invalid bit width: '" +
                             std::to_string(bit_width) + "'");
    }
  } else if (name == "date") {
    std::string unit = object.value("unit", "");
    if (unit == "day") {
      datatype = arrow::date32();
    } else if (unit == "millisecond") {
      datatype = arrow::date64();
    } else {
      return Status::Invalid("Invalid date unit: '" + unit + "'");
    }
  } else if (name == "timestamp") {
    json time_unit_object = object.value("unit", json());
    arrow::TimeUnit::type time_unit;
    RETURN_ON_ERROR(detail::TimeUnitFromJSON(time_unit_object, time_unit));
    std::string timezone = object.value("timezone", "");
    if (timezone.empty()) {
      datatype = arrow::timestamp(time_unit);
    } else {
      datatype = arrow::timestamp(time_unit, timezone);
    }
  } else if (name == "interval") {
    std::string unit = object.value("unit", "");
    if (unit == "month") {
      datatype = arrow::month_interval();
    } else if (unit == "day_time") {
      datatype = arrow::day_time_interval();
    } else if (unit == "month_day_nano") {
      datatype = arrow::month_day_nano_interval();
    } else {
      return Status::Invalid("Invalid interval unit: '" + unit + "'");
    }
  } else if (name == "duration") {
    json time_unit_object = object.value("unit", json());
    arrow::TimeUnit::type time_unit;
    RETURN_ON_ERROR(detail::TimeUnitFromJSON(time_unit_object, time_unit));
    datatype = arrow::duration(time_unit);
  } else if (name == "decimal") {
    int precision = object.value("precision", -1);
    int scale = object.value("scale", -1);
    int bit_width = object.value("bit_width", -1);
    if (bit_width == 128) {
      datatype = arrow::decimal128(precision, scale);
    } else if (bit_width == 256) {
      datatype = arrow::decimal256(precision, scale);
    } else {
      return Status::Invalid("Invalid bit width: '" +
                             std::to_string(bit_width) + "'");
    }
  } else if (name == "dictionary") {
    json index_type_object = object.value("index_type", json());
    std::shared_ptr<arrow::DataType> index_type;
    RETURN_ON_ERROR(DataTypeFromJSON(index_type_object, index_type));
    json value_type_object = object.value("value_type", json());
    std::shared_ptr<arrow::DataType> value_type;
    RETURN_ON_ERROR(DataTypeFromJSON(value_type_object, value_type));
    datatype = arrow::dictionary(index_type, value_type);
  } else if (name == "struct") {
    json fields_object = object.value("fields", json());
    if (!fields_object.is_array()) {
      return Status::Invalid("Invalid fields object: '" + fields_object.dump() +
                             "'");
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& field_object : fields_object) {
      std::shared_ptr<arrow::Field> field;
      RETURN_ON_ERROR(FieldFromJSON(field_object, field));
      fields.push_back(field);
    }
    datatype = arrow::struct_(fields);
  } else if (name == "union") {
    std::string mode = object.value("mode", "");
    json fields_object = object.value("fields", json());
    if (!fields_object.is_array()) {
      return Status::Invalid("Invalid fields object: '" + fields_object.dump() +
                             "'");
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& field_object : fields_object) {
      std::shared_ptr<arrow::Field> field;
      RETURN_ON_ERROR(FieldFromJSON(field_object, field));
      fields.push_back(field);
    }
    if (mode == "sparse") {
      datatype = arrow::sparse_union(fields);
    } else if (mode == "dense") {
      datatype = arrow::dense_union(fields);
    } else {
      return Status::Invalid("Invalid union mode: '" + mode + "'");
    }
  } else {
    return Status::Invalid("Invalid data type: '" + name + "'");
  }
  return Status::OK();
}

Status FieldToJSON(const std::shared_ptr<arrow::Field>& field, json& object) {
  if (field == nullptr) {
    return Status::Invalid("Invalid field object");
  }
  json type_object;
  RETURN_ON_ERROR(DataTypeToJSON(field->type(), type_object));
  object = json{{
                    "name",
                    field->name(),
                },
                {
                    "type",
                    type_object,
                },
                {
                    "nullable",
                    field->nullable(),
                }};
  return Status::OK();
}

Status FieldFromJSON(const json& object, std::shared_ptr<arrow::Field>& field) {
  if (!object.is_object()) {
    return Status::Invalid("Invalid field object: '" + object.dump() + "'");
  }
  std::string name = object.value("name", "");
  json type_object = object.value("type", json());
  std::shared_ptr<arrow::DataType> type;
  RETURN_ON_ERROR(DataTypeFromJSON(type_object, type));
  bool nullable = object.value("nullable", true);
  field = arrow::field(name, type, nullable);
  return Status::OK();
}

}  // namespace arrow_shim

}  // namespace vineyard
