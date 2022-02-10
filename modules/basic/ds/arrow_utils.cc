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

#include "basic/ds/arrow_utils.h"

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

namespace vineyard {

std::shared_ptr<arrow::Table> ConcatenateTables(
    std::vector<std::shared_ptr<arrow::Table>>& tables) {
  if (tables.size() == 1) {
    return tables[0];
  }
  auto col_names = tables[0]->ColumnNames();
  for (size_t i = 1; i < tables.size(); ++i) {
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(tables[i]->RenameColumns(col_names, &tables[i]));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(tables[i],
                                 tables[i]->RenameColumns(col_names));
#endif
  }
  std::shared_ptr<arrow::Table> table;
  CHECK_ARROW_ERROR_AND_ASSIGN(table, arrow::ConcatenateTables(tables));
  return table;
}

std::shared_ptr<arrow::DataType> FromAnyType(AnyType type) {
  switch (type) {
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
  default:
    return arrow::null();
  }
}

Status GetRecordBatchStreamSize(const arrow::RecordBatch& batch, size_t* size) {
  // emulates the behavior of Write without actually writing
  arrow::io::MockOutputStream dst;

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  RETURN_ON_ARROW_ERROR(
      arrow::ipc::RecordBatchStreamWriter::Open(&dst, batch.schema(), &writer));
#elif defined(ARROW_VERSION) && ARROW_VERSION < 2000000
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

Status SerializeRecordBatchesToAllocatedBuffer(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer) {
  arrow::io::FixedSizeBufferWriter stream(*buffer);
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      batches, arrow::ipc::IpcOptions::Defaults(), &stream));
  return Status::OK();
}

Status SerializeRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer) {
  std::shared_ptr<arrow::io::BufferOutputStream> out_stream;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(out_stream,
                                   arrow::io::BufferOutputStream::Create(1024));
  RETURN_ON_ARROW_ERROR(arrow::ipc::WriteRecordBatchStream(
      batches, arrow::ipc::IpcOptions::Defaults(), out_stream.get()));
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(*buffer, out_stream->Finish());
  return Status::OK();
}

Status DeserializeRecordBatches(
    const std::shared_ptr<arrow::Buffer>& buffer,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches) {
  arrow::io::BufferReader reader(buffer);
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  RETURN_ON_ARROW_ERROR(
      arrow::ipc::RecordBatchStreamReader::Open(&reader, &batch_reader));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      batch_reader, arrow::ipc::RecordBatchStreamReader::Open(&reader));
#endif
  RETURN_ON_ARROW_ERROR(batch_reader->ReadAll(batches));
  return Status::OK();
}

Status RecordBatchesToTable(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table) {
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  RETURN_ON_ARROW_ERROR(arrow::Table::FromRecordBatches(batches, table));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(*table,
                                   arrow::Table::FromRecordBatches(batches));
#endif
  return Status::OK();
}

Status CombineRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::Table> table, combined_table;
  RETURN_ON_ERROR(RecordBatchesToTable(batches, &table));
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  RETURN_ON_ARROW_ERROR(
      table->CombineChunks(arrow::default_memory_pool(), &combined_table));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      combined_table, table->CombineChunks(arrow::default_memory_pool()));
#endif
  arrow::TableBatchReader tbreader(*combined_table);
  RETURN_ON_ARROW_ERROR(tbreader.ReadNext(batch));
  std::shared_ptr<arrow::RecordBatch> test_batch;
  RETURN_ON_ARROW_ERROR(tbreader.ReadNext(&test_batch));
  RETURN_ON_ASSERT(test_batch == nullptr);
  return Status::OK();
}

Status TableToRecordBatches(
    std::shared_ptr<arrow::Table> table,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches) {
  arrow::TableBatchReader tbr(*table);
  RETURN_ON_ARROW_ERROR(tbr.ReadAll(batches));
  return Status::OK();
}

Status SerializeTableToAllocatedBuffer(std::shared_ptr<arrow::Table> table,
                                       std::shared_ptr<arrow::Buffer>* buffer) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(TableToRecordBatches(table, &batches));
  RETURN_ON_ERROR(SerializeRecordBatchesToAllocatedBuffer(batches, buffer));
  return Status::OK();
}

Status SerializeTable(std::shared_ptr<arrow::Table> table,
                      std::shared_ptr<arrow::Buffer>* buffer) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(TableToRecordBatches(table, &batches));
  RETURN_ON_ERROR(SerializeRecordBatches(batches, buffer));
  return Status::OK();
}

Status DeserializeTable(std::shared_ptr<arrow::Buffer> buffer,
                        std::shared_ptr<arrow::Table>* table) {
  arrow::io::BufferReader reader(buffer);
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  RETURN_ON_ARROW_ERROR(
      arrow::ipc::RecordBatchStreamReader::Open(&reader, &batch_reader));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      batch_reader, arrow::ipc::RecordBatchStreamReader::Open(&reader));
#endif
  RETURN_ON_ARROW_ERROR(batch_reader->ReadAll(table));
  return Status::OK();
}

Status EmptyTableBuilder::Build(const std::shared_ptr<arrow::Schema>& schema,
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
    } else if (type == arrow::utf8()) {
      arrow::StringBuilder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (type == arrow::large_utf8()) {
      arrow::LargeStringBuilder builder;
      RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
    } else if (type == arrow::list(arrow::uint64())) {
      auto builder = std::make_shared<arrow::UInt64Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (type == arrow::list(arrow::int64())) {
      auto builder = std::make_shared<arrow::Int64Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (type == arrow::list(arrow::uint32())) {
      auto builder = std::make_shared<arrow::UInt32Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (type == arrow::list(arrow::int32())) {
      auto builder = std::make_shared<arrow::Int32Builder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (type == arrow::list(arrow::float64())) {
      auto builder = std::make_shared<arrow::DoubleBuilder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
    } else if (type == arrow::list(arrow::int64())) {
      auto builder = std::make_shared<arrow::FloatBuilder>();
      arrow::ListBuilder list_builder(arrow::default_memory_pool(), builder);
      RETURN_ON_ARROW_ERROR(list_builder.Finish(&dummy));
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

std::shared_ptr<arrow::Schema> EmptyTableBuilder::EmptySchema() {
#if defined(ARROW_VERSION) && ARROW_VERSION >= 4000000
  return std::shared_ptr<arrow::Schema>(
      new arrow::Schema({}, arrow::Endianness::Native));
#else
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({}));
#endif
}

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
  } else if (name == "large_list<item: int32>") {
    return arrow::large_list(arrow::int32());
  } else if (name == "large_list<item: uint32>") {
    return arrow::large_list(arrow::uint32());
  } else if (name == "large_list<item: int64>") {
    return arrow::large_list(arrow::int64());
  } else if (name == "large_list<item: uint64>") {
    return arrow::large_list(arrow::int32());
  } else if (name == "large_list<item: float>") {
    return arrow::large_list(arrow::float32());
  } else if (name == "large_list<item: double>") {
    return arrow::large_list(arrow::float64());
  } else if (name == "null" || name == "NULL") {
    return arrow::null();
  } else {
    LOG(ERROR) << "Unsupported data type: " << name;
    return nullptr;
  }
}

}  // namespace vineyard
