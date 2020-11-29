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

#include "basic/ds/arrow_utils.h"

#include "arrow/io/memory.h"
#include "arrow/ipc/options.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/record_batch.h"
#include "arrow/util/config.h"

namespace vineyard {

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

}  // namespace vineyard
