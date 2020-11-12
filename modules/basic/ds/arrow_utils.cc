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

bool SameShape(std::shared_ptr<arrow::ChunkedArray> ca1,
               std::shared_ptr<arrow::ChunkedArray> ca2) {
  if (ca1->length() != ca2->length()) {
    return false;
  }
  if (ca1->num_chunks() != ca2->num_chunks()) {
    return false;
  }
  size_t num_chunks = ca1->num_chunks();
  for (size_t i = 0; i < num_chunks; ++i) {
    if (ca1->chunk(i)->length() != ca2->chunk(i)->length()) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<Column> CreateColumn(
    std::shared_ptr<arrow::ChunkedArray> chunked_array, size_t chunk_size) {
  std::shared_ptr<arrow::DataType> type = chunked_array->type();
  if (type == arrow::int32()) {
    return std::make_shared<Int32Column>(chunked_array, chunk_size);
  } else if (type == arrow::int64()) {
    return std::make_shared<Int64Column>(chunked_array, chunk_size);
  } else if (type == arrow::uint32()) {
    return std::make_shared<UInt32Column>(chunked_array, chunk_size);
  } else if (type == arrow::uint64()) {
    return std::make_shared<UInt64Column>(chunked_array, chunk_size);
  } else if (type == arrow::float32()) {
    return std::make_shared<FloatColumn>(chunked_array, chunk_size);
  } else if (type == arrow::float64()) {
    return std::make_shared<DoubleColumn>(chunked_array, chunk_size);
  } else if (type == arrow::utf8()) {
    return std::make_shared<StringColumn>(chunked_array, chunk_size);
  } else if (type == arrow::binary()) {
    return std::make_shared<StringColumn>(chunked_array, chunk_size);
  } else if (type->id() == arrow::Type::TIMESTAMP) {
    return std::make_shared<TimestampColumn>(chunked_array, chunk_size);
  } else {
    LOG(ERROR) << "Invalid type when creating column...";
    return nullptr;
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

TableAppender::TableAppender(std::shared_ptr<arrow::Schema> schema) {
  for (const auto& field : schema->fields()) {
    std::shared_ptr<arrow::DataType> type = field->type();
    if (type == arrow::uint64()) {
      funcs_.push_back(AppendHelper<uint64_t>::append);
    } else if (type == arrow::int64()) {
      funcs_.push_back(AppendHelper<int64_t>::append);
    } else if (type == arrow::uint32()) {
      funcs_.push_back(AppendHelper<uint32_t>::append);
    } else if (type == arrow::int32()) {
      funcs_.push_back(AppendHelper<int32_t>::append);
    } else if (type == arrow::float32()) {
      funcs_.push_back(AppendHelper<float>::append);
    } else if (type == arrow::float64()) {
      funcs_.push_back(AppendHelper<double>::append);
    } else if (type == arrow::binary()) {
      funcs_.push_back(AppendHelper<std::string>::append);
    } else if (type == arrow::utf8()) {
      funcs_.push_back(AppendHelper<std::string>::append);
    } else if (type == arrow::null()) {
      funcs_.push_back(AppendHelper<void>::append);
    } else if (type->id() == arrow::Type::TIMESTAMP) {
      funcs_.push_back(AppendHelper<arrow::TimestampType>::append);
    } else {
      LOG(FATAL) << "Datatype [" << type->ToString() << "] not implemented...";
    }
  }
  col_num_ = funcs_.size();
}

Status TableAppender::Apply(
    std::unique_ptr<arrow::RecordBatchBuilder>& builder,
    std::shared_ptr<arrow::RecordBatch> batch, size_t offset,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches_out) {
  for (size_t i = 0; i < col_num_; ++i) {
    funcs_[i](builder->GetField(i), batch->column(i), offset);
  }
  if (builder->GetField(0)->length() == builder->initial_capacity()) {
    std::shared_ptr<arrow::RecordBatch> tmp_batch;
    RETURN_ON_ARROW_ERROR(builder->Flush(&tmp_batch));
    batches_out.emplace_back(std::move(tmp_batch));
  }
  return Status::OK();
}

Status TableAppender::Flush(
    std::unique_ptr<arrow::RecordBatchBuilder>& builder,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches_out) {
  // If there's no batch, we need an empty batch to make an empty table
  if (builder->GetField(0)->length() != 0 || batches_out.size() == 0) {
    std::shared_ptr<arrow::RecordBatch> batch;
    RETURN_ON_ARROW_ERROR(builder->Flush(&batch));
    batches_out.emplace_back(std::move(batch));
  }
  return Status::OK();
}

}  // namespace vineyard
