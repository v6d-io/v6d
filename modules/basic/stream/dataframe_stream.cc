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

#include "basic/stream/dataframe_stream.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

Status DataframeStreamWriter::GetNext(
    size_t const size, std::unique_ptr<arrow::MutableBuffer>& buffer) {
  return client_.GetNextStreamChunk(id_, size, buffer);
}

Status DataframeStreamWriter::Abort() {
  if (stoped_) {
    return Status::OK();
  }
  stoped_ = true;
  return client_.StopStream(id_, true);
}

Status DataframeStreamWriter::Finish() {
  if (stoped_) {
    return Status::OK();
  }
  stoped_ = true;
  return client_.StopStream(id_, false);
}

Status DataframeStreamWriter::WriteTable(std::shared_ptr<arrow::Table> table) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(TableToRecordBatches(table, &batches));
  for (auto const& batch : batches) {
    RETURN_ON_ERROR(WriteBatch(batch));
  }
  return Status::OK();
}

Status DataframeStreamWriter::WriteBatch(
    std::shared_ptr<arrow::RecordBatch> batch) {
  size_t size = 0;
  RETURN_ON_ERROR(GetRecordBatchStreamSize(*batch, &size));
  std::unique_ptr<arrow::MutableBuffer> buffer;
  RETURN_ON_ERROR(GetNext(size, buffer));
  arrow::io::FixedSizeBufferWriter stream(std::move(buffer));

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  RETURN_ON_ARROW_ERROR(arrow::ipc::RecordBatchStreamWriter::Open(
      &stream, batch->schema(), &writer));
#elif defined(ARROW_VERSION) && ARROW_VERSION < 2000000
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::NewStreamWriter(&stream, batch->schema()));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::MakeStreamWriter(&stream, batch->schema()));
#endif
  RETURN_ON_ARROW_ERROR(writer->WriteRecordBatch(*batch));
  RETURN_ON_ARROW_ERROR(writer->Close());
  return Status::OK();
}

Status DataframeStreamWriter::WriteDataframe(std::shared_ptr<DataFrame> df) {
  return WriteBatch(df->AsBatch());
}

Status DataframeStreamReader::GetNext(std::unique_ptr<arrow::Buffer>& buffer) {
  return client_.PullNextStreamChunk(id_, buffer);
}

Status DataframeStreamReader::ReadRecordBatches(
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches) {
  std::shared_ptr<arrow::RecordBatch> batch;
  while (ReadBatch(batch).ok()) {
    batches.emplace_back(batch);
  }
  return Status::OK();
}

Status DataframeStreamReader::ReadTable(std::shared_ptr<arrow::Table>& table) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(this->ReadRecordBatches(batches));
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  RETURN_ON_ARROW_ERROR(arrow::Table::FromRecordBatches(batches, &table));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(table,
                                   arrow::Table::FromRecordBatches(batches));
#endif
  return Status::OK();
}

Status DataframeStreamReader::ReadBatch(
    std::shared_ptr<arrow::RecordBatch>& batch) {
  std::unique_ptr<arrow::Buffer> buf;

  auto status = GetNext(buf);
  if (status.ok()) {
    std::shared_ptr<arrow::Buffer> copied_buffer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    RETURN_ON_ARROW_ERROR(buf->Copy(0, buf->size(), &copied_buffer));
#else
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(copied_buffer,
                                     buf->CopySlice(0, buf->size()));
#endif
    auto buffer_reader =
        std::make_shared<arrow::io::BufferReader>(copied_buffer);
    std::shared_ptr<arrow::ipc::RecordBatchReader> reader;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    RETURN_ON_ARROW_ERROR(
        arrow::ipc::RecordBatchStreamReader::Open(buffer_reader, &reader));
#else
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        reader, arrow::ipc::RecordBatchStreamReader::Open(buffer_reader));
#endif
    RETURN_ON_ARROW_ERROR(reader->ReadNext(&batch));

    std::shared_ptr<arrow::KeyValueMetadata> metadata;
    if (batch->schema()->metadata() != nullptr) {
      metadata = batch->schema()->metadata()->Copy();
    } else {
      metadata.reset(new arrow::KeyValueMetadata());
    }

#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    std::unordered_map<std::string, std::string> metakv;
    metadata->ToUnorderedMap(&metakv);
    for (auto const& kv : params_) {
      metakv[kv.first] = kv.second;
    }
    metadata = std::make_shared<arrow::KeyValueMetadata>();
    for (auto const& kv : metakv) {
      metadata->Append(kv.first, kv.second);
    }
#else
    for (auto const& kv : params_) {
      CHECK_ARROW_ERROR(metadata->Set(kv.first, kv.second));
    }
#endif

    batch = batch->ReplaceSchemaMetadata(metadata);
  }
  return status;
}

Status DataframeStreamReader::GetHeaderLine(bool& header_row,
                                            std::string& header_line) {
  if (params_.find("header_row") != params_.end()) {
    header_row = (params_["header_row"] == "1");
    if (params_.find("header_line") != params_.end()) {
      header_line = params_["header_line"];
    } else {
      header_line = "";
    }
  } else {
    header_row = false;
    header_line = "";
  }
  return Status::OK();
}

Status DataframeStream::OpenReader(
    Client& client, std::unique_ptr<DataframeStreamReader>& reader) {
  RETURN_ON_ERROR(client.OpenStream(id_, StreamOpenMode::read));
  reader = std::unique_ptr<DataframeStreamReader>(
      new DataframeStreamReader(client, id_, meta_, params_));
  return Status::OK();
}

Status DataframeStream::OpenWriter(
    Client& client, std::unique_ptr<DataframeStreamWriter>& writer) {
  RETURN_ON_ERROR(client.OpenStream(id_, StreamOpenMode::write));
  writer = std::unique_ptr<DataframeStreamWriter>(
      new DataframeStreamWriter(client, id_, meta_));
  return Status::OK();
}

}  // namespace vineyard
