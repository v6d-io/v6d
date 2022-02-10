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

#include "basic/stream/recordbatch_stream.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/arrow.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

Status RecordBatchStreamWriter::Push(std::shared_ptr<Object> const& chunk) {
  return client_.ClientBase::PushNextStreamChunk(id_, chunk->id());
}

Status RecordBatchStreamWriter::Push(ObjectMeta const& chunk) {
  return client_.ClientBase::PushNextStreamChunk(id_, chunk.GetId());
}

Status RecordBatchStreamWriter::Push(ObjectID const& chunk) {
  return client_.ClientBase::PushNextStreamChunk(id_, chunk);
}

Status RecordBatchStreamWriter::Abort() {
  if (stoped_) {
    return Status::OK();
  }
  stoped_ = true;
  return client_.StopStream(id_, true);
}

Status RecordBatchStreamWriter::Finish() {
  if (stoped_) {
    return Status::OK();
  }
  stoped_ = true;
  return client_.StopStream(id_, false);
}

Status RecordBatchStreamWriter::WriteTable(
    std::shared_ptr<arrow::Table> table) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(TableToRecordBatches(table, &batches));
  for (auto const& batch : batches) {
    RETURN_ON_ERROR(WriteBatch(batch));
  }
  return Status::OK();
}

Status RecordBatchStreamWriter::WriteBatch(
    std::shared_ptr<arrow::RecordBatch> batch) {
  RecordBatchBuilder builder(client_, batch);
  return this->Push(builder.Seal(client_));
}

Status RecordBatchStreamWriter::WriteDataframe(std::shared_ptr<DataFrame> df) {
  return WriteBatch(df->AsBatch());
}

Status RecordBatchStreamReader::GetNext(std::shared_ptr<Object>& chunk) {
  return client_.ClientBase::PullNextStreamChunk(id_, chunk);
}

Status RecordBatchStreamReader::ReadRecordBatches(
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches) {
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    auto status = ReadBatch(batch);
    if (status.ok()) {
      batches.emplace_back(
          std::dynamic_pointer_cast<RecordBatch>(batch)->GetRecordBatch());
    } else if (status.IsStreamDrained()) {
      break;
    } else {
      return status;
    }
  }
  return Status::OK();
}

Status RecordBatchStreamReader::ReadTable(
    std::shared_ptr<arrow::Table>& table) {
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

Status RecordBatchStreamReader::ReadBatch(
    std::shared_ptr<arrow::RecordBatch>& batch) {
  std::shared_ptr<Object> recordbatch;

  auto status = this->GetNext(recordbatch);
  if (status.ok()) {
    batch =
        std::dynamic_pointer_cast<RecordBatch>(recordbatch)->GetRecordBatch();
  }
  return status;
}

Status RecordBatchStreamReader::GetHeaderLine(bool& header_row,
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

Status RecordBatchStream::OpenReader(
    Client& client, std::unique_ptr<RecordBatchStreamReader>& reader) {
  RETURN_ON_ERROR(client.OpenStream(id_, StreamOpenMode::read));
  reader = std::unique_ptr<RecordBatchStreamReader>(
      new RecordBatchStreamReader(client, id_, meta_, params_));
  return Status::OK();
}

Status RecordBatchStream::OpenWriter(
    Client& client, std::unique_ptr<RecordBatchStreamWriter>& writer) {
  RETURN_ON_ERROR(client.OpenStream(id_, StreamOpenMode::write));
  writer = std::unique_ptr<RecordBatchStreamWriter>(
      new RecordBatchStreamWriter(client, id_, meta_));
  return Status::OK();
}

}  // namespace vineyard
