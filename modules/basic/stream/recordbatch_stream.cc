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

Status RecordBatchStream::WriteTable(
    std::shared_ptr<arrow::Table> const& table) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(TableToRecordBatches(table, &batches));
  for (auto const& batch : batches) {
    RETURN_ON_ERROR(WriteBatch(batch));
  }
  return Status::OK();
}

Status RecordBatchStream::WriteBatch(
    std::shared_ptr<arrow::RecordBatch> const& batch) {
  RecordBatchBuilder builder(*client_, batch);
  return this->Push(builder.Seal(*client_));
}

Status RecordBatchStream::WriteDataframe(std::shared_ptr<DataFrame> const& df) {
  // TODO: needs optimization to make it zero-copy
  return WriteBatch(df->AsBatch());
}

Status RecordBatchStream::ReadRecordBatches(
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

Status RecordBatchStream::ReadTable(std::shared_ptr<arrow::Table>& table) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(this->ReadRecordBatches(batches));
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(table,
                                   arrow::Table::FromRecordBatches(batches));
  return Status::OK();
}

Status RecordBatchStream::ReadBatch(
    std::shared_ptr<arrow::RecordBatch>& batch) {
  std::shared_ptr<RecordBatch> recordbatch;
  auto status = this->Next(recordbatch);
  if (status.ok()) {
    batch = recordbatch->GetRecordBatch();
  }
  return status;
}

}  // namespace vineyard
