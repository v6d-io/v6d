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

#include "graph/utils/table_shuffler.h"

namespace vineyard {

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
    } else if (type == arrow::large_binary()) {
      funcs_.push_back(AppendHelper<std::string>::append);
    } else if (type == arrow::large_utf8()) {
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
