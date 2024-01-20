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

#include "graph/utils/table_shuffler.h"

#include <mpi.h>

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "boost/leaf.hpp"

#include "grape/communication/sync_comm.h"
#include "grape/utils/concurrent_queue.h"
#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "common/util/functions.h"
#include "common/util/status.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/thread_group.h"

namespace vineyard {

namespace detail {

void send_array_data(const std::shared_ptr<arrow::ArrayData>& data,
                     bool include_data_type, int dst_worker_id, MPI_Comm comm,
                     int tag) {
  // if data is null, send a null pointer
  int64_t is_null = data == nullptr ? 1 : 0;
  MPI_Send(&is_null, 1, MPI_INT64_T, dst_worker_id, tag, comm);
  if (is_null) {
    return;
  }

  // type
  if (include_data_type) {
    std::shared_ptr<arrow::Buffer> buffer;
    // shouldn't fail
    VINEYARD_CHECK_OK(SerializeDataType(data->type, &buffer));
    SendArrowBuffer(buffer, dst_worker_id, comm, tag);
  }

  // length
  int64_t length = data->length;
  MPI_Send(&length, 1, MPI_INT64_T, dst_worker_id, tag, comm);

  // null_count
  int64_t null_count = data->null_count.load();
  MPI_Send(&null_count, 1, MPI_INT64_T, dst_worker_id, tag, comm);

  // offset
  int64_t offset = data->offset;
  MPI_Send(&offset, 1, MPI_INT64_T, dst_worker_id, tag, comm);

  // buffers length
  int64_t buffers_length = data->buffers.size();
  MPI_Send(&buffers_length, 1, MPI_INT64_T, dst_worker_id, tag, comm);

  // buffers
  for (auto& buffer : data->buffers) {
    SendArrowBuffer(buffer, dst_worker_id, comm, tag);
  }

  // child data length
  int64_t child_data_length = data->child_data.size();
  MPI_Send(&child_data_length, 1, MPI_INT64_T, dst_worker_id, tag, comm);

  // child data
  for (auto& child_data : data->child_data) {
    send_array_data(child_data, true, dst_worker_id, comm, tag);
  }

  // dictionary
  send_array_data(data->dictionary, true, dst_worker_id, comm, tag);
}

void recv_array_data(std::shared_ptr<arrow::ArrayData>& data,
                     const std::shared_ptr<arrow::DataType> known_type,
                     int src_worker_id, MPI_Comm comm, int tag) {
  // if data is null, recv a null pointer
  int64_t is_null;
  MPI_Recv(&is_null, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);
  if (is_null == 1) {
    data = nullptr;
    return;
  } else {
    data = std::make_shared<arrow::ArrayData>();
  }

  // type
  if (known_type == nullptr) {
    std::shared_ptr<arrow::Buffer> buffer;
    RecvArrowBuffer(buffer, src_worker_id, comm, tag);
    // shouldn't fail
    ARROW_CHECK_OK(DeserializeDataType(buffer, &data->type));
  } else {
    data->type = known_type;
  }

  // length
  MPI_Recv(&data->length, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);

  // null_count
  int64_t null_count;
  MPI_Recv(&null_count, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);
  data->null_count.store(null_count);

  // offset
  MPI_Recv(&data->offset, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);

  // buffers length
  int64_t buffers_length;
  MPI_Recv(&buffers_length, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);

  // buffers
  for (int64_t i = 0; i < buffers_length; ++i) {
    std::shared_ptr<arrow::Buffer> buffer;
    RecvArrowBuffer(buffer, src_worker_id, comm, tag);
    data->buffers.push_back(buffer);
  }

  // child data length
  int64_t child_data_length;
  MPI_Recv(&child_data_length, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);

  // child data
  for (int64_t i = 0; i < child_data_length; ++i) {
    std::shared_ptr<arrow::ArrayData> child;
    recv_array_data(child, nullptr, src_worker_id, comm, tag);
    data->child_data.push_back(child);
  }

  // dictionary
  recv_array_data(data->dictionary, nullptr, src_worker_id, comm, tag);
}

template <typename T>
static inline void serialize_typed_items(grape::InArchive& arc,
                                         std::shared_ptr<arrow::Array> array) {
  auto ptr = std::dynamic_pointer_cast<ArrowArrayType<T>>(array)->raw_values();
  for (int64_t x = 0; x < array->length(); ++x) {
    arc << ptr[x];
  }
}

template <typename T>
static inline void serialize_typed_items(grape::InArchive& arc,
                                         std::shared_ptr<arrow::Array> array,
                                         const std::vector<int64_t>& offset) {
  auto ptr = std::dynamic_pointer_cast<ArrowArrayType<T>>(array)->raw_values();
  for (auto x : offset) {
    arc << ptr[x];
  }
}

static inline void serialize_string_items(grape::InArchive& arc,
                                          std::shared_ptr<arrow::Array> array,
                                          const std::vector<int64_t>& offset) {
  auto* ptr = std::dynamic_pointer_cast<arrow::LargeStringArray>(array).get();
  for (auto x : offset) {
    arc << ptr->GetView(x);
  }
}

static inline void serialize_null_items(grape::InArchive& arc,
                                        std::shared_ptr<arrow::Array> array,
                                        const std::vector<int64_t>& offset) {
  return;
}

template <typename T>
static inline void serialize_list_items(grape::InArchive& arc,
                                        std::shared_ptr<arrow::Array> array,
                                        const std::vector<int64_t>& offset) {
  auto* ptr = std::dynamic_pointer_cast<arrow::LargeListArray>(array).get();
  for (auto x : offset) {
    arrow::LargeListArray::offset_type length = ptr->value_length(x);
    arc << length;
    auto value = ptr->value_slice(x);
    serialize_typed_items<T>(arc, value);
  }
}

template <typename T>
static inline void deserialize_typed_items(grape::OutArchive& arc, int64_t num,
                                           arrow::ArrayBuilder* builder) {
  auto casted_builder = dynamic_cast<ArrowBuilderType<T>*>(builder);
  ArrowValueType<T> val;
  for (int64_t i = 0; i != num; ++i) {
    arc >> val;
    CHECK_ARROW_ERROR(casted_builder->Append(val));
  }
}

static inline void deserialize_string_items(grape::OutArchive& arc, int64_t num,
                                            arrow::ArrayBuilder* builder) {
  auto casted_builder = dynamic_cast<arrow::LargeStringBuilder*>(builder);
  arrow_string_view val;
  for (int64_t i = 0; i != num; ++i) {
    arc >> val;
    CHECK_ARROW_ERROR(casted_builder->Append(val));
  }
}

static inline void deserialize_null_items(grape::OutArchive& arc, int64_t num,
                                          arrow::ArrayBuilder* builder) {
  auto casted_builder = dynamic_cast<arrow::NullBuilder*>(builder);
  CHECK_ARROW_ERROR(casted_builder->AppendNulls(num));
}

template <typename T>
static inline void deserialize_list_items(grape::OutArchive& arc, int64_t num,
                                          arrow::ArrayBuilder* builder) {
  auto casted_builder = dynamic_cast<arrow::LargeListBuilder*>(builder);
  auto value_builder = casted_builder->value_builder();
  arrow::LargeListArray::offset_type length;
  for (int64_t i = 0; i != num; ++i) {
    arc >> length;
    deserialize_typed_items<T>(arc, length, value_builder);
    CHECK_ARROW_ERROR(casted_builder->Append(true));
  }
}

template <typename T>
static inline void select_typed_items(std::shared_ptr<arrow::Array> array,
                                      arrow::ArrayBuilder* builder) {
  auto ptr = std::dynamic_pointer_cast<ArrowArrayType<T>>(array)->raw_values();
  auto casted_builder = dynamic_cast<ArrowBuilderType<T>*>(builder);
  CHECK_ARROW_ERROR(casted_builder->AppendValues(ptr, array->length()));
}

template <typename T>
static inline void select_typed_items(std::shared_ptr<arrow::Array> array,
                                      const std::vector<int64_t>& offset,
                                      arrow::ArrayBuilder* builder) {
  auto ptr = std::dynamic_pointer_cast<ArrowArrayType<T>>(array)->raw_values();
  auto casted_builder = dynamic_cast<ArrowBuilderType<T>*>(builder);
  for (auto x : offset) {
    CHECK_ARROW_ERROR(casted_builder->Append(ptr[x]));
  }
}

static inline void select_string_items(std::shared_ptr<arrow::Array> array,
                                       const std::vector<int64_t>& offset,
                                       arrow::ArrayBuilder* builder) {
  auto* ptr = std::dynamic_pointer_cast<arrow::LargeStringArray>(array).get();
  auto casted_builder = dynamic_cast<arrow::LargeStringBuilder*>(builder);
  for (auto x : offset) {
    CHECK_ARROW_ERROR(casted_builder->Append(ptr->GetView(x)));
  }
}

static inline void select_null_items(std::shared_ptr<arrow::Array> array,
                                     const std::vector<int64_t>& offset,
                                     arrow::ArrayBuilder* builder) {
  arrow::NullBuilder* casted_builder =
      dynamic_cast<arrow::NullBuilder*>(builder);
  CHECK_ARROW_ERROR(casted_builder->AppendNulls(offset.size()));
}

template <typename T>
static inline void select_list_items(std::shared_ptr<arrow::Array> array,
                                     const std::vector<int64_t>& offset,
                                     arrow::ArrayBuilder* builder) {
  auto ptr = std::dynamic_pointer_cast<arrow::LargeListArray>(array).get();

  auto casted_builder = dynamic_cast<arrow::LargeListBuilder*>(builder);
  auto value_builder = casted_builder->value_builder();
  for (auto x : offset) {
    select_typed_items<T>(ptr->value_slice(x), value_builder);
    CHECK_ARROW_ERROR(casted_builder->Append(true));
  }
}

}  // namespace detail

TableAppender::TableAppender(std::shared_ptr<arrow::Schema> schema) {
  for (const auto& field : schema->fields()) {
    std::shared_ptr<arrow::DataType> type = field->type();
    if (arrow::uint64()->Equals(type)) {
      funcs_.push_back(AppendHelper<uint64_t>::append);
    } else if (arrow::int64()->Equals(type)) {
      funcs_.push_back(AppendHelper<int64_t>::append);
    } else if (arrow::uint32()->Equals(type)) {
      funcs_.push_back(AppendHelper<uint32_t>::append);
    } else if (arrow::int32()->Equals(type)) {
      funcs_.push_back(AppendHelper<int32_t>::append);
    } else if (arrow::float32()->Equals(type)) {
      funcs_.push_back(AppendHelper<float>::append);
    } else if (arrow::float64()->Equals(type)) {
      funcs_.push_back(AppendHelper<double>::append);
    } else if (arrow::large_binary()->Equals(type)) {
      funcs_.push_back(AppendHelper<std::string>::append);
    } else if (arrow::large_utf8()->Equals(type)) {
      funcs_.push_back(AppendHelper<std::string>::append);
    } else if (arrow::null()->Equals(type)) {
      funcs_.push_back(AppendHelper<void>::append);
    } else if (arrow::date32()->Equals(type)) {
      funcs_.push_back(AppendHelper<arrow::Date32Type>::append);
    } else if (arrow::date64()->Equals(type)) {
      funcs_.push_back(AppendHelper<arrow::Date64Type>::append);
    } else if (type->id() == arrow::Type::TIME32) {
      funcs_.push_back(AppendHelper<arrow::Time32Type>::append);
    } else if (type->id() == arrow::Type::TIME64) {
      funcs_.push_back(AppendHelper<arrow::Time64Type>::append);
    } else if (type->id() == arrow::Type::TIMESTAMP) {
      funcs_.push_back(AppendHelper<arrow::TimestampType>::append);
    } else {
      LOG(ERROR) << "Datatype [" << type->ToString() << "] not implemented...";
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
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
    RETURN_ON_ARROW_ERROR(builder->Flush(&tmp_batch));
#else
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(tmp_batch, builder->Flush());
#endif
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
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
    RETURN_ON_ARROW_ERROR(builder->Flush(&batch));
#else
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(batch, builder->Flush());
#endif
    batches_out.emplace_back(std::move(batch));
  }
  return Status::OK();
}

void SendArrowBuffer(const std::shared_ptr<arrow::Buffer>& buffer,
                     int dst_worker_id, MPI_Comm comm, int tag) {
  int64_t size = -1;
  if (buffer == nullptr) {
    MPI_Send(&size, 1, MPI_INT64_T, dst_worker_id, tag, comm);
    return;
  }
  size = buffer->size();
  MPI_Send(&size, 1, MPI_INT64_T, dst_worker_id, tag, comm);
  if (size != 0) {
    grape::sync_comm::send_buffer<uint8_t>(buffer->data(), size, dst_worker_id,
                                           tag, comm);
  }
}

void RecvArrowBuffer(std::shared_ptr<arrow::Buffer>& buffer, int src_worker_id,
                     MPI_Comm comm, int tag) {
  int64_t size = -1;
  MPI_Recv(&size, 1, MPI_INT64_T, src_worker_id, tag, comm, MPI_STATUS_IGNORE);
  if (size == -1) {
    buffer = nullptr;
    return;
  }
  if (size != 0) {
    ARROW_CHECK_OK_AND_ASSIGN(
        buffer, arrow::AllocateBuffer(size, arrow::default_memory_pool()));
    grape::sync_comm::recv_buffer<uint8_t>(buffer->mutable_data(), size,
                                           src_worker_id, tag, comm);
  } else {
    // empty, but not null
    buffer = std::make_shared<arrow::Buffer>(nullptr, 0);
  }
}

template <typename ArrayType>
Status FragmentAllGatherArray(
    const grape::CommSpec& comm_spec, std::shared_ptr<ArrayType> data_in,
    std::vector<std::shared_ptr<ArrayType>>& data_out) {
  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();

  // reserve space
  data_out.resize(comm_spec.fnum());

  auto send_procedure = [&]() -> Status {
    int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
    while (dst_worker_id != worker_id) {
      SendArrowArray(data_in, dst_worker_id, comm_spec.comm());
      dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
    }
    return Status::OK();
  };

  auto recv_procedure = [&]() -> Status {
    int src_worker_id = (worker_id + 1) % worker_num;
    while (src_worker_id != worker_id) {
      fid_t src_fid = comm_spec.WorkerToFrag(src_worker_id);
      std::shared_ptr<ArrayType>& out = data_out[src_fid];
      RecvArrowArray(out, src_worker_id, comm_spec.comm());
      src_worker_id = (src_worker_id + 1) % worker_num;
    }
    data_out[comm_spec.fid()] = data_in;
    return Status::OK();
  };

  // ensure both have thread resources
  DynamicThreadGroup tg(2 /* sender + receiver */);
  tg.AddTask(send_procedure);
  tg.AddTask(recv_procedure);

  Status status;
  for (auto& res : tg.TakeResults()) {
    status += res;
  }
  return status;
}

template Status FragmentAllGatherArray<arrow::Array>(
    const grape::CommSpec& comm_spec, std::shared_ptr<arrow::Array> data_in,
    std::vector<std::shared_ptr<arrow::Array>>& data_out);

template Status FragmentAllGatherArray<arrow::ChunkedArray>(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<arrow::ChunkedArray> data_in,
    std::vector<std::shared_ptr<arrow::ChunkedArray>>& data_out);

template Status FragmentAllGatherArray<arrow::Int32Array>(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<arrow::Int32Array> data_in,
    std::vector<std::shared_ptr<arrow::Int32Array>>& data_out);

template Status FragmentAllGatherArray<arrow::UInt32Array>(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<arrow::UInt32Array> data_in,
    std::vector<std::shared_ptr<arrow::UInt32Array>>& data_out);

template Status FragmentAllGatherArray<arrow::Int64Array>(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<arrow::Int64Array> data_in,
    std::vector<std::shared_ptr<arrow::Int64Array>>& data_out);

template Status FragmentAllGatherArray<arrow::UInt64Array>(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<arrow::UInt64Array> data_in,
    std::vector<std::shared_ptr<arrow::UInt64Array>>& data_out);

template Status FragmentAllGatherArray<arrow::StringArray>(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<arrow::StringArray> data_in,
    std::vector<std::shared_ptr<arrow::StringArray>>& data_out);

template Status FragmentAllGatherArray<arrow::LargeStringArray>(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<arrow::LargeStringArray> data_in,
    std::vector<std::shared_ptr<arrow::LargeStringArray>>& data_out);

Status CheckSchemaConsistency(const arrow::Schema& schema,
                              const grape::CommSpec& comm_spec) {
  std::shared_ptr<arrow::Buffer> buffer;
  auto serialized_status = SerializeSchema(schema, &buffer);

  if (!serialized_status.ok()) {
    int flag = 1;
    int sum = 0;
    MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec.comm());
    return Status::Wrap(serialized_status, "failed to serialize the schema");
  } else {
    int flag = 0;
    int sum = 0;
    MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec.comm());
    if (sum != 0) {
      return ArrowError(
          arrow::Status(arrow::StatusCode::Invalid,
                        "failed to serialize the schema on peer worker"));
    }
  }

  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();

  std::thread send_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int dst_worker_id = (worker_id + i) % worker_num;
      SendArrowBuffer(buffer, dst_worker_id, comm_spec.comm());
    }
  });
  bool consistent = true;
  Status status;
  std::thread recv_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int src_worker_id = (worker_id + worker_num - i) % worker_num;
      std::shared_ptr<arrow::Buffer> got_buffer;
      RecvArrowBuffer(got_buffer, src_worker_id, comm_spec.comm());

      std::shared_ptr<arrow::Schema> got_schema;
      status += DeserializeSchema(got_buffer, &got_schema);
      if (status.ok()) {
        consistent &= (got_schema->Equals(schema));
      } else {
        consistent = false;
      }
    }
  });

  send_thread.join();
  recv_thread.join();

  MPI_Barrier(comm_spec.comm());

  if (!consistent) {
    if (status.ok()) {
      return ArrowError(
          arrow::Status(arrow::StatusCode::Invalid,
                        "Schemas of edge tables are not consistent."));
    } else {
      return Status::Wrap(status, "schemas of edge tables are not consistent.");
    }
  }
  return Status::OK();
}

void SerializeSelectedItems(grape::InArchive& arc,
                            std::shared_ptr<arrow::Array> array,
                            const std::vector<int64_t>& offset) {
  if (array->type()->Equals(arrow::null())) {
    detail::serialize_null_items(arc, array, offset);
  } else if (array->type()->Equals(arrow::float64())) {
    detail::serialize_typed_items<double>(arc, array, offset);
  } else if (array->type()->Equals(arrow::float32())) {
    detail::serialize_typed_items<float>(arc, array, offset);
  } else if (array->type()->Equals(arrow::int64())) {
    detail::serialize_typed_items<int64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::int32())) {
    detail::serialize_typed_items<int32_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::uint64())) {
    detail::serialize_typed_items<uint64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::uint32())) {
    detail::serialize_typed_items<uint32_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_utf8())) {
    detail::serialize_string_items(arc, array, offset);
  } else if (array->type()->Equals(arrow::null())) {
    detail::serialize_null_items(arc, array, offset);
  } else if (array->type()->Equals(arrow::date32())) {
    detail::serialize_typed_items<arrow::Date32Type>(arc, array, offset);
  } else if (array->type()->Equals(arrow::date64())) {
    detail::serialize_typed_items<arrow::Date64Type>(arc, array, offset);
  } else if (array->type()->id() == arrow::Type::TIME32) {
    detail::serialize_typed_items<arrow::Time32Type>(arc, array, offset);
  } else if (array->type()->id() == arrow::Type::TIME64) {
    detail::serialize_typed_items<arrow::Time64Type>(arc, array, offset);
  } else if (array->type()->id() == arrow::Type::TIMESTAMP) {
    detail::serialize_typed_items<arrow::TimestampType>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::float64()))) {
    detail::serialize_list_items<double>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::float32()))) {
    detail::serialize_list_items<float>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::int64()))) {
    detail::serialize_list_items<int64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::int32()))) {
    detail::serialize_list_items<int32_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint64()))) {
    detail::serialize_list_items<uint64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint32()))) {
    detail::serialize_list_items<uint32_t>(arc, array, offset);
  } else {
    LOG(ERROR) << "Unsupported data type - " << array->type()->ToString();
  }
}

void SerializeSelectedRows(grape::InArchive& arc,
                           std::shared_ptr<arrow::RecordBatch> record_batch,
                           const std::vector<int64_t>& offset) {
  arc << static_cast<int64_t>(offset.size());
  if (record_batch == nullptr) {  // skip
    return;
  }
  int col_num = record_batch->num_columns();
  for (int col_id = 0; col_id != col_num; ++col_id) {
    SerializeSelectedItems(arc, record_batch->column(col_id), offset);
  }
}

void DeserializeSelectedItems(grape::OutArchive& arc, int64_t num,
                              arrow::ArrayBuilder* builder) {
  if (builder->type()->Equals(arrow::null())) {
    detail::deserialize_null_items(arc, num, builder);
  } else if (builder->type()->Equals(arrow::float64())) {
    detail::deserialize_typed_items<double>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::float32())) {
    detail::deserialize_typed_items<float>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::int64())) {
    detail::deserialize_typed_items<int64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::int32())) {
    detail::deserialize_typed_items<int32_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::uint64())) {
    detail::deserialize_typed_items<uint64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::uint32())) {
    detail::deserialize_typed_items<uint32_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_utf8())) {
    detail::deserialize_string_items(arc, num, builder);
  } else if (builder->type()->Equals(arrow::date32())) {
    detail::deserialize_typed_items<arrow::Date32Type>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::date64())) {
    detail::deserialize_typed_items<arrow::Date64Type>(arc, num, builder);
  } else if (builder->type()->id() == arrow::Type::TIME32) {
    detail::deserialize_typed_items<arrow::Time32Type>(arc, num, builder);
  } else if (builder->type()->id() == arrow::Type::TIME64) {
    detail::deserialize_typed_items<arrow::Time64Type>(arc, num, builder);
  } else if (builder->type()->id() == arrow::Type::TIMESTAMP) {
    detail::deserialize_typed_items<arrow::TimestampType>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::float64()))) {
    detail::deserialize_list_items<double>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::float32()))) {
    detail::deserialize_list_items<float>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::int64()))) {
    detail::deserialize_list_items<int64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::int32()))) {
    detail::deserialize_list_items<int32_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::uint64()))) {
    detail::deserialize_list_items<uint64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::uint32()))) {
    detail::deserialize_list_items<uint32_t>(arc, num, builder);
  } else {
    LOG(ERROR) << "Unsupported data type - " << builder->type()->ToString();
  }
}

void DeserializeSelectedRows(grape::OutArchive& arc,
                             std::shared_ptr<arrow::Schema> schema,
                             std::shared_ptr<arrow::RecordBatch>& batch_out) {
  int64_t row_num;
  arc >> row_num;
  std::unique_ptr<arrow::RecordBatchBuilder> builder;
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  ARROW_CHECK_OK(arrow::RecordBatchBuilder::Make(
      schema, arrow::default_memory_pool(), row_num, &builder));
#else
  ARROW_CHECK_OK_AND_ASSIGN(builder,
                            arrow::RecordBatchBuilder::Make(
                                schema, arrow::default_memory_pool(), row_num));
#endif
  int col_num = builder->num_fields();
  for (int col_id = 0; col_id != col_num; ++col_id) {
    DeserializeSelectedItems(arc, row_num, builder->GetField(col_id));
  }
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  ARROW_CHECK_OK(builder->Flush(&batch_out));
#else
  ARROW_CHECK_OK_AND_ASSIGN(batch_out, builder->Flush());
#endif
}

void SelectItems(std::shared_ptr<arrow::Array> array,
                 const std::vector<int64_t> offset,
                 arrow::ArrayBuilder* builder) {
  if (array->type()->Equals(arrow::null())) {
    detail::select_null_items(array, offset, builder);
  } else if (array->type()->Equals(arrow::float64())) {
    detail::select_typed_items<double>(array, offset, builder);
  } else if (array->type()->Equals(arrow::float32())) {
    detail::select_typed_items<float>(array, offset, builder);
  } else if (array->type()->Equals(arrow::int64())) {
    detail::select_typed_items<int64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::int32())) {
    detail::select_typed_items<int32_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::uint64())) {
    detail::select_typed_items<uint64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::uint32())) {
    detail::select_typed_items<uint32_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_utf8())) {
    detail::select_string_items(array, offset, builder);
  } else if (array->type()->Equals(arrow::date32())) {
    detail::select_typed_items<arrow::Date32Type>(array, offset, builder);
  } else if (array->type()->Equals(arrow::date64())) {
    detail::select_typed_items<arrow::Date64Type>(array, offset, builder);
  } else if (array->type()->id() == arrow::Type::TIME32) {
    detail::select_typed_items<arrow::Time32Type>(array, offset, builder);
  } else if (array->type()->id() == arrow::Type::TIME64) {
    detail::select_typed_items<arrow::Time64Type>(array, offset, builder);
  } else if (array->type()->id() == arrow::Type::TIMESTAMP) {
    detail::select_typed_items<arrow::TimestampType>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::float64()))) {
    detail::select_list_items<double>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::float32()))) {
    detail::select_list_items<float>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::int64()))) {
    detail::select_list_items<int64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::int32()))) {
    detail::select_list_items<int32_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint64()))) {
    detail::select_list_items<uint64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint32()))) {
    detail::select_list_items<uint32_t>(array, offset, builder);
  } else {
    LOG(ERROR) << "Unsupported data type - " << builder->type()->ToString();
  }
}

void SelectRows(std::shared_ptr<arrow::RecordBatch> record_batch_in,
                const std::vector<int64_t>& offset,
                std::shared_ptr<arrow::RecordBatch>& record_batch_out) {
  int64_t row_num = offset.size();
  if (record_batch_in == nullptr) {  // skip
    record_batch_out = nullptr;
    return;
  }
  std::unique_ptr<arrow::RecordBatchBuilder> builder;
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  ARROW_CHECK_OK(arrow::RecordBatchBuilder::Make(record_batch_in->schema(),
                                                 arrow::default_memory_pool(),
                                                 row_num, &builder));
#else
  ARROW_CHECK_OK_AND_ASSIGN(
      builder,
      arrow::RecordBatchBuilder::Make(record_batch_in->schema(),
                                      arrow::default_memory_pool(), row_num));
#endif
  int col_num = builder->num_fields();
  for (int col_id = 0; col_id != col_num; ++col_id) {
    SelectItems(record_batch_in->column(col_id), offset,
                builder->GetField(col_id));
  }
#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  ARROW_CHECK_OK(builder->Flush(&record_batch_out));
#else
  ARROW_CHECK_OK_AND_ASSIGN(record_batch_out, builder->Flush());
#endif
}

boost::leaf::result<void> ShuffleTableByOffsetLists(
    const grape::CommSpec& comm_spec,
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_send,
    const std::vector<std::vector<std::vector<int64_t>>>& offset_lists,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_recv) {
  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();
  size_t record_batches_out_num = record_batches_send.size();

  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
      comm_spec.local_num();
  int deserialize_thread_num = std::max(1, (thread_num - 2) / 2);
  int serialize_thread_num =
      std::max(1, thread_num - 2 - deserialize_thread_num);
  std::vector<std::thread> serialize_threads(serialize_thread_num);
  std::vector<std::thread> deserialize_threads(deserialize_thread_num);

  grape::BlockingQueue<std::pair<grape::fid_t, grape::InArchive>> msg_out;
  grape::BlockingQueue<grape::OutArchive> msg_in;

  msg_out.SetProducerNum(serialize_thread_num);
  msg_in.SetProducerNum(1);

  int64_t record_batches_to_send = static_cast<int64_t>(record_batches_out_num);
  int64_t total_record_batches;
  MPI_Allreduce(&record_batches_to_send, &total_record_batches, 1, MPI_INT64_T,
                MPI_SUM, comm_spec.comm());
  int64_t record_batches_to_recv =
      total_record_batches - record_batches_to_send;

  std::thread send_thread([&]() {
    std::pair<grape::fid_t, grape::InArchive> item;
    while (msg_out.Get(item)) {
      int dst_worker_id = comm_spec.FragToWorker(item.first);
      auto& arc = item.second;
      grape::sync_comm::Send(arc, dst_worker_id, 0, comm_spec.comm());
    }
  });

  std::thread recv_thread([&]() {
    int64_t remaining_msg_num = record_batches_to_recv;
    while (remaining_msg_num != 0) {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_spec.comm(), &status);
      grape::OutArchive arc;
      grape::sync_comm::Recv(arc, status.MPI_SOURCE, 0, comm_spec.comm());
      msg_in.Put(std::move(arc));
      --remaining_msg_num;
    }
    msg_in.DecProducerNum();
  });

  std::atomic<size_t> cur_batch_out(0);
  for (int i = 0; i != serialize_thread_num; ++i) {
    serialize_threads[i] = std::thread([&]() {
      while (true) {
        size_t got_batch = cur_batch_out.fetch_add(1);
        if (got_batch >= record_batches_out_num) {
          break;
        }
        auto cur_rb = record_batches_send[got_batch];
        auto& cur_offset_lists = offset_lists[got_batch];

        for (int i = 1; i != worker_num; ++i) {
          int dst_worker_id = (worker_id + i) % worker_num;
          grape::fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
          std::pair<grape::fid_t, grape::InArchive> item;
          item.first = dst_fid;
          SerializeSelectedRows(item.second, cur_rb, cur_offset_lists[dst_fid]);
          msg_out.Put(std::move(item));
        }
      }
      msg_out.DecProducerNum();
    });
  }

  std::atomic<int64_t> cur_batch_in(0);
  record_batches_recv.resize(record_batches_to_recv);
  for (int i = 0; i != deserialize_thread_num; ++i) {
    deserialize_threads[i] = std::thread([&]() {
      grape::OutArchive arc;
      while (msg_in.Get(arc)) {
        int64_t got_batch = cur_batch_in.fetch_add(1);
        DeserializeSelectedRows(arc, schema, record_batches_recv[got_batch]);
      }
    });
  }

  send_thread.join();
  recv_thread.join();
  for (auto& thrd : serialize_threads) {
    thrd.join();
  }
  for (auto& thrd : deserialize_threads) {
    thrd.join();
  }

  for (size_t rb_i = 0; rb_i != record_batches_out_num; ++rb_i) {
    std::shared_ptr<arrow::RecordBatch> rb;
    SelectRows(record_batches_send[rb_i], offset_lists[rb_i][comm_spec.fid()],
               rb);
    record_batches_recv.emplace_back(std::move(rb));
  }
  MPI_Barrier(comm_spec.comm());
  return {};
}

boost::leaf::result<void> ShuffleTableByOffsetLists(
    const grape::CommSpec& comm_spec,
    const std::shared_ptr<arrow::Schema> schema,
    const std::shared_ptr<ITablePipeline>& record_batches_send,
    std::function<void(const std::shared_ptr<arrow::RecordBatch> batch,
                       std::vector<std::vector<int64_t>>& offset_list)>
        genoffset,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_recv) {
  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();
  size_t record_batches_out_num = record_batches_send->num_batches();

  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
      comm_spec.local_num();
  // after pipelining, the serialize thread would be responsible for trigger
  // the execution of whole pipeline, including things like generate eid,
  // resolve oid -> gid mapping, etc., and become more computation intensive.
  int deserialize_thread_num = std::max(1, (thread_num - 2) / 6);
  int serialize_thread_num =
      std::max(1, thread_num - 2 - deserialize_thread_num);
  std::vector<std::thread> serialize_threads(serialize_thread_num);
  std::vector<std::thread> deserialize_threads(deserialize_thread_num);

  grape::BlockingQueue<std::pair<grape::fid_t, grape::InArchive>> msg_out;
  grape::BlockingQueue<grape::OutArchive> msg_in;

  msg_out.SetProducerNum(serialize_thread_num);
  msg_in.SetProducerNum(1);

  int64_t record_batches_to_send = static_cast<int64_t>(record_batches_out_num);
  int64_t total_record_batches;
  MPI_Allreduce(&record_batches_to_send, &total_record_batches, 1, MPI_INT64_T,
                MPI_SUM, comm_spec.comm());
  int64_t record_batches_to_recv =
      total_record_batches - record_batches_to_send;

  // preserve spaces for recv and self-recv
  record_batches_recv.resize(total_record_batches);
  VLOG(100) << "[worker-" << comm_spec.worker_id()
            << "] ShuffleTableByOffsetLists: batches: total = "
            << total_record_batches << ", to send = " << record_batches_to_send
            << ", to recv = " << record_batches_to_recv
            << ", serialization thread: " << serialize_thread_num
            << ", deserialization thread: " << deserialize_thread_num;

  std::thread send_thread([&]() {
    std::pair<grape::fid_t, grape::InArchive> item;
    while (msg_out.Get(item)) {
      int dst_worker_id = comm_spec.FragToWorker(item.first);
      auto& arc = item.second;
      grape::sync_comm::Send(arc, dst_worker_id, 0, comm_spec.comm());
    }
  });

  std::thread recv_thread([&]() {
    int64_t remaining_msg_num = record_batches_to_recv;
    while (remaining_msg_num != 0) {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_spec.comm(), &status);
      grape::OutArchive arc;
      grape::sync_comm::Recv(arc, status.MPI_SOURCE, 0, comm_spec.comm());
      msg_in.Put(std::move(arc));
      --remaining_msg_num;
    }
    msg_in.DecProducerNum();
  });

  // self: starts from `record_batches_to_recv`.
  std::atomic<int64_t> cur_self_batch_in(record_batches_to_recv);
  std::vector<vineyard::Status> processing_errors(serialize_thread_num);
  for (int thread_idx = 0; thread_idx != serialize_thread_num; ++thread_idx) {
    serialize_threads[thread_idx] = std::thread(
        [&](const int thread_index) {
          std::vector<std::vector<int64_t>> offset_lists(comm_spec.fnum());
          while (true) {
            std::shared_ptr<arrow::RecordBatch> batch;
            auto status = record_batches_send->Next(batch);
            if (status.IsStreamDrained()) {
              break;
            }
            if (!status.ok()) {
              LOG(ERROR) << "Failed to fetch a batch from the table pipeline: "
                         << status.ToString();
              processing_errors[thread_index] = Status::Wrap(
                  status, "Failed to fetch a batch from the table pipeline");
            }
            // NB: when error occurs, keep drain the input queue, and generate
            // empty serialized outputs
            if (!processing_errors[thread_index].ok()) {
              batch = nullptr;
            }

            // generate offset lists
            genoffset(batch, offset_lists);

            // send to other workers
            for (int i = 1; i != worker_num; ++i) {
              int dst_worker_id = (worker_id + i) % worker_num;
              grape::fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
              std::pair<grape::fid_t, grape::InArchive> item;
              item.first = dst_fid;
              SerializeSelectedRows(item.second, batch, offset_lists[dst_fid]);
              msg_out.Put(std::move(item));
            }

            // select to self and put to recv
            int64_t got_batch = cur_self_batch_in.fetch_add(1);
            SelectRows(batch, offset_lists[comm_spec.fid()],
                       record_batches_recv[got_batch]);
          }
          msg_out.DecProducerNum();
        },
        thread_idx);
  }

  std::atomic<int64_t> cur_batch_in(0);
  for (int i = 0; i != deserialize_thread_num; ++i) {
    deserialize_threads[i] = std::thread([&]() {
      grape::OutArchive arc;
      while (msg_in.Get(arc)) {
        int64_t got_batch = cur_batch_in.fetch_add(1);
        DeserializeSelectedRows(arc, schema, record_batches_recv[got_batch]);
      }
    });
  }

  send_thread.join();
  recv_thread.join();
  for (auto& thrd : serialize_threads) {
    thrd.join();
  }
  for (auto& thrd : deserialize_threads) {
    thrd.join();
  }
  vineyard::Status error;
  MPI_Barrier(comm_spec.comm());
  for (auto& err : processing_errors) {
    error += err;
  }
  VY_OK_OR_RAISE(error);
  return {};
}

}  // namespace vineyard
