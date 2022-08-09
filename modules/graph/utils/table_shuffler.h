/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#ifndef MODULES_GRAPH_UTILS_TABLE_SHUFFLER_H_
#define MODULES_GRAPH_UTILS_TABLE_SHUFFLER_H_

#include <mpi.h>

#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "boost/leaf.hpp"

#include "grape/communication/sync_comm.h"
#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "common/util/status.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/thread_group.h"

namespace vineyard {

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
struct AppendHelper<arrow::Date32Type> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::Date32Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::Date32Array>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<arrow::Date64Type> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::Date64Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::Date64Array>(array)->GetView(offset)));
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

const int chunk_size = 409600;

template <typename T>
inline void send_buffer(const T* ptr, size_t len, int dst_worker_id,
                        MPI_Comm comm, int tag = 0) {
  const size_t chunk_size_in_bytes = chunk_size * sizeof(T);
  int iter = len / chunk_size;
  size_t remaining = (len % chunk_size) * sizeof(T);
  for (int i = 0; i < iter; ++i) {
    MPI_Send(ptr, chunk_size_in_bytes, MPI_CHAR, dst_worker_id, tag, comm);
    ptr += chunk_size;
  }
  if (remaining != 0) {
    MPI_Send(ptr, remaining, MPI_CHAR, dst_worker_id, tag, comm);
  }
}

template <typename T>
inline void recv_buffer(T* ptr, size_t len, int src_worker_id, MPI_Comm comm,
                        int tag = 0) {
  const size_t chunk_size_in_bytes = chunk_size * sizeof(T);
  int iter = len / chunk_size;
  size_t remaining = (len % chunk_size) * sizeof(T);
  for (int i = 0; i < iter; ++i) {
    MPI_Recv(ptr, chunk_size_in_bytes, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
    ptr += chunk_size;
  }
  if (remaining != 0) {
    MPI_Recv(ptr, remaining, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
  }
}

inline std::shared_ptr<arrow::RecordBatch> ArrayToRecordBatch(
    const std::shared_ptr<arrow::Array>& array) {
  auto field = std::make_shared<arrow::Field>("data", array->type());
  std::vector<std::shared_ptr<arrow::Field>> fields = {field};
  auto schema = std::make_shared<arrow::Schema>(fields);
  int64_t num_rows = array->length();
  std::vector<std::shared_ptr<arrow::Array>> array_list = {array};
  return arrow::RecordBatch::Make(schema, num_rows, array_list);
}

inline void SendArrowBuffer(const std::shared_ptr<arrow::Buffer>& buffer,
                            int dst_worker_id, MPI_Comm comm, int tag = 0) {
  int64_t size = buffer->size();
  MPI_Send(&size, 1, MPI_INT64_T, dst_worker_id, tag, comm);
  if (size != 0) {
    send_buffer<uint8_t>(buffer->data(), size, dst_worker_id, comm, tag);
  }
}

inline Status RecvArrowBuffer(std::shared_ptr<arrow::Buffer>& buffer,
                              int src_worker_id, MPI_Comm comm, int tag = 0) {
  int64_t size;
  MPI_Recv(&size, 1, MPI_INT64_T, src_worker_id, tag, comm, MPI_STATUS_IGNORE);

  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      buffer, arrow::AllocateBuffer(size, arrow::default_memory_pool()));

  if (size != 0) {
    recv_buffer<uint8_t>(buffer->mutable_data(), size, src_worker_id, comm,
                         tag);
  }
  return Status::OK();
}

template <typename T>
inline Status send_numeric_array(
    const std::shared_ptr<typename ConvertToArrowType<T>::ArrayType>& array,
    int dst_worker_id, MPI_Comm comm, int tag = 0) {
  size_t len = array->length();
  MPI_Send(&len, sizeof(size_t), MPI_CHAR, dst_worker_id, tag, comm);
  send_buffer<T>(array->raw_values(), len, dst_worker_id, comm, tag);
  return Status::OK();
}

template <>
inline Status send_numeric_array<std::string>(
    const std::shared_ptr<arrow::LargeStringArray>& array, int dst_worker_id,
    MPI_Comm comm, int tag) {
  auto batch =
      ArrayToRecordBatch(std::dynamic_pointer_cast<arrow::Array>(array));
  std::shared_ptr<arrow::Buffer> buffer;
  RETURN_ON_ERROR(SerializeRecordBatches({batch}, &buffer));
  SendArrowBuffer(buffer, dst_worker_id, comm, tag);
  return Status::OK();
}

template <typename T>
inline Status recv_numeric_array(
    std::shared_ptr<typename ConvertToArrowType<T>::ArrayType>& array,
    int src_worker_id, MPI_Comm comm, int tag = 0) {
  size_t len;
  MPI_Recv(&len, sizeof(size_t), MPI_CHAR, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);
  typename ConvertToArrowType<T>::BuilderType builder;
  RETURN_ON_ARROW_ERROR(builder.Resize(len));
  recv_buffer<T>(&builder[0], len, src_worker_id, comm, tag);
  RETURN_ON_ARROW_ERROR(builder.Advance(len));
  RETURN_ON_ARROW_ERROR(builder.Finish(&array));
  return Status::OK();
}

template <>
inline Status recv_numeric_array<std::string>(
    std::shared_ptr<typename ConvertToArrowType<std::string>::ArrayType>& array,
    int src_worker_id, MPI_Comm comm, int tag) {
  std::shared_ptr<arrow::Buffer> buffer;
  RETURN_ON_ERROR(RecvArrowBuffer(buffer, src_worker_id, comm, tag));
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  RETURN_ON_ERROR(DeserializeRecordBatches(buffer, &batches));
  CHECK_EQ(batches.size(), 1);
  CHECK_EQ(batches[0]->num_columns(), 1);
  CHECK_EQ(batches[0]->column(0)->type(), arrow::large_utf8());
  array =
      std::dynamic_pointer_cast<arrow::LargeStringArray>(batches[0]->column(0));
  return Status::OK();
}

template <typename T>
Status FragmentAllGatherArray(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<typename ConvertToArrowType<T>::ArrayType> data_in,
    std::vector<std::shared_ptr<typename ConvertToArrowType<T>::ArrayType>>&
        data_out) {
  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();

  data_out.resize(comm_spec.fnum());

  auto send_procedure = [&]() {
    int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
    while (dst_worker_id != worker_id) {
      RETURN_ON_ERROR(
          send_numeric_array<T>(data_in, dst_worker_id, comm_spec.comm()));
      dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
    }
    return Status::OK();
  };

  auto recv_procedure = [&]() {
    int src_worker_id = (worker_id + 1) % worker_num;
    while (src_worker_id != worker_id) {
      fid_t src_fid = comm_spec.WorkerToFrag(src_worker_id);
      RETURN_ON_ERROR(recv_numeric_array<T>(data_out[src_fid], src_worker_id,
                                            comm_spec.comm()));
      src_worker_id = (src_worker_id + 1) % worker_num;
    }
    data_out[comm_spec.fid()] = data_in;
    return Status::OK();
  };

  ThreadGroup tg;

  tg.AddTask(send_procedure);
  tg.AddTask(recv_procedure);
  auto results = tg.TakeResults();

  for (auto& res : results) {
    RETURN_ON_ERROR(res);
  }
  return Status::OK();
}

template <typename PARTITIONER_T>
Status ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                                  const PARTITIONER_T& partitioner,
                                  const std::shared_ptr<arrow::Table>& table_in,
                                  std::shared_ptr<arrow::Table>& table_out) {
  using oid_t = typename PARTITIONER_T::oid_t;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_type = typename ConvertToArrowType<oid_t>::ArrayType;
  auto fnum = comm_spec.fnum();
  std::vector<std::unique_ptr<arrow::RecordBatchBuilder>>
      divided_table_builders;
  std::vector<std::vector<std::shared_ptr<arrow::RecordBatch>>> divided_records(
      fnum);
  for (fid_t i = 0; i < fnum; ++i) {
    std::unique_ptr<arrow::RecordBatchBuilder> builder;
    RETURN_ON_ARROW_ERROR(arrow::RecordBatchBuilder::Make(
        table_in->schema(), arrow::default_memory_pool(), 4096, &builder));
    divided_table_builders.emplace_back(std::move(builder));
  }
  TableAppender appender(table_in->schema());
  arrow::TableBatchReader tbreader(*table_in);
  std::shared_ptr<arrow::RecordBatch> batch;

  while (true) {
    RETURN_ON_ARROW_ERROR(tbreader.ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    auto id_col = std::dynamic_pointer_cast<oid_array_type>(batch->column(0));
    size_t row_num = batch->num_rows();
    for (size_t i = 0; i < row_num; ++i) {
      internal_oid_t rs = id_col->GetView(i);
      fid_t fid = partitioner.GetPartitionId(oid_t(rs));
      RETURN_ON_ERROR(appender.Apply(divided_table_builders[fid], batch, i,
                                     divided_records[fid]));
    }
  }

  for (fid_t i = 0; i < fnum; ++i) {
    std::unique_ptr<arrow::RecordBatchBuilder> builder =
        std::move(divided_table_builders[i]);
    RETURN_ON_ERROR(appender.Flush(builder, divided_records[i]));
  }
  divided_table_builders.clear();

  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();

  auto send_procedure = [&]() -> Status {
    int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
    while (dst_worker_id != worker_id) {
      fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
      std::shared_ptr<arrow::Buffer> buffer;
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches =
          std::move(divided_records[dst_fid]);
      RETURN_ON_ERROR(SerializeRecordBatches(batches, &buffer));
      SendArrowBuffer(buffer, dst_worker_id, comm_spec.comm());
      dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
    }
    return Status::OK();
  };

  auto recv_procedure = [&]() -> Status {
    int src_worker_id = (worker_id + 1) % worker_num;
    while (src_worker_id != worker_id) {
      std::shared_ptr<arrow::Buffer> buffer;
      RETURN_ON_ERROR(RecvArrowBuffer(buffer, src_worker_id, comm_spec.comm()));
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      RETURN_ON_ERROR(DeserializeRecordBatches(buffer, &batches));
      for (const auto& batch : batches) {
        if (batch->num_rows() > 0) {
          divided_records[comm_spec.fid()].push_back(batch);
        }
      }
      src_worker_id = (src_worker_id + 1) % worker_num;
    }
    return Status::OK();
  };

  ThreadGroup tg;

  tg.AddTask(send_procedure);
  tg.AddTask(recv_procedure);
  std::vector<Status> results = tg.TakeResults();

  for (auto& st : results) {
    if (!st.ok()) {
      return st;
    }
  }

  auto& batches = divided_records[comm_spec.fid()];
  // remove empty batches
  batches.erase(std::remove_if(batches.begin(), batches.end(),
                               [](std::shared_ptr<arrow::RecordBatch>& e) {
                                 return e->num_rows() == 0;
                               }),
                batches.end());

  if (batches.empty()) {
    return EmptyTableBuilder::Build(table_in->schema(), table_out);
  }
  std::shared_ptr<arrow::Table> tmp_table;
  RETURN_ON_ERROR(RecordBatchesToTable(batches, &tmp_table));
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      table_out, tmp_table->CombineChunks(arrow::default_memory_pool()));
  return Status::OK();
}

template <typename VID_TYPE>
Status ShufflePropertyEdgeTable(const grape::CommSpec& comm_spec,
                                IdParser<VID_TYPE>& id_parser, int src_col_id,
                                int dst_col_id,
                                const std::shared_ptr<arrow::Table>& table_in,
                                std::shared_ptr<arrow::Table>& table_out) {
  fid_t fnum = comm_spec.fnum();
  std::vector<std::vector<std::shared_ptr<arrow::RecordBatch>>>
      divided_record_batches(fnum);

  TableAppender appender(table_in->schema());
  std::vector<std::unique_ptr<arrow::RecordBatchBuilder>>
      divided_table_builders;

  for (fid_t i = 0; i < fnum; ++i) {
    std::unique_ptr<arrow::RecordBatchBuilder> builder;
    RETURN_ON_ARROW_ERROR(arrow::RecordBatchBuilder::Make(
        table_in->schema(), arrow::default_memory_pool(), 4096, &builder));
    divided_table_builders.emplace_back(std::move(builder));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> tmp_batch_vec;
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;

  RETURN_ON_ERROR(TableToRecordBatches(table_in, &record_batches));
  for (const auto& rb : record_batches) {
    size_t row_num = rb->num_rows();
    auto src_col = std::dynamic_pointer_cast<
        typename ConvertToArrowType<VID_TYPE>::ArrayType>(
        rb->column(src_col_id));
    auto dst_col = std::dynamic_pointer_cast<
        typename ConvertToArrowType<VID_TYPE>::ArrayType>(
        rb->column(dst_col_id));

    for (size_t i = 0; i < row_num; ++i) {
      VID_TYPE src_gid = src_col->Value(i);
      VID_TYPE dst_gid = dst_col->Value(i);
      fid_t src_fid = id_parser.GetFid(src_gid);
      fid_t dst_fid = id_parser.GetFid(dst_gid);
      RETURN_ON_ERROR(appender.Apply(divided_table_builders[src_fid], rb, i,
                                     tmp_batch_vec));
      if (!tmp_batch_vec.empty()) {
        divided_record_batches[src_fid].emplace_back(
            std::move(tmp_batch_vec[0]));
        tmp_batch_vec.clear();
      }
      if (src_fid != dst_fid) {
        RETURN_ON_ERROR(appender.Apply(divided_table_builders[dst_fid], rb, i,
                                       tmp_batch_vec));
        if (!tmp_batch_vec.empty()) {
          divided_record_batches[dst_fid].emplace_back(
              std::move(tmp_batch_vec[0]));
          tmp_batch_vec.clear();
        }
      }
    }
  }

  for (fid_t fid = 0; fid < fnum; ++fid) {
    RETURN_ON_ERROR(appender.Flush(divided_table_builders[fid], tmp_batch_vec));
    if (!tmp_batch_vec.empty()) {
      divided_record_batches[fid].emplace_back(std::move(tmp_batch_vec[0]));
      tmp_batch_vec.clear();
    }
  }

  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();

  auto send_procedure = [&]() {
    int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
    while (dst_worker_id != worker_id) {
      fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
      std::shared_ptr<arrow::Buffer> buffer;
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches =
          std::move(divided_record_batches[dst_fid]);
      if (batches.empty()) {
        std::shared_ptr<arrow::io::BufferOutputStream> out_stream;
        RETURN_ON_ARROW_ERROR_AND_ASSIGN(
            out_stream, arrow::io::BufferOutputStream::Create(1024));
        RETURN_ON_ARROW_ERROR_AND_ASSIGN(buffer, out_stream->Finish());
      } else {
        RETURN_ON_ERROR(SerializeRecordBatches(batches, &buffer));
      }
      SendArrowBuffer(buffer, dst_worker_id, comm_spec.comm());
      dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
    }

    return Status::OK();
  };

  auto recv_procedure = [&]() {
    int src_worker_id = (worker_id + 1) % worker_num;

    while (src_worker_id != worker_id) {
      std::shared_ptr<arrow::Buffer> buffer;
      RETURN_ON_ERROR(RecvArrowBuffer(buffer, src_worker_id, comm_spec.comm()));

      if (buffer->size() > 0) {
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        RETURN_ON_ERROR(DeserializeRecordBatches(buffer, &batches));
        for (const auto& batch : batches) {
          divided_record_batches[comm_spec.fid()].push_back(batch);
        }
      }
      src_worker_id = (src_worker_id + 1) % worker_num;
    }
    return Status::OK();
  };

  ThreadGroup tg;

  tg.AddTask(send_procedure);
  tg.AddTask(recv_procedure);
  auto results = tg.TakeResults();

  for (auto& res : results) {
    RETURN_ON_ERROR(res);
  }

  auto batches = divided_record_batches[comm_spec.fid()];
  batches.erase(std::remove_if(batches.begin(), batches.end(),
                               [](std::shared_ptr<arrow::RecordBatch>& e) {
                                 return e->num_rows() == 0;
                               }),
                batches.end());

  // N.B.: we need an empty table for non-existing labels.
  if (batches.empty()) {
    RETURN_ON_ERROR(EmptyTableBuilder::Build(table_in->schema(), table_out));
  } else {
    std::shared_ptr<arrow::Table> tmp_table;
    RETURN_ON_ERROR(RecordBatchesToTable(batches, &tmp_table));
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        table_out, tmp_table->CombineChunks(arrow::default_memory_pool()));
  }
  return Status::OK();
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TABLE_SHUFFLER_H_
