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

#ifndef MODULES_GRAPH_UTILS_TABLE_SHUFFLER_H_
#define MODULES_GRAPH_UTILS_TABLE_SHUFFLER_H_

#include <mpi.h>

#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/leaf/all.hpp>

#include "arrow/buffer.h"
#include "arrow/table.h"
#include "arrow/util/config.h"

#include "grape/communication/sync_comm.h"
#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"
#include "graph/utils/error.h"

namespace vineyard {

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

inline void SendArrowBuffer(const std::shared_ptr<arrow::Buffer>& buffer,
                            int dst_worker_id, MPI_Comm comm, int tag = 0) {
  int64_t size = buffer->size();
  MPI_Send(&size, 1, MPI_INT64_T, dst_worker_id, tag, comm);
  if (size != 0) {
    send_buffer<uint8_t>(buffer->data(), size, dst_worker_id, comm, tag);
  }
}

inline boost::leaf::result<std::shared_ptr<arrow::Buffer>> RecvArrowBuffer(
    int src_worker_id, MPI_Comm comm, int tag = 0) {
  int64_t size;
  MPI_Recv(&size, 1, MPI_INT64_T, src_worker_id, tag, comm, MPI_STATUS_IGNORE);

  std::shared_ptr<arrow::Buffer> buffer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  ARROW_OK_OR_RAISE(
      arrow::AllocateBuffer(arrow::default_memory_pool(), size, &buffer));
#else
  ARROW_OK_ASSIGN_OR_RAISE(
      buffer, arrow::AllocateBuffer(size, arrow::default_memory_pool()));
#endif

  if (size != 0) {
    recv_buffer<uint8_t>(buffer->mutable_data(), size, src_worker_id, comm,
                         tag);
  }
  return buffer;
}

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyVertexTable(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    std::shared_ptr<arrow::Table>& table_in) {
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
    ARROW_OK_OR_RAISE(arrow::RecordBatchBuilder::Make(
        table_in->schema(), arrow::default_memory_pool(), 4096, &builder));
    divided_table_builders.emplace_back(std::move(builder));
  }
  TableAppender appender(table_in->schema());
  arrow::TableBatchReader tbreader(*table_in);
  std::shared_ptr<arrow::RecordBatch> batch;

  while (true) {
    ARROW_OK_OR_RAISE(tbreader.ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    std::shared_ptr<oid_array_type> id_col =
        std::dynamic_pointer_cast<oid_array_type>(batch->column(0));
    size_t row_num = batch->num_rows();
    for (size_t i = 0; i < row_num; ++i) {
      internal_oid_t rs = id_col->GetView(i);
      fid_t fid = partitioner.GetPartitionId(oid_t(rs));
      ARROW_OK_OR_RAISE(appender.Apply(divided_table_builders[fid], batch, i,
                                       divided_records[fid]));
    }
  }

  for (fid_t i = 0; i < fnum; ++i) {
    std::unique_ptr<arrow::RecordBatchBuilder> builder =
        std::move(divided_table_builders[i]);
    ARROW_OK_OR_RAISE(appender.Flush(builder, divided_records[i]));
  }
  divided_table_builders.clear();

  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();
  std::vector<std::string> error_msgs;
  auto error_handlers = std::make_tuple([&](const GSError& e) {
    auto msg = "Shuffle vertex table error: " + e.error_msg;
    LOG(ERROR) << msg;
    error_msgs.push_back(msg);
  });

  auto send_procedure = [&]() -> boost::leaf::result<void> {
    int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
    while (dst_worker_id != worker_id) {
      fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
      std::shared_ptr<arrow::Buffer> buffer;
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches =
          std::move(divided_records[dst_fid]);
      VY_OK_OR_RAISE(SerializeRecordBatches(batches, &buffer));
      SendArrowBuffer(buffer, dst_worker_id, comm_spec.comm());
      dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
    }

    return boost::leaf::result<void>();
  };

  auto recv_procedure = [&]() -> boost::leaf::result<void> {
    int src_worker_id = (worker_id + 1) % worker_num;
    while (src_worker_id != worker_id) {
      BOOST_LEAF_AUTO(buffer, RecvArrowBuffer(src_worker_id, comm_spec.comm()));
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      VY_OK_OR_RAISE(DeserializeRecordBatches(buffer, &batches));
      for (const auto& batch : batches) {
        if (batch->num_rows() > 0) {
          divided_records[comm_spec.fid()].push_back(batch);
        }
      }
      src_worker_id = (src_worker_id + 1) % worker_num;
    }
    return boost::leaf::result<void>();
  };

  std::vector<std::future<boost::leaf::result<void>>> fut;

  fut.push_back(std::async(std::launch::async, [&] {
    return boost::leaf::capture(
        boost::leaf::make_shared_context(error_handlers), send_procedure);
  }));
  fut.push_back(std::async(std::launch::async, [&] {
    return boost::leaf::capture(
        boost::leaf::make_shared_context(error_handlers), recv_procedure);
  }));

  for (auto& f : fut) {
    f.wait();
    boost::leaf::try_handle_some(
        [&]() -> boost::leaf::result<void> {
          auto res = f.get();
          if (!res) {
            return res.error();
          }
          return boost::leaf::result<void>();
        },
        error_handlers);
  }

  if (!error_msgs.empty()) {
    auto msgs = std::accumulate(
        error_msgs.begin(), error_msgs.end(), std::string(),
        [](const std::string& a, const std::string& b) -> std::string {
          return a + (!a.empty() ? "," : "") + b;
        });

    return boost::leaf::new_error(GSError(ErrorCode::kIOError, msgs));
  }

  auto& batches = divided_records[comm_spec.fid()];
  // remove empty batches
  batches.erase(std::remove_if(batches.begin(), batches.end(),
                               [](std::shared_ptr<arrow::RecordBatch>& e) {
                                 return e->num_rows() == 0;
                               }),
                batches.end());

  std::shared_ptr<arrow::Table> table_out;
  if (batches.empty()) {
    VY_OK_OR_RAISE(EmptyTableBuilder::Build(table_in->schema(), table_out));
  } else {
    std::shared_ptr<arrow::Table> tmp_table;
    VY_OK_OR_RAISE(RecordBatchesToTable(batches, &tmp_table));
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    ARROW_OK_OR_RAISE(
        tmp_table->CombineChunks(arrow::default_memory_pool(), &table_out));
#else
    ARROW_OK_ASSIGN_OR_RAISE(
        table_out, tmp_table->CombineChunks(arrow::default_memory_pool()));
#endif
  }
  return table_out;
}

template <typename T>
inline void send_numeric_array(
    std::shared_ptr<typename ConvertToArrowType<T>::ArrayType> array,
    int dst_worker_id, MPI_Comm comm, int tag = 0) {
  size_t len = array->length();
  MPI_Send(&len, sizeof(size_t), MPI_CHAR, dst_worker_id, tag, comm);
  send_buffer<T>(array->raw_values(), len, dst_worker_id, comm, tag);
}

inline void ArrayToRecordBatch(const std::shared_ptr<arrow::Array>& array,
                               std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::Field> field =
      std::make_shared<arrow::Field>("data", array->type());
  std::vector<std::shared_ptr<arrow::Field>> fields = {field};
  std::shared_ptr<arrow::Schema> schema =
      std::make_shared<arrow::Schema>(fields);
  int64_t num_rows = array->length();
  std::vector<std::shared_ptr<arrow::Array>> array_list = {array};
  *batch = arrow::RecordBatch::Make(schema, num_rows, array_list);
}

template <>
inline void send_numeric_array<std::string>(
    std::shared_ptr<arrow::StringArray> array, int dst_worker_id, MPI_Comm comm,
    int tag) {
  std::shared_ptr<arrow::RecordBatch> batch;
  ArrayToRecordBatch(std::dynamic_pointer_cast<arrow::Array>(array), &batch);
  std::shared_ptr<arrow::Buffer> buffer;
  VINEYARD_CHECK_OK(SerializeRecordBatches({batch}, &buffer));
  SendArrowBuffer(buffer, dst_worker_id, comm, tag);
}

template <typename T>
inline boost::leaf::result<void> recv_numeric_array(
    std::shared_ptr<typename ConvertToArrowType<T>::ArrayType>& array,
    int src_worker_id, MPI_Comm comm, int tag = 0) {
  size_t len;
  MPI_Recv(&len, sizeof(size_t), MPI_CHAR, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);
  typename ConvertToArrowType<T>::BuilderType builder;
  ARROW_OK_OR_RAISE(builder.Resize(len));
  recv_buffer<T>(&builder[0], len, src_worker_id, comm, tag);
  ARROW_OK_OR_RAISE(builder.Advance(len));
  ARROW_OK_OR_RAISE(builder.Finish(&array));
  return boost::leaf::result<void>();
}

template <>
inline boost::leaf::result<void> recv_numeric_array<std::string>(
    std::shared_ptr<arrow::StringArray>& array, int src_worker_id,
    MPI_Comm comm, int tag) {
  BOOST_LEAF_AUTO(buffer, RecvArrowBuffer(src_worker_id, comm, tag));
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  VY_OK_OR_RAISE(DeserializeRecordBatches(buffer, &batches));
  CHECK_EQ(batches.size(), 1);
  CHECK_EQ(batches[0]->num_columns(), 1);
  CHECK_EQ(batches[0]->column(0)->type(), arrow::utf8());
  array = std::dynamic_pointer_cast<arrow::StringArray>(batches[0]->column(0));
  return boost::leaf::result<void>();
}

template <typename T>
boost::leaf::result<
    std::vector<std::shared_ptr<typename ConvertToArrowType<T>::ArrayType>>>
FragmentAllGatherArray(
    const grape::CommSpec& comm_spec,
    std::shared_ptr<typename ConvertToArrowType<T>::ArrayType> data_in) {
  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();
  std::vector<std::shared_ptr<typename ConvertToArrowType<T>::ArrayType>>
      data_out(comm_spec.fnum());

  auto send_procedure = [&]() -> boost::leaf::result<void> {
    int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
    while (dst_worker_id != worker_id) {
      send_numeric_array<T>(data_in, dst_worker_id, comm_spec.comm());
      dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
    }
    return boost::leaf::result<void>();
  };

  auto recv_procedure = [&]() -> boost::leaf::result<void> {
    int src_worker_id = (worker_id + 1) % worker_num;
    while (src_worker_id != worker_id) {
      fid_t src_fid = comm_spec.WorkerToFrag(src_worker_id);
      recv_numeric_array<T>(data_out[src_fid], src_worker_id, comm_spec.comm());
      src_worker_id = (src_worker_id + 1) % worker_num;
    }
    data_out[comm_spec.fid()] = data_in;
    return boost::leaf::result<void>();
  };

  std::vector<std::future<boost::leaf::result<void>>> fut;
  std::vector<std::string> error_msgs;
  auto error_handlers = std::make_tuple([&](const GSError& e) {
    auto msg = "Gather array error: " + e.error_msg;
    LOG(ERROR) << msg;
    error_msgs.push_back(msg);
  });

  fut.push_back(std::async(std::launch::async, [&] {
    return boost::leaf::capture(
        boost::leaf::make_shared_context(error_handlers), send_procedure);
  }));
  fut.push_back(std::async(std::launch::async, [&] {
    return boost::leaf::capture(
        boost::leaf::make_shared_context(error_handlers), recv_procedure);
  }));

  for (auto& f : fut) {
    f.wait();
    boost::leaf::try_handle_some(
        [&]() -> boost::leaf::result<void> {
          auto res = f.get();
          if (!res) {
            return res.error();
          }
          return boost::leaf::result<void>();
        },
        error_handlers);
  }

  if (!error_msgs.empty()) {
    auto msgs = std::accumulate(
        error_msgs.begin(), error_msgs.end(), std::string(),
        [](const std::string& a, const std::string& b) -> std::string {
          return a + (!a.empty() ? "," : "") + b;
        });

    return boost::leaf::new_error(GSError(ErrorCode::kIOError, msgs));
  }

  return data_out;
}

template <typename VID_TYPE>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyEdgeTable(
    const grape::CommSpec& comm_spec, IdParser<VID_TYPE>& id_parser,
    int src_col_id, int dst_col_id, std::shared_ptr<arrow::Table>& table_in) {
  fid_t fnum = comm_spec.fnum();
  std::vector<std::vector<std::shared_ptr<arrow::RecordBatch>>>
      divided_record_batches(fnum);

  TableAppender appender(table_in->schema());
  std::vector<std::unique_ptr<arrow::RecordBatchBuilder>>
      divided_table_builders;

  for (fid_t i = 0; i < fnum; ++i) {
    std::unique_ptr<arrow::RecordBatchBuilder> builder;
    ARROW_OK_OR_RAISE(arrow::RecordBatchBuilder::Make(
        table_in->schema(), arrow::default_memory_pool(), 4096, &builder));
    divided_table_builders.emplace_back(std::move(builder));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> tmp_batch_vec;
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;

  VY_OK_OR_RAISE(TableToRecordBatches(table_in, &record_batches));
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
      ARROW_OK_OR_RAISE(appender.Apply(divided_table_builders[src_fid], rb, i,
                                       tmp_batch_vec));
      if (!tmp_batch_vec.empty()) {
        divided_record_batches[src_fid].emplace_back(
            std::move(tmp_batch_vec[0]));
        tmp_batch_vec.clear();
      }
      if (src_fid != dst_fid) {
        ARROW_OK_OR_RAISE(appender.Apply(divided_table_builders[dst_fid], rb, i,
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
    ARROW_OK_OR_RAISE(
        appender.Flush(divided_table_builders[fid], tmp_batch_vec));
    if (!tmp_batch_vec.empty()) {
      divided_record_batches[fid].emplace_back(std::move(tmp_batch_vec[0]));
      tmp_batch_vec.clear();
    }
  }

  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();
  std::vector<std::string> error_msgs;
  auto error_handlers = std::make_tuple([&](const GSError& e) {
    auto msg = "Shuffle edge table error: " + e.error_msg;
    LOG(ERROR) << msg;
    error_msgs.push_back(msg);
  });

  auto send_procedure = [&]() -> boost::leaf::result<void> {
    int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
    while (dst_worker_id != worker_id) {
      fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
      std::shared_ptr<arrow::Buffer> buffer;
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches =
          std::move(divided_record_batches[dst_fid]);
      if (batches.empty()) {
        std::shared_ptr<arrow::io::BufferOutputStream> out_stream;
        ARROW_OK_ASSIGN_OR_RAISE(out_stream,
                                 arrow::io::BufferOutputStream::Create(1024));
        ARROW_OK_ASSIGN_OR_RAISE(buffer, out_stream->Finish());
      } else {
        VY_OK_OR_RAISE(SerializeRecordBatches(batches, &buffer));
      }
      SendArrowBuffer(buffer, dst_worker_id, comm_spec.comm());
      dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
    }

    return boost::leaf::result<void>();
  };

  auto recv_procedure = [&]() -> boost::leaf::result<void> {
    int src_worker_id = (worker_id + 1) % worker_num;

    while (src_worker_id != worker_id) {
      BOOST_LEAF_AUTO(buffer, RecvArrowBuffer(src_worker_id, comm_spec.comm()));

      if (buffer->size() > 0) {
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        VY_OK_OR_RAISE(DeserializeRecordBatches(buffer, &batches));
        for (const auto& batch : batches) {
          divided_record_batches[comm_spec.fid()].push_back(batch);
        }
      }
      src_worker_id = (src_worker_id + 1) % worker_num;
    }
    return boost::leaf::result<void>();
  };

  std::vector<std::future<boost::leaf::result<void>>> fut;
  fut.push_back(std::async(std::launch::async, [&] {
    return boost::leaf::capture(
        boost::leaf::make_shared_context(error_handlers), send_procedure);
  }));
  fut.push_back(std::async(std::launch::async, [&] {
    return boost::leaf::capture(
        boost::leaf::make_shared_context(error_handlers), recv_procedure);
  }));

  for (auto& f : fut) {
    f.wait();
    boost::leaf::try_handle_some(
        [&]() -> boost::leaf::result<void> {
          auto res = f.get();
          if (!res) {
            return res.error();
          }
          return boost::leaf::result<void>();
        },
        error_handlers);
  }

  if (!error_msgs.empty()) {
    auto msgs = std::accumulate(
        error_msgs.begin(), error_msgs.end(), std::string(),
        [](const std::string& a, const std::string& b) -> std::string {
          return a + (!a.empty() ? "," : "") + b;
        });

    return boost::leaf::new_error(GSError(ErrorCode::kIOError, msgs));
  }

  auto batches = divided_record_batches[comm_spec.fid()];
  batches.erase(std::remove_if(batches.begin(), batches.end(),
                               [](std::shared_ptr<arrow::RecordBatch>& e) {
                                 return e->num_rows() == 0;
                               }),
                batches.end());

  // N.B.: we need an empty table for non-existing labels.
  std::shared_ptr<arrow::Table> table_out;
  if (batches.empty()) {
    VY_OK_OR_RAISE(EmptyTableBuilder::Build(table_in->schema(), table_out));
  } else {
    std::shared_ptr<arrow::Table> tmp_table;
    VY_OK_OR_RAISE(RecordBatchesToTable(batches, &tmp_table));
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    ARROW_OK_OR_RAISE(
        tmp_table->CombineChunks(arrow::default_memory_pool(), &table_out));
#else
    ARROW_OK_ASSIGN_OR_RAISE(
        table_out, tmp_table->CombineChunks(arrow::default_memory_pool()));
#endif
  }
  return table_out;
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TABLE_SHUFFLER_H_
