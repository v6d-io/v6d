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

#ifndef MODULES_GRAPH_UTILS_TABLE_SHUFFLER_BETA_H_
#define MODULES_GRAPH_UTILS_TABLE_SHUFFLER_BETA_H_

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "grape/communication/sync_comm.h"
#include "grape/utils/concurrent_queue.h"
#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"

namespace vineyard {

namespace beta {

inline void SendArrowBuffer(const std::shared_ptr<arrow::Buffer>& buffer,
                            int dst_worker_id, MPI_Comm comm) {
  int64_t size = buffer->size();
  MPI_Send(&size, 1, MPI_INT64_T, dst_worker_id, 0, comm);
  if (size != 0) {
    grape::sync_comm::send_buffer<uint8_t>(buffer->data(), size, dst_worker_id,
                                           0, comm);
  }
}

inline void RecvArrowBuffer(std::shared_ptr<arrow::Buffer>& buffer,
                            int src_worker_id, MPI_Comm comm) {
  int64_t size;
  MPI_Recv(&size, 1, MPI_INT64_T, src_worker_id, 0, comm, MPI_STATUS_IGNORE);
  ARROW_CHECK_OK_AND_ASSIGN(
      buffer, arrow::AllocateBuffer(size, arrow::default_memory_pool()));
  if (size != 0) {
    grape::sync_comm::recv_buffer<uint8_t>(buffer->mutable_data(), size,
                                           src_worker_id, 0, comm);
  }
}

inline boost::leaf::result<void> SchemaConsistent(
    const arrow::Schema& schema, const grape::CommSpec& comm_spec) {
  std::shared_ptr<arrow::Buffer> buffer;
  arrow::Status serialized_status;
#if defined(ARROW_VERSION) && ARROW_VERSION < 2000000
  arrow::ipc::DictionaryMemo out_memo;
  auto ret = arrow::ipc::SerializeSchema(schema, &out_memo,
                                         arrow::default_memory_pool());
  serialized_status = ret.status();
  if (ret.ok()) {
    buffer = std::move(ret).ValueOrDie();
  }
#else
  auto ret = arrow::ipc::SerializeSchema(schema, arrow::default_memory_pool());
  serialized_status = ret.status();
  if (ret.ok()) {
    buffer = std::move(ret).ValueOrDie();
  }
#endif
  if (!serialized_status.ok()) {
    int flag = 1;
    int sum;
    MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec.comm());
    RETURN_GS_ERROR(ErrorCode::kArrowError, "Serializing schema failed.");
  } else {
    int flag = 0;
    int sum;
    MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec.comm());
    if (sum != 0) {
      RETURN_GS_ERROR(ErrorCode::kArrowError, "Serializing schema failed.");
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
  std::thread recv_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int src_worker_id = (worker_id + worker_num - i) % worker_num;
      std::shared_ptr<arrow::Buffer> got_buffer;
      RecvArrowBuffer(got_buffer, src_worker_id, comm_spec.comm());
      arrow::ipc::DictionaryMemo in_memo;
      arrow::io::BufferReader reader(got_buffer);
      std::shared_ptr<arrow::Schema> got_schema;
      ARROW_CHECK_OK_AND_ASSIGN(got_schema,
                                arrow::ipc::ReadSchema(&reader, &in_memo));
      consistent &= (got_schema->Equals(schema));
    }
  });

  send_thread.join();
  recv_thread.join();

  MPI_Barrier(comm_spec.comm());

  if (!consistent) {
    RETURN_GS_ERROR(ErrorCode::kInvalidOperationError,
                    "Schemas of edge tables are not consistent.");
  }

  return {};
}

template <typename T>
inline void serialize_selected_typed_items(
    grape::InArchive& arc, std::shared_ptr<arrow::Array> array) {
  auto ptr =
      std::dynamic_pointer_cast<typename ConvertToArrowType<T>::ArrayType>(
          array)
          ->raw_values();
  for (int64_t x = 0; x < array->length(); ++x) {
    arc << ptr[x];
  }
}

template <typename T>
inline void serialize_selected_typed_items(grape::InArchive& arc,
                                           std::shared_ptr<arrow::Array> array,
                                           const std::vector<int64_t>& offset) {
  auto ptr =
      std::dynamic_pointer_cast<typename ConvertToArrowType<T>::ArrayType>(
          array)
          ->raw_values();
  for (auto x : offset) {
    arc << ptr[x];
  }
}

void serialize_string_items(grape::InArchive& arc,
                            std::shared_ptr<arrow::Array> array,
                            const std::vector<int64_t>& offset);

void serialize_null_items(grape::InArchive& arc,
                          std::shared_ptr<arrow::Array> array,
                          const std::vector<int64_t>& offset);

template <typename T>
void serialize_list_items(grape::InArchive& arc,
                          std::shared_ptr<arrow::Array> array,
                          const std::vector<int64_t>& offset) {
  auto* ptr = std::dynamic_pointer_cast<arrow::LargeListArray>(array).get();
  for (auto x : offset) {
    arrow::LargeListArray::offset_type length = ptr->value_length(x);
    arc << length;
    auto value = ptr->value_slice(x);
    serialize_selected_typed_items<T>(arc, value);
  }
}

void SerializeSelectedItems(grape::InArchive& arc,
                            std::shared_ptr<arrow::Array> array,
                            const std::vector<int64_t>& offset);

void SerializeSelectedRows(grape::InArchive& arc,
                           std::shared_ptr<arrow::RecordBatch> record_batch,
                           const std::vector<int64_t>& offset);
template <typename T>
inline void deserialize_selected_typed_items(grape::OutArchive& arc,
                                             int64_t num,
                                             arrow::ArrayBuilder* builder) {
  auto casted_builder =
      dynamic_cast<typename ConvertToArrowType<T>::BuilderType*>(builder);
  T val;
  for (int64_t i = 0; i != num; ++i) {
    arc >> val;
    CHECK_ARROW_ERROR(casted_builder->Append(val));
  }
}

inline void deserialize_string_items(grape::OutArchive& arc, int64_t num,
                                     arrow::ArrayBuilder* builder) {
  auto casted_builder = dynamic_cast<arrow::LargeStringBuilder*>(builder);
  arrow_string_view val;
  for (int64_t i = 0; i != num; ++i) {
    arc >> val;
    CHECK_ARROW_ERROR(casted_builder->Append(val));
  }
}

inline void deserialize_null_items(grape::OutArchive& arc, int64_t num,
                                   arrow::ArrayBuilder* builder) {
  auto casted_builder = dynamic_cast<arrow::NullBuilder*>(builder);
  CHECK_ARROW_ERROR(casted_builder->AppendNulls(num));
}

template <typename T>
inline void deserialize_list_items(grape::OutArchive& arc, int64_t num,
                                   arrow::ArrayBuilder* builder) {
  auto casted_builder = dynamic_cast<arrow::LargeListBuilder*>(builder);
  auto value_builder = casted_builder->value_builder();
  arrow::LargeListArray::offset_type length;
  for (int64_t i = 0; i != num; ++i) {
    arc >> length;
    deserialize_selected_typed_items<T>(arc, length, value_builder);
    CHECK_ARROW_ERROR(casted_builder->Append(true));
  }
}

void DeserializeSelectedItems(grape::OutArchive& arc, int64_t num,
                              arrow::ArrayBuilder* builder);

void DeserializeSelectedRows(grape::OutArchive& arc,
                             std::shared_ptr<arrow::Schema> schema,
                             std::shared_ptr<arrow::RecordBatch>& batch_out);

template <typename T>
inline void select_typed_items(std::shared_ptr<arrow::Array> array,
                               arrow::ArrayBuilder* builder) {
  auto ptr =
      std::dynamic_pointer_cast<typename ConvertToArrowType<T>::ArrayType>(
          array)
          ->raw_values();
  auto casted_builder =
      dynamic_cast<typename ConvertToArrowType<T>::BuilderType*>(builder);
  CHECK_ARROW_ERROR(casted_builder->AppendValues(ptr, array->length()));
}

template <typename T>
inline void select_typed_items(std::shared_ptr<arrow::Array> array,
                               const std::vector<int64_t>& offset,
                               arrow::ArrayBuilder* builder) {
  auto ptr =
      std::dynamic_pointer_cast<typename ConvertToArrowType<T>::ArrayType>(
          array)
          ->raw_values();
  auto casted_builder =
      dynamic_cast<typename ConvertToArrowType<T>::BuilderType*>(builder);
  for (auto x : offset) {
    CHECK_ARROW_ERROR(casted_builder->Append(ptr[x]));
  }
}

inline void select_string_items(std::shared_ptr<arrow::Array> array,
                                const std::vector<int64_t>& offset,
                                arrow::ArrayBuilder* builder) {
  auto* ptr = std::dynamic_pointer_cast<arrow::LargeStringArray>(array).get();
  auto casted_builder = dynamic_cast<arrow::LargeStringBuilder*>(builder);
  for (auto x : offset) {
    CHECK_ARROW_ERROR(casted_builder->Append(ptr->GetView(x)));
  }
}

inline void select_null_items(std::shared_ptr<arrow::Array> array,
                              const std::vector<int64_t>& offset,
                              arrow::ArrayBuilder* builder) {
  arrow::NullBuilder* casted_builder =
      dynamic_cast<arrow::NullBuilder*>(builder);
  CHECK_ARROW_ERROR(casted_builder->AppendNulls(offset.size()));
}

template <typename T>
inline void select_list_items(std::shared_ptr<arrow::Array> array,
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

inline void SelectItems(std::shared_ptr<arrow::Array> array,
                        const std::vector<int64_t> offset,
                        arrow::ArrayBuilder* builder) {
  if (array->type()->Equals(arrow::float64())) {
    select_typed_items<double>(array, offset, builder);
  } else if (array->type()->Equals(arrow::float32())) {
    select_typed_items<float>(array, offset, builder);
  } else if (array->type()->Equals(arrow::int64())) {
    select_typed_items<int64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::int32())) {
    select_typed_items<int32_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::uint64())) {
    select_typed_items<uint64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::uint32())) {
    select_typed_items<uint32_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_utf8())) {
    select_string_items(array, offset, builder);
  } else if (array->type()->Equals(arrow::null())) {
    select_null_items(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::float64()))) {
    select_list_items<double>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::float32()))) {
    select_list_items<float>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::int64()))) {
    select_list_items<int64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::int32()))) {
    select_list_items<int32_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint64()))) {
    select_list_items<uint64_t>(array, offset, builder);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint32()))) {
    select_list_items<uint32_t>(array, offset, builder);
  } else {
    LOG(FATAL) << "Unsupported data type - " << builder->type()->ToString();
  }
}

inline void SelectRows(std::shared_ptr<arrow::RecordBatch> record_batch_in,
                       const std::vector<int64_t>& offset,
                       std::shared_ptr<arrow::RecordBatch>& record_batch_out) {
  int64_t row_num = offset.size();
  std::unique_ptr<arrow::RecordBatchBuilder> builder;
  ARROW_CHECK_OK(arrow::RecordBatchBuilder::Make(record_batch_in->schema(),
                                                 arrow::default_memory_pool(),
                                                 row_num, &builder));
  int col_num = builder->num_fields();
  for (int col_id = 0; col_id != col_num; ++col_id) {
    SelectItems(record_batch_in->column(col_id), offset,
                builder->GetField(col_id));
  }
  ARROW_CHECK_OK(builder->Flush(&record_batch_out));
}

void ShuffleTableByOffsetLists(
    std::shared_ptr<arrow::Schema> schema,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_out,
    const std::vector<std::vector<std::vector<int64_t>>>& offset_lists,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_in,
    const grape::CommSpec& comm_spec);

template <typename VID_TYPE>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyEdgeTable(
    const grape::CommSpec& comm_spec, IdParser<VID_TYPE>& id_parser,
    int src_col_id, int dst_col_id, std::shared_ptr<arrow::Table>& table_in) {
  BOOST_LEAF_CHECK(SchemaConsistent(*table_in->schema(), comm_spec));

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
  VY_OK_OR_RAISE(TableToRecordBatches(table_in, &record_batches));

  size_t record_batch_num = record_batches.size();
  // record_batch_num, fragment_num, row_ids
  std::vector<std::vector<std::vector<int64_t>>> offset_lists(record_batch_num);

  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
      comm_spec.local_num();
  std::vector<std::thread> scan_threads(thread_num);
  std::atomic<size_t> cur(0);

  for (int i = 0; i < thread_num; ++i) {
    scan_threads[i] = std::thread([&]() {
      while (true) {
        size_t got = cur.fetch_add(1);
        if (got >= record_batch_num) {
          break;
        }

        auto& offset_list = offset_lists[got];
        offset_list.resize(comm_spec.fnum());
        auto cur_batch = record_batches[got];
        int64_t row_num = cur_batch->num_rows();

        const VID_TYPE* src_col =
            std::dynamic_pointer_cast<
                typename ConvertToArrowType<VID_TYPE>::ArrayType>(
                cur_batch->column(src_col_id))
                ->raw_values();
        const VID_TYPE* dst_col =
            std::dynamic_pointer_cast<
                typename ConvertToArrowType<VID_TYPE>::ArrayType>(
                cur_batch->column(dst_col_id))
                ->raw_values();

        for (int64_t row_id = 0; row_id < row_num; ++row_id) {
          VID_TYPE src_gid = src_col[row_id];
          VID_TYPE dst_gid = dst_col[row_id];

          grape::fid_t src_fid = id_parser.GetFid(src_gid);
          grape::fid_t dst_fid = id_parser.GetFid(dst_gid);

          offset_list[src_fid].push_back(row_id);
          if (src_fid != dst_fid) {
            offset_list[dst_fid].push_back(row_id);
          }
        }
      }
    });
  }
  for (auto& thrd : scan_threads) {
    thrd.join();
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_in;

  ShuffleTableByOffsetLists(table_in->schema(), record_batches, offset_lists,
                            batches_in, comm_spec);

  batches_in.erase(std::remove_if(batches_in.begin(), batches_in.end(),
                                  [](std::shared_ptr<arrow::RecordBatch>& e) {
                                    return e->num_rows() == 0;
                                  }),
                   batches_in.end());

  // N.B.: we need an empty table for non-existing labels.
  std::shared_ptr<arrow::Table> table_out;
  if (batches_in.empty()) {
    VY_OK_OR_RAISE(EmptyTableBuilder::Build(table_in->schema(), table_out));
  } else {
    std::shared_ptr<arrow::Table> tmp_table;
    VY_OK_OR_RAISE(RecordBatchesToTable(batches_in, &tmp_table));
    ARROW_OK_ASSIGN_OR_RAISE(
        table_out, tmp_table->CombineChunks(arrow::default_memory_pool()));
  }
  return table_out;
}

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyVertexTable(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    std::shared_ptr<arrow::Table>& table_in) {
  using oid_t = typename PARTITIONER_T::oid_t;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_type = typename ConvertToArrowType<oid_t>::ArrayType;

  BOOST_LEAF_CHECK(SchemaConsistent(*table_in->schema(), comm_spec));

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
  VY_OK_OR_RAISE(TableToRecordBatches(table_in, &record_batches));

  size_t record_batch_num = record_batches.size();
  std::vector<std::vector<std::vector<int64_t>>> offset_lists(record_batch_num);

  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
      comm_spec.local_num();
  std::vector<std::thread> scan_threads(thread_num);
  std::atomic<size_t> cur(0);

  for (int i = 0; i < thread_num; ++i) {
    scan_threads[i] = std::thread([&]() {
      while (true) {
        size_t got = cur.fetch_add(1);
        if (got >= record_batch_num) {
          break;
        }

        auto& offset_list = offset_lists[got];
        offset_list.resize(comm_spec.fnum());
        auto cur_batch = record_batches[got];
        int64_t row_num = cur_batch->num_rows();

        std::shared_ptr<oid_array_type> id_col =
            std::dynamic_pointer_cast<oid_array_type>(cur_batch->column(0));

        for (int64_t row_id = 0; row_id < row_num; ++row_id) {
          internal_oid_t rs = id_col->GetView(row_id);
          grape::fid_t fid = partitioner.GetPartitionId(oid_t(rs));
          offset_list[fid].push_back(row_id);
        }
      }
    });
  }
  for (auto& thrd : scan_threads) {
    thrd.join();
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_in;

  ShuffleTableByOffsetLists(table_in->schema(), record_batches, offset_lists,
                            batches_in, comm_spec);

  batches_in.erase(std::remove_if(batches_in.begin(), batches_in.end(),
                                  [](std::shared_ptr<arrow::RecordBatch>& e) {
                                    return e->num_rows() == 0;
                                  }),
                   batches_in.end());

  std::shared_ptr<arrow::Table> table_out;
  if (batches_in.empty()) {
    VY_OK_OR_RAISE(EmptyTableBuilder::Build(table_in->schema(), table_out));
  } else {
    std::shared_ptr<arrow::Table> tmp_table;
    VY_OK_OR_RAISE(RecordBatchesToTable(batches_in, &tmp_table));
    ARROW_OK_ASSIGN_OR_RAISE(
        table_out, tmp_table->CombineChunks(arrow::default_memory_pool()));
  }
  return table_out;
}

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(const grape::CommSpec& comm_spec,
                                    const PARTITIONER_T& partitioner,
                                    int src_col_id, int dst_col_id,
                                    std::shared_ptr<arrow::Table>& table_in) {
  using oid_t = typename PARTITIONER_T::oid_t;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_type = typename ConvertToArrowType<oid_t>::ArrayType;

  BOOST_LEAF_CHECK(SchemaConsistent(*table_in->schema(), comm_spec));

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
  VY_OK_OR_RAISE(TableToRecordBatches(table_in, &record_batches));

  size_t record_batch_num = record_batches.size();
  // record_batch_num, fragment_num, row_ids
  std::vector<std::vector<std::vector<int64_t>>> offset_lists(record_batch_num);

  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
      comm_spec.local_num();
  std::vector<std::thread> scan_threads(thread_num);
  std::atomic<size_t> cur(0);

  for (int i = 0; i < thread_num; ++i) {
    scan_threads[i] = std::thread([&]() {
      while (true) {
        size_t got = cur.fetch_add(1);
        if (got >= record_batch_num) {
          break;
        }

        auto& offset_list = offset_lists[got];
        offset_list.resize(comm_spec.fnum());
        auto cur_batch = record_batches[got];
        int64_t row_num = cur_batch->num_rows();

        auto src_col = std::dynamic_pointer_cast<oid_array_type>(
            cur_batch->column(src_col_id));
        auto dst_col = std::dynamic_pointer_cast<oid_array_type>(
            cur_batch->column(dst_col_id));

        for (int64_t row_id = 0; row_id < row_num; ++row_id) {
          internal_oid_t src_oid = src_col->GetView(row_id);
          internal_oid_t dst_oid = dst_col->GetView(row_id);

          grape::fid_t src_fid = partitioner.GetPartitionId(oid_t(src_oid));
          grape::fid_t dst_fid = partitioner.GetPartitionId(oid_t(dst_oid));

          offset_list[src_fid].push_back(row_id);
          if (src_fid != dst_fid) {
            offset_list[dst_fid].push_back(row_id);
          }
        }
      }
    });
  }
  for (auto& thrd : scan_threads) {
    thrd.join();
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_in;

  ShuffleTableByOffsetLists(table_in->schema(), record_batches, offset_lists,
                            batches_in, comm_spec);

  batches_in.erase(std::remove_if(batches_in.begin(), batches_in.end(),
                                  [](std::shared_ptr<arrow::RecordBatch>& e) {
                                    return e->num_rows() == 0;
                                  }),
                   batches_in.end());

  // N.B.: we need an empty table for non-existing labels.
  std::shared_ptr<arrow::Table> table_out;
  if (batches_in.empty()) {
    VY_OK_OR_RAISE(EmptyTableBuilder::Build(table_in->schema(), table_out));
  } else {
    std::shared_ptr<arrow::Table> tmp_table;
    VY_OK_OR_RAISE(RecordBatchesToTable(batches_in, &tmp_table));
    ARROW_OK_ASSIGN_OR_RAISE(
        table_out, tmp_table->CombineChunks(arrow::default_memory_pool()));
  }
  return table_out;
}

}  // namespace beta

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TABLE_SHUFFLER_BETA_H_
