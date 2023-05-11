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

#ifndef MODULES_GRAPH_UTILS_TABLE_SHUFFLER_IMPL_H_
#define MODULES_GRAPH_UTILS_TABLE_SHUFFLER_IMPL_H_

#include "graph/utils/table_shuffler.h"

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
#include "grape/utils/concurrent_queue.h"
#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "common/util/status.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/thread_group.h"

namespace vineyard {

template <typename ArrayType>
void SendArrowArray(const std::shared_ptr<ArrayType>& array, int dst_worker_id,
                    MPI_Comm comm, int tag) {
  std::shared_ptr<arrow::ArrayData> data =
      array == nullptr ? nullptr : array->data();
  detail::send_array_data(data, true, dst_worker_id, comm, tag);
}

template <typename ArrayType>
void RecvArrowArray(std::shared_ptr<ArrayType>& array, int src_worker_id,
                    MPI_Comm comm, int tag) {
  std::shared_ptr<arrow::ArrayData> data;
  detail::recv_array_data(data, nullptr, src_worker_id, comm, tag);
  array = std::dynamic_pointer_cast<ArrayType>(arrow::MakeArray(data));
}

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send) {
  using oid_t = typename PARTITIONER_T::oid_t;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_type = ArrowArrayType<oid_t>;

  VY_OK_OR_RAISE(CheckSchemaConsistency(*table_send->schema(), comm_spec));

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
  VY_OK_OR_RAISE(TableToRecordBatches(table_send, &record_batches));

  size_t record_batch_num = record_batches.size();
  // record_batch_num, fragment_num, row_ids
  std::vector<std::vector<std::vector<int64_t>>> offset_lists(record_batch_num);

  auto fn = [&](const size_t batch_index) -> Status {
    auto& offset_list = offset_lists[batch_index];
    offset_list.resize(comm_spec.fnum());
    auto current_batch = record_batches[batch_index];
    int64_t row_num = current_batch->num_rows();

    auto src_col = std::dynamic_pointer_cast<oid_array_type>(
        current_batch->column(src_col_id));
    auto dst_col = std::dynamic_pointer_cast<oid_array_type>(
        current_batch->column(dst_col_id));

    for (int64_t row_id = 0; row_id < row_num; ++row_id) {
      internal_oid_t src_oid = src_col->GetView(row_id);
      internal_oid_t dst_oid = dst_col->GetView(row_id);

      grape::fid_t src_fid = partitioner.GetPartitionId(src_oid);
      grape::fid_t dst_fid = partitioner.GetPartitionId(dst_oid);

      offset_list[src_fid].push_back(row_id);
      if (src_fid != dst_fid) {
        offset_list[dst_fid].push_back(row_id);
      }
    }
    return Status::OK();
  };

  ThreadGroup tg(comm_spec);
  for (size_t batch_index = 0; batch_index < record_batch_num; ++batch_index) {
    tg.AddTask(fn, batch_index);
  }
  Status status;
  for (const Status& s : tg.TakeResults()) {
    status += s;
  }
  VY_OK_OR_RAISE(status);

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_recv;
  BOOST_LEAF_CHECK(ShuffleTableByOffsetLists(comm_spec, table_send->schema(),
                                             record_batches, offset_lists,
                                             batches_recv));

  batches_recv.erase(std::remove_if(batches_recv.begin(), batches_recv.end(),
                                    [](std::shared_ptr<arrow::RecordBatch>& e) {
                                      return e == nullptr || e->num_rows() == 0;
                                    }),
                     batches_recv.end());

  // N.B.: we need an empty table for labels that doesn't have effective data.
  std::shared_ptr<arrow::Table> table_out;
  VY_OK_OR_RAISE(
      RecordBatchesToTable(table_send->schema(), batches_recv, &table_out));
  // ARROW_OK_ASSIGN_OR_RAISE(table_out,
  // table_out->CombineChunks(arrow::default_memory_pool()));
  return table_out;
}

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send) {
  using oid_t = typename PARTITIONER_T::oid_t;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_type = ArrowArrayType<oid_t>;

  VY_OK_OR_RAISE(CheckSchemaConsistency(*table_send->schema(), comm_spec));

  fid_t fnum = comm_spec.fnum();

  auto offsetfn = [fnum, &partitioner, src_col_id, dst_col_id](
                      const std::shared_ptr<arrow::RecordBatch> batch,
                      std::vector<std::vector<int64_t>>& offset_list) -> void {
    // per thread
    offset_list.resize(fnum);
    for (auto& offsets : offset_list) {
      offsets.clear();
    }

    int64_t row_num = batch ? batch->num_rows() : 0;
    auto src_col = batch ? std::dynamic_pointer_cast<oid_array_type>(
                               batch->column(src_col_id))
                         : nullptr;
    auto dst_col = batch ? std::dynamic_pointer_cast<oid_array_type>(
                               batch->column(dst_col_id))
                         : nullptr;

    for (int64_t row_id = 0; row_id < row_num; ++row_id) {
      internal_oid_t src_oid = src_col->GetView(row_id);
      internal_oid_t dst_oid = dst_col->GetView(row_id);

      grape::fid_t src_fid = partitioner.GetPartitionId(src_oid);
      grape::fid_t dst_fid = partitioner.GetPartitionId(dst_oid);

      offset_list[src_fid].push_back(row_id);
      if (src_fid != dst_fid) {
        offset_list[dst_fid].push_back(row_id);
      }
    }
  };

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_recv;
  BOOST_LEAF_CHECK(ShuffleTableByOffsetLists(
      comm_spec, table_send->schema(), table_send, offsetfn, batches_recv));

  batches_recv.erase(std::remove_if(batches_recv.begin(), batches_recv.end(),
                                    [](std::shared_ptr<arrow::RecordBatch>& e) {
                                      return e == nullptr || e->num_rows() == 0;
                                    }),
                     batches_recv.end());

  // N.B.: we need an empty table for labels that doesn't have effective data.
  std::shared_ptr<arrow::Table> table_out;
  VY_OK_OR_RAISE(
      RecordBatchesToTable(table_send->schema(), batches_recv, &table_out));
  // ARROW_OK_ASSIGN_OR_RAISE(table_out,
  // table_out->CombineChunks(arrow::default_memory_pool()));
  return table_out;
}

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyVertexTable(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    const std::shared_ptr<arrow::Table>& table_send) {
  using oid_t = typename PARTITIONER_T::oid_t;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_type = ArrowArrayType<oid_t>;

  VY_OK_OR_RAISE(CheckSchemaConsistency(*table_send->schema(), comm_spec));

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
  VY_OK_OR_RAISE(TableToRecordBatches(table_send, &record_batches));

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
          grape::fid_t fid = partitioner.GetPartitionId(rs);
          offset_list[fid].push_back(row_id);
        }
      }
    });
  }
  for (auto& thrd : scan_threads) {
    thrd.join();
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_recv;

  BOOST_LEAF_CHECK(ShuffleTableByOffsetLists(comm_spec, table_send->schema(),
                                             record_batches, offset_lists,
                                             batches_recv));

  batches_recv.erase(std::remove_if(batches_recv.begin(), batches_recv.end(),
                                    [](std::shared_ptr<arrow::RecordBatch>& e) {
                                      return e == nullptr || e->num_rows() == 0;
                                    }),
                     batches_recv.end());

  std::shared_ptr<arrow::Table> table_out;
  VY_OK_OR_RAISE(
      RecordBatchesToTable(table_send->schema(), batches_recv, &table_out));
  // ARROW_OK_ASSIGN_OR_RAISE(table_out,
  // table_out->CombineChunks(arrow::default_memory_pool()));
  return table_out;
}

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyVertexTable(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    const std::shared_ptr<ITablePipeline>& table_send) {
  using oid_t = typename PARTITIONER_T::oid_t;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_type = ArrowArrayType<oid_t>;
  fid_t fnum = comm_spec.fnum();

  VY_OK_OR_RAISE(CheckSchemaConsistency(*table_send->schema(), comm_spec));

  auto offsetfn = [fnum, &partitioner](
                      const std::shared_ptr<arrow::RecordBatch> batch,
                      std::vector<std::vector<int64_t>>& offset_list) -> void {
    // per thread
    offset_list.resize(fnum);
    for (auto& offsets : offset_list) {
      offsets.clear();
    }

    int64_t row_num = batch ? batch->num_rows() : 0;
    std::shared_ptr<oid_array_type> id_col =
        batch ? std::dynamic_pointer_cast<oid_array_type>(batch->column(0))
              : nullptr;

    for (int64_t row_id = 0; row_id < row_num; ++row_id) {
      internal_oid_t rs = id_col->GetView(row_id);
      grape::fid_t fid = partitioner.GetPartitionId(rs);
      offset_list[fid].push_back(row_id);
    }
  };

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_recv;
  BOOST_LEAF_CHECK(ShuffleTableByOffsetLists(
      comm_spec, table_send->schema(), table_send, offsetfn, batches_recv));

  batches_recv.erase(std::remove_if(batches_recv.begin(), batches_recv.end(),
                                    [](std::shared_ptr<arrow::RecordBatch>& e) {
                                      return e == nullptr || e->num_rows() == 0;
                                    }),
                     batches_recv.end());

  VLOG(100) << "[worker-" << comm_spec.worker_id()
            << "] Vertices: after shuffle by offset lists: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  // N.B.: we need an empty table for labels that doesn't have effective data.
  std::shared_ptr<arrow::Table> table_out;
  VY_OK_OR_RAISE(
      RecordBatchesToTable(table_send->schema(), batches_recv, &table_out));
  // ARROW_OK_ASSIGN_OR_RAISE(table_out,
  // table_out->CombineChunks(arrow::default_memory_pool()));
  return table_out;
}

template <typename VID_TYPE>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyEdgeTable(
    const grape::CommSpec& comm_spec, IdParser<VID_TYPE>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send) {
  VY_OK_OR_RAISE(CheckSchemaConsistency(*table_send->schema(), comm_spec));

  using vid_array_t = ArrowArrayType<VID_TYPE>;
  fid_t fnum = comm_spec.fnum();

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
  VY_OK_OR_RAISE(TableToRecordBatches(table_send, &record_batches));

  size_t record_batch_num = record_batches.size();
  // record_batch_num, fragment_num, row_ids
  std::vector<std::vector<std::vector<int64_t>>> offset_lists(record_batch_num);

  auto fn = [&](const size_t batch_index) -> Status {
    auto& offset_list = offset_lists[batch_index];
    offset_list.resize(fnum);
    auto current_batch = record_batches[batch_index];
    int64_t row_num = current_batch->num_rows();

    const VID_TYPE* src_col = std::dynamic_pointer_cast<vid_array_t>(
                                  current_batch->column(src_col_id))
                                  ->raw_values();
    const VID_TYPE* dst_col = std::dynamic_pointer_cast<vid_array_t>(
                                  current_batch->column(dst_col_id))
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
    return Status::OK();
  };

  ThreadGroup tg(comm_spec);
  for (size_t batch_index = 0; batch_index < record_batch_num; ++batch_index) {
    tg.AddTask(fn, batch_index);
  }
  Status status;
  for (const Status& s : tg.TakeResults()) {
    status += s;
  }
  VY_OK_OR_RAISE(status);

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_recv;
  BOOST_LEAF_CHECK(ShuffleTableByOffsetLists(comm_spec, table_send->schema(),
                                             record_batches, offset_lists,
                                             batches_recv));

  batches_recv.erase(std::remove_if(batches_recv.begin(), batches_recv.end(),
                                    [](std::shared_ptr<arrow::RecordBatch>& e) {
                                      return e == nullptr || e->num_rows() == 0;
                                    }),
                     batches_recv.end());

  // N.B.: we need an empty table for labels that doesn't have effective data.
  std::shared_ptr<arrow::Table> table_out;
  VY_OK_OR_RAISE(
      RecordBatchesToTable(table_send->schema(), batches_recv, &table_out));
  // ARROW_OK_ASSIGN_OR_RAISE(table_out,
  // table_out->CombineChunks(arrow::default_memory_pool()));
  return table_out;
}

template <typename VID_TYPE>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyEdgeTable(
    const grape::CommSpec& comm_spec, IdParser<VID_TYPE>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send) {
  VY_OK_OR_RAISE(CheckSchemaConsistency(*table_send->schema(), comm_spec));

  using vid_array_t = ArrowArrayType<VID_TYPE>;
  fid_t fnum = comm_spec.fnum();

  auto offsetfn = [fnum, id_parser, src_col_id, dst_col_id](
                      const std::shared_ptr<arrow::RecordBatch> batch,
                      std::vector<std::vector<int64_t>>& offset_list) -> void {
    // per thread
    offset_list.resize(fnum);
    for (auto& offsets : offset_list) {
      offsets.clear();
    }

    int64_t row_num = batch ? batch->num_rows() : 0;
    const VID_TYPE* src_col =
        batch
            ? std::dynamic_pointer_cast<vid_array_t>(batch->column(src_col_id))
                  ->raw_values()
            : nullptr;
    const VID_TYPE* dst_col =
        batch
            ? std::dynamic_pointer_cast<vid_array_t>(batch->column(dst_col_id))
                  ->raw_values()
            : nullptr;

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
  };

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_recv;
  BOOST_LEAF_CHECK(ShuffleTableByOffsetLists(
      comm_spec, table_send->schema(), table_send, offsetfn, batches_recv));

  batches_recv.erase(std::remove_if(batches_recv.begin(), batches_recv.end(),
                                    [](std::shared_ptr<arrow::RecordBatch>& e) {
                                      return e == nullptr || e->num_rows() == 0;
                                    }),
                     batches_recv.end());

  VLOG(100) << "[worker-" << comm_spec.worker_id()
            << "] Edges: after shuffle by offset lists: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  // N.B.: we need an empty table for labels that doesn't have effective data.
  std::shared_ptr<arrow::Table> table_out;
  VY_OK_OR_RAISE(
      RecordBatchesToTable(table_send->schema(), batches_recv, &table_out));
  // ARROW_OK_ASSIGN_OR_RAISE(table_out,
  // table_out->CombineChunks(arrow::default_memory_pool()));
  VLOG(100) << "[worker-" << comm_spec.worker_id()
            << "] Edges: after combine chunks: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  return table_out;
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TABLE_SHUFFLER_IMPL_H_
