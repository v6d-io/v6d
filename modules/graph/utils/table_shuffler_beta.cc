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

#include "graph/utils/table_shuffler_beta.h"

namespace vineyard {

namespace beta {

void serialize_string_items(grape::InArchive& arc,
                            std::shared_ptr<arrow::Array> array,
                            const std::vector<int64_t>& offset) {
  auto* ptr = std::dynamic_pointer_cast<arrow::LargeStringArray>(array).get();
  for (auto x : offset) {
    arc << ptr->GetView(x);
  }
}

void serialize_null_items(grape::InArchive& arc,
                          std::shared_ptr<arrow::Array> array,
                          const std::vector<int64_t>& offset) {
  return;
}

void SerializeSelectedItems(grape::InArchive& arc,
                            std::shared_ptr<arrow::Array> array,
                            const std::vector<int64_t>& offset) {
  if (array->type()->Equals(arrow::float64())) {
    serialize_selected_typed_items<double>(arc, array, offset);
  } else if (array->type()->Equals(arrow::float32())) {
    serialize_selected_typed_items<float>(arc, array, offset);
  } else if (array->type()->Equals(arrow::int64())) {
    serialize_selected_typed_items<int64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::int32())) {
    serialize_selected_typed_items<int32_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::uint64())) {
    serialize_selected_typed_items<uint64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::uint32())) {
    serialize_selected_typed_items<uint32_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_utf8())) {
    serialize_string_items(arc, array, offset);
  } else if (array->type()->Equals(arrow::null())) {
    serialize_null_items(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::float64()))) {
    serialize_list_items<double>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::float32()))) {
    serialize_list_items<float>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::int64()))) {
    serialize_list_items<int64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::int32()))) {
    serialize_list_items<int32_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint64()))) {
    serialize_list_items<uint64_t>(arc, array, offset);
  } else if (array->type()->Equals(arrow::large_list(arrow::uint32()))) {
    serialize_list_items<uint32_t>(arc, array, offset);
  } else {
    LOG(FATAL) << "Unsupported data type - " << array->type()->ToString();
  }
}

void SerializeSelectedRows(grape::InArchive& arc,
                           std::shared_ptr<arrow::RecordBatch> record_batch,
                           const std::vector<int64_t>& offset) {
  int col_num = record_batch->num_columns();
  arc << static_cast<int64_t>(offset.size());
  for (int col_id = 0; col_id != col_num; ++col_id) {
    SerializeSelectedItems(arc, record_batch->column(col_id), offset);
  }
}

void DeserializeSelectedItems(grape::OutArchive& arc, int64_t num,
                              arrow::ArrayBuilder* builder) {
  if (builder->type()->Equals(arrow::float64())) {
    deserialize_selected_typed_items<double>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::float32())) {
    deserialize_selected_typed_items<float>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::int64())) {
    deserialize_selected_typed_items<int64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::int32())) {
    deserialize_selected_typed_items<int32_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::uint64())) {
    deserialize_selected_typed_items<uint64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::uint32())) {
    deserialize_selected_typed_items<uint32_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_utf8())) {
    deserialize_string_items(arc, num, builder);
  } else if (builder->type()->Equals(arrow::null())) {
    deserialize_null_items(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::float64()))) {
    deserialize_list_items<double>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::float32()))) {
    deserialize_list_items<float>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::int64()))) {
    deserialize_list_items<int64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::int32()))) {
    deserialize_list_items<int32_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::uint64()))) {
    deserialize_list_items<uint64_t>(arc, num, builder);
  } else if (builder->type()->Equals(arrow::large_list(arrow::uint32()))) {
    deserialize_list_items<uint32_t>(arc, num, builder);
  } else {
    LOG(FATAL) << "Unsupported data type - " << builder->type()->ToString();
  }
}

void DeserializeSelectedRows(grape::OutArchive& arc,
                             std::shared_ptr<arrow::Schema> schema,
                             std::shared_ptr<arrow::RecordBatch>& batch_out) {
  int64_t row_num;
  arc >> row_num;
  std::unique_ptr<arrow::RecordBatchBuilder> builder;
  ARROW_CHECK_OK(arrow::RecordBatchBuilder::Make(
      schema, arrow::default_memory_pool(), row_num, &builder));
  int col_num = builder->num_fields();
  for (int col_id = 0; col_id != col_num; ++col_id) {
    DeserializeSelectedItems(arc, row_num, builder->GetField(col_id));
  }
  ARROW_CHECK_OK(builder->Flush(&batch_out));
}

void ShuffleTableByOffsetLists(
    std::shared_ptr<arrow::Schema> schema,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_out,
    const std::vector<std::vector<std::vector<int64_t>>>& offset_lists,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_in,
    const grape::CommSpec& comm_spec) {
  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();
  size_t record_batches_out_num = record_batches_out.size();
#if 1
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
        auto cur_rb = record_batches_out[got_batch];
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
  record_batches_in.resize(record_batches_to_recv);
  for (int i = 0; i != deserialize_thread_num; ++i) {
    deserialize_threads[i] = std::thread([&]() {
      grape::OutArchive arc;
      while (msg_in.Get(arc)) {
        int64_t got_batch = cur_batch_in.fetch_add(1);
        DeserializeSelectedRows(arc, schema, record_batches_in[got_batch]);
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
    SelectRows(record_batches_out[rb_i], offset_lists[rb_i][comm_spec.fid()],
               rb);
    record_batches_in.emplace_back(std::move(rb));
  }
#else
  std::thread send_thread([&]() {
    for (int i = 1; i != worker_num; ++i) {
      int dst_worker_id = (worker_id + worker_num - i) % worker_num;
      grape::fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
      grape::InArchive arc;
      arc << record_batches_out_num;
      for (size_t rb_i = 0; rb_i != record_batches_out_num; ++rb_i) {
        SerializeSelectedRows(arc, record_batches_out[rb_i],
                              offset_lists[rb_i][dst_fid]);
      }
      grape::sync_comm::Send(arc, dst_worker_id, 0, comm_spec.comm());
    }
  });
  std::thread recv_thread([&]() {
    for (int i = 1; i != worker_num; ++i) {
      int src_worker_id = (worker_id + i) % worker_num;
      grape::OutArchive arc;
      grape::sync_comm::Recv(arc, src_worker_id, 0, comm_spec.comm());
      size_t rb_num;
      arc >> rb_num;
      for (size_t rb_i = 0; rb_i != rb_num; ++rb_i) {
        std::shared_ptr<arrow::RecordBatch> rb;
        DeserializeSelectedRows(arc, schema, rb);
        record_batches_in.emplace_back(std::move(rb));
      }
    }

    for (size_t rb_i = 0; rb_i != record_batches_out_num; ++rb_i) {
      std::shared_ptr<arrow::RecordBatch> rb;
      SelectRows(record_batches_out[rb_i], offset_lists[rb_i][comm_spec.fid()],
                 rb);
      record_batches_in.emplace_back(std::move(rb));
    }
  });

  send_thread.join();
  recv_thread.join();
#endif

  MPI_Barrier(comm_spec.comm());
}
}  // namespace beta

}  // namespace vineyard
