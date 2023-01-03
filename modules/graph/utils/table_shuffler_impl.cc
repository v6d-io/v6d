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

#include "graph/utils/table_shuffler_impl.h"

#include <string>

#include "graph/utils/partitioner.h"

namespace vineyard {

template void SendArrowArray<arrow::Array>(
    const std::shared_ptr<arrow::Array>& array, int dst_worker_id,
    MPI_Comm comm, int tag);
template void SendArrowArray<arrow::Int32Array>(
    const std::shared_ptr<arrow::Int32Array>& array, int dst_worker_id,
    MPI_Comm comm, int tag);
template void SendArrowArray<arrow::UInt32Array>(
    const std::shared_ptr<arrow::UInt32Array>& array, int dst_worker_id,
    MPI_Comm comm, int tag);
template void SendArrowArray<arrow::Int64Array>(
    const std::shared_ptr<arrow::Int64Array>& array, int dst_worker_id,
    MPI_Comm comm, int tag);
template void SendArrowArray<arrow::UInt64Array>(
    const std::shared_ptr<arrow::UInt64Array>& array, int dst_worker_id,
    MPI_Comm comm, int tag);
template void SendArrowArray<arrow::StringArray>(
    const std::shared_ptr<arrow::StringArray>& array, int dst_worker_id,
    MPI_Comm comm, int tag);
template void SendArrowArray<arrow::LargeStringArray>(
    const std::shared_ptr<arrow::LargeStringArray>& array, int dst_worker_id,
    MPI_Comm comm, int tag);

template void RecvArrowArray<arrow::Array>(std::shared_ptr<arrow::Array>& array,
                                           int src_worker_id, MPI_Comm comm,
                                           int tag);
template void RecvArrowArray<arrow::Int32Array>(
    std::shared_ptr<arrow::Int32Array>& array, int src_worker_id, MPI_Comm comm,
    int tag);
template void RecvArrowArray<arrow::UInt32Array>(
    std::shared_ptr<arrow::UInt32Array>& array, int src_worker_id,
    MPI_Comm comm, int tag);
template void RecvArrowArray<arrow::Int64Array>(
    std::shared_ptr<arrow::Int64Array>& array, int src_worker_id, MPI_Comm comm,
    int tag);
template void RecvArrowArray<arrow::UInt64Array>(
    std::shared_ptr<arrow::UInt64Array>& array, int src_worker_id,
    MPI_Comm comm, int tag);
template void RecvArrowArray<arrow::StringArray>(
    std::shared_ptr<arrow::StringArray>& array, int src_worker_id,
    MPI_Comm comm, int tag);
template void RecvArrowArray<arrow::LargeStringArray>(
    std::shared_ptr<arrow::LargeStringArray>& array, int src_worker_id,
    MPI_Comm comm, int tag);

template <>
void SendArrowArray<arrow::ChunkedArray>(
    const std::shared_ptr<arrow::ChunkedArray>& array, int dst_worker_id,
    MPI_Comm comm, int tag) {
  // type
  std::shared_ptr<arrow::Buffer> buffer;
  // shouldn't fail
  ARROW_CHECK_OK(SerializeDataType(array->type(), &buffer));
  SendArrowBuffer(buffer, dst_worker_id, comm, tag);

  // length
  int64_t length = array->length();
  MPI_Send(&length, 1, MPI_INT64_T, dst_worker_id, tag, comm);

  // chunk_size
  int64_t num_chunks = array->num_chunks();
  MPI_Send(&num_chunks, 1, MPI_INT64_T, dst_worker_id, tag, comm);

  // chunks
  for (int64_t i = 0; i < num_chunks; ++i) {
    std::shared_ptr<arrow::Array> chunk = array->chunk(i);
    detail::send_array_data(chunk->data(), false /* no duplicate data type */,
                            dst_worker_id, comm, tag);
  }
}

template <>
void RecvArrowArray<arrow::ChunkedArray>(
    std::shared_ptr<arrow::ChunkedArray>& array, int src_worker_id,
    MPI_Comm comm, int tag) {
  // type
  std::shared_ptr<arrow::DataType> type;
  std::shared_ptr<arrow::Buffer> buffer;
  RecvArrowBuffer(buffer, src_worker_id, comm, tag);
  // shouldn't fail
  ARROW_CHECK_OK(DeserializeDataType(buffer, &type));

  // length
  int64_t length;
  MPI_Recv(&length, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);

  // chunk_size
  int64_t num_chunks;
  MPI_Recv(&num_chunks, 1, MPI_INT64_T, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);

  // chunks
  std::vector<std::shared_ptr<arrow::Array>> chunks;
  for (int64_t i = 0; i < num_chunks; ++i) {
    std::shared_ptr<arrow::ArrayData> data;
    detail::recv_array_data(data, type /* no duplicate data type */,
                            src_worker_id, comm, tag);
    chunks.push_back(arrow::MakeArray(data));
  }

  array = std::make_shared<arrow::ChunkedArray>(chunks, type);
}

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<int32_t>& partitioner, int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<int64_t>& partitioner, int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<std::string>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const SegmentedPartitioner<int32_t>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const SegmentedPartitioner<int64_t>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const SegmentedPartitioner<std::string>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<int32_t>& partitioner, int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<int64_t>& partitioner, int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<std::string>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const SegmentedPartitioner<int32_t>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const SegmentedPartitioner<int64_t>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec,
    const SegmentedPartitioner<std::string>& partitioner, int src_col_id,
    int dst_col_id, const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTable<uint32_t>(
    const grape::CommSpec& comm_spec, IdParser<uint32_t>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTable<uint64_t>(
    const grape::CommSpec& comm_spec, IdParser<uint64_t>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTable<uint32_t>(
    const grape::CommSpec& comm_spec, IdParser<uint32_t>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTable<uint64_t>(
    const grape::CommSpec& comm_spec, IdParser<uint64_t>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const HashPartitioner<int32_t>& partitioner,
                           const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const HashPartitioner<int64_t>& partitioner,
                           const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const HashPartitioner<std::string>& partitioner,
                           const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const SegmentedPartitioner<int32_t>& partitioner,
                           const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const SegmentedPartitioner<int64_t>& partitioner,
                           const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const SegmentedPartitioner<std::string>& partitioner,
                           const std::shared_ptr<arrow::Table>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const HashPartitioner<int32_t>& partitioner,
                           const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const HashPartitioner<int64_t>& partitioner,
                           const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const HashPartitioner<std::string>& partitioner,
                           const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const SegmentedPartitioner<int32_t>& partitioner,
                           const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const SegmentedPartitioner<int64_t>& partitioner,
                           const std::shared_ptr<ITablePipeline>& table_send);

template boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyVertexTable(const grape::CommSpec& comm_spec,
                           const SegmentedPartitioner<std::string>& partitioner,
                           const std::shared_ptr<ITablePipeline>& table_send);

}  // namespace vineyard
