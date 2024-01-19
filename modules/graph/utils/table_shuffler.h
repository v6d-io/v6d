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

#include "common/util/status.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/table_pipeline.h"

namespace grape {
class CommSpec;
}  // namespace grape

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
struct AppendHelper<arrow::Time32Type> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::Time32Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::Time32Array>(array)->GetView(offset)));
    return Status::OK();
  }
};

template <>
struct AppendHelper<arrow::Time64Type> {
  static Status append(arrow::ArrayBuilder* builder,
                       std::shared_ptr<arrow::Array> array, size_t offset) {
    RETURN_ON_ARROW_ERROR(dynamic_cast<arrow::Time64Builder*>(builder)->Append(
        std::dynamic_pointer_cast<arrow::Time64Array>(array)->GetView(offset)));
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

namespace detail {

void send_array_data(const std::shared_ptr<arrow::ArrayData>& data,
                     bool include_data_type, int dst_worker_id, MPI_Comm comm,
                     int tag = 0);

void recv_array_data(std::shared_ptr<arrow::ArrayData>& data,
                     const std::shared_ptr<arrow::DataType> known_type,
                     int src_worker_id, MPI_Comm comm, int tag = 0);

}  // namespace detail

void SendArrowBuffer(const std::shared_ptr<arrow::Buffer>& buffer,
                     int dst_worker_id, MPI_Comm comm, int tag = 0);

void RecvArrowBuffer(std::shared_ptr<arrow::Buffer>& buffer, int src_worker_id,
                     MPI_Comm comm, int tag = 0);

template <typename ArrayType>
void SendArrowArray(const std::shared_ptr<ArrayType>& array, int dst_worker_id,
                    MPI_Comm comm, int tag = 0);

template <typename ArrayType>
void RecvArrowArray(std::shared_ptr<ArrayType>& array, int src_worker_id,
                    MPI_Comm comm, int tag = 0);

template <>
void SendArrowArray<arrow::ChunkedArray>(
    const std::shared_ptr<arrow::ChunkedArray>& array, int dst_worker_id,
    MPI_Comm comm, int tag);

template <>
void RecvArrowArray<arrow::ChunkedArray>(
    std::shared_ptr<arrow::ChunkedArray>& array, int src_worker_id,
    MPI_Comm comm, int tag);

template <typename ArrayType = arrow::Array>
Status FragmentAllGatherArray(
    const grape::CommSpec& comm_spec, std::shared_ptr<ArrayType> data_in,
    std::vector<std::shared_ptr<ArrayType>>& data_out);

Status CheckSchemaConsistency(const arrow::Schema& schema,
                              const grape::CommSpec& comm_spec);

void SerializeSelectedItems(grape::InArchive& arc,
                            std::shared_ptr<arrow::Array> array,
                            const std::vector<int64_t>& offset);

void SerializeSelectedRows(grape::InArchive& arc,
                           std::shared_ptr<arrow::RecordBatch> record_batch,
                           const std::vector<int64_t>& offset);

void DeserializeSelectedItems(grape::OutArchive& arc, int64_t num,
                              arrow::ArrayBuilder* builder);

void DeserializeSelectedRows(grape::OutArchive& arc,
                             std::shared_ptr<arrow::Schema> schema,
                             std::shared_ptr<arrow::RecordBatch>& batch_out);

void SelectItems(const std::shared_ptr<arrow::Array> array,
                 const std::vector<int64_t> offset,
                 arrow::ArrayBuilder* builder);

void SelectRows(const std::shared_ptr<arrow::RecordBatch> record_batch_in,
                const std::vector<int64_t>& offset,
                std::shared_ptr<arrow::RecordBatch>& record_batch_out);

boost::leaf::result<void> ShuffleTableByOffsetLists(
    const grape::CommSpec& comm_spec,
    const std::shared_ptr<arrow::Schema> schema,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_send,
    const std::vector<std::vector<std::vector<int64_t>>>& offset_lists,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_recv);

boost::leaf::result<void> ShuffleTableByOffsetLists(
    const grape::CommSpec& comm_spec,
    const std::shared_ptr<arrow::Schema> schema,
    const std::shared_ptr<ITablePipeline>& record_batches_send,
    std::function<void(const std::shared_ptr<arrow::RecordBatch> batch,
                       std::vector<std::vector<int64_t>>& offset_list)>
        genoffset,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches_recv);

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send);

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>>
ShufflePropertyEdgeTableByPartition(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send);

template <typename VID_TYPE>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyEdgeTable(
    const grape::CommSpec& comm_spec, IdParser<VID_TYPE>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<arrow::Table>& table_send);

template <typename VID_TYPE>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyEdgeTable(
    const grape::CommSpec& comm_spec, IdParser<VID_TYPE>& id_parser,
    int src_col_id, int dst_col_id,
    const std::shared_ptr<ITablePipeline>& table_send);

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyVertexTable(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    const std::shared_ptr<arrow::Table>& table_send);

template <typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<arrow::Table>> ShufflePropertyVertexTable(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    const std::shared_ptr<ITablePipeline>& table_send);

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TABLE_SHUFFLER_H_
