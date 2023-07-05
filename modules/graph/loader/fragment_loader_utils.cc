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

#include "graph/loader/fragment_loader_utils.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "basic/ds/hashmap.h"
#include "graph/fragment/arrow_fragment_base.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/utils/partitioner.h"
#include "graph/utils/table_shuffler.h"
#include "graph/utils/table_shuffler_beta.h"

namespace vineyard {

boost::leaf::result<std::vector<std::string>> GatherVertexLabels(
    const grape::CommSpec& comm_spec, const std::vector<InputTable>& tables) {
  std::set<std::string> local_vertex_labels;
  for (auto& tab : tables) {
    local_vertex_labels.insert(tab.src_label);
    local_vertex_labels.insert(tab.dst_label);
  }
  std::vector<std::string> local_vertex_label_list;
  for (auto& label : local_vertex_labels) {
    local_vertex_label_list.push_back(label);
  }
  std::vector<std::vector<std::string>> gathered_vertex_label_lists;
  GlobalAllGatherv(local_vertex_label_list, gathered_vertex_label_lists,
                   comm_spec);
  std::vector<std::string> sorted_labels;
  for (auto& vec : gathered_vertex_label_lists) {
    sorted_labels.insert(sorted_labels.end(), vec.begin(), vec.end());
  }
  std::sort(sorted_labels.begin(), sorted_labels.end());
  auto iter = std::unique(sorted_labels.begin(), sorted_labels.end());
  sorted_labels.erase(iter, sorted_labels.end());
  return sorted_labels;
}

/**
 * @brief Build a label to index map
 *
 * @param tables vector of labels, each items must be unique.
 * @return label to index map
 */
boost::leaf::result<std::map<std::string, property_graph_types::LABEL_ID_TYPE>>
GetVertexLabelToIndex(const std::vector<std::string>& labels) {
  std::map<std::string, property_graph_types::LABEL_ID_TYPE> label_to_index;
  for (size_t i = 0; i < labels.size(); ++i) {
    label_to_index[labels[i]] =
        static_cast<property_graph_types::LABEL_ID_TYPE>(i);
  }
  return label_to_index;
}

template <typename PARTITIONER_T>
boost::leaf::result<std::map<std::string, std::shared_ptr<arrow::Table>>>
BuildVertexTableFromEdges(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    const std::vector<InputTable>& edge_tables,
    const std::map<std::string, property_graph_types::LABEL_ID_TYPE>&
        vertex_label_to_index,
    const std::set<std::string>& deduced_labels) {
  static constexpr int src_column = 0;
  static constexpr int dst_column = 1;

  using oid_t = typename PARTITIONER_T::oid_t;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;

  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{
          arrow::field("id", vineyard::ConvertToArrowType<oid_t>::TypeValue()),
      });

  size_t vertex_label_num = vertex_label_to_index.size();
  std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>> chunks(
      vertex_label_num);

  // group src/dst columns by label
  for (auto& table : edge_tables) {
    if (deduced_labels.find(table.src_label) != deduced_labels.end()) {
      label_id_t src_label_id = vertex_label_to_index.at(table.src_label);
      chunks[src_label_id].emplace_back(table.table->column(src_column));
    }

    if (deduced_labels.find(table.dst_label) != deduced_labels.end()) {
      label_id_t dst_label_id = vertex_label_to_index.at(table.dst_label);
      chunks[dst_label_id].emplace_back(table.table->column(dst_column));
    }
  }

  // construct table pipelines for each label
  std::vector<std::shared_ptr<ITablePipeline>> table_pipelines(
      vertex_label_num);
  for (const auto& label : deduced_labels) {
    label_id_t label_id = vertex_label_to_index.at(label);
    table_pipelines[label_id] =
        std::make_shared<TablePipeline>(arrow::Table::Make(
            schema,
            std::vector<std::shared_ptr<arrow::ChunkedArray>>{
                ConcatenateChunkedArrays(std::move(chunks[label_id]))}));
  }

  // filter as unique ids
  std::vector<std::shared_ptr<ITablePipeline>> table_pipelines_deduped(
      vertex_label_num);
  std::vector<ConcurrentOidSet<oid_t>> oid_sets(vertex_label_num);

  auto dedupfn = [&oid_sets, &schema](
                     const std::shared_ptr<arrow::RecordBatch>& from,
                     std::mutex&, label_id_t& label_id,
                     std::shared_ptr<arrow::RecordBatch>& to) -> Status {
    auto& oid_set = oid_sets[label_id];
    auto chunk = from->column(0);
    std::shared_ptr<arrow::Array> out_chunk;
    RETURN_ON_ERROR(oid_set.Insert(chunk, out_chunk));
    to = arrow::RecordBatch::Make(
        schema, out_chunk->length(),
        std::vector<std::shared_ptr<arrow::Array>>{out_chunk});
    return Status::OK();
  };

  for (const auto& label : deduced_labels) {
    label_id_t label_id = vertex_label_to_index.at(label);
    table_pipelines_deduped[label_id] =
        std::make_shared<MapTablePipeline<label_id_t>>(
            table_pipelines[label_id], dedupfn, label_id, schema);
  }

  int concurrency =
      (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
      comm_spec.local_num();

  // shuffle: gather vertex ids from all fragments
  std::map<std::string, std::shared_ptr<arrow::Table>> results;
  for (const auto& label : deduced_labels) {
    label_id_t label_id = vertex_label_to_index.at(label);
    BOOST_LEAF_AUTO(
        out, ShufflePropertyVertexTable(comm_spec, partitioner,
                                        table_pipelines_deduped[label_id]));

    oid_sets[label_id].Clear();
    auto sink = std::make_shared<TablePipelineSink>(
        std::make_shared<MapTablePipeline<label_id_t>>(
            std::make_shared<TablePipeline>(out), dedupfn, label_id, schema),
        schema, concurrency);
    std::shared_ptr<arrow::Table> local_vtable;
    VY_OK_OR_RAISE(sink->Result(local_vtable));
    results[label] = local_vtable;
  }
  VLOG(100) << "finished collecting vtable from etable: " << get_rss_pretty()
            << ", peak rss: " << get_peak_rss_pretty();
  return results;
}

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
BuildVertexTableFromEdges<HashPartitioner<int32_t>>(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<int32_t>& partitioner,
    const std::vector<InputTable>& edge_tables,
    const std::map<std::string, property_graph_types::LABEL_ID_TYPE>&
        vertex_label_to_index,
    const std::set<std::string>& deduced_labels);

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
BuildVertexTableFromEdges<HashPartitioner<int64_t>>(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<int64_t>& partitioner,
    const std::vector<InputTable>& edge_tables,
    const std::map<std::string, property_graph_types::LABEL_ID_TYPE>&
        vertex_label_to_index,
    const std::set<std::string>& deduced_labels);

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
BuildVertexTableFromEdges<HashPartitioner<std::string>>(
    const grape::CommSpec& comm_spec,
    const HashPartitioner<std::string>& partitioner,
    const std::vector<InputTable>& edge_tables,
    const std::map<std::string, property_graph_types::LABEL_ID_TYPE>&
        vertex_label_to_index,
    const std::set<std::string>& deduced_labels);

boost::leaf::result<std::shared_ptr<arrow::Table>> SyncSchema(
    const std::shared_ptr<arrow::Table>& table,
    const grape::CommSpec& comm_spec) {
  std::shared_ptr<arrow::Schema> local_schema =
      table != nullptr ? table->schema() : nullptr;
  std::vector<std::shared_ptr<arrow::Schema>> schemas;

  GlobalAllGatherv(local_schema, schemas, comm_spec);
  std::shared_ptr<arrow::Schema> normalized_schema;
  VY_OK_OR_RAISE(TypeLoosen(schemas, normalized_schema));

  std::shared_ptr<arrow::Table> table_out;
  if (table == nullptr) {
    VY_OK_OR_RAISE(
        vineyard::EmptyTableBuilder::Build(normalized_schema, table_out));
  } else {
    VY_OK_OR_RAISE(CastTableToSchema(table, normalized_schema, table_out));
  }
  return table_out;
}

boost::leaf::result<ObjectID> ConstructFragmentGroup(
    Client& client, ObjectID frag_id, const grape::CommSpec& comm_spec) {
  ObjectID group_object_id;
  uint64_t instance_id = client.instance_id();

  MPI_Barrier(comm_spec.comm());
  VINEYARD_DISCARD(client.SyncMetaData());

  if (comm_spec.worker_id() == 0) {
    std::vector<uint64_t> gathered_instance_ids(comm_spec.worker_num());
    std::vector<ObjectID> gathered_object_ids(comm_spec.worker_num());
    MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR,
               &gathered_instance_ids[0], sizeof(uint64_t), MPI_CHAR, 0,
               comm_spec.comm());

    MPI_Gather(&frag_id, sizeof(ObjectID), MPI_CHAR, &gathered_object_ids[0],
               sizeof(ObjectID), MPI_CHAR, 0, comm_spec.comm());

    ArrowFragmentGroupBuilder builder;
    builder.set_total_frag_num(comm_spec.fnum());
    typename ArrowFragmentBase::label_id_t vertex_label_num = 0,
                                           edge_label_num = 0;

    ObjectMeta meta;
    if (client.GetMetaData(frag_id, meta).ok()) {
      if (meta.HasKey("vertex_label_num_")) {
        vertex_label_num =
            meta.GetKeyValue<typename ArrowFragmentBase::label_id_t>(
                "vertex_label_num_");
      }
      if (meta.HasKey("edge_label_num_")) {
        edge_label_num =
            meta.GetKeyValue<typename ArrowFragmentBase::label_id_t>(
                "edge_label_num_");
      }
    }

    builder.set_vertex_label_num(vertex_label_num);
    builder.set_edge_label_num(edge_label_num);
    for (fid_t i = 0; i < comm_spec.fnum(); ++i) {
      builder.AddFragmentObject(
          i, gathered_object_ids[comm_spec.FragToWorker(i)],
          gathered_instance_ids[comm_spec.FragToWorker(i)]);
    }

    std::shared_ptr<Object> fragment_group_object;
    VY_OK_OR_RAISE(builder.Seal(client, fragment_group_object));
    group_object_id = fragment_group_object->id();
    VY_OK_OR_RAISE(client.Persist(group_object_id));

    MPI_Bcast(&group_object_id, sizeof(ObjectID), MPI_CHAR, 0,
              comm_spec.comm());
  } else {
    // if not 0 worker, collect self data and send to the 0 worker
    MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR, NULL, sizeof(uint64_t),
               MPI_CHAR, 0, comm_spec.comm());
    MPI_Gather(&frag_id, sizeof(ObjectID), MPI_CHAR, NULL, sizeof(ObjectID),
               MPI_CHAR, 0, comm_spec.comm());
    MPI_Bcast(&group_object_id, sizeof(ObjectID), MPI_CHAR, 0,
              comm_spec.comm());
  }

  MPI_Barrier(comm_spec.comm());
  VINEYARD_DISCARD(client.SyncMetaData());
  return group_object_id;
}

EdgeTableInfo FlattenTableInfos(const std::vector<EdgeTableInfo>& edge_tables) {
  if (edge_tables.size() == 1) {
    return edge_tables[0];
  }
  std::vector<std::shared_ptr<arrow::Table>> adj_list_tables;
  std::vector<std::shared_ptr<arrow::Table>> property_tables;
  for (auto& table : edge_tables) {
    adj_list_tables.push_back(table.adj_list_table);
    property_tables.push_back(table.property_table);
  }
  auto adj_list_table = arrow::ConcatenateTables(adj_list_tables).ValueOrDie();
  auto property_table = arrow::ConcatenateTables(property_tables).ValueOrDie();
  return EdgeTableInfo(adj_list_table, nullptr, property_table, 0, true);
}

std::pair<int64_t, int64_t> BinarySearchChunkPair(
    const std::vector<int64_t>& agg_num, int64_t got) {
  // binary search;
  size_t low = 0, high = agg_num.size() - 1;
  while (low <= high) {
    size_t mid = (low + high) / 2;
    if (agg_num[mid] <= got &&
        (mid == agg_num.size() - 1 || agg_num[mid + 1] > got)) {
      return std::make_pair(mid, got - agg_num[mid]);
    } else if (agg_num[mid] > got) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return std::make_pair(low, got - agg_num[low]);
}

Status parallel_prefix_sum_chunks(
    std::vector<std::shared_ptr<arrow::Int64Array>>& in_chunks,
    std::shared_ptr<arrow::Int64Array>& out) {
  if (in_chunks.size() == 1) {
    out = in_chunks[0];
    return Status::OK();
  }
  std::vector<int64_t> block_sum(in_chunks.size());
  size_t bsize = static_cast<size_t>(in_chunks[0]->length() - 1);
  size_t chunk_size = in_chunks.size();
  int thread_num = static_cast<int>(chunk_size);
  block_sum[0] = in_chunks[0]->Value(in_chunks[0]->length() - 1);
  size_t length = in_chunks[0]->length();
  for (size_t i = 1; i < in_chunks.size(); ++i) {
    block_sum[i] =
        block_sum[i - 1] + in_chunks[i]->Value(in_chunks[i]->length() - 1);
    length += in_chunks[i]->length() - 1;
  }

  std::unique_ptr<arrow::Buffer> buffer;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      buffer, arrow::AllocateBuffer(length * sizeof(int64_t)));
  int64_t* builder = reinterpret_cast<int64_t*>(buffer->mutable_data());
  auto block_add = [&](int i) {
    size_t begin = std::min(static_cast<size_t>(i) * bsize, length);
    size_t end = std::min(begin + bsize, length);
    if (i == 0) {
      for (; begin < end; ++begin) {
        builder[begin] =
            in_chunks[i]->Value(begin - static_cast<size_t>(i) * bsize);
      }
    } else {
      for (; begin < end; ++begin) {
        builder[begin] =
            in_chunks[i]->Value(begin - static_cast<size_t>(i) * bsize) +
            block_sum[i - 1];
      }
    }
  };

  std::vector<std::thread> threads_sum;
  for (int i = 0; i < thread_num; ++i) {
    threads_sum.emplace_back(block_add, i);
  }
  for (auto& thrd : threads_sum) {
    thrd.join();
  }
  // the last element is the total sum
  builder[length - 1] = block_sum[chunk_size - 1];
  out = std::make_shared<arrow::Int64Array>(length, std::move(buffer), nullptr,
                                            0);
  return Status::OK();
}

}  // namespace vineyard
