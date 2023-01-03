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

template <typename OID_T, typename PARTITIONER_T>
boost::leaf::result<std::map<std::string, std::shared_ptr<arrow::Table>>>
FragmentLoaderUtils<OID_T, PARTITIONER_T>::BuildVertexTableFromEdges(
    const std::vector<InputTable>& edge_tables,
    const std::map<std::string, label_id_t>& vertex_label_to_index,
    const std::set<std::string>& deduced_labels) {
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
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();

  // shuffle: gather vertex ids from all fragments
  std::map<std::string, std::shared_ptr<arrow::Table>> results;
  for (const auto& label : deduced_labels) {
    label_id_t label_id = vertex_label_to_index.at(label);
    BOOST_LEAF_AUTO(
        out, ShufflePropertyVertexTable(comm_spec_, partitioner_,
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
            << ", peek rss: " << get_peak_rss_pretty();
  return results;
}

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
FragmentLoaderUtils<int32_t, HashPartitioner<int32_t>>::
    BuildVertexTableFromEdges(
        const std::vector<InputTable>& edge_tables,
        const std::map<std::string, label_id_t>& vertex_label_to_index,
        const std::set<std::string>& deduced_labels);

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
FragmentLoaderUtils<int64_t, HashPartitioner<int64_t>>::
    BuildVertexTableFromEdges(
        const std::vector<InputTable>& edge_tables,
        const std::map<std::string, label_id_t>& vertex_label_to_index,
        const std::set<std::string>& deduced_labels);

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
FragmentLoaderUtils<std::string, HashPartitioner<std::string>>::
    BuildVertexTableFromEdges(
        const std::vector<InputTable>& edge_tables,
        const std::map<std::string, label_id_t>& vertex_label_to_index,
        const std::set<std::string>& deduced_labels);

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
FragmentLoaderUtils<int32_t, SegmentedPartitioner<int32_t>>::
    BuildVertexTableFromEdges(
        const std::vector<InputTable>& edge_tables,
        const std::map<std::string, label_id_t>& vertex_label_to_index,
        const std::set<std::string>& deduced_labels);

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
FragmentLoaderUtils<int64_t, SegmentedPartitioner<int64_t>>::
    BuildVertexTableFromEdges(
        const std::vector<InputTable>& edge_tables,
        const std::map<std::string, label_id_t>& vertex_label_to_index,
        const std::set<std::string>& deduced_labels);

template boost::leaf::result<
    std::map<std::string, std::shared_ptr<arrow::Table>>>
FragmentLoaderUtils<std::string, SegmentedPartitioner<std::string>>::
    BuildVertexTableFromEdges(
        const std::vector<InputTable>& edge_tables,
        const std::map<std::string, label_id_t>& vertex_label_to_index,
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
      if (meta.Haskey("vertex_label_num_")) {
        vertex_label_num =
            meta.GetKeyValue<typename ArrowFragmentBase::label_id_t>(
                "vertex_label_num_");
      }
      if (meta.Haskey("edge_label_num_")) {
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

    auto group_object =
        std::dynamic_pointer_cast<ArrowFragmentGroup>(builder.Seal(client));
    group_object_id = group_object->id();
    VY_OK_OR_RAISE(client.Persist(group_object_id));

    MPI_Bcast(&group_object_id, sizeof(ObjectID), MPI_CHAR, 0,
              comm_spec.comm());
  } else {
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

}  // namespace vineyard
