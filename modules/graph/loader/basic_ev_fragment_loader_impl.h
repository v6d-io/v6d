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

#ifndef MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_IMPL_H_
#define MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_IMPL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "grape/worker/comm_spec.h"

#include "common/util/env.h"
#include "common/util/static_if.h"
#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/loader/basic_ev_fragment_loader.h"
#include "graph/loader/fragment_loader_utils.h"
#include "graph/utils/error.h"
#include "graph/utils/table_shuffler.h"
#include "graph/utils/table_shuffler_beta.h"
#include "graph/vertex_map/arrow_local_vertex_map.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::BasicEVFragmentLoader(
    Client& client, const grape::CommSpec& comm_spec,
    const PARTITIONER_T& partitioner, bool directed, bool generate_eid,
    bool retain_oid, bool local_vertex_map, bool compact_edges,
    bool use_perfect_hash)
    : client_(client),
      comm_spec_(comm_spec),
      partitioner_(partitioner),
      directed_(directed),
      generate_eid_(generate_eid),
      retain_oid_(retain_oid),
      local_vertex_map_(local_vertex_map),
      compact_edges_(compact_edges),
      use_perfect_hash_(use_perfect_hash) {}

/**
 * @brief Add a loaded vertex table.
 *
 * @param label vertex label name.
 * @param vertex_table
 *  | id : OID_T | property_1 | ... | property_n |
 * @return
 */
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::AddVertexTable(
    const std::string& label, std::shared_ptr<arrow::Table> vertex_table) {
  auto id_column_type = vertex_table->column(id_column)->type();

  if (!id_column_type->Equals(ConvertToArrowType<oid_t>::TypeValue())) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "OID_T '" + type_name<OID_T>() +
                        "' is not same with arrow::Column(" +
                        std::to_string(id_column) + ") for label '" + label +
                        "': '" + id_column_type->ToString() + "'");
  }

  if (input_vertex_tables_.find(label) == input_vertex_tables_.end()) {
    vertex_labels_.push_back(label);
    input_vertex_tables_[label] = vertex_table;
  } else {
    std::vector<std::shared_ptr<arrow::Table>> tables;
    tables.push_back(input_vertex_tables_.at(label));
    tables.push_back(vertex_table);
    VY_OK_OR_RAISE(ConcatenateTables(tables, input_vertex_tables_[label]));
  }
  return {};
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::ConstructVertices(
    ObjectID vm_id) {
  for (size_t i = 0; i < vertex_labels_.size(); ++i) {
    vertex_label_to_index_[vertex_labels_[i]] = i;
  }
  vertex_label_num_ = vertex_labels_.size();

  ordered_vertex_tables_.clear();
  ordered_vertex_tables_.resize(vertex_label_num_, nullptr);

  for (auto& pair : input_vertex_tables_) {
    VLOG(100) << "[worker-" << comm_spec_.worker_id()
              << "] un-shuffled vertex table size for label "
              << vertex_label_to_index_[pair.first] << ": "
              << pair.second->num_rows();
    ordered_vertex_tables_[vertex_label_to_index_[pair.first]] =
        std::make_shared<TablePipeline>(pair.second);
  }

  input_vertex_tables_.clear();

  output_vertex_tables_.resize(vertex_label_num_);
  boost::leaf::result<void> result;
  if (local_vertex_map_) {
    result = constructVerticesImplLocal(vm_id);
  } else {
    result = constructVerticesImpl(vm_id);
  }
  ordered_vertex_tables_.clear();
  return result;
}

/**
 * @brief Add a loaded edge table.
 *
 * @param src_label src vertex label name.
 * @param dst_label dst vertex label name.
 * @param edge_label edge label name.
 * @param edge_table
 *  | src : OID_T | dst : OID_T | property_1 | ... | property_m |
 * @return
 */
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::AddEdgeTable(
    const std::string& src_label, const std::string& dst_label,
    const std::string& edge_label, std::shared_ptr<arrow::Table> edge_table) {
  label_id_t src_label_id, dst_label_id;
  auto iter = vertex_label_to_index_.find(src_label);
  if (iter == vertex_label_to_index_.end()) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "Invalid src vertex label " + src_label);
  }
  src_label_id = iter->second;
  iter = vertex_label_to_index_.find(dst_label);
  if (iter == vertex_label_to_index_.end()) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "Invalid dst vertex label " + dst_label);
  }
  dst_label_id = iter->second;

  auto src_column_type = edge_table->column(src_column)->type();
  auto dst_column_type = edge_table->column(dst_column)->type();

  if (!src_column_type->Equals(
          vineyard::ConvertToArrowType<oid_t>::TypeValue())) {
    RETURN_GS_ERROR(
        ErrorCode::kInvalidValueError,
        "OID_T '" + type_name<OID_T>() +
            "' is not consistent with src id of edge table for label " +
            src_label + "'" + src_column_type->ToString() +
            "', please specify 'column_types' for your input files");
  }
  if (!dst_column_type->Equals(
          vineyard::ConvertToArrowType<oid_t>::TypeValue())) {
    RETURN_GS_ERROR(
        ErrorCode::kInvalidValueError,
        "OID_T '" + type_name<OID_T>() +
            "' is not consistent with dst id of edge table for label " +
            dst_label + "'" + dst_column_type->ToString() +
            "', please specify 'column_types' for your input files");
  }

  input_edge_tables_[edge_label].emplace_back(
      std::make_pair(src_label_id, dst_label_id), edge_table);
  if (std::find(std::begin(edge_labels_), std::end(edge_labels_), edge_label) ==
      std::end(edge_labels_)) {
    edge_labels_.push_back(edge_label);
  }
  return {};
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::ConstructEdges(
    int label_offset, int vertex_label_num) {
  if (vertex_label_num == 0) {
    vertex_label_num = vertex_label_num_;
  }
  for (size_t i = 0; i < edge_labels_.size(); ++i) {
    edge_label_to_index_[edge_labels_[i]] = i;
  }
  edge_label_num_ = edge_labels_.size();

  ordered_edge_tables_.clear();
  ordered_edge_tables_.resize(edge_label_num_);

  for (auto& pair : input_edge_tables_) {
    for (auto const& item : pair.second) {
      VLOG(100) << "[worker-" << comm_spec_.worker_id()
                << "] un-shuffled edge table size for label "
                << edge_label_to_index_[pair.first] << ": "
                << item.second->num_rows();
      ordered_edge_tables_[edge_label_to_index_[pair.first]].push_back(
          std::make_pair(item.first,
                         std::make_shared<TablePipeline>(item.second)));
    }
  }
  input_edge_tables_.clear();

  if (generate_eid_) {
    BOOST_LEAF_CHECK(
        generateEdgeId(comm_spec_, ordered_edge_tables_, label_offset));
  }

  edge_relations_.resize(edge_label_num_);
  for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
    std::vector<std::pair<label_id_t, label_id_t>> relations;
    for (auto const& pair : ordered_edge_tables_[e_label]) {
      relations.push_back(pair.first);
    }
    std::vector<std::vector<std::pair<label_id_t, label_id_t>>>
        gathered_relations;
    GlobalAllGatherv(relations, gathered_relations, comm_spec_);
    for (auto& pair_vec : gathered_relations) {
      for (auto& pair : pair_vec) {
        edge_relations_[e_label].insert(pair);
      }
    }
  }

  VLOG(100) << "Starting constructing property edge tables: "
            << get_rss_pretty() << ", peak = " << get_peak_rss_pretty();
  boost::leaf::result<void> result;
  if (local_vertex_map_) {
    result = constructEdgesImplLocal(label_offset, vertex_label_num);
  } else {
    result = constructEdgesImpl(label_offset, vertex_label_num);
  }
  ordered_edge_tables_.clear();
  return result;
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<ObjectID>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::AddVerticesToFragment(
    std::shared_ptr<ArrowFragmentBase> frag) {
  if (local_vertex_map_) {
    RETURN_GS_ERROR(
        vineyard::ErrorCode::kUnsupportedOperationError,
        "Cannot only add vertices to fragment with local vertex map");
  }
  int pre_vlabel_num = frag->schema().all_vertex_label_num();
  std::map<label_id_t, std::shared_ptr<arrow::Table>> vertex_tables_map;
  for (size_t i = 0; i < output_vertex_tables_.size(); ++i) {
    vertex_tables_map[pre_vlabel_num + i] = output_vertex_tables_[i];
  }
  return frag->AddVertices(client_, std::move(vertex_tables_map),
                           vm_ptr_ ? vm_ptr_->id() : local_vm_ptr_->id());
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<ObjectID>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::AddEdgesToFragment(
    std::shared_ptr<ArrowFragmentBase> frag) {
  std::vector<std::set<std::pair<std::string, std::string>>> edge_relations(
      edge_label_num_);
  int pre_vlabel_num = frag->schema().all_vertex_label_num();
  int pre_elabel_num = frag->schema().all_edge_label_num();
  std::map<label_id_t, std::shared_ptr<arrow::Table>> edge_tables_map;
  for (size_t i = 0; i < output_edge_tables_.size(); ++i) {
    edge_tables_map[pre_elabel_num + i] = output_edge_tables_[i];
  }

  vertex_labels_.resize(pre_vlabel_num);
  for (auto& pair : vertex_label_to_index_) {
    vertex_labels_[pair.second] = pair.first;
  }
  for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
    for (auto& pair : edge_relations_[e_label]) {
      std::string src_label = vertex_labels_[pair.first];
      std::string dst_label = vertex_labels_[pair.second];
      edge_relations[e_label].insert({src_label, dst_label});
    }
  }
  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();
  return frag->AddEdges(client_, std::move(edge_tables_map), edge_relations,
                        thread_num);
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<ObjectID>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::
    AddVerticesAndEdgesToFragment(std::shared_ptr<ArrowFragmentBase> frag) {
  // handle two special cases
  if (output_edge_tables_.empty()) {
    return AddVerticesToFragment(frag);
  }
  if (output_vertex_tables_.empty()) {
    return AddEdgesToFragment(frag);
  }

  int pre_vlabel_num = frag->schema().all_vertex_label_num();
  int pre_elabel_num = frag->schema().all_edge_label_num();
  std::map<label_id_t, std::shared_ptr<arrow::Table>> vertex_tables_map;
  for (size_t i = 0; i < output_vertex_tables_.size(); ++i) {
    vertex_tables_map[pre_vlabel_num + i] = output_vertex_tables_[i];
  }

  std::map<label_id_t, std::shared_ptr<arrow::Table>> edge_tables_map;
  for (size_t i = 0; i < output_edge_tables_.size(); ++i) {
    edge_tables_map[pre_elabel_num + i] = output_edge_tables_[i];
  }

  // The size is all vertex label number.
  vertex_labels_.resize(pre_vlabel_num + output_vertex_tables_.size());
  for (auto& pair : vertex_label_to_index_) {
    vertex_labels_[pair.second] = pair.first;
  }
  std::vector<std::set<std::pair<std::string, std::string>>> edge_relations(
      edge_label_num_);
  for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
    for (auto& pair : edge_relations_[e_label]) {
      std::string src_label = vertex_labels_[pair.first];
      std::string dst_label = vertex_labels_[pair.second];
      edge_relations[e_label].insert({src_label, dst_label});
    }
  }
  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();
  return frag->AddVerticesAndEdges(
      client_, std::move(vertex_tables_map), std::move(edge_tables_map),
      vm_ptr_ ? vm_ptr_->id() : local_vm_ptr_->id(), edge_relations,
      thread_num);
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<ObjectID>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::ConstructFragment() {
  int concurrency =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();
  PropertyGraphSchema schema;
  BOOST_LEAF_CHECK(initSchema(schema));

  VLOG(100) << "Start constructing fragment: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();
#define CONSTRUCT_FRAGMENT_BODY                                               \
  frag_builder.SetPropertyGraphSchema(std::move(schema));                     \
  BOOST_LEAF_CHECK(frag_builder.Init(                                         \
      comm_spec_.fid(), comm_spec_.fnum(), std::move(output_vertex_tables_),  \
      std::move(output_edge_tables_), directed_, concurrency));               \
  VLOG(100) << "Finished fragment builder construction: " << get_rss_pretty() \
            << ", peak: " << get_peak_rss_pretty();                           \
                                                                              \
  std::shared_ptr<Object> fragment_object;                                    \
  VY_OK_OR_RAISE(frag_builder.Seal(client_, fragment_object));                \
  VLOG(100) << "Finished fragment builder seal: " << get_rss_pretty()         \
            << ", peak: " << get_peak_rss_pretty();                           \
                                                                              \
  VY_OK_OR_RAISE(client_.Persist(fragment_object->id()));                     \
  return fragment_object->id()

  if (!local_vertex_map_ && !compact_edges_) {
    BasicArrowFragmentBuilder<oid_t, vid_t, vertex_map_t, false> frag_builder(
        client_, vm_ptr_);
    CONSTRUCT_FRAGMENT_BODY;
  } else if (!local_vertex_map_ && compact_edges_) {
    BasicArrowFragmentBuilder<oid_t, vid_t, vertex_map_t, true> frag_builder(
        client_, vm_ptr_);
    CONSTRUCT_FRAGMENT_BODY;
  } else if (local_vertex_map_ && !compact_edges_) {
    BasicArrowFragmentBuilder<oid_t, vid_t, local_vertex_map_t, false>
        frag_builder(client_, local_vm_ptr_);
    CONSTRUCT_FRAGMENT_BODY;
  } else {
    BasicArrowFragmentBuilder<oid_t, vid_t, local_vertex_map_t, true>
        frag_builder(client_, local_vm_ptr_);
    CONSTRUCT_FRAGMENT_BODY;
  }
#undef CONSTRUCT_FRAGMENT_BODY
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<std::shared_ptr<ITablePipeline>>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::edgesId2Gid(
    const std::shared_ptr<ITablePipeline> edge_table, label_id_t src_label,
    label_id_t dst_label) {
  std::shared_ptr<arrow::Field> src_gid_field = std::make_shared<arrow::Field>(
      "src", vineyard::ConvertToArrowType<vid_t>::TypeValue());
  std::shared_ptr<arrow::Field> dst_gid_field = std::make_shared<arrow::Field>(
      "dst", vineyard::ConvertToArrowType<vid_t>::TypeValue());

  auto fn = [this, src_label, dst_label, src_gid_field, dst_gid_field](
                const std::shared_ptr<arrow::RecordBatch>& from, std::mutex& mu,
                std::nullptr_t&,
                std::shared_ptr<arrow::RecordBatch>& to) -> Status {
    to = from;  // initializing
    {
      std::shared_ptr<arrow::Array> src_gid_array;
      RETURN_ON_ERROR(this->parseOidChunkedArrayChunk(
          src_label, to->column(src_column), src_gid_array));
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(
          to, to->SetColumn(src_column, src_gid_field, src_gid_array));
    }
    {
      std::shared_ptr<arrow::Array> dst_gid_array;
      RETURN_ON_ERROR(this->parseOidChunkedArrayChunk(
          dst_label, to->column(dst_column), dst_gid_array));
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(
          to, to->SetColumn(dst_column, dst_gid_field, dst_gid_array));
    }
    return Status::OK();
  };
  auto schema = edge_table->schema();
  ARROW_OK_ASSIGN_OR_RAISE(schema, schema->SetField(src_column, src_gid_field));
  ARROW_OK_ASSIGN_OR_RAISE(schema, schema->SetField(dst_column, dst_gid_field));
  return std::make_shared<MapTablePipeline<std::nullptr_t>>(edge_table, fn,
                                                            nullptr, schema);
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
Status BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::parseOidChunkedArray(
    label_id_t label_id,
    const std::shared_ptr<arrow::ChunkedArray> oid_arrays_in,
    std::shared_ptr<arrow::ChunkedArray>& out) {
  size_t chunk_num = oid_arrays_in->num_chunks();
  std::vector<std::shared_ptr<arrow::Array>> chunks_out(chunk_num);

  auto parsefn = [&](const size_t chunk_index,
                     const std::shared_ptr<arrow::Array> oid_array) -> Status {
    return parseOidChunkedArrayChunk(label_id, oid_array,
                                     chunks_out[chunk_index]);
  };

  ThreadGroup tg(comm_spec_);
  for (size_t chunk_index = 0; chunk_index < chunk_num; ++chunk_index) {
    std::shared_ptr<arrow::Array> oid_array = oid_arrays_in->chunk(chunk_index);
    tg.AddTask(parsefn, chunk_index, oid_array);
  }

  Status status;
  for (auto& status : tg.TakeResults()) {
    status += status;
  }
  RETURN_ON_ERROR(status);
  out = std::make_shared<arrow::ChunkedArray>(chunks_out);
  return Status::OK();
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
Status
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::parseOidChunkedArrayChunk(
    label_id_t label_id, const std::shared_ptr<arrow::Array> oid_arrays_in,
    std::shared_ptr<arrow::Array>& out) {
  std::shared_ptr<oid_array_t> oid_array =
      std::dynamic_pointer_cast<oid_array_t>(oid_arrays_in);
  vertex_map_t* vm = vm_ptr_.get();
  local_vertex_map_t* local_vm = local_vm_ptr_.get();

  // prepare buffer
  std::unique_ptr<arrow::Buffer> buffer;
  {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        buffer, arrow::AllocateBuffer(oid_array->length() * sizeof(vid_t)));
  }

  vid_t* builder = reinterpret_cast<vid_t*>(buffer->mutable_data());
  for (int64_t k = 0; k != oid_array->length(); ++k) {
    internal_oid_t oid = oid_array->GetView(k);
    fid_t fid = partitioner_.GetPartitionId(oid);
    bool found = false;
    if (vm != nullptr) {
      found = vm->GetGid(fid, label_id, oid, builder[k]);
    } else {
      found = local_vm->GetGid(fid, label_id, oid, builder[k]);
    }
    if (!found) {
      std::stringstream buffer;
      buffer << "Mapping vertex '" << oid << "' failed. All src/dst in edges "
             << "must present in corresponding vertices first";
      std::string error_message = buffer.str();
      LOG(ERROR) << error_message;
      return Status::Invalid(error_message);
    }
  }
  out = std::make_shared<ArrowArrayType<VID_T>>(
      oid_array->length(), std::shared_ptr<arrow::Buffer>(std::move(buffer)),
      nullptr, 0);
  return Status::OK();
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::initSchema(
    PropertyGraphSchema& schema) {
  schema.set_fnum(comm_spec_.fnum());
  for (label_id_t v_label = 0; v_label != vertex_label_num_; ++v_label) {
    std::string vertex_label = vertex_labels_[v_label];
    auto entry =
        schema.CreateEntry(vertex_label, PropertyGraphSchema::VERTEX_TYPE_NAME);

    auto table = output_vertex_tables_[v_label];

    if (retain_oid_) {
      int col_id = table->num_columns() - 1;
      entry->AddPrimaryKey(table->schema()->field(col_id)->name());
    }

    for (int i = 0; i < table->num_columns(); ++i) {
      entry->AddProperty(table->schema()->field(i)->name(),
                         table->schema()->field(i)->type());
    }
  }
  for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
    std::string edge_label = edge_labels_[e_label];
    auto entry =
        schema.CreateEntry(edge_label, PropertyGraphSchema::EDGE_TYPE_NAME);

    auto& relation_set = edge_relations_[e_label];
    for (auto& pair : relation_set) {
      std::string src_label = vertex_labels_[pair.first];
      std::string dst_label = vertex_labels_[pair.second];
      entry->AddRelation(src_label, dst_label);
    }

    auto table = output_edge_tables_[e_label];

    for (int i = 2; i < table->num_columns(); ++i) {
      entry->AddProperty(table->schema()->field(i)->name(),
                         table->schema()->field(i)->type());
    }
  }
  std::string message;
  if (!schema.Validate(message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, message);
  }
  return {};
}

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::generateEdgeId(
    const grape::CommSpec& comm_spec,
    std::vector<std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                      std::shared_ptr<ITablePipeline>>>>&
        edge_tables,
    int label_offset) {
  label_id_t edge_label_num = edge_tables.size();
  IdParser<uint64_t> eid_parser;
  eid_parser.Init(comm_spec.fnum(), edge_label_num + label_offset);

  auto eid_field = std::make_shared<arrow::Field>(
      "eid", ConvertToArrowType<int64_t>::TypeValue());

  auto genfn = [eid_field](const std::shared_ptr<arrow::RecordBatch>& from,
                           std::mutex& mu, int64_t& current_edge_id,
                           std::shared_ptr<arrow::RecordBatch>& to) -> Status {
    int64_t chunk_size = from->num_rows();
    int64_t edge_id_from = 0;
    {  // get and update the state
      std::lock_guard<std::mutex> lock(mu);
      edge_id_from = current_edge_id;
      current_edge_id += chunk_size;
    }
    std::unique_ptr<arrow::Buffer> buffer;
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        buffer, arrow::AllocateBuffer(from->num_rows() * sizeof(int64_t)));
    int64_t* edge_ids = reinterpret_cast<int64_t*>(buffer->mutable_data());
    for (int64_t i = 0; i != chunk_size; ++i) {
      edge_ids[i] = edge_id_from++;
    }
    std::shared_ptr<arrow::Array> eid_array =
        std::make_shared<arrow::Int64Array>(
            from->num_rows(), std::shared_ptr<arrow::Buffer>(std::move(buffer)),
            nullptr, 0, 0);
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        to, from->AddColumn(edge_id_column, eid_field, eid_array));
    return Status::OK();
  };

  for (label_id_t e_label = 0; e_label < edge_label_num; ++e_label) {
    auto& edge_table_list = edge_tables[e_label];
    uint64_t cur_id =
        eid_parser.GenerateId(comm_spec.fid(), e_label + label_offset, 0);
    for (size_t edge_table_index = 0;
         edge_table_index != edge_table_list.size(); ++edge_table_index) {
      auto& edge_table = edge_table_list[edge_table_index].second;
      std::shared_ptr<arrow::Schema> schema;
      ARROW_OK_ASSIGN_OR_RAISE(
          schema, edge_table->schema()->AddField(edge_id_column, eid_field));

      edge_table = std::make_shared<MapTablePipeline<int64_t>>(
          edge_table, genfn, cur_id, schema);
    }
  }
  return {};
}

// constructVertices implementation for ArrowVertexMap
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::constructVerticesImpl(
    ObjectID vm_id) {
  VLOG(100) << "Starting constructing vertices: " << get_rss_pretty();
  std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>> oid_lists(
      vertex_label_num_);

  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    auto vertex_table = ordered_vertex_tables_[v_label];
    ordered_vertex_tables_[v_label].reset();  // release memory

    auto shuffle_procedure =
        [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
      BOOST_LEAF_AUTO(table, ShufflePropertyVertexTable<partitioner_t>(
                                 comm_spec_, partitioner_, vertex_table));
      VLOG(100) << "[worker-" << comm_spec_.worker_id()
                << "] shuffled vertex table size for label " << v_label << ": "
                << table->num_rows();

      std::vector<std::shared_ptr<arrow::ChunkedArray>> shuffled_oid_array;
      auto local_oid_array = table->column(id_column);
      VY_OK_OR_RAISE(FragmentAllGatherArray(comm_spec_, local_oid_array,
                                            shuffled_oid_array));
      for (auto const& array : shuffled_oid_array) {
        oid_lists[v_label].emplace_back(
            std::dynamic_pointer_cast<arrow::ChunkedArray>(array));
      }

      auto id_field = table->schema()->field(id_column);
      auto id_array = table->column(id_column);
      CHECK_ARROW_ERROR_AND_ASSIGN(table, table->RemoveColumn(id_column));
      if (retain_oid_) {  // move the id column to the end
        CHECK_ARROW_ERROR_AND_ASSIGN(
            table, table->AddColumn(table->num_columns(), id_field, id_array));
      }
      return table;
    };
    BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));

    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append("label", vertex_labels_[v_label]);
    metadata->Append("label_id", std::to_string(v_label));
    metadata->Append("type", PropertyGraphSchema::VERTEX_TYPE_NAME);
    metadata->Append("retain_oid", std::to_string(retain_oid_));
    output_vertex_tables_[v_label] = table->ReplaceSchemaMetadata(metadata);
  }
  VLOG(100) << "Constructing vertices: after shuffle: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  ObjectID new_vm_id = InvalidObjectID();
  if (vm_id == InvalidObjectID()) {
    // fill oid array.
    BasicArrowVertexMapBuilder<internal_oid_t, vid_t> vm_builder(
        client_, comm_spec_.fnum(), vertex_label_num_, std::move(oid_lists),
        use_perfect_hash_);
    // oid_lists.clear();

    // vm_object -> vertex map
    std::shared_ptr<Object> vm_object;
    VY_OK_OR_RAISE(vm_builder.Seal(client_, vm_object));
    new_vm_id = vm_object->id();
  } else {
    auto old_vm_ptr =
        std::dynamic_pointer_cast<vertex_map_t>(client_.GetObject(vm_id));
    label_id_t pre_label_num = old_vm_ptr->label_num();
    std::map<label_id_t, std::vector<std::shared_ptr<arrow::ChunkedArray>>>
        oid_lists_map;
    for (size_t i = 0; i < oid_lists.size(); ++i) {
      oid_lists_map[pre_label_num + i] = oid_lists[i];
    }
    // No new vertices.
    if (oid_lists_map.empty()) {
      new_vm_id = vm_id;
    } else {
      new_vm_id = old_vm_ptr->AddVertices(client_, std::move(oid_lists_map));
    }
  }
  vm_ptr_ =
      std::dynamic_pointer_cast<vertex_map_t>(client_.GetObject(new_vm_id));

  VLOG(100) << "Constructing vertices: after constructing vertex map: "
            << get_rss_pretty() << ", peak = " << get_peak_rss_pretty();
  return {};
}

// constructVertices implementation for ArrowLocalVertexMap
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::constructVerticesImplLocal(
    ObjectID vm_id) {
  if (vm_id != InvalidObjectID()) {
    auto old_vm_ptr =
        std::dynamic_pointer_cast<local_vertex_map_t>(client_.GetObject(vm_id));
    label_id_t pre_label_num = old_vm_ptr->label_num();
    if (pre_label_num != 0) {
      std::string msg =
          "Not support adding vertex label to existing labels with local "
          "vertex map";
      RETURN_GS_ERROR(ErrorCode::kUnsupportedOperationError, msg);
    }
  }
  local_vm_builder_ =
      std::make_shared<ArrowLocalVertexMapBuilder<internal_oid_t, vid_t>>(
          client_, comm_spec_.fnum(), comm_spec_.fid(), vertex_label_num_);

  std::vector<std::shared_ptr<arrow::ChunkedArray>> local_oid_array(
      vertex_label_num_);
  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    auto vertex_table = ordered_vertex_tables_[v_label];
    auto shuffle_procedure =
        [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
      BOOST_LEAF_AUTO(table, ShufflePropertyVertexTable<partitioner_t>(
                                 comm_spec_, partitioner_, vertex_table));
      VLOG(100) << "[worker-" << comm_spec_.worker_id()
                << "] shuffled vertex table size for label " << v_label << ": "
                << table->num_rows();

      local_oid_array[v_label] = table->column(id_column);

      // TODO: check about add a new label on old vertex map
      // local_vm_builder_->AddLocalVertices(v_label, local_oid_array);

      auto id_field = table->schema()->field(id_column);
      auto id_array = table->column(id_column);
      CHECK_ARROW_ERROR_AND_ASSIGN(table, table->RemoveColumn(id_column));
      if (retain_oid_) {
        CHECK_ARROW_ERROR_AND_ASSIGN(
            table, table->AddColumn(table->num_columns(), id_field, id_array));
      }
      return table;
    };
    BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));

    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append("label", vertex_labels_[v_label]);
    metadata->Append("label_id", std::to_string(v_label));
    metadata->Append("type", PropertyGraphSchema::VERTEX_TYPE_NAME);
    metadata->Append("retain_oid", std::to_string(retain_oid_));
    output_vertex_tables_[v_label] = table->ReplaceSchemaMetadata(metadata);
  }
  VY_OK_OR_RAISE(local_vm_builder_->AddLocalVertices(
      comm_spec_, std::move(local_oid_array)));
  local_oid_array.clear();
  ordered_vertex_tables_.clear();
  return {};
}

// constructEdges implementation for ArrowVertexMap
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::constructEdgesImpl(
    int label_offset, int vertex_label_num) {
  vineyard::IdParser<vid_t> id_parser;
  id_parser.Init(comm_spec_.fnum(), vertex_label_num);

  output_edge_tables_.resize(edge_label_num_);

  for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
    auto& edge_table_list = ordered_edge_tables_[e_label];
    auto shuffle_procedure =
        [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
      std::vector<std::shared_ptr<ITablePipeline>> processed_table_list;
      for (size_t i = 0; i < edge_table_list.size(); ++i) {
        label_id_t src_label = edge_table_list[i].first.first;
        label_id_t dst_label = edge_table_list[i].first.second;
        BOOST_LEAF_AUTO(table, edgesId2Gid(edge_table_list[i].second, src_label,
                                           dst_label));
        edge_table_list[i].second.reset();  // ensure release memory
        processed_table_list.emplace_back(table);
      }

      std::shared_ptr<ITablePipeline> table =
          std::make_shared<ConcatTablePipeline>(processed_table_list);
      // Shuffle the edge table with gid
      BOOST_LEAF_AUTO(
          table_out, ShufflePropertyEdgeTable<vid_t>(
                         comm_spec_, id_parser, src_column, dst_column, table));
      VLOG(100) << "[worker-" << comm_spec_.worker_id()
                << "] shuffled edge table size for label " << e_label << ": "
                << table_out->num_rows();
      return table_out;
    };
    BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));

    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append("label", edge_labels_[e_label]);
    metadata->Append("label_id", std::to_string(e_label));
    metadata->Append("type", PropertyGraphSchema::EDGE_TYPE_NAME);
    output_edge_tables_[e_label] = table->ReplaceSchemaMetadata(metadata);
    ordered_edge_tables_[e_label].clear();
  }
  return {};
}

// constructEdges implementation for ArrowLocalVertexMap
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
boost::leaf::result<void>
BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>::constructEdgesImplLocal(
    int label_offset, int vertex_label_num) {
  vineyard::IdParser<vid_t> id_parser;
  id_parser.Init(comm_spec_.fnum(), vertex_label_num);

  output_edge_tables_.resize(edge_label_num_);

  std::vector<std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                    std::shared_ptr<arrow::Table>>>>
      shuffled_tables(edge_label_num_);

  // Shuffle property tables
  for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
    std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                          std::shared_ptr<arrow::Table>>>
        shuffled;
    for (auto& item : ordered_edge_tables_[e_label]) {
      auto shuffle_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        BOOST_LEAF_AUTO(
            table_out,
            ShufflePropertyEdgeTableByPartition<partitioner_t>(
                comm_spec_, partitioner_, src_column, dst_column, item.second));
        VLOG(100) << "[worker-" << comm_spec_.worker_id()
                  << "] shuffled edge table size for label " << e_label << ": "
                  << table_out->num_rows();
        return table_out;
      };
      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));
      shuffled.emplace_back(item.first, table);
    }
    ordered_edge_tables_[e_label].clear();  // release those table pipelines
    shuffled_tables[e_label] = shuffled;
  }

  // Finish the local vertex map with outer vertices
  VLOG(100) << "Finalizing local vertex map: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  // outer_vertex_oid_sets[fid][vlabel_id]
  std::vector<std::vector<std::shared_ptr<ConcurrentOidSet<oid_t>>>>
      outer_vertex_oid_sets(comm_spec_.fnum());
  // collected unique oid arrays: outer_vertex_oids[fid][vlabel_id]
  std::vector<std::vector<std::shared_ptr<oid_array_t>>> outer_vertex_oids(
      comm_spec_.fnum());

  for (fid_t fid = 0; fid < comm_spec_.fnum(); fid++) {
    outer_vertex_oid_sets[fid].resize(vertex_label_num_);
    for (label_id_t label_id = 0; label_id < vertex_label_num_; ++label_id) {
      outer_vertex_oid_sets[fid][label_id] =
          std::make_shared<ConcurrentOidSet<oid_t>>();
    }
    outer_vertex_oids[fid].resize(vertex_label_num_);
  }

  ThreadGroup tg(comm_spec_);
  auto fn = [&](const label_id_t label_id,
                std::shared_ptr<arrow::Array> chunk) -> Status {
    auto array = std::dynamic_pointer_cast<oid_array_t>(chunk);
    for (int64_t index = 0; index < array->length(); ++index) {
      internal_oid_t internal_oid = array->GetView(index);
      fid_t fid = partitioner_.GetPartitionId(internal_oid);
      if (fid != comm_spec_.fid()) {
        outer_vertex_oid_sets[fid][label_id]->Insert(internal_oid);
      }
    }
    return Status::OK();
  };

  for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
    for (auto& item : shuffled_tables[e_label]) {
      auto src_label = item.first.first;
      auto dst_label = item.first.second;
      auto table = item.second;

      auto srcs = table->column(src_column);
      auto dsts = table->column(dst_column);

      for (int64_t chunk = 0; chunk < srcs->num_chunks(); ++chunk) {
        tg.AddTask(fn, src_label, srcs->chunk(chunk));
      }
      for (int64_t chunk = 0; chunk < srcs->num_chunks(); ++chunk) {
        tg.AddTask(fn, dst_label, dsts->chunk(chunk));
      }
    }
  }
  {
    Status status;
    for (auto const& s : tg.TakeResults()) {
      status += s;
    }
    VY_OK_OR_RAISE(status);
  }

  // copy set to oid array
  ThreadGroup tgcopy(comm_spec_);
  auto copyfn = [&](const fid_t fid, const label_id_t label_id) -> Status {
    RETURN_ON_ERROR(outer_vertex_oid_sets[fid][label_id]->ToArray(
        outer_vertex_oids[fid][label_id]));
    outer_vertex_oid_sets[fid][label_id].reset();  // release the memory
    return Status::OK();
  };

  for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
    for (label_id_t label_id = 0; label_id < vertex_label_num_; label_id++) {
      tgcopy.AddTask(copyfn, fid, label_id);
    }
  }
  {
    Status status;
    for (auto const& s : tgcopy.TakeResults()) {
      status += s;
    }
    VY_OK_OR_RAISE(status);
  }

  VLOG(100) << "Finish collect outer vertices: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  // Exchange the outer vertex oid index between workers
  int worker_id = comm_spec_.worker_id();
  int worker_num = comm_spec_.worker_num();
  std::vector<std::vector<std::vector<vid_t>>> index_lists(comm_spec_.fnum());

  std::thread request_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int dst_worker_id = (worker_id + i) % worker_num;
      fid_t fid = comm_spec_.WorkerToFrag(dst_worker_id);
      for (label_id_t label_id = 0; label_id < vertex_label_num_; ++label_id) {
        SendArrowArray<oid_array_t>(outer_vertex_oids[fid][label_id],
                                    dst_worker_id, comm_spec_.comm(), 0);
      }
      grape::sync_comm::Recv(index_lists[fid], dst_worker_id, 1,
                             comm_spec_.comm());
    }
  });
  std::thread response_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int src_worker_id = (worker_id + worker_num - i) % worker_num;
      std::vector<std::shared_ptr<oid_array_t>> oids(vertex_label_num_);
      for (label_id_t label_id = 0; label_id < vertex_label_num_; ++label_id) {
        RecvArrowArray<oid_array_t>(oids[label_id], src_worker_id,
                                    comm_spec_.comm(), 0);
      }
      std::vector<std::vector<vid_t>> index_list;
      VINEYARD_DISCARD(local_vm_builder_->GetIndexOfOids(oids, index_list));
      grape::sync_comm::Send(index_list, src_worker_id, 1, comm_spec_.comm());
    }
  });

  request_thread.join();
  response_thread.join();
  MPI_Barrier(comm_spec_.comm());
  // Construct the outer vertex map with response o2i
  VY_OK_OR_RAISE(
      local_vm_builder_->template AddOuterVerticesMapping<internal_oid_t>(
          std::move(outer_vertex_oids), std::move(index_lists)));

  VLOG(100) << "Finish adding outer vertices mapping: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  std::shared_ptr<Object> vm_object;
  VY_OK_OR_RAISE(local_vm_builder_->Seal(client_, vm_object));
  local_vm_ptr_ = std::dynamic_pointer_cast<local_vertex_map_t>(vm_object);

  // Concatenate and add metadata to final edge tables
  VLOG(100) << "Transforming ids of edge tables and concatenate them: "
            << get_rss_pretty() << ", peak = " << get_peak_rss_pretty();
  size_t concurrency =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();

  for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
    std::vector<std::shared_ptr<ITablePipeline>> processed_table_list;
    std::shared_ptr<arrow::Schema> schema;
    for (auto& item : shuffled_tables[e_label]) {
      BOOST_LEAF_AUTO(
          table,
          edgesId2Gid(std::make_shared<TablePipeline>(std::move(item.second)),
                      item.first.first, item.first.second));
      schema = table->schema();
      processed_table_list.emplace_back(table);
    }
    shuffled_tables[e_label].clear();

    TablePipelineSink sink(
        std::make_shared<ConcatTablePipeline>(processed_table_list), schema,
        concurrency);
    std::shared_ptr<arrow::Table> table;
    VY_OK_OR_RAISE(sink.Result(table));

    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append("label", edge_labels_[e_label]);
    metadata->Append("label_id", std::to_string(e_label));
    metadata->Append("type", PropertyGraphSchema::EDGE_TYPE_NAME);
    output_edge_tables_[e_label] = table->ReplaceSchemaMetadata(metadata);
  }
  return {};
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_IMPL_H_
