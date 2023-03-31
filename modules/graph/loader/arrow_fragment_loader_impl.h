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

#ifndef MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_IMPL_H_
#define MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_IMPL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/worker/comm_spec.h"

#include "basic/ds/dataframe.h"
#include "basic/ds/tensor.h"
#include "client/client.h"
#include "io/io/io_factory.h"

#include "graph/fragment/property_graph_utils.h"
#include "graph/loader/arrow_fragment_loader.h"
#include "graph/loader/fragment_loader_utils.h"
#include "graph/utils/thread_group.h"

namespace vineyard {

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadFragment() {
  BOOST_LEAF_CHECK(initPartitioner());
  BOOST_LEAF_AUTO(raw_v_e_tables, LoadVertexEdgeTables());
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after loading tables: " << get_rss_pretty();

  return LoadFragment(std::move(raw_v_e_tables));
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadFragment(
    const std::vector<std::string>& efiles,
    const std::vector<std::string>& vfiles) {
  this->efiles_ = efiles;
  this->vfiles_ = vfiles;
  return this->LoadFragment();
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadFragment(
    std::pair<table_vec_t, std::vector<table_vec_t>> raw_v_e_tables) {
  auto& partial_v_tables = raw_v_e_tables.first;
  auto& partial_e_tables = raw_v_e_tables.second;

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "PROCESS-INPUTS-0";
  BOOST_LEAF_AUTO(v_e_tables,
                  preprocessInputs(partial_v_tables, partial_e_tables));
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "PROCESS-INPUTS-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after normalize tables: " << get_rss_pretty();

  // clear after grouped by labels
  partial_v_tables.clear();
  partial_e_tables.clear();

  auto& vertex_tables_with_label = v_e_tables.first;
  auto& edge_tables_with_label = v_e_tables.second;

  auto basic_fragment_loader = std::make_shared<basic_fragment_loader_t>(
      client_, comm_spec_, partitioner_, directed_, generate_eid_, retain_oid_);

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-0";
  for (auto const& pair : vertex_tables_with_label) {
    BOOST_LEAF_CHECK(
        basic_fragment_loader->AddVertexTable(pair.first, pair.second));
  }

  vertex_tables_with_label.clear();
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after freeing vertex tables: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-50";
  BOOST_LEAF_CHECK(basic_fragment_loader->ConstructVertices());
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after constructing vertices: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-EDGE-0";
  for (auto& table : edge_tables_with_label) {
    BOOST_LEAF_CHECK(basic_fragment_loader->AddEdgeTable(
        table.src_label, table.dst_label, table.edge_label, table.table));
  }

  edge_tables_with_label.clear();
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after freeing edge tables: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-EDGE-50";
  BOOST_LEAF_CHECK(basic_fragment_loader->ConstructEdges());
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-EDGE-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after constructing edges: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "SEAL-0";
  return basic_fragment_loader->ConstructFragment();
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadFragmentAsFragmentGroup() {
  BOOST_LEAF_AUTO(frag_id, LoadFragment());
  auto frag =
      std::dynamic_pointer_cast<ArrowFragment<OID_T, VID_T, vertex_map_t>>(
          client_.GetObject(frag_id));
  if (frag == nullptr) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "fragment is null, means it is failed to be constructed");
  }

  BOOST_LEAF_AUTO(group_id,
                  ConstructFragmentGroup(client_, frag_id, comm_spec_));
  return group_id;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::initPartitioner() {
#ifdef HASH_PARTITION
  partitioner_.Init(comm_spec_.fnum());
#else
  if (vfiles_.empty()) {
    RETURN_GS_ERROR(ErrorCode::kInvalidOperationError,
                    "Segmented partitioner is not supported when the v-file is "
                    "not provided");
  }
  std::vector<std::shared_ptr<arrow::Table>> vtables;
  {
    BOOST_LEAF_AUTO(tmp, loadVertexTables(vfiles_, 0, 1));
    vtables = tmp;
  }
  std::vector<oid_t> oid_list;

  for (auto& table : vtables) {
    std::shared_ptr<arrow::ChunkedArray> oid_array_chunks =
        table->column(id_column);
    size_t chunk_num = oid_array_chunks->num_chunks();

    for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
      std::shared_ptr<oid_array_t> array =
          std::dynamic_pointer_cast<oid_array_t>(
              oid_array_chunks->chunk(chunk_i));
      int64_t length = array->length();
      for (int64_t i = 0; i < length; ++i) {
        oid_list.emplace_back(oid_t(array->GetView(i)));
      }
    }
  }

  partitioner_.Init(comm_spec_.fnum(), oid_list);
#endif
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadFragmentAsFragmentGroup(
    const std::vector<std::string>& efiles,
    const std::vector<std::string>& vfiles) {
  this->efiles_ = efiles;
  this->vfiles_ = vfiles;
  return this->LoadFragmentAsFragmentGroup();
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<std::pair<table_vec_t, std::vector<table_vec_t>>>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadVertexEdgeTables() {
  BOOST_LEAF_AUTO(v_tables, LoadVertexTables());
  BOOST_LEAF_AUTO(e_tables, LoadEdgeTables());
  return std::make_pair(v_tables, e_tables);
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<table_vec_t>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadVertexTables() {
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-VERTEX-0";
  table_vec_t v_tables;
  if (!vfiles_.empty()) {
    auto load_v_procedure = [&]() {
      return loadVertexTables(vfiles_, comm_spec_.local_id(),
                              comm_spec_.local_num());
    };
    BOOST_LEAF_ASSIGN(v_tables,
                      vineyard::sync_gs_error(comm_spec_, load_v_procedure));
  } else if (!partial_v_tables_.empty()) {
    v_tables = std::move(partial_v_tables_);
    partial_v_tables_.clear();
  }
  for (const auto& table : v_tables) {
    BOOST_LEAF_CHECK(sanityChecks(table));
  }
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-VERTEX-100";
  return v_tables;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<std::vector<table_vec_t>>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadEdgeTables() {
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-EDGE-0";
  std::vector<table_vec_t> e_tables;
  if (!efiles_.empty()) {
    auto load_e_procedure = [&]() {
      return loadEdgeTables(efiles_, comm_spec_.local_id(),
                            comm_spec_.local_num());
    };
    BOOST_LEAF_ASSIGN(e_tables,
                      vineyard::sync_gs_error(comm_spec_, load_e_procedure));
  } else if (!partial_e_tables_.empty()) {
    e_tables = std::move(partial_e_tables_);
    partial_e_tables_.clear();
  }
  for (const auto& table_vec : e_tables) {
    for (const auto& table : table_vec) {
      BOOST_LEAF_CHECK(sanityChecks(table));
    }
  }
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-EDGE-100";
  return e_tables;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::AddLabelsToFragment(
    vineyard::ObjectID frag_id) {
  BOOST_LEAF_CHECK(initPartitioner());
  BOOST_LEAF_AUTO(raw_v_e_tables, LoadVertexEdgeTables());
  return addVerticesAndEdges(frag_id, raw_v_e_tables);
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::
    AddLabelsToFragmentAsFragmentGroup(vineyard::ObjectID frag_id) {
  BOOST_LEAF_AUTO(new_frag_id, AddLabelsToFragment(frag_id));
  return vineyard::ConstructFragmentGroup(client_, new_frag_id, comm_spec_);
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::resolveVineyardObject(
    std::string const& source) {
  vineyard::ObjectID sourceId = vineyard::InvalidObjectID();
  // encoding: 'o' prefix for object id, and 's' prefix for object name.
  CHECK_OR_RAISE(!source.empty() && (source[0] == 'o' || source[0] == 's'));
  if (source[0] == 'o') {
    sourceId = vineyard::ObjectIDFromString(source.substr(1));
  } else {
    VY_OK_OR_RAISE(client_.GetName(source.substr(1), sourceId, true));
  }
  CHECK_OR_RAISE(sourceId != vineyard::InvalidObjectID());
  return sourceId;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::loadVertexTables(
    const std::vector<std::string>& files, int index, int total_parts) {
  auto label_num = static_cast<label_id_t>(files.size());
  std::vector<std::shared_ptr<arrow::Table>> tables(label_num);

  for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
    auto read_procedure =
        [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
      std::shared_ptr<arrow::Table> table;
      if (files[label_id].rfind("vineyard://", 0) == 0) {
        BOOST_LEAF_AUTO(sourceId,
                        resolveVineyardObject(files[label_id].substr(11)));
        VY_OK_OR_RAISE(ReadTableFromVineyard(client_, sourceId, table, index,
                                             total_parts));
      } else {
        VY_OK_OR_RAISE(
            ReadTableFromLocation(files[label_id], table, index, total_parts));
      }
      return table;
    };
    BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure));

    auto sync_schema_procedure =
        [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
      return SyncSchema(table, comm_spec_);
    };

    // normailize the schema of this distributed table
    BOOST_LEAF_AUTO(normalized_table,
                    sync_gs_error(comm_spec_, sync_schema_procedure));

    auto meta = normalized_table->schema()->metadata();
    if (meta == nullptr || meta->FindKey(LABEL_TAG) == -1) {
      RETURN_GS_ERROR(
          ErrorCode::kIOError,
          "Metadata of input vertex files should contain label name");
    }
    tables[label_id] = normalized_table;
  }
  return tables;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::loadEdgeTables(
    const std::vector<std::string>& files, int index, int total_parts) {
  auto label_num = static_cast<label_id_t>(files.size());
  std::vector<std::vector<std::shared_ptr<arrow::Table>>> tables(label_num);

  try {
    for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
      std::vector<std::string> sub_label_files;
      boost::split(sub_label_files, files[label_id], boost::is_any_of(";"));

      for (size_t j = 0; j < sub_label_files.size(); ++j) {
        auto read_procedure =
            [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
          std::shared_ptr<arrow::Table> table;
          if (files[label_id].rfind("vineyard://", 0) == 0) {
            BOOST_LEAF_AUTO(sourceId,
                            resolveVineyardObject(files[label_id].substr(11)));
            VY_OK_OR_RAISE(ReadTableFromVineyard(client_, sourceId, table,
                                                 index, total_parts));
          } else {
            VY_OK_OR_RAISE(
                ReadTableFromLocation(sub_label_files[j] + "#header_row=true",
                                      table, index, total_parts));
          }
          return table;
        };
        BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure));

        // normailize the schema of this distributed table
        auto sync_schema_procedure =
            [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
          return SyncSchema(table, comm_spec_);
        };
        BOOST_LEAF_AUTO(normalized_table,
                        sync_gs_error(comm_spec_, sync_schema_procedure));

        auto meta = normalized_table->schema()->metadata();
        if (meta == nullptr || meta->FindKey(LABEL_TAG) == -1) {
          RETURN_GS_ERROR(
              ErrorCode::kIOError,
              "Metadata of input edge files should contain label name");
        }
        if (meta == nullptr || meta->FindKey(SRC_LABEL_TAG) == -1) {
          RETURN_GS_ERROR(
              ErrorCode::kIOError,
              "Metadata of input edge files should contain src label name");
        }
        if (meta == nullptr || meta->FindKey(DST_LABEL_TAG) == -1) {
          RETURN_GS_ERROR(
              ErrorCode::kIOError,
              "Metadata of input edge files should contain dst label name");
        }
        tables[label_id].emplace_back(normalized_table);
      }
    }
  } catch (std::exception& e) {
    RETURN_GS_ERROR(ErrorCode::kIOError, std::string(e.what()));
  }
  return tables;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<
    std::pair<typename ArrowFragmentLoader<OID_T, VID_T,
                                           VERTEX_MAP_T>::vertex_table_info_t,
              typename ArrowFragmentLoader<OID_T, VID_T,
                                           VERTEX_MAP_T>::edge_table_info_t>>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::preprocessInputs(
    const std::vector<std::shared_ptr<arrow::Table>>& v_tables,
    const std::vector<std::vector<std::shared_ptr<arrow::Table>>>& e_tables,
    const std::set<std::string>& previous_vertex_labels) {
  vertex_table_info_t vertex_tables_with_label;
  edge_table_info_t edge_tables_with_label;
  std::set<std::string> deduced_labels;
  for (auto table : v_tables) {
    auto meta = table->schema()->metadata();
    if (meta == nullptr) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Metadata of input vertex files shouldn't be empty");
    }

    int label_meta_index = meta->FindKey(LABEL_TAG);
    if (label_meta_index == -1) {
      RETURN_GS_ERROR(
          ErrorCode::kInvalidValueError,
          "Metadata of input vertex files should contain label name");
    }
    std::string label_name = meta->value(label_meta_index);
    vertex_tables_with_label[label_name] = table;
  }

  auto label_not_exists = [&](const std::string& label) {
    return vertex_tables_with_label.find(label) ==
               vertex_tables_with_label.end() &&
           previous_vertex_labels.find(label) == previous_vertex_labels.end();
  };

  for (auto& table_vec : e_tables) {
    for (auto table : table_vec) {
      auto meta = table->schema()->metadata();
      int label_meta_index = meta->FindKey(LABEL_TAG);
      std::string label_name = meta->value(label_meta_index);
      int src_label_meta_index = meta->FindKey(SRC_LABEL_TAG);
      std::string src_label_name = meta->value(src_label_meta_index);
      int dst_label_meta_index = meta->FindKey(DST_LABEL_TAG);
      std::string dst_label_name = meta->value(dst_label_meta_index);
      edge_tables_with_label.emplace_back(src_label_name, dst_label_name,
                                          label_name, table);
      // Find vertex labels that need to be deduced, i.e. not assigned by user
      // directly

      if (label_not_exists(src_label_name)) {
        deduced_labels.insert(src_label_name);
      }
      if (label_not_exists(dst_label_name)) {
        deduced_labels.insert(dst_label_name);
      }
    }
  }

  if (!deduced_labels.empty()) {
    FragmentLoaderUtils<OID_T, partitioner_t> loader_utils(comm_spec_,
                                                           partitioner_);
    BOOST_LEAF_AUTO(vertex_labels,
                    loader_utils.GatherVertexLabels(edge_tables_with_label));
    BOOST_LEAF_AUTO(vertex_label_to_index,
                    loader_utils.GetVertexLabelToIndex(vertex_labels));
    BOOST_LEAF_AUTO(v_tables_map, loader_utils.BuildVertexTableFromEdges(
                                      edge_tables_with_label,
                                      vertex_label_to_index, deduced_labels));
    for (auto& pair : v_tables_map) {
      vertex_tables_with_label[pair.first] = pair.second;
    }
  }
  return std::make_pair(vertex_tables_with_label, edge_tables_with_label);
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::sanityChecks(
    std::shared_ptr<arrow::Table> table) {
  // We require that there are no identical column names
  auto names = table->ColumnNames();
  std::sort(names.begin(), names.end());
  const auto duplicate = std::adjacent_find(names.begin(), names.end());
  if (duplicate != names.end()) {
    auto meta = table->schema()->metadata();
    int label_meta_index = meta->FindKey(LABEL_TAG);
    std::string label_name = meta->value(label_meta_index);
    std::stringstream msg;
    msg << "Label " << label_name
        << " has identical property names, which is not allowed. The "
           "original names are: ";
    auto origin_names = table->ColumnNames();
    msg << "[";
    for (size_t i = 0; i < origin_names.size(); ++i) {
      if (i != 0) {
        msg << ", ";
      }
      msg << origin_names[i];
    }
    msg << "]";
    RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError, msg.str());
  }
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::addVerticesAndEdges(
    vineyard::ObjectID frag_id,
    std::pair<table_vec_t, std::vector<table_vec_t>> raw_v_e_tables) {
  auto& partial_v_tables = raw_v_e_tables.first;
  auto& partial_e_tables = raw_v_e_tables.second;

  auto frag = std::dynamic_pointer_cast<fragment_t>(client_.GetObject(frag_id));
  const PropertyGraphSchema& schema = frag->schema();

  std::map<std::string, label_id_t> vertex_label_to_index;
  std::set<std::string> previous_labels;

  for (auto& entry : schema.vertex_entries()) {
    vertex_label_to_index[entry.label] = entry.id;
    previous_labels.insert(entry.label);
  }

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "PROCESS-INPUTS-0";
  BOOST_LEAF_AUTO(
      v_e_tables,
      preprocessInputs(partial_v_tables, partial_e_tables, previous_labels));
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "PROCESS-INPUTS-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after normalize tables: " << get_rss_pretty();

  // clear after grouped by labels
  partial_v_tables.clear();
  partial_e_tables.clear();

  auto& vertex_tables_with_label = v_e_tables.first;
  auto& edge_tables_with_label = v_e_tables.second;

  auto basic_fragment_loader = std::make_shared<basic_fragment_loader_t>(
      client_, comm_spec_, partitioner_, directed_, generate_eid_, retain_oid_);

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-0";
  for (auto& pair : vertex_tables_with_label) {
    BOOST_LEAF_CHECK(
        basic_fragment_loader->AddVertexTable(pair.first, pair.second));
  }
  vertex_tables_with_label.clear();
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after freeing vertex tables: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  auto old_vm_ptr = frag->GetVertexMap();
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-50";
  BOOST_LEAF_CHECK(basic_fragment_loader->ConstructVertices(old_vm_ptr->id()));
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after constructing vertices: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  label_id_t pre_label_num = old_vm_ptr->label_num();

  auto new_labels_index = basic_fragment_loader->get_vertex_label_to_index();
  for (auto& pair : new_labels_index) {
    vertex_label_to_index[pair.first] = pair.second + pre_label_num;
  }
  basic_fragment_loader->set_vertex_label_to_index(
      std::move(vertex_label_to_index));
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-EDGE-0";
  for (auto& table : edge_tables_with_label) {
    BOOST_LEAF_CHECK(basic_fragment_loader->AddEdgeTable(
        table.src_label, table.dst_label, table.edge_label, table.table));
  }
  edge_tables_with_label.clear();
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after freeing edge tables: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-EDGE-50";
  BOOST_LEAF_CHECK(basic_fragment_loader->ConstructEdges(
      schema.all_edge_label_num(), schema.all_vertex_label_num()));
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-EDGE-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after constructing edges: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "SEAL-0";
  return basic_fragment_loader->AddVerticesAndEdgesToFragment(frag);
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_IMPL_H_
