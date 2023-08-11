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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BUILDER_IMPL_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BUILDER_IMPL_H_

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/fragment/fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/utils/vertex_array.h"

#include "client/ds/core_types.h"
#include "client/ds/object_meta.h"

#include "basic/ds/arrow.h"
#include "basic/ds/arrow_utils.h"
#include "common/util/functions.h"
#include "common/util/typename.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/fragment_traits.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"
#include "graph/utils/context_protocols.h"
#include "graph/utils/error.h"
#include "graph/utils/thread_group.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::AddVerticesAndEdges(
    Client& client,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& vertex_tables_map,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& edge_tables_map,
    ObjectID vm_id,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    const int concurrency) {
  int extra_vertex_label_num = vertex_tables_map.size();
  int total_vertex_label_num = vertex_label_num_ + extra_vertex_label_num;

  std::vector<std::shared_ptr<arrow::Table>> vertex_tables;
  vertex_tables.resize(extra_vertex_label_num);
  for (auto& pair : vertex_tables_map) {
    if (pair.first < vertex_label_num_ ||
        pair.first >= total_vertex_label_num) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid vertex label id: " + std::to_string(pair.first));
    }
    vertex_tables[pair.first - vertex_label_num_] = pair.second;
  }
  int extra_edge_label_num = edge_tables_map.size();
  int total_edge_label_num = edge_label_num_ + extra_edge_label_num;

  std::vector<std::shared_ptr<arrow::Table>> edge_tables;
  edge_tables.resize(extra_edge_label_num);
  for (auto& pair : edge_tables_map) {
    if (pair.first < edge_label_num_ || pair.first >= total_edge_label_num) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid edge label id: " + std::to_string(pair.first));
    }
    edge_tables[pair.first - edge_label_num_] = pair.second;
  }
  return AddNewVertexEdgeLabels(client, std::move(vertex_tables),
                                std::move(edge_tables), vm_id, edge_relations,
                                concurrency);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::AddVertices(
    Client& client,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& vertex_tables_map,
    ObjectID vm_id, const int concurrency) {
  int extra_vertex_label_num = vertex_tables_map.size();
  int total_vertex_label_num = vertex_label_num_ + extra_vertex_label_num;

  std::vector<std::shared_ptr<arrow::Table>> vertex_tables;
  vertex_tables.resize(extra_vertex_label_num);
  for (auto& pair : vertex_tables_map) {
    if (pair.first < vertex_label_num_ ||
        pair.first >= total_vertex_label_num) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid vertex label id: " + std::to_string(pair.first));
    }
    vertex_tables[pair.first - vertex_label_num_] = pair.second;
  }
  return AddNewVertexLabels(client, std::move(vertex_tables), vm_id);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::AddEdges(
    Client& client,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& edge_tables_map,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    const int concurrency) {
  int extra_edge_label_num = edge_tables_map.size();
  int total_edge_label_num = edge_label_num_ + extra_edge_label_num;

  std::vector<std::shared_ptr<arrow::Table>> edge_tables;
  edge_tables.resize(extra_edge_label_num);
  for (auto& pair : edge_tables_map) {
    if (pair.first < edge_label_num_ || pair.first >= total_edge_label_num) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid edge label id: " + std::to_string(pair.first));
    }
    edge_tables[pair.first - edge_label_num_] = pair.second;
  }
  return AddNewEdgeLabels(client, std::move(edge_tables), edge_relations,
                          concurrency);
}

/// Add a set of new vertex labels and a set of new edge labels to graph.
/// Vertex label id started from vertex_label_num_, and edge label id
/// started from edge_label_num_.
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::AddNewVertexEdgeLabels(
    Client& client, std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
    std::vector<std::shared_ptr<arrow::Table>>&& edge_tables, ObjectID vm_id,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    const int concurrency) {
  int extra_vertex_label_num = vertex_tables.size();
  int total_vertex_label_num = vertex_label_num_ + extra_vertex_label_num;
  int extra_edge_label_num = edge_tables.size();
  int total_edge_label_num = edge_label_num_ + extra_edge_label_num;

  // Init size
  auto vm_ptr =
      std::dynamic_pointer_cast<vertex_map_t>(client.GetObject(vm_id));

  std::vector<vid_t> ivnums(total_vertex_label_num);
  std::vector<vid_t> ovnums(total_vertex_label_num);
  std::vector<vid_t> tvnums(total_vertex_label_num);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    ivnums[i] = ivnums_[i];
  }
  for (size_t i = 0; i < vertex_tables.size(); ++i) {
    ivnums[vertex_label_num_ + i] =
        vm_ptr->GetInnerVertexSize(fid_, vertex_label_num_ + i);
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs(extra_edge_label_num),
      dsts(extra_edge_label_num);
  for (label_id_t label = 0; label < extra_edge_label_num; ++label) {
    srcs[label] = edge_tables[label]->column(0);
    dsts[label] = edge_tables[label]->column(1);
    // remove src and dst columns from edge tables
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[label],
                             edge_tables[label]->RemoveColumn(0));
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[label],
                             edge_tables[label]->RemoveColumn(0));
  }

  // Construct the new start value of lid of extra outer vertices
  std::vector<vid_t> start_ids(total_vertex_label_num);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]) + ovnums_[i];
  }
  for (label_id_t i = vertex_label_num_; i < total_vertex_label_num; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums[i]);
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new vertices and edges: before init the new vertex map: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  // Make a copy of ovg2l map, since we need to add some extra outer vertices
  // pulled in this fragment by new edges.
  std::vector<ovg2l_map_t> ovg2l_maps(total_vertex_label_num);
  for (int i = 0; i < vertex_label_num_; ++i) {
    for (auto iter = ovg2l_maps_ptr_[i]->begin();
         iter != ovg2l_maps_ptr_[i]->end(); ++iter) {
      ovg2l_maps[i].emplace(iter->first, iter->second);
    }
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] After init the new vertex map: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  std::vector<std::shared_ptr<vid_array_t>> extra_ovgid_lists(
      total_vertex_label_num);

  // Add extra outer vertices to ovg2l map, and collect distinct gid of extra
  // outer vertices.
  generate_outer_vertices_map(vid_parser_, fid_, total_vertex_label_num, srcs,
                              dsts, start_ids, ovg2l_maps, extra_ovgid_lists);

  VLOG(100)
      << "[frag-" << this->fid_
      << "] Add new vertices and edges: after generate_outer_vertices_map: "
      << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  std::vector<std::shared_ptr<vid_vineyard_builder_t>> ovgid_lists(
      total_vertex_label_num);

  // Append extra ovgid_lists with origin ovgid_lists to make it complete
  for (label_id_t i = 0; i < total_vertex_label_num; ++i) {
    // If the ovgid have no new entries, leave it empty to indicate using the
    // old ovgid when seal.
    if (extra_ovgid_lists[i]->length() != 0 || i >= vertex_label_num_) {
      std::vector<std::shared_ptr<vid_array_t>> chunks;
      if (i < vertex_label_num_) {
        chunks.push_back(ovgid_lists_[i]->GetArray());
        chunks.push_back(extra_ovgid_lists[i]);
      } else {
        chunks.push_back(extra_ovgid_lists[i]);
      }
      ovgid_lists[i] =
          std::make_shared<vid_vineyard_builder_t>(client, std::move(chunks));
    } else {
      ovgid_lists[i] = nullptr;
    }

    ovnums[i] = i < vertex_label_num_ ? ovgid_lists_[i]->length() : 0;
    ovnums[i] += extra_ovgid_lists[i]->length();
    tvnums[i] = ivnums[i] + ovnums[i];

    extra_ovgid_lists[i].reset();  // release the reference
  }

  // Gather all local id of new edges.
  // And delete the src/dst column in edge tables.
  std::vector<std::vector<std::shared_ptr<vid_array_t>>> edge_src, edge_dst;
  edge_src.resize(extra_edge_label_num);
  edge_dst.resize(extra_edge_label_num);

  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::MemoryPool> recorder;
  if (VLOG_IS_ON(1000)) {
    recorder = std::make_shared<arrow::LoggingMemoryPool>(pool);
    pool = recorder.get();
  }
  for (int i = 0; i < extra_edge_label_num; ++i) {
    generate_local_id_list(vid_parser_, std::move(srcs[i]), fid_, ovg2l_maps,
                           concurrency, edge_src[i], pool);
    generate_local_id_list(vid_parser_, std::move(dsts[i]), fid_, ovg2l_maps,
                           concurrency, edge_dst[i], pool);
  }
  VLOG(100) << "[frag-" << this->fid_
            << "] Add new vertices and edges: after generate_local_id_list: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  // Generate CSR vector of new edge tables.
  std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>>
      ie_lists(total_vertex_label_num);
  std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>>
      oe_lists(total_vertex_label_num);
  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>> ie_offsets_lists(
      total_vertex_label_num);
  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>> oe_offsets_lists(
      total_vertex_label_num);

  for (label_id_t v_label = 0; v_label < total_vertex_label_num; ++v_label) {
    oe_lists[v_label].resize(total_edge_label_num);
    oe_offsets_lists[v_label].resize(total_edge_label_num);
    if (directed_) {
      ie_lists[v_label].resize(total_edge_label_num);
      ie_offsets_lists[v_label].resize(total_edge_label_num);
    }
  }

  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      vid_t prev_offset_size = tvnums_[v_label] + 1;
      vid_t cur_offset_size = tvnums[v_label] + 1;
      if (directed_) {
        ie_offsets_lists[v_label][e_label] =
            std::make_shared<FixedInt64Builder>(client, cur_offset_size);
        int64_t* offsets = ie_offsets_lists[v_label][e_label]->data();
        const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
        for (vid_t k = 0; k < prev_offset_size; ++k) {
          offsets[k] = offset_array[k];
        }
        for (vid_t k = prev_offset_size; k < cur_offset_size; ++k) {
          offsets[k] = offsets[k - 1];
        }
      }
      oe_offsets_lists[v_label][e_label] =
          std::make_shared<FixedInt64Builder>(client, cur_offset_size);
      int64_t* offsets = oe_offsets_lists[v_label][e_label]->data();
      const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
      for (size_t k = 0; k < prev_offset_size; ++k) {
        offsets[k] = offset_array[k];
      }
      for (size_t k = prev_offset_size; k < cur_offset_size; ++k) {
        offsets[k] = offsets[k - 1];
      }
    }
  }

  for (label_id_t e_label = 0; e_label < total_edge_label_num; ++e_label) {
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_ie_lists(
        total_vertex_label_num);
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_oe_lists(
        total_vertex_label_num);
    std::vector<std::shared_ptr<FixedInt64Builder>> sub_ie_offset_lists(
        total_vertex_label_num);
    std::vector<std::shared_ptr<FixedInt64Builder>> sub_oe_offset_lists(
        total_vertex_label_num);

    // Process v_num...total_v_num  X  0...e_num  part.
    if (e_label < edge_label_num_) {
      if (directed_) {
        for (label_id_t v_label = vertex_label_num_;
             v_label < total_vertex_label_num; ++v_label) {
          sub_ie_lists[v_label] =
              std::make_shared<PodArrayBuilder<nbr_unit_t>>(client, 0);
          sub_ie_offset_lists[v_label] =
              std::make_shared<FixedInt64Builder>(client, tvnums[v_label] + 1);
          memset(sub_ie_offset_lists[v_label]->data(), 0x00,
                 sizeof(int64_t) * (tvnums[v_label] + 1));
        }
      }
      for (label_id_t v_label = vertex_label_num_;
           v_label < total_vertex_label_num; ++v_label) {
        sub_oe_lists[v_label] =
            std::make_shared<PodArrayBuilder<nbr_unit_t>>(client, 0);
        sub_oe_offset_lists[v_label] =
            std::make_shared<FixedInt64Builder>(client, tvnums[v_label] + 1);
        memset(sub_oe_offset_lists[v_label]->data(), 0x00,
               sizeof(int64_t) * (tvnums[v_label] + 1));
      }
    } else {
      auto cur_label_index = e_label - edge_label_num_;
      // Process v_num...total_v_num  X  0...e_num  part.
      if (directed_) {
        // Process 0...total_v_num  X  e_num...total_e_num  part.
        generate_directed_csr<vid_t, eid_t>(
            client, vid_parser_, std::move(edge_src[cur_label_index]),
            std::move(edge_dst[cur_label_index]), tvnums,
            total_vertex_label_num, concurrency, sub_oe_lists,
            sub_oe_offset_lists, is_multigraph_);
        generate_directed_csc<vid_t, eid_t>(
            client, vid_parser_, tvnums, total_vertex_label_num, concurrency,
            sub_oe_lists, sub_oe_offset_lists, sub_ie_lists,
            sub_ie_offset_lists, is_multigraph_);
      } else {
        generate_undirected_csr_memopt<vid_t, eid_t>(
            client, vid_parser_, std::move(edge_src[cur_label_index]),
            std::move(edge_dst[cur_label_index]), tvnums,
            total_vertex_label_num, concurrency, sub_oe_lists,
            sub_oe_offset_lists, is_multigraph_);
      }
    }

    for (label_id_t v_label = 0; v_label < total_vertex_label_num; ++v_label) {
      if (v_label < vertex_label_num_ && e_label < edge_label_num_) {
        continue;
      }
      if (directed_) {
        ie_lists[v_label][e_label] = sub_ie_lists[v_label];
        ie_offsets_lists[v_label][e_label] = sub_ie_offset_lists[v_label];
      }
      oe_lists[v_label][e_label] = sub_oe_lists[v_label];
      oe_offsets_lists[v_label][e_label] = sub_oe_offset_lists[v_label];
    }
  }

  // FIXME: varint encoding
  if (this->compact_edges_) {
    RETURN_GS_ERROR(
        ErrorCode::kUnimplementedMethod,
        "Varint encoding is not implemented for adding vertices/edges");
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new vertices and edges: after generate CSR: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT> builder(*this);
  builder.set_vertex_label_num_(total_vertex_label_num);
  builder.set_edge_label_num_(total_edge_label_num);

  auto schema = schema_;  // make a copy

  // Extra vertex table
  for (label_id_t extra_label_id = 0; extra_label_id < extra_vertex_label_num;
       ++extra_label_id) {
    int label_id = vertex_label_num_ + extra_label_id;
    auto& table = vertex_tables[extra_label_id];

    // build schema entry for the new vertex
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);

    auto entry =
        schema.CreateEntry(kvs["label"], PropertyGraphSchema::VERTEX_TYPE_NAME);
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    std::string retain_oid = kvs["retain_oid"];
    if (retain_oid == "1" || retain_oid == "true") {
      int column_index = table->num_columns() - 1;
      entry->AddPrimaryKey(table->schema()->field(column_index)->name());
    }

    builder.set_vertex_tables_(
        label_id, std::make_shared<TableBuilder>(client, std::move(table),
                                                 true /* merge chunks */));
  }

  // extra edge tables
  for (label_id_t extra_label_id = 0; extra_label_id < extra_edge_label_num;
       ++extra_label_id) {
    label_id_t label_id = edge_label_num_ + extra_label_id;
    auto& table = edge_tables[extra_label_id];

    // build schema entry for the new edge
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);
    auto entry =
        schema.CreateEntry(kvs["label"], PropertyGraphSchema::EDGE_TYPE_NAME);
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    for (const auto& rel : edge_relations[extra_label_id]) {
      entry->AddRelation(rel.first, rel.second);
    }

    builder.set_edge_tables_(
        label_id, std::make_shared<TableBuilder>(client, std::move(table),
                                                 true /* merge chunks */));
  }

  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());

  ThreadGroup tg;
  {
    // ivnums, ovnums, tvnums
    auto fn = [&builder, &ivnums, &ovnums, &tvnums](Client* client) -> Status {
      vineyard::ArrayBuilder<vid_t> ivnums_builder(*client, ivnums);
      vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums);
      vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums);
      std::shared_ptr<Object> object;
      RETURN_ON_ERROR(ivnums_builder.Seal(*client, object));
      builder.set_ivnums_(object);
      RETURN_ON_ERROR(ovnums_builder.Seal(*client, object));
      builder.set_ovnums_(object);
      RETURN_ON_ERROR(tvnums_builder.Seal(*client, object));
      builder.set_tvnums_(object);
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  // Extra ovgid, ovg2l
  //
  // If the map have no new entries, clear it to indicate using the old map
  // when seal.
  for (int i = 0; i < vertex_label_num_; ++i) {
    if (ovg2l_maps_ptr_[i]->size() == ovg2l_maps[i].size()) {
      ovg2l_maps[i].clear();
    }
  }
  builder.ovgid_lists_.resize(total_vertex_label_num);
  builder.ovg2l_maps_.resize(total_vertex_label_num);
  for (label_id_t i = 0; i < total_vertex_label_num; ++i) {
    auto fn = [this, &builder, i, &ovgid_lists,
               &ovg2l_maps](Client* client) -> Status {
      if (ovgid_lists[i] != nullptr) {
        builder.set_ovgid_lists_(i, ovgid_lists[i]);
      }

      if (i >= vertex_label_num_ || !ovg2l_maps[i].empty()) {
        vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
            *client, std::move(ovg2l_maps[i]));
        std::shared_ptr<Object> ovg2l_map;
        RETURN_ON_ERROR(ovg2l_builder.Seal(*client, ovg2l_map));
        builder.set_ovg2l_maps_(i, ovg2l_map);
      }
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  // Extra ie_list, oe_list, ie_offset_list, oe_offset_list
  if (directed_) {
    builder.ie_lists_.resize(total_vertex_label_num);
    builder.ie_offsets_lists_.resize(total_vertex_label_num);
  }
  builder.oe_lists_.resize(total_vertex_label_num);
  builder.oe_offsets_lists_.resize(total_vertex_label_num);
  for (label_id_t i = 0; i < total_vertex_label_num; ++i) {
    if (directed_) {
      builder.ie_lists_[i].resize(total_edge_label_num);
      builder.ie_offsets_lists_[i].resize(total_edge_label_num);
    }
    builder.oe_lists_[i].resize(total_edge_label_num);
    builder.oe_offsets_lists_[i].resize(total_edge_label_num);
    for (label_id_t j = 0; j < total_edge_label_num; ++j) {
      auto fn = [this, &builder, i, j, &ie_lists, &oe_lists, &ie_offsets_lists,
                 &oe_offsets_lists](Client* client) -> Status {
        if (directed_) {
          if (!(i < vertex_label_num_ && j < edge_label_num_)) {
            builder.set_ie_lists_(i, j, ie_lists[i][j]);
          }
          builder.set_ie_offsets_lists_(i, j, ie_offsets_lists[i][j]);
        }
        if (!(i < vertex_label_num_ && j < edge_label_num_)) {
          builder.set_oe_lists_(i, j, oe_lists[i][j]);
        }
        builder.set_oe_offsets_lists_(i, j, oe_offsets_lists[i][j]);
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new vertices and edges: after building into vineyard: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  // wait all
  tg.TakeResults();

  builder.set_vm_ptr_(vm_ptr);

  std::shared_ptr<Object> object;
  VY_OK_OR_RAISE(builder.Seal(client, object));
  return object->id();
}

/// Add a set of new vertex labels to graph. Vertex label id started from
/// vertex_label_num_.
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::AddNewVertexLabels(
    Client& client, std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
    ObjectID vm_id, const int concurrency) {
  int extra_vertex_label_num = vertex_tables.size();
  int total_vertex_label_num = vertex_label_num_ + extra_vertex_label_num;

  auto vm_ptr =
      std::dynamic_pointer_cast<vertex_map_t>(client.GetObject(vm_id));

  std::vector<vid_t> ivnums(total_vertex_label_num);
  std::vector<vid_t> ovnums(total_vertex_label_num);
  std::vector<vid_t> tvnums(total_vertex_label_num);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    ivnums[i] = ivnums_[i];
    ovnums[i] = ovnums_[i];
    tvnums[i] = tvnums_[i];
  }
  for (size_t i = 0; i < vertex_tables.size(); ++i) {
    ivnums[vertex_label_num_ + i] =
        vm_ptr->GetInnerVertexSize(fid_, vertex_label_num_ + i);
    ovnums[vertex_label_num_ + i] = 0;
    tvnums[vertex_label_num_ + i] = ivnums[vertex_label_num_ + i];
  }

  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT> builder(*this);
  builder.set_vertex_label_num_(total_vertex_label_num);

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new vertices: start: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  auto schema = schema_;
  for (int extra_label_id = 0; extra_label_id < extra_vertex_label_num;
       ++extra_label_id) {
    int label_id = vertex_label_num_ + extra_label_id;
    auto& table = vertex_tables[extra_label_id];

    // build schema entry for the new vertex
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);

    auto entry =
        schema.CreateEntry(kvs["label"], PropertyGraphSchema::VERTEX_TYPE_NAME);
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    std::string retain_oid = kvs["retain_oid"];
    if (retain_oid == "1" || retain_oid == "true") {
      int col_id = table->num_columns() - 1;
      entry->AddPrimaryKey(table->schema()->field(col_id)->name());
    }

    builder.set_vertex_tables_(
        label_id, std::make_shared<TableBuilder>(client, std::move(table),
                                                 true /* merge chunks */));
  }
  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());

  vineyard::ArrayBuilder<vid_t> ivnums_builder(client, ivnums);
  vineyard::ArrayBuilder<vid_t> ovnums_builder(client, ovnums);
  vineyard::ArrayBuilder<vid_t> tvnums_builder(client, tvnums);

  std::shared_ptr<Object> object;
  VY_OK_OR_RAISE(ivnums_builder.Seal(client, object));
  builder.set_ivnums_(object);
  VY_OK_OR_RAISE(ovnums_builder.Seal(client, object));
  builder.set_ovnums_(object);
  VY_OK_OR_RAISE(tvnums_builder.Seal(client, object));
  builder.set_tvnums_(object);

  // Assign additional meta for new vertex labels
  std::vector<std::vector<std::shared_ptr<vineyard::FixedSizeBinaryArray>>>
      vy_ie_lists, vy_oe_lists;
  std::vector<std::vector<std::shared_ptr<vineyard::NumericArray<int64_t>>>>
      vy_ie_offsets_lists, vy_oe_offsets_lists;

  for (label_id_t extra_label_id = 0; extra_label_id < extra_vertex_label_num;
       ++extra_label_id) {
    label_id_t label_id = vertex_label_num_ + extra_label_id;
    builder.set_ovgid_lists_(label_id,
                             std::make_shared<vid_vineyard_builder_t>(client));
    builder.set_ovg2l_maps_(
        label_id,
        std::make_shared<vineyard::HashmapBuilder<vid_t, vid_t>>(client));
  }

  for (label_id_t i = 0; i < extra_vertex_label_num; ++i) {
    for (label_id_t j = 0; j < edge_label_num_; ++j) {
      label_id_t vertex_label_id = vertex_label_num_ + i;
      if (directed_) {
        vineyard::FixedSizeBinaryArrayBuilder ie_builder(
            client, arrow::fixed_size_binary(sizeof(nbr_unit_t)));
        std::shared_ptr<Object> ie_object;
        ARROW_OK_OR_RAISE(ie_builder.Seal(client, ie_object));
        builder.set_ie_lists_(vertex_label_id, j, ie_object);

        arrow::Int64Builder int64_builder;
        // Offset vector's length is tvnum + 1
        std::vector<int64_t> offset_vec(tvnums[vertex_label_id] + 1, 0);
        ARROW_OK_OR_RAISE(int64_builder.AppendValues(offset_vec));
        std::shared_ptr<arrow::Int64Array> ie_offset_array;
        ARROW_OK_OR_RAISE(int64_builder.Finish(&ie_offset_array));

        vineyard::NumericArrayBuilder<int64_t> ie_offset_builder(
            client, ie_offset_array);
        std::shared_ptr<Object> ie_offset_object;
        ARROW_OK_OR_RAISE(ie_offset_builder.Seal(client, ie_offset_object));
        builder.set_ie_offsets_lists_(vertex_label_id, j, ie_offset_object);
      }

      vineyard::FixedSizeBinaryArrayBuilder oe_builder(
          client, arrow::fixed_size_binary(sizeof(nbr_unit_t)));
      std::shared_ptr<Object> oe_object;
      ARROW_OK_OR_RAISE(oe_builder.Seal(client, oe_object));
      builder.set_oe_lists_(vertex_label_id, j, oe_object);

      arrow::Int64Builder int64_builder;
      // Offset vector's length is tvnum + 1
      std::vector<int64_t> offset_vec(tvnums[vertex_label_id] + 1, 0);
      ARROW_OK_OR_RAISE(int64_builder.AppendValues(offset_vec));
      std::shared_ptr<arrow::Int64Array> oe_offset_array;
      ARROW_OK_OR_RAISE(int64_builder.Finish(&oe_offset_array));
      vineyard::NumericArrayBuilder<int64_t> oe_offset_builder(client,
                                                               oe_offset_array);
      std::shared_ptr<Object> oe_offset_object;
      ARROW_OK_OR_RAISE(oe_offset_builder.Seal(client, oe_offset_object));
      builder.set_oe_offsets_lists_(vertex_label_id, j, oe_offset_object);
    }
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new vertices: after building into vineyard: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  builder.set_vm_ptr_(vm_ptr);
  std::shared_ptr<Object> vm_object;
  VY_OK_OR_RAISE(builder.Seal(client, vm_object));
  return vm_object->id();
}

/// Add a set of new edge labels to graph. Edge label id started from
/// edge_label_num_.
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::AddNewEdgeLabels(
    Client& client, std::vector<std::shared_ptr<arrow::Table>>&& edge_tables,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    const int concurrency) {
  int extra_edge_label_num = edge_tables.size();
  int total_edge_label_num = edge_label_num_ + extra_edge_label_num;

  std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs(extra_edge_label_num),
      dsts(extra_edge_label_num);
  for (label_id_t label = 0; label < extra_edge_label_num; ++label) {
    srcs[label] = edge_tables[label]->column(0);
    dsts[label] = edge_tables[label]->column(1);
    // remove src and dst columns from edge tables
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[label],
                             edge_tables[label]->RemoveColumn(0));
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[label],
                             edge_tables[label]->RemoveColumn(0));
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new edges: before init the new vertex map: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  // Make a copy of ovg2l map, since we need to add some extra outer vertices
  // pulled in this fragment by new edges.
  std::vector<ovg2l_map_t> ovg2l_maps(vertex_label_num_);
  for (int i = 0; i < vertex_label_num_; ++i) {
    for (auto iter = ovg2l_maps_ptr_[i]->begin();
         iter != ovg2l_maps_ptr_[i]->end(); ++iter) {
      ovg2l_maps[i].emplace(iter->first, iter->second);
    }
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new edges: after init the new vertex map: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  std::vector<vid_t> start_ids(vertex_label_num_);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]) + ovnums_[i];
  }
  // Add extra outer vertices to ovg2l map, and collect distinct gid of extra
  // outer vertices.
  std::vector<std::shared_ptr<vid_array_t>> extra_ovgid_lists(
      vertex_label_num_);
  generate_outer_vertices_map(vid_parser_, fid_, vertex_label_num_, srcs, dsts,
                              start_ids, ovg2l_maps, extra_ovgid_lists);

  VLOG(100) << "[frag-" << this->fid_
            << "] Init edges: after generate_outer_vertices_map: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  std::vector<vid_t> ovnums(vertex_label_num_), tvnums(vertex_label_num_);
  std::vector<std::shared_ptr<vid_vineyard_builder_t>> ovgid_lists(
      vertex_label_num_);
  // Append extra ovgid_lists with origin ovgid_lists to make it complete
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    vid_builder_t ovgid_list_builder;
    // If the ovgid have no new entries, leave it empty to indicate using the
    // old ovgid when seal.
    if (extra_ovgid_lists[i]->length() != 0) {
      std::vector<std::shared_ptr<vid_array_t>> chunks;
      chunks.push_back(ovgid_lists_[i]->GetArray());
      chunks.push_back(extra_ovgid_lists[i]);
      ovgid_lists[i] = std::make_shared<vid_vineyard_builder_t>(client, chunks);
    } else {
      ovgid_lists[i] = nullptr;
    }

    ovnums[i] = ovgid_lists_[i]->length() + extra_ovgid_lists[i]->length();
    tvnums[i] = ivnums_[i] + ovnums[i];

    extra_ovgid_lists[i].reset();  // release the reference
  }

  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>
      ie_offsets_lists_expanded(vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>
      oe_offsets_lists_expanded(vertex_label_num_);

  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    if (directed_) {
      ie_offsets_lists_expanded[v_label].resize(edge_label_num_);
    }
    oe_offsets_lists_expanded[v_label].resize(edge_label_num_);
  }
  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      vid_t prev_offset_size = tvnums_[v_label] + 1;
      vid_t current_offset_size = tvnums[v_label] + 1;
      if (directed_) {
        ie_offsets_lists_expanded[v_label][e_label] =
            std::make_shared<FixedInt64Builder>(client, current_offset_size);
        int64_t* offsets = ie_offsets_lists_expanded[v_label][e_label]->data();
        const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
        for (vid_t k = 0; k < prev_offset_size; ++k) {
          offsets[k] = offset_array[k];
        }
        for (vid_t k = prev_offset_size; k < current_offset_size; ++k) {
          offsets[k] = offsets[k - 1];
        }
      }
      oe_offsets_lists_expanded[v_label][e_label] =
          std::make_shared<FixedInt64Builder>(client, current_offset_size);
      int64_t* offsets = oe_offsets_lists_expanded[v_label][e_label]->data();
      const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
      for (size_t k = 0; k < prev_offset_size; ++k) {
        offsets[k] = offset_array[k];
      }
      for (size_t k = prev_offset_size; k < current_offset_size; ++k) {
        offsets[k] = offsets[k - 1];
      }
    }
  }
  // Gather all local id of new edges.
  // And delete the src/dst column in edge tables.
  std::vector<std::vector<std::shared_ptr<vid_array_t>>> edge_src, edge_dst;
  edge_src.resize(extra_edge_label_num);
  edge_dst.resize(extra_edge_label_num);

  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::MemoryPool> recorder;
  if (VLOG_IS_ON(1000)) {
    recorder = std::make_shared<arrow::LoggingMemoryPool>(pool);
    pool = recorder.get();
  }
  for (int i = 0; i < extra_edge_label_num; ++i) {
    generate_local_id_list(vid_parser_, std::move(srcs[i]), fid_, ovg2l_maps,
                           concurrency, edge_src[i], pool);
    generate_local_id_list(vid_parser_, std::move(dsts[i]), fid_, ovg2l_maps,
                           concurrency, edge_dst[i], pool);
  }
  VLOG(100) << "[frag-" << this->fid_
            << "] Add new edges: after generate_local_id_list: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  // Generate CSR vector of new edge tables.
  std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>>
      ie_lists(vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>>
      oe_lists(vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>> ie_offsets_lists(
      vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>> oe_offsets_lists(
      vertex_label_num_);

  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    oe_lists[v_label].resize(extra_edge_label_num);
    oe_offsets_lists[v_label].resize(extra_edge_label_num);
    if (directed_) {
      ie_lists[v_label].resize(extra_edge_label_num);
      ie_offsets_lists[v_label].resize(extra_edge_label_num);
    }
  }

  for (label_id_t e_label = 0; e_label < extra_edge_label_num; ++e_label) {
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_ie_lists(
        vertex_label_num_);
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_oe_lists(
        vertex_label_num_);
    std::vector<std::shared_ptr<FixedInt64Builder>> sub_ie_offset_lists(
        vertex_label_num_);
    std::vector<std::shared_ptr<FixedInt64Builder>> sub_oe_offset_lists(
        vertex_label_num_);
    if (directed_) {
      generate_directed_csr<vid_t, eid_t>(
          client, vid_parser_, std::move(edge_src[e_label]),
          std::move(edge_dst[e_label]), tvnums, vertex_label_num_, concurrency,
          sub_oe_lists, sub_oe_offset_lists, is_multigraph_);
      generate_directed_csc<vid_t, eid_t>(
          client, vid_parser_, tvnums, vertex_label_num_, concurrency,
          sub_oe_lists, sub_oe_offset_lists, sub_ie_lists, sub_ie_offset_lists,
          is_multigraph_);
    } else {
      generate_undirected_csr_memopt<vid_t, eid_t>(
          client, vid_parser_, std::move(edge_src[e_label]),
          std::move(edge_dst[e_label]), tvnums, vertex_label_num_, concurrency,
          sub_oe_lists, sub_oe_offset_lists, is_multigraph_);
    }

    for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
      if (directed_) {
        ie_lists[v_label][e_label] = sub_ie_lists[v_label];
        ie_offsets_lists[v_label][e_label] = sub_ie_offset_lists[v_label];
      }
      oe_lists[v_label][e_label] = sub_oe_lists[v_label];
      oe_offsets_lists[v_label][e_label] = sub_oe_offset_lists[v_label];
    }
  }

  // FIXME: varint encoding
  if (this->compact_edges_) {
    RETURN_GS_ERROR(
        ErrorCode::kUnimplementedMethod,
        "Varint encoding is not implemented for adding vertices/edges");
  }

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new edges: after generate CSR: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT> builder(*this);
  builder.set_edge_label_num_(total_edge_label_num);

  auto schema = schema_;
  for (label_id_t extra_label_id = 0; extra_label_id < extra_edge_label_num;
       ++extra_label_id) {
    label_id_t label_id = edge_label_num_ + extra_label_id;
    auto& table = edge_tables[extra_label_id];

    // build schema entry for the new edge
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);
    auto entry =
        schema.CreateEntry(kvs["label"], PropertyGraphSchema::EDGE_TYPE_NAME);
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    for (const auto& rel : edge_relations[extra_label_id]) {
      entry->AddRelation(rel.first, rel.second);
    }

    builder.set_edge_tables_(
        label_id, std::make_shared<TableBuilder>(client, std::move(table),
                                                 true /* merge chunks */));
  }

  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());

  ThreadGroup tg;
  {
    auto fn = [&builder, &ovnums, &tvnums](Client* client) -> Status {
      vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums);
      vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums);
      std::shared_ptr<Object> object;
      RETURN_ON_ERROR(ovnums_builder.Seal(*client, object));
      builder.set_ovnums_(object);
      RETURN_ON_ERROR(tvnums_builder.Seal(*client, object));
      builder.set_tvnums_(object);
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  // If the map have no new entries, clear it to indicate using the old map
  // when seal.
  for (int i = 0; i < vertex_label_num_; ++i) {
    if (ovg2l_maps_ptr_[i]->size() == ovg2l_maps[i].size()) {
      ovg2l_maps[i].clear();
    }
  }
  builder.ovgid_lists_.resize(vertex_label_num_);
  builder.ovg2l_maps_.resize(vertex_label_num_);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    auto fn = [&builder, i, &ovgid_lists,
               &ovg2l_maps](Client* client) -> Status {
      if (ovgid_lists[i] != nullptr) {
        builder.set_ovgid_lists_(i, ovgid_lists[i]);
      }

      if (!ovg2l_maps[i].empty()) {
        vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
            *client, std::move(ovg2l_maps[i]));
        std::shared_ptr<Object> ovg2l_map_object;
        RETURN_ON_ERROR(ovg2l_builder.Seal(*client, ovg2l_map_object));
        builder.set_ovg2l_maps_(i, ovg2l_map_object);
      }
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  // Extra ie_list, oe_list, ie_offset_list, oe_offset_list
  if (directed_) {
    builder.ie_lists_.resize(vertex_label_num_);
    builder.ie_offsets_lists_.resize(vertex_label_num_);
  }
  builder.oe_lists_.resize(vertex_label_num_);
  builder.oe_offsets_lists_.resize(vertex_label_num_);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    if (directed_) {
      builder.ie_lists_[i].resize(edge_label_num_ + extra_edge_label_num);
      builder.ie_offsets_lists_[i].resize(edge_label_num_ +
                                          extra_edge_label_num);
    }
    builder.oe_lists_[i].resize(edge_label_num_ + extra_edge_label_num);
    builder.oe_offsets_lists_[i].resize(edge_label_num_ + extra_edge_label_num);
    for (label_id_t j = 0; j < extra_edge_label_num; ++j) {
      auto fn = [this, &builder, i, j, &ie_lists, &oe_lists, &ie_offsets_lists,
                 &oe_offsets_lists](Client* client) -> Status {
        label_id_t edge_label_id = edge_label_num_ + j;
        if (directed_) {
          builder.set_ie_lists_(i, edge_label_id, ie_lists[i][j]);
          builder.set_ie_offsets_lists_(i, edge_label_id,
                                        ie_offsets_lists[i][j]);
        }
        builder.set_oe_lists_(i, edge_label_id, oe_lists[i][j]);
        builder.set_oe_offsets_lists_(i, edge_label_id, oe_offsets_lists[i][j]);
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }
  }
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    for (label_id_t j = 0; j < edge_label_num_; ++j) {
      auto fn = [this, &builder, &ie_offsets_lists_expanded,
                 &oe_offsets_lists_expanded](Client* client, const label_id_t i,
                                             const label_id_t j) -> Status {
        if (directed_) {
          builder.set_ie_offsets_lists_(i, j, ie_offsets_lists_expanded[i][j]);
        }
        builder.set_oe_offsets_lists_(i, j, oe_offsets_lists_expanded[i][j]);
        return Status::OK();
      };
      tg.AddTask(fn, &client, i, j);
    }
  }
  tg.TakeResults();

  VLOG(100) << "[frag-" << this->fid_
            << "] Add new edges: after building into vineyard: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();
  std::shared_ptr<Object> fragment_object;
  VY_OK_OR_RAISE(builder.Seal(client, fragment_object));
  return fragment_object->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
vineyard::Status
BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::Build(
    vineyard::Client& client) {
  VLOG(100) << "[frag-" << this->fid_
            << "] Start building into vineyard: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  ThreadGroup tg;
  {
    auto fn = [this](Client* client) -> Status {
      vineyard::ArrayBuilder<vid_t> ivnums_builder(*client, ivnums_);
      vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums_);
      vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums_);
      std::shared_ptr<Object> object;
      RETURN_ON_ERROR(ivnums_builder.Seal(*client, object));
      this->set_ivnums_(object);
      RETURN_ON_ERROR(ovnums_builder.Seal(*client, object));
      this->set_ovnums_(object);
      RETURN_ON_ERROR(tvnums_builder.Seal(*client, object));
      this->set_tvnums_(object);
      return Status::OK();
    };

    tg.AddTask(fn, &client);
  }

  Base::vertex_tables_.resize(this->vertex_label_num_);
  Base::ovgid_lists_.resize(this->vertex_label_num_);
  Base::ovg2l_maps_.resize(this->vertex_label_num_);

  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    auto fn = [this, i](Client* client) -> Status {
      this->set_vertex_tables_(i, std::make_shared<vineyard::TableBuilder>(
                                      *client, std::move(vertex_tables_[i]),
                                      true /* merge chunks */));

      vineyard::NumericArrayBuilder<vid_t> ovgid_list_builder(
          *client, std::move(ovgid_lists_[i]));
      std::shared_ptr<Object> ovgid_list_object;
      RETURN_ON_ERROR(ovgid_list_builder.Seal(*client, ovgid_list_object));
      this->set_ovgid_lists_(i, ovgid_list_object);

      vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
          *client, std::move(ovg2l_maps_[i]));
      std::shared_ptr<Object> ovg2l_map_object;
      RETURN_ON_ERROR(ovg2l_builder.Seal(*client, ovg2l_map_object));
      this->set_ovg2l_maps_(i, ovg2l_map_object);
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  Base::edge_tables_.resize(this->edge_label_num_);
  for (label_id_t i = 0; i < this->edge_label_num_; ++i) {
    auto fn = [this, i](Client* client) -> Status {
      this->set_edge_tables_(
          i, std::make_shared<vineyard::TableBuilder>(
                 *client, std::move(edge_tables_[i]), true /* merge chunks */));
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  if (this->directed_) {
    if (this->compact_edges_) {
      Base::compact_ie_lists_.resize(this->vertex_label_num_);
      Base::ie_boffsets_lists_.resize(this->vertex_label_num_);
    } else {
      Base::ie_lists_.resize(this->vertex_label_num_);
    }
    Base::ie_offsets_lists_.resize(this->vertex_label_num_);
  }
  if (this->compact_edges_) {
    Base::compact_oe_lists_.resize(this->vertex_label_num_);
    Base::oe_boffsets_lists_.resize(this->vertex_label_num_);
  } else {
    Base::oe_lists_.resize(this->vertex_label_num_);
  }
  Base::oe_offsets_lists_.resize(this->vertex_label_num_);

  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    if (this->directed_) {
      if (this->compact_edges_) {
        Base::compact_ie_lists_[i].resize(this->edge_label_num_);
        Base::ie_boffsets_lists_[i].resize(this->edge_label_num_);
      } else {
        Base::ie_lists_[i].resize(this->edge_label_num_);
      }
      Base::ie_offsets_lists_[i].resize(this->edge_label_num_);
    }
    if (this->compact_edges_) {
      Base::compact_oe_lists_[i].resize(this->edge_label_num_);
      Base::oe_boffsets_lists_[i].resize(this->edge_label_num_);
    } else {
      Base::oe_lists_[i].resize(this->edge_label_num_);
    }
    Base::oe_offsets_lists_[i].resize(this->edge_label_num_);

    for (label_id_t j = 0; j < this->edge_label_num_; ++j) {
      auto fn = [this, i, j](Client* client) -> Status {
        std::shared_ptr<Object> object;
        if (this->directed_) {
          if (this->compact_edges_) {
            RETURN_ON_ERROR(compact_ie_lists_[i][j]->Seal(*client, object));
            this->set_compact_ie_lists_(i, j, object);
            RETURN_ON_ERROR(ie_boffsets_lists_[i][j]->Seal(*client, object));
            this->set_ie_boffsets_lists_(i, j, object);
          } else {
            RETURN_ON_ERROR(ie_lists_[i][j]->Seal(*client, object));
            this->set_ie_lists_(i, j, object);
          }
          RETURN_ON_ERROR(ie_offsets_lists_[i][j]->Seal(*client, object));
          this->set_ie_offsets_lists_(i, j, object);
        }
        if (this->compact_edges_) {
          RETURN_ON_ERROR(compact_oe_lists_[i][j]->Seal(*client, object));
          this->set_compact_oe_lists_(i, j, object);
          RETURN_ON_ERROR(oe_boffsets_lists_[i][j]->Seal(*client, object));
          this->set_oe_boffsets_lists_(i, j, object);
        } else {
          RETURN_ON_ERROR(oe_lists_[i][j]->Seal(*client, object));
          this->set_oe_lists_(i, j, object);
        }
        RETURN_ON_ERROR(oe_offsets_lists_[i][j]->Seal(*client, object));
        this->set_oe_offsets_lists_(i, j, object);
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }
  }

  tg.TakeResults();

  this->set_vm_ptr_(vm_ptr_);

  this->set_oid_type(type_name<oid_t>());
  this->set_vid_type(type_name<vid_t>());

  VLOG(100) << "[frag-" << this->fid_
            << "] Finish building into vineyard: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  return Status::OK();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<void>
BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::Init(
    fid_t fid, fid_t fnum,
    std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
    std::vector<std::shared_ptr<arrow::Table>>&& edge_tables, bool directed,
    int concurrency) {
  this->fid_ = fid;
  this->fnum_ = fnum;
  this->directed_ = directed;
  this->is_multigraph_ = false;
  this->vertex_label_num_ = vertex_tables.size();
  this->edge_label_num_ = edge_tables.size();
  this->local_vertex_map_ = is_local_vertex_map<VERTEX_MAP_T>::value;
  this->compact_edges_ = COMPACT;

  vid_parser_.Init(this->fnum_, this->vertex_label_num_);

  VLOG(100) << "[frag-" << this->fid_
            << "] Init: start init vertices: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();
  BOOST_LEAF_CHECK(initVertices(std::move(vertex_tables)));
  VLOG(100) << "[frag-" << this->fid_
            << "] Init: start init edges: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();
  BOOST_LEAF_CHECK(initEdges(std::move(edge_tables), concurrency));
  VLOG(100) << "[frag-" << this->fid_
            << "] Init: finish init vertices and edges: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();
  return {};
}

// | prop_0 | prop_1 | ... |
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<void>
BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::initVertices(
    std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables) {
  assert(vertex_tables.size() == static_cast<size_t>(this->vertex_label_num_));
  vertex_tables_ = vertex_tables;
  ivnums_.resize(this->vertex_label_num_);
  ovnums_.resize(this->vertex_label_num_);
  tvnums_.resize(this->vertex_label_num_);
  for (size_t i = 0; i < vertex_tables_.size(); ++i) {
    ivnums_[i] = vm_ptr_->GetInnerVertexSize(this->fid_, i);
  }
  return {};
}

// | src_id(generated) | dst_id(generated) | prop_0 | prop_1
// | ... |
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
boost::leaf::result<void>
BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT>::initEdges(
    std::vector<std::shared_ptr<arrow::Table>>&& edge_tables, int concurrency) {
  auto gen_edge_start_time = vineyard::GetCurrentTime();
  assert(edge_tables.size() == static_cast<size_t>(this->edge_label_num_));
  edge_tables_.resize(this->edge_label_num_);

  std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs(this->edge_label_num_),
      dsts(this->edge_label_num_);
  for (label_id_t label = 0; label < this->edge_label_num_; ++label) {
    srcs[label] = edge_tables[label]->column(0);
    dsts[label] = edge_tables[label]->column(1);
    std::shared_ptr<arrow::Table> table = std::move(edge_tables[label]);
    // remove src and dst columns from edge tables
    ARROW_OK_ASSIGN_OR_RAISE(table, table->RemoveColumn(0));
    ARROW_OK_ASSIGN_OR_RAISE(table, table->RemoveColumn(0));
    edge_tables[label].reset();
    edge_tables_[label] = table;
  }

  VLOG(100) << "[frag-" << this->fid_ << "] Init edges: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  std::vector<vid_t> start_ids(this->vertex_label_num_);
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]);
  }
  generate_outer_vertices_map<vid_t>(vid_parser_, this->fid_,
                                     this->vertex_label_num_, srcs, dsts,
                                     start_ids, ovg2l_maps_, ovgid_lists_);
  VLOG(100) << "[frag-" << this->fid_
            << "] Init edges: after generate_outer_vertices_map: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  std::vector<std::vector<std::shared_ptr<vid_array_t>>> edge_src, edge_dst;
  edge_src.resize(this->edge_label_num_);
  edge_dst.resize(this->edge_label_num_);

  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    ovnums_[i] = ovgid_lists_[i]->length();
    tvnums_[i] = ivnums_[i] + ovnums_[i];
  }
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::MemoryPool> recorder;
  if (VLOG_IS_ON(1000)) {
    recorder = std::make_shared<arrow::LoggingMemoryPool>(pool);
    pool = recorder.get();
  }
  for (size_t i = 0; i < edge_tables.size(); ++i) {
    generate_local_id_list(vid_parser_, std::move(srcs[i]), this->fid_,
                           ovg2l_maps_, concurrency, edge_src[i], pool);
    generate_local_id_list(vid_parser_, std::move(dsts[i]), this->fid_,
                           ovg2l_maps_, concurrency, edge_dst[i], pool);
  }
  VLOG(100) << "[frag-" << this->fid_
            << "] Init edges: after generate_local_id_list: "
            << get_rss_pretty() << ", peak: " << get_peak_rss_pretty();

  oe_lists_.resize(this->vertex_label_num_);
  oe_offsets_lists_.resize(this->vertex_label_num_);
  if (this->directed_) {
    ie_lists_.resize(this->vertex_label_num_);
    ie_offsets_lists_.resize(this->vertex_label_num_);
  }

  for (label_id_t v_label = 0; v_label < this->vertex_label_num_; ++v_label) {
    oe_lists_[v_label].resize(this->edge_label_num_);
    oe_offsets_lists_[v_label].resize(this->edge_label_num_);
    if (this->directed_) {
      ie_lists_[v_label].resize(this->edge_label_num_);
      ie_offsets_lists_[v_label].resize(this->edge_label_num_);
    }
  }
  for (label_id_t e_label = 0; e_label < this->edge_label_num_; ++e_label) {
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_ie_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_oe_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<FixedInt64Builder>> sub_ie_offset_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<FixedInt64Builder>> sub_oe_offset_lists(
        this->vertex_label_num_);
    if (this->directed_) {
      generate_directed_csr<vid_t, eid_t>(
          client_, vid_parser_, std::move(edge_src[e_label]),
          std::move(edge_dst[e_label]), tvnums_, this->vertex_label_num_,
          concurrency, sub_oe_lists, sub_oe_offset_lists, this->is_multigraph_);
      generate_directed_csc<vid_t, eid_t>(
          client_, vid_parser_, tvnums_, this->vertex_label_num_, concurrency,
          sub_oe_lists, sub_oe_offset_lists, sub_ie_lists, sub_ie_offset_lists,
          this->is_multigraph_);
    } else {
      generate_undirected_csr_memopt<vid_t, eid_t>(
          client_, vid_parser_, std::move(edge_src[e_label]),
          std::move(edge_dst[e_label]), tvnums_, this->vertex_label_num_,
          concurrency, sub_oe_lists, sub_oe_offset_lists, this->is_multigraph_);
    }

    for (label_id_t v_label = 0; v_label < this->vertex_label_num_; ++v_label) {
      if (this->directed_) {
        ie_lists_[v_label][e_label] = sub_ie_lists[v_label];
        ie_offsets_lists_[v_label][e_label] = sub_ie_offset_lists[v_label];
      }
      oe_lists_[v_label][e_label] = sub_oe_lists[v_label];
      oe_offsets_lists_[v_label][e_label] = sub_oe_offset_lists[v_label];
    }
  }
  VLOG(100) << "[frag-" << this->fid_
            << "] Init edges: after generate CSR: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  VLOG(100) << "Generate edge time usage: "
            << (vineyard::GetCurrentTime() - gen_edge_start_time) << " seconds";

  if (this->compact_edges_) {
    BOOST_LEAF_CHECK(varint_encoding_edges(
        client_, this->directed_, this->vertex_label_num_,
        this->edge_label_num_, ie_lists_, oe_lists_, compact_ie_lists_,
        compact_oe_lists_, ie_offsets_lists_, oe_offsets_lists_,
        ie_boffsets_lists_, oe_boffsets_lists_, concurrency));
  }
  return {};
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BUILDER_IMPL_H_
