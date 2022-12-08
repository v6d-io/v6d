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

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddVerticesAndEdges(
    Client& client,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& vertex_tables_map,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& edge_tables_map,
    ObjectID vm_id,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    int concurrency) {
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

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddVertices(
    Client& client,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& vertex_tables_map,
    ObjectID vm_id) {
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

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddEdges(
    Client& client,
    std::map<label_id_t, std::shared_ptr<arrow::Table>>&& edge_tables_map,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    int concurrency) {
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
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddNewVertexEdgeLabels(
    Client& client, std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
    std::vector<std::shared_ptr<arrow::Table>>&& edge_tables, ObjectID vm_id,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    int concurrency) {
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
    ARROW_OK_ASSIGN_OR_RAISE(
        vertex_tables[i],
        vertex_tables[i]->CombineChunks(arrow::default_memory_pool()));
    ivnums[vertex_label_num_ + i] =
        vm_ptr->GetInnerVertexSize(fid_, vertex_label_num_ + i);
  }

  // Collect extra outer vertices.
  auto collect_extra_outer_vertices =
      [this](const std::shared_ptr<vid_array_t>& gid_array,
             std::vector<std::vector<vid_t>>& extra_ovgids) {
        const VID_T* arr = gid_array->raw_values();
        for (int64_t i = 0; i < gid_array->length(); ++i) {
          fid_t fid = vid_parser_.GetFid(arr[i]);
          label_id_t label_id = vid_parser_.GetLabelId(arr[i]);
          bool flag = true;
          if (fid != fid_) {
            if (label_id < vertex_label_num_) {
              auto cur_map = ovg2l_maps_ptr_[label_id];
              flag = cur_map->find(arr[i]) == cur_map->end();
            }
          } else {
            flag = false;
          }

          if (flag) {
            extra_ovgids[label_id].push_back(arr[i]);
          }
        }
      };

  std::vector<std::vector<vid_t>> extra_ovgids(total_vertex_label_num);
  for (int i = 0; i < extra_edge_label_num; ++i) {
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[i], edge_tables[i]->CombineChunks(
                                                 arrow::default_memory_pool()));

    collect_extra_outer_vertices(
        std::dynamic_pointer_cast<
            typename vineyard::ConvertToArrowType<vid_t>::ArrayType>(
            edge_tables[i]->column(0)->chunk(0)),
        extra_ovgids);
    collect_extra_outer_vertices(
        std::dynamic_pointer_cast<
            typename vineyard::ConvertToArrowType<vid_t>::ArrayType>(
            edge_tables[i]->column(1)->chunk(0)),
        extra_ovgids);
  }

  // Construct the new start value of lid of extra outer vertices
  std::vector<vid_t> start_ids(total_vertex_label_num);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]) + ovnums_[i];
  }
  for (label_id_t i = vertex_label_num_; i < total_vertex_label_num; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums[i]);
  }

  // Make a copy of ovg2l map, since we need to add some extra outer vertices
  // pulled in this fragment by new edges.
  std::vector<ovg2l_map_t> ovg2l_maps(total_vertex_label_num);
  for (int i = 0; i < vertex_label_num_; ++i) {
    for (auto iter = ovg2l_maps_ptr_[i]->begin();
         iter != ovg2l_maps_ptr_[i]->end(); ++iter) {
      ovg2l_maps[i].emplace(iter->first, iter->second);
    }
  }

  std::vector<std::shared_ptr<vid_array_t>> extra_ovgid_lists(
      total_vertex_label_num);
  // Add extra outer vertices to ovg2l map, and collect distinct gid of extra
  // outer vertices.
  generate_outer_vertices_map(extra_ovgids, start_ids, total_vertex_label_num,
                              ovg2l_maps, extra_ovgid_lists);
  extra_ovgids.clear();

  std::vector<std::shared_ptr<vid_array_t>> ovgid_lists(total_vertex_label_num);
  // Append extra ovgid_lists with origin ovgid_lists to make it complete
  for (label_id_t i = 0; i < total_vertex_label_num; ++i) {
    vid_builder_t ovgid_list_builder;
    // If the ovgid have no new entries, leave it empty to indicate using the
    // old ovgid when seal.
    if (extra_ovgid_lists[i]->length() != 0) {
      if (i < vertex_label_num_) {
        ARROW_OK_OR_RAISE(ovgid_list_builder.AppendValues(
            ovgid_lists_[i]->raw_values(), ovgid_lists_[i]->length()));
      }
      ARROW_OK_OR_RAISE(ovgid_list_builder.AppendValues(
          extra_ovgid_lists[i]->raw_values(), extra_ovgid_lists[i]->length()));
    }
    ARROW_OK_OR_RAISE(ovgid_list_builder.Finish(&ovgid_lists[i]));

    ovnums[i] = i < vertex_label_num_ ? ovgid_lists_[i]->length() : 0;
    ovnums[i] += extra_ovgid_lists[i]->length();
    tvnums[i] = ivnums[i] + ovnums[i];
  }

  // Gather all local id of new edges.
  // And delete the src/dst column in edge tables.
  std::vector<std::shared_ptr<vid_array_t>> edge_src, edge_dst;
  edge_src.resize(extra_edge_label_num);
  edge_dst.resize(extra_edge_label_num);
  for (int i = 0; i < extra_edge_label_num; ++i) {
    generate_local_id_list(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(0)->chunk(0)),
                           fid_, ovg2l_maps, concurrency, edge_src[i]);
    generate_local_id_list(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(1)->chunk(0)),
                           fid_, ovg2l_maps, concurrency, edge_dst[i]);
    std::shared_ptr<arrow::Table> tmp_table0;
    ARROW_OK_ASSIGN_OR_RAISE(tmp_table0, edge_tables[i]->RemoveColumn(0));
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[i], tmp_table0->RemoveColumn(0));
  }

  // Generate CSR vector of new edge tables.

  std::vector<std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>>>
      ie_lists(total_vertex_label_num);
  std::vector<std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>>>
      oe_lists(total_vertex_label_num);
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>> ie_offsets_lists(
      total_vertex_label_num);
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>> oe_offsets_lists(
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
        std::vector<int64_t> offsets(cur_offset_size);
        const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
        for (vid_t k = 0; k < prev_offset_size; ++k) {
          offsets[k] = offset_array[k];
        }
        for (vid_t k = prev_offset_size; k < cur_offset_size; ++k) {
          offsets[k] = offsets[k - 1];
        }
        arrow::Int64Builder builder;
        ARROW_OK_OR_RAISE(builder.AppendValues(offsets));
        ARROW_OK_OR_RAISE(builder.Finish(&ie_offsets_lists[v_label][e_label]));
      }
      std::vector<int64_t> offsets(cur_offset_size);
      const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
      for (size_t k = 0; k < prev_offset_size; ++k) {
        offsets[k] = offset_array[k];
      }
      for (size_t k = prev_offset_size; k < cur_offset_size; ++k) {
        offsets[k] = offsets[k - 1];
      }
      arrow::Int64Builder builder;
      ARROW_OK_OR_RAISE(builder.AppendValues(offsets));
      ARROW_OK_OR_RAISE(builder.Finish(&oe_offsets_lists[v_label][e_label]));
    }
  }

  for (label_id_t e_label = 0; e_label < total_edge_label_num; ++e_label) {
    std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_ie_lists(
        total_vertex_label_num);
    std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_oe_lists(
        total_vertex_label_num);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_ie_offset_lists(
        total_vertex_label_num);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_oe_offset_lists(
        total_vertex_label_num);

    // Process v_num...total_v_num  X  0...e_num  part.
    if (e_label < edge_label_num_) {
      if (directed_) {
        for (label_id_t v_label = vertex_label_num_;
             v_label < total_vertex_label_num; ++v_label) {
          PodArrayBuilder<nbr_unit_t> binary_builder;
          std::shared_ptr<arrow::FixedSizeBinaryArray> ie_array;
          ARROW_OK_OR_RAISE(binary_builder.Finish(&ie_array));

          sub_ie_lists[v_label] = ie_array;

          arrow::Int64Builder int64_builder;
          std::vector<int64_t> offset_vec(tvnums[v_label] + 1, 0);
          ARROW_OK_OR_RAISE(int64_builder.AppendValues(offset_vec));
          std::shared_ptr<arrow::Int64Array> ie_offset_array;
          ARROW_OK_OR_RAISE(int64_builder.Finish(&ie_offset_array));
          sub_ie_offset_lists[v_label] = ie_offset_array;
        }
      }
      for (label_id_t v_label = vertex_label_num_;
           v_label < total_vertex_label_num; ++v_label) {
        PodArrayBuilder<nbr_unit_t> binary_builder;
        std::shared_ptr<arrow::FixedSizeBinaryArray> oe_array;
        ARROW_OK_OR_RAISE(binary_builder.Finish(&oe_array));
        sub_oe_lists[v_label] = oe_array;

        arrow::Int64Builder int64_builder;
        std::vector<int64_t> offset_vec(tvnums[v_label] + 1, 0);
        ARROW_OK_OR_RAISE(int64_builder.AppendValues(offset_vec));
        std::shared_ptr<arrow::Int64Array> oe_offset_array;
        ARROW_OK_OR_RAISE(int64_builder.Finish(&oe_offset_array));
        sub_oe_offset_lists[v_label] = oe_offset_array;
      }
    } else {
      auto cur_label_index = e_label - edge_label_num_;
      // Process v_num...total_v_num  X  0...e_num  part.
      if (directed_) {
        // Process 0...total_v_num  X  e_num...total_e_num  part.
        generate_directed_csr<vid_t, eid_t>(
            vid_parser_, edge_src[cur_label_index], edge_dst[cur_label_index],
            tvnums, total_vertex_label_num, concurrency, sub_oe_lists,
            sub_oe_offset_lists, is_multigraph_);
        generate_directed_csr<vid_t, eid_t>(
            vid_parser_, edge_dst[cur_label_index], edge_src[cur_label_index],
            tvnums, total_vertex_label_num, concurrency, sub_ie_lists,
            sub_ie_offset_lists, is_multigraph_);
      } else {
        generate_undirected_csr<vid_t, eid_t>(
            vid_parser_, edge_src[cur_label_index], edge_dst[cur_label_index],
            tvnums, total_vertex_label_num, concurrency, sub_oe_lists,
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

  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  builder.set_vertex_label_num_(total_vertex_label_num);
  builder.set_edge_label_num_(total_edge_label_num);

  auto schema = schema_;  // make a copy

  // Extra vertex table
  for (label_id_t extra_label_id = 0; extra_label_id < extra_vertex_label_num;
       ++extra_label_id) {
    int label_id = vertex_label_num_ + extra_label_id;
    auto const& table = vertex_tables[extra_label_id];
    builder.set_vertex_tables_(label_id,
                               TableBuilder(client, table).Seal(client));

    // build schema entry for the new vertex
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);

    auto entry = schema.CreateEntry(kvs["label"], "VERTEX");
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    std::string retain_oid = kvs["retain_oid"];
    if (retain_oid == "1" || retain_oid == "true") {
      int column_index = table->num_columns() - 1;
      entry->AddPrimaryKey(table->schema()->field(column_index)->name());
    }
  }

  // extra edge tables
  for (label_id_t extra_label_id = 0; extra_label_id < extra_edge_label_num;
       ++extra_label_id) {
    label_id_t label_id = edge_label_num_ + extra_label_id;
    auto const& table = edge_tables[extra_label_id];
    builder.set_edge_tables_(label_id,
                             TableBuilder(client, table).Seal(client));

    // build schema entry for the new edge
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);
    auto entry = schema.CreateEntry(kvs["label"], "EDGE");
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    for (const auto& rel : edge_relations[extra_label_id]) {
      entry->AddRelation(rel.first, rel.second);
    }
  }

  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());

  ThreadGroup tg;
  {
    // ivnums, ovnums, tvnums
    auto fn = [this, &builder, &ivnums, &ovnums, &tvnums](Client* client) {
      vineyard::ArrayBuilder<vid_t> ivnums_builder(*client, ivnums);
      vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums);
      vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums);
      builder.set_ivnums_(ivnums_builder.Seal(*client));
      builder.set_ovnums_(ovnums_builder.Seal(*client));
      builder.set_tvnums_(tvnums_builder.Seal(*client));
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
    auto fn = [this, &builder, i, &ovgid_lists, &ovg2l_maps](Client* client) {
      if (i >= vertex_label_num_ || ovgid_lists[i]->length() != 0) {
        vineyard::NumericArrayBuilder<vid_t> ovgid_list_builder(*client,
                                                                ovgid_lists[i]);
        builder.set_ovgid_lists_(i, ovgid_list_builder.Seal(*client));
      }

      if (i >= vertex_label_num_ || !ovg2l_maps[i].empty()) {
        vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
            *client, std::move(ovg2l_maps[i]));
        builder.set_ovg2l_maps_(i, ovg2l_builder.Seal(*client));
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
                 &oe_offsets_lists](Client* client) {
        if (directed_) {
          if (!(i < vertex_label_num_ && j < edge_label_num_)) {
            vineyard::FixedSizeBinaryArrayBuilder ie_builder(*client,
                                                             ie_lists[i][j]);
            builder.set_ie_lists_(i, j, ie_builder.Seal(*client));
          }
          vineyard::NumericArrayBuilder<int64_t> ieo_builder(
              *client, ie_offsets_lists[i][j]);
          builder.set_ie_offsets_lists_(i, j, ieo_builder.Seal(*client));
        }
        if (!(i < vertex_label_num_ && j < edge_label_num_)) {
          vineyard::FixedSizeBinaryArrayBuilder oe_builder(*client,
                                                           oe_lists[i][j]);
          builder.set_oe_lists_(i, j, oe_builder.Seal(*client));
        }
        vineyard::NumericArrayBuilder<int64_t> oeo_builder(
            *client, oe_offsets_lists[i][j]);
        builder.set_oe_offsets_lists_(i, j, oeo_builder.Seal(*client));
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }
  }

  // wait all
  tg.TakeResults();

  builder.set_vm_ptr_(vm_ptr);

  return builder.Seal(client)->id();
}

/// Add a set of new vertex labels to graph. Vertex label id started from
/// vertex_label_num_.
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddNewVertexLabels(
    Client& client, std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
    ObjectID vm_id) {
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
    ARROW_OK_ASSIGN_OR_RAISE(
        vertex_tables[i],
        vertex_tables[i]->CombineChunks(arrow::default_memory_pool()));
    ivnums[vertex_label_num_ + i] =
        vm_ptr->GetInnerVertexSize(fid_, vertex_label_num_ + i);
    ovnums[vertex_label_num_ + i] = 0;
    tvnums[vertex_label_num_ + i] = ivnums[vertex_label_num_ + i];
  }

  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  builder.set_vertex_label_num_(total_vertex_label_num);

  auto schema = schema_;
  for (int extra_label_id = 0; extra_label_id < extra_vertex_label_num;
       ++extra_label_id) {
    int label_id = vertex_label_num_ + extra_label_id;
    auto const& table = vertex_tables[extra_label_id];
    builder.set_vertex_tables_(label_id,
                               TableBuilder(client, table).Seal(client));

    // build schema entry for the new vertex
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);

    auto entry = schema.CreateEntry(kvs["label"], "VERTEX");
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    std::string retain_oid = kvs["retain_oid"];
    if (retain_oid == "1" || retain_oid == "true") {
      int col_id = table->num_columns() - 1;
      entry->AddPrimaryKey(table->schema()->field(col_id)->name());
    }
  }
  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());

  vineyard::ArrayBuilder<vid_t> ivnums_builder(client, ivnums);
  vineyard::ArrayBuilder<vid_t> ovnums_builder(client, ovnums);
  vineyard::ArrayBuilder<vid_t> tvnums_builder(client, tvnums);

  builder.set_ivnums_(ivnums_builder.Seal(client));
  builder.set_ovnums_(ovnums_builder.Seal(client));
  builder.set_tvnums_(tvnums_builder.Seal(client));

  // Assign additional meta for new vertex labels
  std::vector<std::vector<std::shared_ptr<vineyard::FixedSizeBinaryArray>>>
      vy_ie_lists, vy_oe_lists;
  std::vector<std::vector<std::shared_ptr<vineyard::NumericArray<int64_t>>>>
      vy_ie_offsets_lists, vy_oe_offsets_lists;

  for (label_id_t extra_label_id = 0; extra_label_id < extra_vertex_label_num;
       ++extra_label_id) {
    label_id_t label_id = vertex_label_num_ + extra_label_id;
    vineyard::NumericArrayBuilder<vid_t> ovgid_list_builder(client);
    builder.set_ovgid_lists_(label_id, ovgid_list_builder.Seal(client));

    vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(client);
    builder.set_ovg2l_maps_(label_id, ovg2l_builder.Seal(client));
  }

  for (label_id_t i = 0; i < extra_vertex_label_num; ++i) {
    for (label_id_t j = 0; j < edge_label_num_; ++j) {
      label_id_t vertex_label_id = vertex_label_num_ + i;
      if (directed_) {
        vineyard::FixedSizeBinaryArrayBuilder ie_builder(
            client, arrow::fixed_size_binary(sizeof(nbr_unit_t)));
        builder.set_ie_lists_(vertex_label_id, j, ie_builder.Seal(client));

        arrow::Int64Builder int64_builder;
        // Offset vector's length is tvnum + 1
        std::vector<int64_t> offset_vec(tvnums[vertex_label_id] + 1, 0);
        ARROW_OK_OR_RAISE(int64_builder.AppendValues(offset_vec));
        std::shared_ptr<arrow::Int64Array> ie_offset_array;
        ARROW_OK_OR_RAISE(int64_builder.Finish(&ie_offset_array));

        vineyard::NumericArrayBuilder<int64_t> ie_offset_builder(
            client, ie_offset_array);
        builder.set_ie_offsets_lists_(vertex_label_id, j,
                                      ie_offset_builder.Seal(client));
      }

      vineyard::FixedSizeBinaryArrayBuilder oe_builder(
          client, arrow::fixed_size_binary(sizeof(nbr_unit_t)));
      builder.set_oe_lists_(vertex_label_id, j, oe_builder.Seal(client));

      arrow::Int64Builder int64_builder;
      // Offset vector's length is tvnum + 1
      std::vector<int64_t> offset_vec(tvnums[vertex_label_id] + 1, 0);
      ARROW_OK_OR_RAISE(int64_builder.AppendValues(offset_vec));
      std::shared_ptr<arrow::Int64Array> oe_offset_array;
      ARROW_OK_OR_RAISE(int64_builder.Finish(&oe_offset_array));
      vineyard::NumericArrayBuilder<int64_t> oe_offset_builder(client,
                                                               oe_offset_array);
      builder.set_oe_offsets_lists_(vertex_label_id, j,
                                    oe_offset_builder.Seal(client));
    }
  }

  builder.set_vm_ptr_(vm_ptr);
  return builder.Seal(client)->id();
}

/// Add a set of new edge labels to graph. Edge label id started from
/// edge_label_num_.
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddNewEdgeLabels(
    Client& client, std::vector<std::shared_ptr<arrow::Table>>&& edge_tables,
    const std::vector<std::set<std::pair<std::string, std::string>>>&
        edge_relations,
    int concurrency) {
  int extra_edge_label_num = edge_tables.size();
  int total_edge_label_num = edge_label_num_ + extra_edge_label_num;

  // Collect extra outer vertices.
  auto collect_extra_outer_vertices =
      [this](const std::shared_ptr<vid_array_t>& gid_array,
             std::vector<std::vector<vid_t>>& extra_ovgids) {
        const VID_T* arr = gid_array->raw_values();
        for (int64_t i = 0; i < gid_array->length(); ++i) {
          fid_t fid = vid_parser_.GetFid(arr[i]);
          label_id_t label_id = vid_parser_.GetLabelId(arr[i]);
          auto cur_map = ovg2l_maps_ptr_[label_id];
          if (fid != fid_ && cur_map->find(arr[i]) == cur_map->end()) {
            extra_ovgids[vid_parser_.GetLabelId(arr[i])].push_back(arr[i]);
          }
        }
      };

  std::vector<std::vector<vid_t>> extra_ovgids(vertex_label_num_);
  for (int i = 0; i < extra_edge_label_num; ++i) {
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[i], edge_tables[i]->CombineChunks(
                                                 arrow::default_memory_pool()));

    collect_extra_outer_vertices(
        std::dynamic_pointer_cast<
            typename vineyard::ConvertToArrowType<vid_t>::ArrayType>(
            edge_tables[i]->column(0)->chunk(0)),
        extra_ovgids);
    collect_extra_outer_vertices(
        std::dynamic_pointer_cast<
            typename vineyard::ConvertToArrowType<vid_t>::ArrayType>(
            edge_tables[i]->column(1)->chunk(0)),
        extra_ovgids);
  }

  // Construct the new start value of lid of extra outer vertices
  std::vector<vid_t> start_ids(vertex_label_num_);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]) + ovnums_[i];
  }

  // Make a copy of ovg2l map, since we need to add some extra outer vertices
  // pulled in this fragment by new edges.
  std::vector<ovg2l_map_t> ovg2l_maps(vertex_label_num_);
  for (int i = 0; i < vertex_label_num_; ++i) {
    for (auto iter = ovg2l_maps_ptr_[i]->begin();
         iter != ovg2l_maps_ptr_[i]->end(); ++iter) {
      ovg2l_maps[i].emplace(iter->first, iter->second);
    }
  }

  std::vector<std::shared_ptr<vid_array_t>> extra_ovgid_lists(
      vertex_label_num_);
  // Add extra outer vertices to ovg2l map, and collect distinct gid of extra
  // outer vertices.
  generate_outer_vertices_map(extra_ovgids, start_ids, vertex_label_num_,
                              ovg2l_maps, extra_ovgid_lists);
  extra_ovgids.clear();

  std::vector<vid_t> ovnums(vertex_label_num_), tvnums(vertex_label_num_);
  std::vector<std::shared_ptr<vid_array_t>> ovgid_lists(vertex_label_num_);
  // Append extra ovgid_lists with origin ovgid_lists to make it complete
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    vid_builder_t ovgid_list_builder;
    // If the ovgid have no new entries, leave it empty to indicate using the
    // old ovgid when seal.
    if (extra_ovgid_lists[i]->length() != 0) {
      ARROW_OK_OR_RAISE(ovgid_list_builder.AppendValues(
          ovgid_lists_[i]->raw_values(), ovgid_lists_[i]->length()));
      ARROW_OK_OR_RAISE(ovgid_list_builder.AppendValues(
          extra_ovgid_lists[i]->raw_values(), extra_ovgid_lists[i]->length()));
    }
    ARROW_OK_OR_RAISE(ovgid_list_builder.Finish(&ovgid_lists[i]));

    ovnums[i] = ovgid_lists_[i]->length() + extra_ovgid_lists[i]->length();
    tvnums[i] = ivnums_[i] + ovnums[i];
  }

  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>>
      ie_offsets_lists_expanded(vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>>
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
      vid_t cur_offset_size = tvnums[v_label] + 1;
      if (directed_) {
        std::vector<int64_t> offsets(cur_offset_size);
        const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
        for (vid_t k = 0; k < prev_offset_size; ++k) {
          offsets[k] = offset_array[k];
        }
        for (vid_t k = prev_offset_size; k < cur_offset_size; ++k) {
          offsets[k] = offsets[k - 1];
        }
        arrow::Int64Builder builder;
        ARROW_OK_OR_RAISE(builder.AppendValues(offsets));
        ARROW_OK_OR_RAISE(
            builder.Finish(&ie_offsets_lists_expanded[v_label][e_label]));
      }
      std::vector<int64_t> offsets(cur_offset_size);
      const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
      for (size_t k = 0; k < prev_offset_size; ++k) {
        offsets[k] = offset_array[k];
      }
      for (size_t k = prev_offset_size; k < cur_offset_size; ++k) {
        offsets[k] = offsets[k - 1];
      }
      arrow::Int64Builder builder;
      ARROW_OK_OR_RAISE(builder.AppendValues(offsets));
      ARROW_OK_OR_RAISE(
          builder.Finish(&oe_offsets_lists_expanded[v_label][e_label]));
    }
  }
  // Gather all local id of new edges.
  // And delete the src/dst column in edge tables.
  std::vector<std::shared_ptr<vid_array_t>> edge_src, edge_dst;
  edge_src.resize(extra_edge_label_num);
  edge_dst.resize(extra_edge_label_num);
  for (int i = 0; i < extra_edge_label_num; ++i) {
    generate_local_id_list(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(0)->chunk(0)),
                           fid_, ovg2l_maps, concurrency, edge_src[i]);
    generate_local_id_list(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(1)->chunk(0)),
                           fid_, ovg2l_maps, concurrency, edge_dst[i]);
    std::shared_ptr<arrow::Table> tmp_table0;
    ARROW_OK_ASSIGN_OR_RAISE(tmp_table0, edge_tables[i]->RemoveColumn(0));
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables[i], tmp_table0->RemoveColumn(0));
  }

  // Generate CSR vector of new edge tables.

  std::vector<std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>>>
      ie_lists(vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>>>
      oe_lists(vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>> ie_offsets_lists(
      vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>> oe_offsets_lists(
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
    std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_ie_lists(
        vertex_label_num_);
    std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_oe_lists(
        vertex_label_num_);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_ie_offset_lists(
        vertex_label_num_);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_oe_offset_lists(
        vertex_label_num_);
    if (directed_) {
      generate_directed_csr<vid_t, eid_t>(
          vid_parser_, edge_src[e_label], edge_dst[e_label], tvnums,
          vertex_label_num_, concurrency, sub_oe_lists, sub_oe_offset_lists,
          is_multigraph_);
      generate_directed_csr<vid_t, eid_t>(
          vid_parser_, edge_dst[e_label], edge_src[e_label], tvnums,
          vertex_label_num_, concurrency, sub_ie_lists, sub_ie_offset_lists,
          is_multigraph_);
    } else {
      generate_undirected_csr<vid_t, eid_t>(
          vid_parser_, edge_src[e_label], edge_dst[e_label], tvnums,
          vertex_label_num_, concurrency, sub_oe_lists, sub_oe_offset_lists,
          is_multigraph_);
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

  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  builder.set_edge_label_num_(total_edge_label_num);

  auto schema = schema_;
  for (label_id_t extra_label_id = 0; extra_label_id < extra_edge_label_num;
       ++extra_label_id) {
    label_id_t label_id = edge_label_num_ + extra_label_id;
    auto const& table = edge_tables[extra_label_id];

    builder.set_edge_tables_(label_id,
                             TableBuilder(client, table).Seal(client));

    // build schema entry for the new edge
    std::unordered_map<std::string, std::string> kvs;
    table->schema()->metadata()->ToUnorderedMap(&kvs);
    auto entry = schema.CreateEntry(kvs["label"], "EDGE");
    for (auto const& field : table->fields()) {
      entry->AddProperty(field->name(), field->type());
    }
    for (const auto& rel : edge_relations[extra_label_id]) {
      entry->AddRelation(rel.first, rel.second);
    }
  }

  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());

  ThreadGroup tg;
  {
    auto fn = [this, &builder, &ovnums, &tvnums](Client* client) {
      vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums);
      vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums);
      builder.set_ovnums_(ovnums_builder.Seal(*client));
      builder.set_tvnums_(tvnums_builder.Seal(*client));
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
    auto fn = [this, &builder, i, &ovgid_lists, &ovg2l_maps](Client* client) {
      if (ovgid_lists[i]->length() != 0) {
        vineyard::NumericArrayBuilder<vid_t> ovgid_list_builder(*client,
                                                                ovgid_lists[i]);
        builder.set_ovgid_lists_(i, ovgid_list_builder.Seal(*client));
      }

      if (!ovg2l_maps[i].empty()) {
        vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
            *client, std::move(ovg2l_maps[i]));
        builder.set_ovg2l_maps_(i, ovg2l_builder.Seal(*client));
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
                 &oe_offsets_lists](Client* client) {
        label_id_t edge_label_id = edge_label_num_ + j;
        if (directed_) {
          vineyard::FixedSizeBinaryArrayBuilder ie_builder(*client,
                                                           ie_lists[i][j]);
          builder.set_ie_lists_(i, edge_label_id, ie_builder.Seal(*client));

          vineyard::NumericArrayBuilder<int64_t> ieo_builder(
              *client, ie_offsets_lists[i][j]);
          builder.set_ie_offsets_lists_(i, edge_label_id,
                                        ieo_builder.Seal(*client));
        }
        vineyard::FixedSizeBinaryArrayBuilder oe_builder(*client,
                                                         oe_lists[i][j]);
        builder.set_oe_lists_(i, edge_label_id, oe_builder.Seal(*client));

        vineyard::NumericArrayBuilder<int64_t> oeo_builder(
            *client, oe_offsets_lists[i][j]);
        builder.set_oe_offsets_lists_(i, edge_label_id,
                                      oeo_builder.Seal(*client));
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }
  }
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    for (label_id_t j = 0; j < edge_label_num_; ++j) {
      auto fn = [this, &builder, i, j, &ie_offsets_lists_expanded,
                 &oe_offsets_lists_expanded](Client* client) {
        label_id_t edge_label_id = j;
        if (directed_) {
          vineyard::NumericArrayBuilder<int64_t> ieo_builder_expanded(
              *client, ie_offsets_lists_expanded[i][j]);
          builder.set_ie_offsets_lists_(i, edge_label_id,
                                        ieo_builder_expanded.Seal(*client));
        }
        vineyard::NumericArrayBuilder<int64_t> oeo_builder_expanded(
            *client, oe_offsets_lists_expanded[i][j]);
        builder.set_oe_offsets_lists_(i, edge_label_id,
                                      oeo_builder_expanded.Seal(*client));
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }
  }
  tg.TakeResults();

  return builder.Seal(client)->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
vineyard::Status BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::Build(
    vineyard::Client& client) {
  ThreadGroup tg;
  {
    auto fn = [this](Client* client) {
      vineyard::ArrayBuilder<vid_t> ivnums_builder(*client, ivnums_);
      vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums_);
      vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums_);
      this->set_ivnums_(std::dynamic_pointer_cast<vineyard::Array<vid_t>>(
          ivnums_builder.Seal(*client)));
      this->set_ovnums_(std::dynamic_pointer_cast<vineyard::Array<vid_t>>(
          ovnums_builder.Seal(*client)));
      this->set_tvnums_(std::dynamic_pointer_cast<vineyard::Array<vid_t>>(
          tvnums_builder.Seal(*client)));
      return Status::OK();
    };

    tg.AddTask(fn, &client);
  }

  this->vertex_tables_.resize(this->vertex_label_num_);
  this->ovgid_lists_.resize(this->vertex_label_num_);
  this->ovg2l_maps_.resize(this->vertex_label_num_);
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    auto fn = [this, i](Client* client) {
      vineyard::TableBuilder vt(*client, vertex_tables_[i]);
      this->set_vertex_tables_(
          i, std::dynamic_pointer_cast<vineyard::Table>(vt.Seal(*client)));

      vineyard::NumericArrayBuilder<vid_t> ovgid_list_builder(*client,
                                                              ovgid_lists_[i]);
      this->set_ovgid_lists_(
          i, std::dynamic_pointer_cast<vineyard::NumericArray<vid_t>>(
                 ovgid_list_builder.Seal(*client)));

      vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
          *client, std::move(ovg2l_maps_[i]));
      this->set_ovg2l_maps_(
          i, std::dynamic_pointer_cast<vineyard::Hashmap<vid_t, vid_t>>(
                 ovg2l_builder.Seal(*client)));
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  this->edge_tables_.resize(this->edge_label_num_);
  for (label_id_t i = 0; i < this->edge_label_num_; ++i) {
    auto fn = [this, i](Client* client) {
      vineyard::TableBuilder et(*client, edge_tables_[i]);
      this->set_edge_tables_(
          i, std::dynamic_pointer_cast<vineyard::Table>(et.Seal(*client)));
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  if (this->directed_) {
    this->ie_lists_.resize(this->vertex_label_num_);
    this->ie_offsets_lists_.resize(this->vertex_label_num_);
  }
  this->oe_lists_.resize(this->vertex_label_num_);
  this->oe_offsets_lists_.resize(this->vertex_label_num_);
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    if (this->directed_) {
      this->ie_lists_[i].resize(this->edge_label_num_);
      this->ie_offsets_lists_[i].resize(this->edge_label_num_);
    }
    this->oe_lists_[i].resize(this->edge_label_num_);
    this->oe_offsets_lists_[i].resize(this->edge_label_num_);
    for (label_id_t j = 0; j < this->edge_label_num_; ++j) {
      auto fn = [this, i, j](Client* client) {
        if (this->directed_) {
          vineyard::FixedSizeBinaryArrayBuilder ie_builder(*client,
                                                           ie_lists_[i][j]);
          this->set_ie_lists_(i, j, ie_builder.Seal(*client));
        }
        {
          vineyard::FixedSizeBinaryArrayBuilder oe_builder(*client,
                                                           oe_lists_[i][j]);
          this->set_oe_lists_(i, j, oe_builder.Seal(*client));
        }
        if (this->directed_) {
          vineyard::NumericArrayBuilder<int64_t> ieo(*client,
                                                     ie_offsets_lists_[i][j]);
          this->set_ie_offsets_lists_(i, j, ieo.Seal(*client));
        }
        {
          vineyard::NumericArrayBuilder<int64_t> oeo(*client,
                                                     oe_offsets_lists_[i][j]);
          this->set_oe_offsets_lists_(i, j, oeo.Seal(*client));
        }
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }
  }

  tg.TakeResults();

  this->set_vm_ptr_(vm_ptr_);

  this->set_oid_type(type_name<oid_t>());
  this->set_vid_type(type_name<vid_t>());

  return Status::OK();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<void>
BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::Init(
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

  vid_parser_.Init(this->fnum_, this->vertex_label_num_);

  BOOST_LEAF_CHECK(initVertices(std::move(vertex_tables)));
  BOOST_LEAF_CHECK(initEdges(std::move(edge_tables), concurrency));
  return {};
}

// | prop_0 | prop_1 | ... |
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<void>
BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::initVertices(
    std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables) {
  assert(vertex_tables.size() == static_cast<size_t>(this->vertex_label_num_));
  vertex_tables_.resize(this->vertex_label_num_);
  ivnums_.resize(this->vertex_label_num_);
  ovnums_.resize(this->vertex_label_num_);
  tvnums_.resize(this->vertex_label_num_);
  for (size_t i = 0; i < vertex_tables.size(); ++i) {
    ARROW_OK_ASSIGN_OR_RAISE(
        vertex_tables_[i],
        vertex_tables[i]->CombineChunks(arrow::default_memory_pool()));
    ivnums_[i] = vm_ptr_->GetInnerVertexSize(this->fid_, i);
  }
  return {};
}

// | src_id(generated) | dst_id(generated) | prop_0 | prop_1
// | ... |
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<void>
BasicArrowFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::initEdges(
    std::vector<std::shared_ptr<arrow::Table>>&& edge_tables, int concurrency) {
  assert(edge_tables.size() == static_cast<size_t>(this->edge_label_num_));
  std::vector<std::shared_ptr<vid_array_t>> edge_src, edge_dst;
  edge_src.resize(this->edge_label_num_);
  edge_dst.resize(this->edge_label_num_);

  edge_tables_.resize(this->edge_label_num_);
  std::vector<std::vector<vid_t>> collected_ovgids(this->vertex_label_num_);

  for (size_t i = 0; i < edge_tables.size(); ++i) {
    std::shared_ptr<arrow::Table> combined_table;
    ARROW_OK_ASSIGN_OR_RAISE(combined_table, edge_tables[i]->CombineChunks(
                                                 arrow::default_memory_pool()));
    edge_tables[i].swap(combined_table);

    collect_outer_vertices(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(0)->chunk(0)),
                           this->fid_, collected_ovgids);
    collect_outer_vertices(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(1)->chunk(0)),
                           this->fid_, collected_ovgids);
  }
  std::vector<vid_t> start_ids(this->vertex_label_num_);
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]);
  }
  generate_outer_vertices_map<vid_t>(collected_ovgids, start_ids,
                                     this->vertex_label_num_, ovg2l_maps_,
                                     ovgid_lists_);
  collected_ovgids.clear();
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    ovnums_[i] = ovgid_lists_[i]->length();
    tvnums_[i] = ivnums_[i] + ovnums_[i];
  }

  for (size_t i = 0; i < edge_tables.size(); ++i) {
    generate_local_id_list(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(0)->chunk(0)),
                           this->fid_, ovg2l_maps_, concurrency, edge_src[i]);
    generate_local_id_list(vid_parser_,
                           std::dynamic_pointer_cast<vid_array_t>(
                               edge_tables[i]->column(1)->chunk(0)),
                           this->fid_, ovg2l_maps_, concurrency, edge_dst[i]);

    std::shared_ptr<arrow::Table> tmp_table0;
    ARROW_OK_ASSIGN_OR_RAISE(tmp_table0, edge_tables[i]->RemoveColumn(0));
    ARROW_OK_ASSIGN_OR_RAISE(edge_tables_[i], tmp_table0->RemoveColumn(0));

    edge_tables[i].reset();
  }

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
    std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_ie_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_oe_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_ie_offset_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_oe_offset_lists(
        this->vertex_label_num_);
    if (this->directed_) {
      generate_directed_csr<vid_t, eid_t>(
          vid_parser_, edge_src[e_label], edge_dst[e_label], tvnums_,
          this->vertex_label_num_, concurrency, sub_oe_lists,
          sub_oe_offset_lists, this->is_multigraph_);
      generate_directed_csr<vid_t, eid_t>(
          vid_parser_, edge_dst[e_label], edge_src[e_label], tvnums_,
          this->vertex_label_num_, concurrency, sub_ie_lists,
          sub_ie_offset_lists, this->is_multigraph_);
    } else {
      generate_undirected_csr<vid_t, eid_t>(
          vid_parser_, edge_src[e_label], edge_dst[e_label], tvnums_,
          this->vertex_label_num_, concurrency, sub_oe_lists,
          sub_oe_offset_lists, this->is_multigraph_);
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
  return {};
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BUILDER_IMPL_H_
