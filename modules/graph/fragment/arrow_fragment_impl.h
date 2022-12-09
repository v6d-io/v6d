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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_IMPL_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_IMPL_H_

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
void ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::PrepareToRunApp(
    const grape::CommSpec& comm_spec, grape::PrepareConf conf) {
  if (conf.message_strategy ==
      grape::MessageStrategy::kAlongEdgeToOuterVertex) {
    initDestFidList(true, true, iodst_, iodoffset_);
  } else if (conf.message_strategy ==
             grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
    initDestFidList(true, false, idst_, idoffset_);
  } else if (conf.message_strategy ==
             grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
    initDestFidList(false, true, odst_, odoffset_);
  }
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddVertexColumns(
    vineyard::Client& client,
    const std::map<
        label_id_t,
        std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>>
        columns,
    bool replace) {
  return AddVertexColumnsImpl<arrow::Array>(client, columns, replace);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddVertexColumns(
    vineyard::Client& client,
    const std::map<label_id_t,
                   std::vector<std::pair<std::string,
                                         std::shared_ptr<arrow::ChunkedArray>>>>
        columns,
    bool replace) {
  return AddVertexColumnsImpl<arrow::ChunkedArray>(client, columns, replace);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
template <typename ArrayType>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddVertexColumnsImpl(
    vineyard::Client& client,
    const std::map<
        label_id_t,
        std::vector<std::pair<std::string, std::shared_ptr<ArrayType>>>>
        columns,
    bool replace) {
  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  auto schema = schema_;

  /// If replace == true, invalidate all previous properties that have new
  /// columns.
  if (replace) {
    for (auto& pair : columns) {
      auto label_id = pair.first;
      auto& entry = schema.GetMutableEntry(label_id, "VERTEX");
      for (size_t i = 0; i < entry.props_.size(); ++i) {
        entry.InvalidateProperty(i);
      }
    }
  }

  for (label_id_t label_id = 0; label_id < vertex_label_num_; ++label_id) {
    std::string table_name =
        generate_name_with_suffix("vertex_tables", label_id);
    if (columns.find(label_id) != columns.end()) {
      auto& table = this->vertex_tables_[label_id];
      vineyard::TableExtender extender(client, table);

      auto& vec = columns.at(label_id);
      for (auto& pair : vec) {
        auto status = extender.AddColumn(client, pair.first, pair.second);
        CHECK(status.ok());
      }
      auto new_table =
          std::dynamic_pointer_cast<vineyard::Table>(extender.Seal(client));
      builder.set_vertex_tables_(label_id, new_table);
      auto& entry =
          schema.GetMutableEntry(schema.GetVertexLabelName(label_id), "VERTEX");
      for (size_t index = table->num_columns();
           index < new_table->num_columns(); ++index) {
        entry.AddProperty(new_table->field(index)->name(),
                          new_table->field(index)->type());
      }
    }
  }
  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());
  return builder.Seal(client)->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddEdgeColumns(
    vineyard::Client& client,
    const std::map<
        label_id_t,
        std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>>
        columns,
    bool replace) {
  return AddEdgeColumnsImpl<arrow::Array>(client, columns, replace);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddEdgeColumns(
    vineyard::Client& client,
    const std::map<label_id_t,
                   std::vector<std::pair<std::string,
                                         std::shared_ptr<arrow::ChunkedArray>>>>
        columns,
    bool replace) {
  return AddEdgeColumnsImpl<arrow::ChunkedArray>(client, columns, replace);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
template <typename ArrayType>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::AddEdgeColumnsImpl(
    vineyard::Client& client,
    const std::map<
        label_id_t,
        std::vector<std::pair<std::string, std::shared_ptr<ArrayType>>>>
        columns,
    bool replace) {
  vineyard::ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  auto schema = schema_;

  if (replace) {
    for (auto& pair : columns) {
      auto label_id = pair.first;
      auto& entry = schema.GetMutableEntry(label_id, "EDGE");
      for (size_t i = 0; i < entry.props_.size(); ++i) {
        entry.InvalidateProperty(i);
      }
    }
  }
  for (label_id_t label_id = 0; label_id < edge_label_num_; ++label_id) {
    if (columns.find(label_id) != columns.end()) {
      auto& table = this->edge_tables_[label_id];
      vineyard::TableExtender extender(client, table);

      auto& vec = columns.at(label_id);
      for (auto& pair : vec) {
        auto status = extender.AddColumn(client, pair.first, pair.second);
        CHECK(status.ok());
      }
      auto new_table =
          std::dynamic_pointer_cast<vineyard::Table>(extender.Seal(client));
      builder.set_edge_tables_(label_id, new_table);
      auto& entry =
          schema.GetMutableEntry(schema.GetEdgeLabelName(label_id), "EDGE");
      for (size_t index = table->num_columns();
           index < new_table->num_columns(); ++index) {
        entry.AddProperty(new_table->field(index)->name(),
                          new_table->field(index)->type());
      }
    }
  }
  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());
  return builder.Seal(client)->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::Project(
    vineyard::Client& client,
    std::map<label_id_t, std::vector<label_id_t>> vertices,
    std::map<label_id_t, std::vector<label_id_t>> edges) {
  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);

  auto schema = schema_;

  std::vector<label_id_t> vertex_labels, edge_labels;
  std::vector<std::vector<prop_id_t>> vertex_properties, edge_properties;

  for (auto& pair : vertices) {
    vertex_labels.push_back(pair.first);
    vertex_properties.push_back(pair.second);
  }
  for (auto& pair : edges) {
    edge_labels.push_back(pair.first);
    edge_properties.push_back(pair.second);
  }

  auto remove_invalid_relation =
      [&schema](const std::vector<label_id_t>& edge_labels,
                std::map<label_id_t, std::vector<label_id_t>> vertices) {
        std::string type = "EDGE";
        for (size_t i = 0; i < edge_labels.size(); ++i) {
          auto& entry = schema.GetMutableEntry(edge_labels[i], type);
          auto& relations = entry.relations;
          std::vector<std::pair<std::string, std::string>> valid_relations;
          for (auto& pair : relations) {
            auto src = schema.GetVertexLabelId(pair.first);
            auto dst = schema.GetVertexLabelId(pair.second);
            if (vertices.find(src) != vertices.end() &&
                vertices.find(dst) != vertices.end()) {
              valid_relations.push_back(pair);
            }
          }
          entry.relations = valid_relations;
        }
      };
  // Compute the set difference of reserved labels and all labels.
  auto invalidate_label = [&schema](const std::vector<label_id_t>& labels,
                                    std::string type, label_id_t label_num) {
    auto it = labels.begin();
    for (label_id_t i = 0; i < label_num; ++i) {
      if (it == labels.end() || i < *it) {
        if (type == "VERTEX") {
          schema.InvalidateVertex(i);
        } else {
          schema.InvalidateEdge(i);
        }
      } else {
        ++it;
      }
    }
  };

  auto invalidate_prop =
      [&schema](const std::vector<label_id_t>& labels, std::string type,
                const std::vector<std::vector<prop_id_t>>& props) {
        for (size_t i = 0; i < labels.size(); ++i) {
          auto& entry = schema.GetMutableEntry(labels[i], type);
          auto it1 = props[i].begin();
          auto it2 = props[i].end();
          prop_id_t prop_num = entry.props_.size();
          for (prop_id_t j = 0; j < prop_num; ++j) {
            if (it1 == it2 || j < *it1) {
              entry.InvalidateProperty(j);
            } else {
              ++it1;
            }
          }
        }
      };
  remove_invalid_relation(edge_labels, vertices);
  invalidate_prop(vertex_labels, "VERTEX", vertex_properties);
  invalidate_prop(edge_labels, "EDGE", edge_properties);
  invalidate_label(vertex_labels, "VERTEX",
                   static_cast<label_id_t>(schema.vertex_entries().size()));
  invalidate_label(edge_labels, "EDGE",
                   static_cast<label_id_t>(schema.edge_entries().size()));

  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());
  return builder.Seal(client)->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::TransformDirection(
    vineyard::Client& client, int concurrency) {
  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  builder.set_directed_(!directed_);

  std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>>
      oe_lists(vertex_label_num_);
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>> oe_offsets_lists(
      vertex_label_num_);

  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    oe_lists[v_label].resize(edge_label_num_);
    oe_offsets_lists[v_label].resize(edge_label_num_);
  }

  if (directed_) {
    bool is_multigraph = is_multigraph_;
    directedCSR2Undirected(client, oe_lists, oe_offsets_lists, concurrency,
                           is_multigraph);

    for (label_id_t i = 0; i < vertex_label_num_; ++i) {
      for (label_id_t j = 0; j < edge_label_num_; ++j) {
        builder.set_oe_lists_(i, j, oe_lists[i][j]);
        vineyard::NumericArrayBuilder<int64_t> oeo_builder(
            client, oe_offsets_lists[i][j]);
        builder.set_oe_offsets_lists_(i, j, oeo_builder.Seal(client));
      }
    }
    builder.set_is_multigraph_(is_multigraph);
  }

  return builder.Seal(client)->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::ConsolidateVertexColumns(
    vineyard::Client& client, const label_id_t vlabel,
    std::vector<std::string> const& prop_names,
    std::string const& consolidate_name) {
  std::vector<prop_id_t> props;
  for (auto const& name : prop_names) {
    int prop = schema_.GetVertexPropertyId(vlabel, name);
    if (prop == -1) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Vertex property '" + name + "' not found");
    }
    props.push_back(prop);
  }
  return ConsolidateVertexColumns(client, vlabel, props, consolidate_name);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::ConsolidateVertexColumns(
    vineyard::Client& client, const label_id_t vlabel,
    std::vector<prop_id_t> const& props, std::string const& consolidate_name) {
  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  auto schema = schema_;

  auto& table = this->vertex_tables_[vlabel];
  vineyard::TableConsolidator consolidator(client, table);
  VY_OK_OR_RAISE(consolidator.ConsolidateColumns(
      client, std::vector<int64_t>{props.begin(), props.end()},
      consolidate_name));
  auto new_table =
      std::dynamic_pointer_cast<vineyard::Table>(consolidator.Seal(client));
  builder.set_vertex_tables_(vlabel, new_table);
  auto& entry = schema.GetMutableEntry(vlabel, "VERTEX");

  // update schema: remove old props and add new merged props
  std::vector<prop_id_t> sorted_props = props;
  std::sort(sorted_props.begin(), sorted_props.end());
  for (size_t index = 0; index < sorted_props.size(); ++index) {
    entry.RemoveProperty(sorted_props[sorted_props.size() - 1 - index]);
  }
  entry.AddProperty(consolidate_name,
                    new_table->field(new_table->num_columns() - 1)->type());

  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());
  return builder.Seal(client)->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::ConsolidateEdgeColumns(
    vineyard::Client& client, const label_id_t elabel,
    std::vector<std::string> const& prop_names,
    std::string const& consolidate_name) {
  std::vector<prop_id_t> props;
  for (auto const& name : prop_names) {
    int prop = schema_.GetEdgePropertyId(elabel, name);
    if (prop == -1) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Edge property '" + name + "' not found");
    }
    props.push_back(prop);
  }
  return ConsolidateEdgeColumns(client, elabel, props, consolidate_name);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<vineyard::ObjectID>
ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::ConsolidateEdgeColumns(
    vineyard::Client& client, const label_id_t elabel,
    std::vector<prop_id_t> const& props, std::string const& consolidate_name) {
  ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> builder(*this);
  auto schema = schema_;

  auto& table = this->edge_tables_[elabel];
  vineyard::TableConsolidator consolidator(client, table);
  VY_OK_OR_RAISE(consolidator.ConsolidateColumns(
      client, std::vector<int64_t>{props.begin(), props.end()},
      consolidate_name));
  auto new_table =
      std::dynamic_pointer_cast<vineyard::Table>(consolidator.Seal(client));
  builder.set_edge_tables_(elabel, new_table);
  auto& entry = schema.GetMutableEntry(elabel, "EDGE");

  // update schema: remove old props and add new merged props
  std::vector<prop_id_t> sorted_props = props;
  std::sort(sorted_props.begin(), sorted_props.end());
  for (size_t index = 0; index < sorted_props.size(); ++index) {
    entry.RemoveProperty(sorted_props[sorted_props.size() - 1 - index]);
  }
  entry.AddProperty(consolidate_name,
                    new_table->field(new_table->num_columns() - 1)->type());

  std::string error_message;
  if (!schema.Validate(error_message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, error_message);
  }
  builder.set_schema_json_(schema.ToJSON());
  return builder.Seal(client)->id();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
void ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::initPointers() {
  edge_tables_columns_.resize(edge_label_num_);
  flatten_edge_tables_columns_.resize(edge_label_num_);
  for (label_id_t i = 0; i < edge_label_num_; ++i) {
    prop_id_t prop_num = static_cast<prop_id_t>(edge_tables_[i]->num_columns());
    edge_tables_columns_[i].resize(prop_num);
    if (edge_tables_[i]->num_rows() == 0) {
      continue;
    }
    for (prop_id_t j = 0; j < prop_num; ++j) {
      // the finalized etables are guaranteed to have been concatenate
      edge_tables_columns_[i][j] =
          get_arrow_array_data(edge_tables_[i]->column(j)->chunk(0));
    }
    flatten_edge_tables_columns_[i] = &edge_tables_columns_[i][0];
  }

  vertex_tables_columns_.resize(vertex_label_num_);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    auto vertex_table = vertex_tables_[i]->GetTable();
    prop_id_t prop_num = static_cast<prop_id_t>(vertex_table->num_columns());
    vertex_tables_columns_[i].resize(prop_num);
    if (vertex_table->num_rows() == 0) {
      continue;
    }
    for (prop_id_t j = 0; j < prop_num; ++j) {
      // the finalized vtables are guaranteed to have been concatenate
      vertex_tables_columns_[i][j] =
          get_arrow_array_data(vertex_table->column(j)->chunk(0));
    }
  }

  oe_ptr_lists_.resize(vertex_label_num_);
  oe_offsets_ptr_lists_.resize(vertex_label_num_);

  idst_.resize(vertex_label_num_);
  odst_.resize(vertex_label_num_);
  iodst_.resize(vertex_label_num_);

  idoffset_.resize(vertex_label_num_);
  odoffset_.resize(vertex_label_num_);
  iodoffset_.resize(vertex_label_num_);

  ovgid_lists_ptr_.resize(vertex_label_num_);
  ovg2l_maps_ptr_.resize(vertex_label_num_);
  for (label_id_t i = 0; i < vertex_label_num_; ++i) {
    ovgid_lists_ptr_[i] = ovgid_lists_[i]->GetArray()->raw_values();
    ovg2l_maps_ptr_[i] = ovg2l_maps_[i].get();

    oe_ptr_lists_[i].resize(edge_label_num_);
    oe_offsets_ptr_lists_[i].resize(edge_label_num_);

    idst_[i].resize(edge_label_num_);
    odst_[i].resize(edge_label_num_);
    iodst_[i].resize(edge_label_num_);

    idoffset_[i].resize(edge_label_num_);
    odoffset_[i].resize(edge_label_num_);
    iodoffset_[i].resize(edge_label_num_);

    for (label_id_t j = 0; j < edge_label_num_; ++j) {
      oe_ptr_lists_[i][j] = reinterpret_cast<const nbr_unit_t*>(
          oe_lists_[i][j]->GetArray()->raw_values());
      oe_offsets_ptr_lists_[i][j] =
          oe_offsets_lists_[i][j]->GetArray()->raw_values();
    }
  }

  if (directed_) {
    ie_ptr_lists_.resize(vertex_label_num_);
    ie_offsets_ptr_lists_.resize(vertex_label_num_);
    for (label_id_t i = 0; i < vertex_label_num_; ++i) {
      ie_ptr_lists_[i].resize(edge_label_num_);
      ie_offsets_ptr_lists_[i].resize(edge_label_num_);
      for (label_id_t j = 0; j < edge_label_num_; ++j) {
        ie_ptr_lists_[i][j] = reinterpret_cast<const nbr_unit_t*>(
            ie_lists_[i][j]->GetArray()->raw_values());
        ie_offsets_ptr_lists_[i][j] =
            ie_offsets_lists_[i][j]->GetArray()->raw_values();
      }
    }
  } else {
    ie_ptr_lists_ = oe_ptr_lists_;
    ie_offsets_ptr_lists_ = oe_offsets_ptr_lists_;
  }
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
void ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::initDestFidList(
    bool in_edge, bool out_edge,
    std::vector<std::vector<std::vector<fid_t>>>& fid_lists,
    std::vector<std::vector<std::vector<fid_t*>>>& fid_lists_offset) {
  for (auto v_label_id = 0; v_label_id < vertex_label_num_; v_label_id++) {
    auto ivnum_ = ivnums_[v_label_id];
    auto inner_vertices = InnerVertices(v_label_id);

    for (auto e_label_id = 0; e_label_id < edge_label_num_; e_label_id++) {
      std::vector<int> id_num(ivnum_, 0);
      std::set<fid_t> dstset;
      vertex_t v = *inner_vertices.begin();
      auto& fid_list = fid_lists[v_label_id][e_label_id];
      auto& fid_list_offset = fid_lists_offset[v_label_id][e_label_id];

      if (!fid_list_offset.empty()) {
        return;
      }
      fid_list_offset.resize(ivnum_ + 1, NULL);
      for (vid_t i = 0; i < ivnum_; ++i) {
        dstset.clear();
        if (in_edge) {
          auto es = GetIncomingAdjList(v, e_label_id);
          for (auto& e : es) {
            fid_t f = GetFragId(e.neighbor());
            if (f != fid_) {
              dstset.insert(f);
            }
          }
        }
        if (out_edge) {
          auto es = GetOutgoingAdjList(v, e_label_id);
          for (auto& e : es) {
            fid_t f = GetFragId(e.neighbor());
            if (f != fid_) {
              dstset.insert(f);
            }
          }
        }
        id_num[i] = dstset.size();
        for (auto fid : dstset) {
          fid_list.push_back(fid);
        }
        ++v;
      }

      fid_list.shrink_to_fit();
      fid_list_offset[0] = fid_list.data();
      for (vid_t i = 0; i < ivnum_; ++i) {
        fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
      }
    }
  }
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
void ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>::directedCSR2Undirected(
    vineyard::Client& client,
    std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>>&
        oe_lists,
    std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>>&
        oe_offsets_lists,
    int concurrency, bool& is_multigraph) {
  for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      const nbr_unit_t* ie = ie_ptr_lists_.at(v_label).at(e_label);
      const nbr_unit_t* oe = oe_ptr_lists_.at(v_label).at(e_label);
      const int64_t* ie_offset = ie_offsets_ptr_lists_.at(v_label).at(e_label);
      const int64_t* oe_offset = oe_offsets_ptr_lists_.at(v_label).at(e_label);

      // Merge edges from two array into one
      auto edge_builder =
          std::make_shared<vineyard::PodArrayBuilder<nbr_unit_t>>(
              client,
              ie_offset[tvnums_[v_label]] + oe_offset[tvnums_[v_label]]);
      nbr_unit_t* data = edge_builder->MutablePointer(0);
      arrow::Int64Builder offset_builder;
      CHECK_ARROW_ERROR(offset_builder.Append(0));
      size_t edge_offset = 0;
      for (size_t offset = 0; offset < static_cast<size_t>(tvnums_[v_label]);
           ++offset) {
        for (size_t k = ie_offset[offset];
             k < static_cast<size_t>(ie_offset[offset + 1]); ++k) {
          data[edge_offset++] = ie[k];
        }
        for (int k = oe_offset[offset]; k < oe_offset[offset + 1]; ++k) {
          data[edge_offset++] = oe[k];
        }
        CHECK_ARROW_ERROR(offset_builder.Append(edge_offset));
      }
      CHECK_ARROW_ERROR(
          offset_builder.Finish(&oe_offsets_lists[v_label][e_label]));

      sort_edges_with_respect_to_vertex(*edge_builder,
                                        oe_offsets_lists[v_label][e_label],
                                        tvnums_[v_label], concurrency);
      if (!is_multigraph) {
        check_is_multigraph(*edge_builder, oe_offsets_lists[v_label][e_label],
                            tvnums_[v_label], concurrency, is_multigraph);
      }

      oe_lists[v_label][e_label] = edge_builder;
    }
  }
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_IMPL_H_
