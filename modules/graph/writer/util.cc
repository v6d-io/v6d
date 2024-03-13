
/** Copyright 2020-2023 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "graph/writer/util.h"

#ifdef ENABLE_GAR

#include <unordered_map>
#include <vector>

#include "gar/util/adj_list_type.h"
#include "gar/util/data_type.h"
#include "gar/util/file_type.h"

namespace GAR = GraphArchive;

namespace vineyard {

boost::leaf::result<std::shared_ptr<GAR::VertexInfo>>
generate_vertex_info_with_schema(
    const Entry& entry, int64_t vertex_block_size, GAR::FileType file_type,
    const std::unordered_map<std::string, std::vector<std::string>>&
        selected_vertex_properties) {
  bool selected_property = false;
  if (selected_vertex_properties.find(entry.label) !=
      selected_vertex_properties.end()) {
    selected_property = true;
  }
  std::vector<GAR::Property> properties;
  for (const auto& prop : entry.props_) {
    if (selected_property &&
        std::find(selected_vertex_properties.at(entry.label).begin(),
                  selected_vertex_properties.at(entry.label).end(),
                  prop.name) ==
            selected_vertex_properties.at(entry.label).end()) {
      // Skip the property that is not selected.
      continue;
    }
    properties.emplace_back(GAR::Property(
        prop.name, GAR::DataType::ArrowDataTypeToDataType(prop.type), false));
  }
  GAR::PropertyGroupVector property_groups;
  if (!properties.empty()) {
    auto pg = GAR::CreatePropertyGroup(properties, file_type);
    if (pg == nullptr) {
      RETURN_GS_ERROR(
          ErrorCode::kGraphArError,
          "Failed to create property group for vertex " + entry.label);
    }
    property_groups.push_back(pg);
  }
  return GAR::CreateVertexInfo(entry.label, vertex_block_size, property_groups);
}

boost::leaf::result<void> generate_edge_infos_with_schema(
    const Entry& entry, int64_t vertex_block_size, int64_t edge_block_size,
    GAR::FileType file_type, const std::vector<std::string>& selected_vertices,
    const std::unordered_map<std::string, std::vector<std::string>>&
        selected_edge_properties,
    GAR::EdgeInfoVector& edge_infos) {
  bool selected_property = false;
  if (selected_edge_properties.find(entry.label) !=
      selected_edge_properties.end()) {
    selected_property = true;
  }
  GAR::AdjacentListVector default_adjacent_lists{
      GAR::CreateAdjacentList(GAR::AdjListType::ordered_by_source, file_type),
      GAR::CreateAdjacentList(GAR::AdjListType::ordered_by_dest, file_type)};
  std::vector<GAR::Property> properties;
  for (const auto& prop : entry.props_) {
    if (selected_property &&
        std::find(selected_edge_properties.at(entry.label).begin(),
                  selected_edge_properties.at(entry.label).end(), prop.name) ==
            selected_edge_properties.at(entry.label).end()) {
      // Skip the property that is not selected.
      continue;
    }
    properties.emplace_back(GAR::Property(
        prop.name, GAR::DataType::ArrowDataTypeToDataType(prop.type), false));
  }
  GAR::PropertyGroupVector property_groups;
  if (!properties.empty()) {
    auto pg = GAR::CreatePropertyGroup(properties, file_type);
    if (pg == nullptr) {
      RETURN_GS_ERROR(
          ErrorCode::kGraphArError,
          "Failed to create property group for edge " + entry.label);
    }
    property_groups.push_back(pg);
  }

  for (const auto& relation : entry.relations) {
    if (!selected_vertices.empty() &&
        (std::find(selected_vertices.begin(), selected_vertices.end(),
                   relation.first) == selected_vertices.end() ||
         std::find(selected_vertices.begin(), selected_vertices.end(),
                   relation.second) == selected_vertices.end())) {
      // Skip the relation that source or destination vertex type is not
      // selected.
      continue;
    }
    auto edge_info = GAR::CreateEdgeInfo(
        relation.first, entry.label, relation.second, edge_block_size,
        vertex_block_size, vertex_block_size, true /* directed */,
        default_adjacent_lists, property_groups);
    if (edge_info == nullptr) {
      RETURN_GS_ERROR(ErrorCode::kGraphArError,
                      "Failed to create edge info for relation " +
                          relation.first + " " + entry.label + " " +
                          relation.second);
    }
    edge_infos.emplace_back(edge_info);
  }
  return {};
}

boost::leaf::result<std::shared_ptr<GAR::GraphInfo>>
generate_graph_info_with_schema(
    const PropertyGraphSchema& schema, const std::string& graph_name,
    const std::string& path, int64_t vertex_block_size, int64_t edge_block_size,
    GAR::FileType file_type, const std::vector<std::string>& selected_vertices,
    const std::vector<std::string>& selected_edges,
    const std::unordered_map<std::string, std::vector<std::string>>&
        selected_vertex_properties,
    const std::unordered_map<std::string, std::vector<std::string>>&
        selected_edge_properties,
    bool store_in_local) {
  GAR::VertexInfoVector vertex_infos;
  GAR::EdgeInfoVector edge_infos;

  for (const auto& entry : schema.vertex_entries()) {
    if (!selected_vertices.empty() &&
        std::find(selected_vertices.begin(), selected_vertices.end(),
                  entry.label) == selected_vertices.end()) {
      // Skip the vertex type that is not selected.
      continue;
    }
    BOOST_LEAF_AUTO(vertex_info, generate_vertex_info_with_schema(
                                     entry, vertex_block_size, file_type,
                                     selected_vertex_properties));
    if (vertex_info == nullptr) {
      RETURN_GS_ERROR(ErrorCode::kGraphArError,
                      "Failed to create vertex info for " + entry.label);
    }
    vertex_infos.emplace_back(vertex_info);
  }

  for (const auto& entry : schema.edge_entries()) {
    if (!selected_edges.empty() &&
        std::find(selected_edges.begin(), selected_edges.end(), entry.label) ==
            selected_edges.end()) {
      // Skip the edge type that is not selected.
      continue;
    }
    BOOST_LEAF_CHECK(generate_edge_infos_with_schema(
        entry, vertex_block_size, edge_block_size, file_type, selected_vertices,
        selected_edge_properties, edge_infos));
  }
  std::unordered_map<std::string, std::string> extra_info;
  if (store_in_local) {
    extra_info[LOCAL_METADATA_KEY] = LOCAL_METADATA_VALUE;
  }
  return GAR::CreateGraphInfo(graph_name, vertex_infos, edge_infos, path,
                              nullptr, extra_info);
}
}  // namespace vineyard

#endif
