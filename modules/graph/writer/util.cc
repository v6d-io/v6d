
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

std::shared_ptr<GraphArchive::GraphInfo> generate_graph_info_with_schema(
    const PropertyGraphSchema& schema, const std::string& graph_name,
    const std::string& path, int64_t vertex_block_size, int64_t edge_block_size,
    GAR::FileType file_type, bool store_in_local) {
  GraphArchive::VertexInfoVector vertex_infos;
  GraphArchive::EdgeInfoVector edge_infos;

  for (const auto& entry : schema.vertex_entries()) {
    std::vector<GraphArchive::Property> properties;
    for (const auto& prop : entry.props_) {
      properties.emplace_back(GAR::Property(
          prop.name, GAR::DataType::ArrowDataTypeToDataType(prop.type), false));
    }
    auto pg = GAR::CreatePropertyGroup(properties, file_type);
    auto vertex_info =
        GAR::CreateVertexInfo(entry.label, vertex_block_size, {pg});
    vertex_infos.emplace_back(vertex_info);
  }
  GAR::AdjacentListVector default_adjacent_lists{
      GAR::CreateAdjacentList(GAR::AdjListType::ordered_by_source, file_type),
      GAR::CreateAdjacentList(GAR::AdjListType::ordered_by_dest, file_type)};
  for (const auto& entry : schema.edge_entries()) {
    std::vector<GraphArchive::Property> properties;
    for (const auto& prop : entry.props_) {
      properties.emplace_back(GAR::Property(
          prop.name, GAR::DataType::ArrowDataTypeToDataType(prop.type), false));
    }
    auto pg = GAR::CreatePropertyGroup(properties, file_type);
    for (const auto& relation : entry.relations) {
      auto edge_info = GAR::CreateEdgeInfo(
          relation.first, entry.label, relation.second, edge_block_size,
          vertex_block_size, vertex_block_size, true /* directed */,
          default_adjacent_lists, {pg});
      edge_infos.emplace_back(edge_info);
    }
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
