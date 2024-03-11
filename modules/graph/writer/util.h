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

#ifndef MODULES_GRAPH_WRITER_UTIL_H_
#define MODULES_GRAPH_WRITER_UTIL_H_

#ifdef ENABLE_GAR

#include <memory>
#include <string>

#include "gar/graph_info.h"
#include "gar/util/file_type.h"
#include "graph/fragment/graph_schema.h"

namespace GAR = GraphArchive;

namespace vineyard {

#define LOCAL_METADATA_KEY "local_meta_prefix"
#define LOCAL_METADATA_VALUE "__local_metadata__"

// A simple struct to store the edge triple, which is used to express the edge
// triple and used as the key of the map.
class EdgeRelation{
 public:
  explicit EdgeRelation(const std::string& src_label, const std::string& edge_label,
                      const std::string& dst_label)
      : src_label(src_label),
        edge_label(edge_label),
        dst_label(dst_label) {}
  explicit EdgeRelation(std::string&& src_label, std::string&& edge_label,
                      std::string&& dst_label)
      : src_label(std::move(src_label)),
        edge_label(std::move(edge_label)),
        dst_label(std::move(dst_label)) {}

  bool operator==(const EdgeRelation& rhs) const {
    return src_label == rhs.src_label && edge_label == rhs.edge_label &&
           dst_label == rhs.dst_label;
  }

  std::string src_label;
  std::string edge_label;
  std::string dst_label;
};

boost::leaf::result<std::shared_ptr<GraphArchive::GraphInfo>> generate_graph_info_with_schema(
    const PropertyGraphSchema& schema, const std::string& graph_name,
    const std::string& path, int64_t vertex_block_size, int64_t edge_block_size,
    GAR::FileType file_type,
    const std::vector<std::string>& selected_vertices,
    const std::vector<std::string>& selected_edges,
    const std::unordered_map<std::string, std::vector<std::string>>& selected_vertex_properties,
    const std::unordered_map<std::string, std::vector<std::string>>& selected_edge_properties,
    bool store_in_local);

}  // namespace vineyard

namespace std {
  template <>
  struct hash<vineyard::EdgeRelation> {
    std::size_t operator()(const vineyard::EdgeRelation& k) const {
      std::size_t h1 = std::hash<std::string>{}(k.src_label);
      std::size_t h2 = std::hash<std::string>{}(k.edge_label);
      std::size_t h3 = std::hash<std::string>{}(k.dst_label);
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };
}  // namespace std

#endif

#endif  // MODULES_GRAPH_WRITER_UTIL_H_
