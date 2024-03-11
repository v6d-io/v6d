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

#ifndef MODULES_GRAPH_LOADER_GAR_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_GAR_FRAGMENT_LOADER_H_

#ifdef ENABLE_GAR

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/worker/comm_spec.h"

#include "client/client.h"
#include "io/io/io_factory.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/gar_fragment_builder.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace GraphArchive {
class GraphInfo;
class VertexInfo;
class EdgeInfo;
class PropertyGroup;
enum class AdjListType : std::uint8_t;
}  // namespace GraphArchive

namespace vineyard {

std::shared_ptr<arrow::Schema> ConstructSchemaFromPropertyGroup(
    const std::shared_ptr<GraphArchive::PropertyGroup>& property_group);

template <typename OID_T = property_graph_types::OID_TYPE,
          typename VID_T = property_graph_types::VID_TYPE,
          template <typename OID_T_ = typename InternalType<OID_T>::type,
                    typename VID_T_ = VID_T>
          class VERTEX_MAP_T = ArrowVertexMap>
class GARFragmentLoader {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = ArrowArrayType<oid_t>;
  using vid_array_t = ArrowArrayType<vid_t>;
  using vertex_map_t = VERTEX_MAP_T<internal_oid_t, vid_t>;
  using fragment_t = ArrowFragment<OID_T, VID_T, vertex_map_t>;
  using gar_id_t = int64_t;

  using table_vec_t = std::vector<std::shared_ptr<arrow::Table>>;
  using oid_array_vec_t = std::vector<std::shared_ptr<oid_array_t>>;
  using vid_array_vec_t = std::vector<std::shared_ptr<vid_array_t>>;

 protected:
  static constexpr const char* CONSOLIDATE_TAG = "consolidate";
  static constexpr const char* MARKER = "PROGRESS--GRAPH-LOADING-";

 public:
  explicit GARFragmentLoader(Client& client, const grape::CommSpec& comm_spec);

  /**
   * @brief Initialize the GARFragmentLoader.
   *    Notes that if vertex_labels or edge_labels are empty, the loader will
   *    load all vertices and edges.
   *
   * @param client vineyard client.
   * @param comm_spec communication spec.
   * @param graph_info_yaml graph info yaml path.
   * @param vertex_labels vertex labels to project subgraph.
   * @param edge_labels edge labels to project subgraph.
   * @param directed whether the graph is directed.
   * @param generate_eid whether to generate edge id.
   * @param store_in_local whether the gar data files are stored in local.
   */
  boost::leaf::result<void> Init(
      const std::string& graph_info_yaml,
      const std::vector<std::string>& selected_vertices = {},
      const std::vector<std::string>& selected_edges = {}, bool directed = true,
      bool generate_eid = false, bool store_in_local = false);

  ~GARFragmentLoader() = default;

  boost::leaf::result<ObjectID> LoadFragment();

  boost::leaf::result<ObjectID> LoadFragmentAsFragmentGroup();

  boost::leaf::result<void> LoadVertexTables();

  boost::leaf::result<void> LoadEdgeTables();

  boost::leaf::result<ObjectID> ConstructFragment();

 protected:  // for subclasses
  boost::leaf::result<void> distributeVertices();

  boost::leaf::result<void> constructVertexMap();

  boost::leaf::result<void> loadVertexTableOfLabel(
      const std::string& vertex_label);

  boost::leaf::result<void> loadEdgeTableOfLabel(
      const std::shared_ptr<GraphArchive::EdgeInfo>& edge_info,
      GraphArchive::AdjListType adj_list_type);

  boost::leaf::result<void> initSchema(PropertyGraphSchema& schema);

  boost::leaf::result<std::shared_ptr<arrow::Table>> parseEdgeIdArrays(
      std::shared_ptr<arrow::Table> adj_list_table, label_id_t src_label,
      label_id_t dst_label, GraphArchive::AdjListType adj_list_type);

  // parse oid to global id
  Status parseIdChunkedArray(
      label_id_t label_id,
      const std::shared_ptr<arrow::ChunkedArray> id_arrays_in,
      bool all_be_local_vertex, std::shared_ptr<arrow::ChunkedArray>& out);

  // parse oid to global id
  Status parseIdChunkedArrayChunk(
      label_id_t label_id, const std::shared_ptr<arrow::Array> id_array_in,
      bool all_be_local_vertex, std::shared_ptr<arrow::Array>& out);

  boost::leaf::result<void> initializeVertexChunkBeginAndNum(
      int vertex_label_index,
      const std::shared_ptr<GraphArchive::VertexInfo>& vertex_info);

  boost::leaf::result<void> checkInitialization();

 private:
  Client& client_;
  grape::CommSpec comm_spec_;
  std::shared_ptr<vertex_map_t> vm_ptr_;
  std::shared_ptr<GraphArchive::GraphInfo> graph_info_;

  bool directed_;

  label_id_t vertex_label_num_;
  label_id_t edge_label_num_;
  std::vector<int64_t> vertex_chunk_sizes_;
  std::vector<std::string> vertex_labels_;
  std::map<std::string, label_id_t> vertex_label_to_index_;
  std::map<std::string, label_id_t> edge_label_to_index_;
  table_vec_t vertex_tables_;
  std::vector<std::string> edge_labels_;
  std::vector<std::set<std::pair<label_id_t, label_id_t>>> edge_relations_;
  std::vector<EdgeTableInfo> csr_edge_tables_;
  std::vector<EdgeTableInfo> csc_edge_tables_;
  std::vector<std::vector<EdgeTableInfo>> csr_edge_tables_with_label_;
  std::vector<std::vector<EdgeTableInfo>> csc_edge_tables_with_label_;

  bool generate_eid_;
  IdParser<vid_t> vid_parser_;

  bool store_in_local_;
  std::vector<int64_t> vertex_chunk_begins_;
  std::vector<int64_t> vertex_chunk_nums_;
};

namespace detail {

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
struct rebind_gar_fragment_loader;

template <typename OID_T, typename VID_T>
struct rebind_gar_fragment_loader<
    OID_T, VID_T,
    vineyard::ArrowVertexMap<typename vineyard::InternalType<OID_T>::type,
                             VID_T>> {
  using type = GARFragmentLoader<OID_T, VID_T, vineyard::ArrowVertexMap>;
};

}  // namespace detail

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
using gar_fragment_loader_t =
    typename detail::rebind_gar_fragment_loader<OID_T, VID_T,
                                                VERTEX_MAP_T>::type;

}  // namespace vineyard

#endif  // ENABLE_GAR
#endif  // MODULES_GRAPH_LOADER_GAR_FRAGMENT_LOADER_H_
