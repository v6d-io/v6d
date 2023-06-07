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

#ifndef MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "grape/worker/comm_spec.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/table_pipeline.h"
#include "graph/vertex_map/arrow_local_vertex_map.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {

struct InputTable {
  InputTable(const std::string& src_label_, const std::string& dst_label_,
             const std::string& edge_label_,
             std::shared_ptr<arrow::Table> table_)
      : src_label(src_label_),
        dst_label(dst_label_),
        edge_label(edge_label_),
        table(table_) {}

  std::string src_label;
  std::string dst_label;
  std::string edge_label;
  std::shared_ptr<arrow::Table> table;
};

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class BasicEVFragmentLoader {
  static constexpr int id_column = 0;
  static constexpr int src_column = 0;
  static constexpr int dst_column = 1;
  static constexpr int edge_id_column = 2;

  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_t = OID_T;
  using vid_t = VID_T;
  using partitioner_t = PARTITIONER_T;
  using vertex_map_t =
      ArrowVertexMap<typename InternalType<OID_T>::type, VID_T>;
  using local_vertex_map_t =
      ArrowLocalVertexMap<typename InternalType<OID_T>::type, VID_T>;
  using oid_array_t = ArrowArrayType<oid_t>;
  using oid_array_builder_t = ArrowBuilderType<oid_t>;
  using internal_oid_t = typename InternalType<oid_t>::type;

 public:
  explicit BasicEVFragmentLoader(
      Client& client, const grape::CommSpec& comm_spec,
      const PARTITIONER_T& partitioner, bool directed = true,
      bool generate_eid = false, bool retain_oid = false,
      bool local_vertex_map = false, bool compact_edges = false,
      bool use_perfect_hash_ = false);

  /**
   * @brief Add a loaded vertex table.
   *
   * @param label vertex label name.
   * @param vertex_table
   *  | id : OID_T | property_1 | ... | property_n |
   * @return
   */
  boost::leaf::result<void> AddVertexTable(
      const std::string& label, std::shared_ptr<arrow::Table> vertex_table);

  boost::leaf::result<void> ConstructVertices(
      ObjectID vm_id = InvalidObjectID());

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
  boost::leaf::result<void> AddEdgeTable(
      const std::string& src_label, const std::string& dst_label,
      const std::string& edge_label, std::shared_ptr<arrow::Table> edge_table);

  boost::leaf::result<void> ConstructEdges(int label_offset = 0,
                                           int vertex_label_num = 0);

  boost::leaf::result<ObjectID> AddVerticesToFragment(
      std::shared_ptr<ArrowFragmentBase> frag);

  boost::leaf::result<ObjectID> AddEdgesToFragment(
      std::shared_ptr<ArrowFragmentBase> frag);

  boost::leaf::result<ObjectID> AddVerticesAndEdgesToFragment(
      std::shared_ptr<ArrowFragmentBase> frag);

  boost::leaf::result<ObjectID> ConstructFragment();

  void set_vertex_label_to_index(std::map<std::string, label_id_t>&& in) {
    vertex_label_to_index_ = std::move(in);
  }

  std::map<std::string, label_id_t> get_vertex_label_to_index() {
    return vertex_label_to_index_;
  }

 private:
  boost::leaf::result<std::shared_ptr<ITablePipeline>> edgesId2Gid(
      const std::shared_ptr<ITablePipeline> edge_table, label_id_t src_label,
      label_id_t dst_label);

  Status parseOidChunkedArray(
      label_id_t label_id,
      const std::shared_ptr<arrow::ChunkedArray> oid_arrays_in,
      std::shared_ptr<arrow::ChunkedArray>& out);

  Status parseOidChunkedArrayChunk(
      label_id_t label_id, const std::shared_ptr<arrow::Array> oid_arrays_in,
      std::shared_ptr<arrow::Array>& out);

  boost::leaf::result<void> initSchema(PropertyGraphSchema& schema);

  boost::leaf::result<void> generateEdgeId(
      const grape::CommSpec& comm_spec,
      std::vector<std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                        std::shared_ptr<ITablePipeline>>>>&
          edge_tables,
      int label_offset);

  // constructVertices implementation for ArrowVertexMap
  boost::leaf::result<void> constructVerticesImpl(ObjectID vm_id);

  // constructVertices implementation for ArrowLocalVertexMap
  boost::leaf::result<void> constructVerticesImplLocal(ObjectID vm_id);

  // constructEdges implementation for ArrowVertexMap
  boost::leaf::result<void> constructEdgesImpl(int label_offset,
                                               int vertex_label_num);

  // constructEdges implementation for ArrowLocalVertexMap
  boost::leaf::result<void> constructEdgesImplLocal(int label_offset,
                                                    int vertex_label_num);

  Client& client_;

  label_id_t vertex_label_num_;
  label_id_t edge_label_num_;

  grape::CommSpec comm_spec_;
  const PARTITIONER_T& partitioner_;

  bool directed_;
  bool generate_eid_ = false;
  bool retain_oid_ = false;
  bool local_vertex_map_ = false;
  bool compact_edges_ = false;
  bool use_perfect_hash_ = false;

  std::map<std::string, label_id_t> vertex_label_to_index_;
  std::vector<std::string> vertex_labels_;
  std::map<std::string, label_id_t> edge_label_to_index_;
  std::vector<std::string> edge_labels_;

  std::map<std::string, std::shared_ptr<arrow::Table>> input_vertex_tables_;
  std::map<std::string, std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                              std::shared_ptr<arrow::Table>>>>
      input_edge_tables_;

  std::vector<std::shared_ptr<ITablePipeline>> ordered_vertex_tables_;
  std::vector<std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                    std::shared_ptr<ITablePipeline>>>>
      ordered_edge_tables_;

  std::vector<std::shared_ptr<arrow::Table>> output_vertex_tables_;
  std::vector<std::shared_ptr<arrow::Table>> output_edge_tables_;
  std::vector<std::set<std::pair<label_id_t, label_id_t>>> edge_relations_;

  std::shared_ptr<vertex_map_t> vm_ptr_;
  std::shared_ptr<local_vertex_map_t> local_vm_ptr_;
  std::shared_ptr<ArrowLocalVertexMapBuilder<internal_oid_t, vid_t>>
      local_vm_builder_;  // temporarily used
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_H_
