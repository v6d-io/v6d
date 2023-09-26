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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BASE_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BASE_H_

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "client/ds/i_object.h"

#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"

namespace gs {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T, bool COMPACT>
class ArrowProjectedFragment;

}  // namespace gs

namespace vineyard {

class ArrowFragmentBase : public vineyard::Object {
 public:
  using prop_id_t = property_graph_types::PROP_ID_TYPE;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;

  virtual ~ArrowFragmentBase() = default;

  virtual boost::leaf::result<ObjectID> AddVerticesAndEdges(
      Client& client,
      std::map<label_id_t, std::shared_ptr<arrow::Table>>&& vertex_tables_map,
      std::map<label_id_t, std::shared_ptr<arrow::Table>>&& edge_tables_map,
      ObjectID vm_id,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  virtual boost::leaf::result<ObjectID> AddVertices(
      Client& client,
      std::map<label_id_t, std::shared_ptr<arrow::Table>>&& vertex_tables_map,
      ObjectID vm_id,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }
  // Add vertices progressively to existed vertex label.
  virtual boost::leaf::result<ObjectID> AddVerticesToExistedLabel(
      Client& client, PropertyGraphSchema::LabelId label_id,
      std::shared_ptr<arrow::Table>&& vertex_table, ObjectID vm_id,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  /**
   * @brief Add edges progressively to existed vertex label.
   *
   * @param client
   * @param label_id the label id of the existed vertex label.
   * @param edge_table the newly added edges
   * @param edge_relations
   * @param concurrency
   * @return boost::leaf::result<ObjectID>
   */
  virtual boost::leaf::result<ObjectID> AddEdgesToExistedLabel(
      Client& client, PropertyGraphSchema::LabelId label_id,
      std::shared_ptr<arrow::Table>&& edge_table,
      const std::set<std::pair<std::string, std::string>>& edge_relations,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  virtual boost::leaf::result<ObjectID> AddEdges(
      Client& client,
      std::map<label_id_t, std::shared_ptr<arrow::Table>>&& edge_tables_map,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  /// Add a set of new vertex labels and a set of new edge labels to graph.
  /// Vertex label id started from vertex_label_num_, and edge label id
  /// started from edge_label_num_.
  virtual boost::leaf::result<ObjectID> AddNewVertexEdgeLabels(
      Client& client,
      std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
      std::vector<std::shared_ptr<arrow::Table>>&& edge_tables, ObjectID vm_id,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  /// Add a set of new vertex labels to graph. Vertex label id started from
  /// vertex_label_num_.
  virtual boost::leaf::result<ObjectID> AddNewVertexLabels(
      Client& client,
      std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
      ObjectID vm_id,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  /// Add a set of new edge labels to graph. Edge label id started from
  /// edge_label_num_.
  virtual boost::leaf::result<ObjectID> AddNewEdgeLabels(
      Client& client, std::vector<std::shared_ptr<arrow::Table>>&& edge_tables,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      const int concurrency = std::thread::hardware_concurrency()) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  virtual boost::leaf::result<vineyard::ObjectID> AddVertexColumns(
      vineyard::Client& client,
      const std::map<
          label_id_t,
          std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>>
          columns,
      bool replace = false) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  virtual boost::leaf::result<vineyard::ObjectID> AddVertexColumns(
      vineyard::Client& client,
      const std::map<label_id_t,
                     std::vector<std::pair<
                         std::string, std::shared_ptr<arrow::ChunkedArray>>>>
          columns,
      bool replace = false) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  virtual boost::leaf::result<vineyard::ObjectID> AddEdgeColumns(
      vineyard::Client& client,
      const std::map<
          label_id_t,
          std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>>
          columns,
      bool replace = false) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  virtual boost::leaf::result<vineyard::ObjectID> AddEdgeColumns(
      vineyard::Client& client,
      const std::map<label_id_t,
                     std::vector<std::pair<
                         std::string, std::shared_ptr<arrow::ChunkedArray>>>>
          columns,
      bool replace = false) {
    VINEYARD_ASSERT(false, "Not implemented");
    return vineyard::InvalidObjectID();
  }

  virtual vineyard::ObjectID vertex_map_id() const = 0;

  virtual bool local_vertex_map() const = 0;

  virtual bool compact_edges() const = 0;

  virtual bool use_perfect_hash() const = 0;

  virtual const PropertyGraphSchema& schema() const = 0;

  virtual bool directed() const = 0;

  virtual bool is_multigraph() const = 0;

  virtual const std::string vid_typename() const = 0;

  virtual const std::string oid_typename() const = 0;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BASE_H_
