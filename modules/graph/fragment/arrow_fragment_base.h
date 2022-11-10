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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BASE_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BASE_H_

#include <cstddef>
#include <map>
#include <memory>
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
          typename VERTEX_MAP_T>
class ArrowProjectedFragment;

}  // namespace gs

namespace vineyard {

class ArrowFragmentBase : public vineyard::Object {
 public:
  using prop_id_t = property_graph_types::PROP_ID_TYPE;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;

  virtual ~ArrowFragmentBase() = default;

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

  virtual const PropertyGraphSchema& schema() const = 0;

  virtual bool directed() const = 0;

  virtual bool is_multigraph() const = 0;

  virtual const std::string vid_typename() const = 0;

  virtual const std::string oid_typename() const = 0;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_BASE_H_
