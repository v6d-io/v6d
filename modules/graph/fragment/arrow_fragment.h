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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/fragment/fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/utils/vertex_array.h"

#include "basic/ds/arrow.h"
#include "basic/ds/arrow_utils.h"
#include "common/util/functions.h"
#include "common/util/typename.h"

#include "graph/fragment/arrow_fragment.vineyard.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"
#include "graph/utils/context_protocols.h"
#include "graph/utils/error.h"
#include "graph/utils/thread_group.h"
#include "graph/vertex_map/arrow_vertex_map.h"
#include "graph/vertex_map/arrow_vertex_map_builder.h"

namespace vineyard {

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
class ArrowFragmentBaseBuilder;

template <typename OID_T, typename VID_T,
          typename VERTEX_MAP_T =
              ArrowVertexMap<typename InternalType<OID_T>::type, VID_T>,
          bool COMPACT = false>
class BasicArrowFragmentBuilder
    : public ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT> {
  using Base = ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T, COMPACT>;

  using oid_t = OID_T;
  using vid_t = VID_T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using eid_t = property_graph_types::EID_TYPE;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using vertex_map_t = VERTEX_MAP_T;
  using nbr_unit_t = property_graph_utils::NbrUnit<vid_t, eid_t>;
  using vid_array_t = vineyard::ArrowArrayType<vid_t>;
  static constexpr bool compact_v = COMPACT;

 public:
  explicit BasicArrowFragmentBuilder(vineyard::Client& client,
                                     std::shared_ptr<vertex_map_t> vm_ptr)
      : ArrowFragmentBaseBuilder<oid_t, vid_t, vertex_map_t, COMPACT>(client),
        client_(client),
        vm_ptr_(vm_ptr) {}

  vineyard::Status Build(vineyard::Client& client) override;

  boost::leaf::result<void> Init(
      fid_t fid, fid_t fnum,
      std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
      std::vector<std::shared_ptr<arrow::Table>>&& edge_tables,
      bool directed = true,
      const int concurrency = std::thread::hardware_concurrency());

  boost::leaf::result<void> SetPropertyGraphSchema(
      PropertyGraphSchema&& schema) {
    this->set_schema_json_(schema.ToJSON());
    return {};
  }

 private:
  // | prop_0 | prop_1 | ... |
  boost::leaf::result<void> initVertices(
      std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables);

  // | src_id(generated) | dst_id(generated) | prop_0 | prop_1
  // | ... |
  boost::leaf::result<void> initEdges(
      std::vector<std::shared_ptr<arrow::Table>>&& edge_tables,
      int concurrency);

  vineyard::Client& client_;

  std::vector<vid_t> ivnums_, ovnums_, tvnums_;

  std::vector<std::shared_ptr<arrow::Table>> vertex_tables_;
  std::vector<std::shared_ptr<vid_array_t>> ovgid_lists_;
  std::vector<typename ArrowFragment<OID_T, VID_T>::ovg2l_map_t> ovg2l_maps_;

  std::vector<std::shared_ptr<arrow::Table>> edge_tables_;

  std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>>
      ie_lists_, oe_lists_;
  std::vector<std::vector<std::shared_ptr<FixedUInt8Builder>>>
      compact_ie_lists_, compact_oe_lists_;
  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>
      ie_offsets_lists_, oe_offsets_lists_;
  std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>
      ie_boffsets_lists_, oe_boffsets_lists_;

  std::shared_ptr<vertex_map_t> vm_ptr_;
  IdParser<vid_t> vid_parser_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_H_
