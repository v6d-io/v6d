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

#ifndef MODULES_GRAPH_LOADER_FRAGMENT_LOADER_UTILS_H_
#define MODULES_GRAPH_LOADER_FRAGMENT_LOADER_UTILS_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/loader/basic_ev_fragment_loader.h"
#include "graph/utils/table_shuffler_beta.h"

namespace vineyard {

template <typename T>
class ConcurrentOidSet {
  using oid_t = T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = ArrowArrayType<oid_t>;
  using oid_array_builder_t = ArrowBuilderType<oid_t>;

 public:
  Status Insert(const std::shared_ptr<arrow::Array>& array,
                std::shared_ptr<arrow::Array>& out) {
    RETURN_ON_ASSERT(
        vineyard::ConvertToArrowType<oid_t>::TypeValue()->Equals(array->type()),
        "OID_T '" + type_name<oid_t>() + "' is not same with arrow::Column(" +
            array->type()->ToString() + ")");
    auto oid_array = std::dynamic_pointer_cast<oid_array_t>(array);
    oid_array_builder_t builder;
    for (int64_t i = 0; i < oid_array->length(); i++) {
      if (oids.insert(oid_array->GetView(i), true /* dummy */)) {
        RETURN_ON_ARROW_ERROR(builder.Append(oid_array->GetView(i)));
      }
    }
    RETURN_ON_ARROW_ERROR(builder.Finish(&out));
    return Status::OK();
  }

  void Insert(internal_oid_t const& oid, ArrowBuilderType<oid_t>& builder) {
    if (oids.insert(oid, true /* dummy */)) {
      DISCARD_ARROW_ERROR(builder.Append(oid));
    }
  }

  void Insert(internal_oid_t const& oid) { oids.insert(oid, true /* dummy */); }

  Status ToArray(std::shared_ptr<oid_array_t>& out) {
    oid_array_builder_t builder;
    RETURN_ON_ARROW_ERROR(builder.Reserve(oids.size()));
    {
      auto locked = oids.lock_table();
      for (auto const& item : locked) {
        RETURN_ON_ARROW_ERROR(builder.Append(item.first));
      }
    }
    RETURN_ON_ARROW_ERROR(builder.Finish(&out));
    return Status::OK();
  }

  void Clear() { oids.clear(); }

 private:
  concurrent_set_t<internal_oid_t> oids;
};

boost::leaf::result<std::vector<std::string>> GatherVertexLabels(
    const grape::CommSpec& comm_spec, const std::vector<InputTable>& tables);

/**
 * @brief Build a label to index map
 *
 * @param tables vector of labels, each items must be unique.
 * @return label to index map
 */
boost::leaf::result<std::map<std::string, property_graph_types::LABEL_ID_TYPE>>
GetVertexLabelToIndex(const std::vector<std::string>& labels);

template <typename PARTITIONER_T>
boost::leaf::result<std::map<std::string, std::shared_ptr<arrow::Table>>>
BuildVertexTableFromEdges(
    const grape::CommSpec& comm_spec, const PARTITIONER_T& partitioner,
    const std::vector<InputTable>& edge_tables,
    const std::map<std::string, property_graph_types::LABEL_ID_TYPE>&
        vertex_label_to_index,
    const std::set<std::string>& deduced_labels);

struct EdgeTableInfo {
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  EdgeTableInfo(std::shared_ptr<arrow::Table> adj_list_table,
                std::shared_ptr<arrow::Int64Array> offsets,
                std::shared_ptr<arrow::Table> property_table, label_id_t label,
                bool flag)
      : adj_list_table(adj_list_table),
        offsets(offsets),
        property_table(property_table),
        vertex_label_id(label),
        flag(flag) {}

  std::shared_ptr<arrow::Table> adj_list_table;
  std::shared_ptr<arrow::Int64Array> offsets;
  std::shared_ptr<arrow::Table> property_table;
  label_id_t vertex_label_id;
  bool flag;
};

EdgeTableInfo FlattenTableInfos(const std::vector<EdgeTableInfo>& edge_tables);

// This method used when several workers is loading a file in parallel, each
// worker will read a chunk of the origin file into a arrow::Table.
// We may get different table schemas as some chunks may have zero rows
// or some chunks' data doesn't have any floating numbers, but others might
// have. We could use this method to gather their schemas, and find out most
// inclusive fields, construct a new schema and broadcast back. Note: We perform
// type loosen, date32 -> int32; timestamp(s) -> int64 -> double -> string (big
// string), and any type is prior to null.
boost::leaf::result<std::shared_ptr<arrow::Table>> SyncSchema(
    const std::shared_ptr<arrow::Table>& table,
    const grape::CommSpec& comm_spec);

boost::leaf::result<ObjectID> ConstructFragmentGroup(
    Client& client, ObjectID frag_id, const grape::CommSpec& comm_spec);

std::pair<int64_t, int64_t> BinarySearchChunkPair(
    const std::vector<int64_t>& agg_num, int64_t got);

// merge offsets of vertex chunk into a whole offset array:
// vertex-chunk-0 offset: [0, 1, 2]
// vertex-chunk-1 offset: [0, 2, 6]
// return: [0, 1, 2, 4, 8]
Status parallel_prefix_sum_chunks(
    std::vector<std::shared_ptr<arrow::Int64Array>>& in_chunks,
    std::shared_ptr<arrow::Int64Array>& out);

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_FRAGMENT_LOADER_UTILS_H_
