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

#ifndef MODULES_GRAPH_LOADER_FRAGMENT_LOADER_UTILS_H_
#define MODULES_GRAPH_LOADER_FRAGMENT_LOADER_UTILS_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "graph/loader/basic_ev_fragment_loader.h"
#include "graph/utils/table_shuffler_beta.h"

namespace vineyard {

template <typename T>
class OidSet {
  using oid_t = T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = ArrowArrayType<oid_t>;

 public:
  boost::leaf::result<void> BatchInsert(
      const std::shared_ptr<arrow::Array>& arr) {
    if (vineyard::ConvertToArrowType<oid_t>::TypeValue() != arr->type()) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "OID_T '" + type_name<oid_t>() +
                          "' is not same with arrow::Column(" +
                          arr->type()->ToString() + ")");
    }
    auto oid_arr = std::dynamic_pointer_cast<oid_array_t>(arr);
    for (int64_t i = 0; i < oid_arr->length(); i++) {
      oids.insert(oid_arr->GetView(i));
    }
    return {};
  }

  boost::leaf::result<void> BatchInsert(
      const std::shared_ptr<arrow::ChunkedArray>& chunked_arr) {
    for (auto chunk_idx = 0; chunk_idx < chunked_arr->num_chunks();
         chunk_idx++) {
      BOOST_LEAF_CHECK(BatchInsert(chunked_arr->chunk(chunk_idx)));
    }
    return {};
  }

  boost::leaf::result<std::shared_ptr<oid_array_t>> ToArrowArray() {
    ArrowBuilderType<oid_t> builder;
    ARROW_OK_OR_RAISE(builder.Reserve(oids.size()));
    for (auto& oid : oids) {
      ARROW_OK_OR_RAISE(builder.Append(oid));
    }

    std::shared_ptr<oid_array_t> oid_arr;
    ARROW_OK_OR_RAISE(builder.Finish(&oid_arr));
    return oid_arr;
  }

  void Clear() { oids.clear(); }

 private:
  std::unordered_set<internal_oid_t> oids;
};

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class FragmentLoaderUtils {
  static constexpr int src_column = 0;
  static constexpr int dst_column = 1;

  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_t = OID_T;
  using vid_t = VID_T;
  using partitioner_t = PARTITIONER_T;
  using oid_array_t = ArrowArrayType<oid_t>;
  using internal_oid_t = typename InternalType<oid_t>::type;

 public:
  explicit FragmentLoaderUtils(const grape::CommSpec& comm_spec,
                               const PARTITIONER_T& partitioner)
      : comm_spec_(comm_spec), partitioner_(partitioner) {}

  /**
   * @brief Gather all vertex labels from each worker, then sort and unique them
   *
   * @param tables vector of edge tables
   * @return processed vector of vertex labels
   */
  boost::leaf::result<std::vector<std::string>> GatherVertexLabels(
      const std::vector<InputTable>& tables) {
    std::set<std::string> local_vertex_labels;
    for (auto& tab : tables) {
      local_vertex_labels.insert(tab.src_label);
      local_vertex_labels.insert(tab.dst_label);
    }
    std::vector<std::string> local_vertex_label_list;
    for (auto& label : local_vertex_labels) {
      local_vertex_label_list.push_back(label);
    }
    std::vector<std::vector<std::string>> gathered_vertex_label_lists;
    GlobalAllGatherv(local_vertex_label_list, gathered_vertex_label_lists,
                     comm_spec_);
    std::vector<std::string> sorted_labels;
    for (auto& vec : gathered_vertex_label_lists) {
      sorted_labels.insert(sorted_labels.end(), vec.begin(), vec.end());
    }
    std::sort(sorted_labels.begin(), sorted_labels.end());
    auto iter = std::unique(sorted_labels.begin(), sorted_labels.end());
    sorted_labels.erase(iter, sorted_labels.end());
    return sorted_labels;
  }

  /**
   * @brief Build a label to index map
   *
   * @param tables vector of labels, each items must be unique.
   * @return label to index map
   */
  boost::leaf::result<std::map<std::string, label_id_t>> GetVertexLabelToIndex(
      const std::vector<std::string>& labels) {
    std::map<std::string, label_id_t> label_to_index;
    for (size_t i = 0; i < labels.size(); ++i) {
      label_to_index[labels[i]] = static_cast<label_id_t>(i);
    }
    return label_to_index;
  }

  boost::leaf::result<std::map<std::string, std::shared_ptr<arrow::Table>>>
  BuildVertexTableFromEdges(
      const std::vector<InputTable>& edge_tables,
      const std::map<std::string, label_id_t>& vertex_label_to_index,
      const std::set<std::string>& deduced_labels) {
    std::map<std::string, std::shared_ptr<arrow::Table>> ret;

    size_t vertex_label_num = vertex_label_to_index.size();
    std::vector<OidSet<oid_t>> oids(vertex_label_num);
    // Collect vertex ids in current fragment
    for (auto& table : edge_tables) {
      if (deduced_labels.find(table.src_label) != deduced_labels.end()) {
        label_id_t src_label_id = vertex_label_to_index.at(table.src_label);
        BOOST_LEAF_CHECK(
            oids[src_label_id].BatchInsert(table.table->column(src_column)));
      }

      if (deduced_labels.find(table.dst_label) != deduced_labels.end()) {
        label_id_t dst_label_id = vertex_label_to_index.at(table.dst_label);
        BOOST_LEAF_CHECK(
            oids[dst_label_id].BatchInsert(table.table->column(dst_column)));
      }
    }

    // Gather vertex ids from all fragments
    std::vector<std::shared_ptr<arrow::Field>> schema_vector{
        arrow::field("id", vineyard::ConvertToArrowType<oid_t>::TypeValue())};
    auto schema = std::make_shared<arrow::Schema>(schema_vector);
    for (const auto& label : deduced_labels) {
      std::shared_ptr<oid_array_t> oid_array;
      {
        label_id_t label_id = vertex_label_to_index.at(label);
        auto& oid_set = oids[label_id];
        BOOST_LEAF_ASSIGN(oid_array, oid_set.ToArrowArray());
        std::vector<std::shared_ptr<arrow::Array>> arrays{oid_array};
        auto v_table = arrow::Table::Make(schema, arrays);
        BOOST_LEAF_AUTO(tmp_table, ShufflePropertyVertexTable(
                                       comm_spec_, partitioner_, v_table));
        oid_set.Clear();
        BOOST_LEAF_CHECK(oid_set.BatchInsert(tmp_table->column(0)));
        BOOST_LEAF_ASSIGN(oid_array, oid_set.ToArrowArray());
      }
      // Build the deduced vertex table
      {
        std::vector<std::shared_ptr<arrow::Array>> arrays{oid_array};
        auto v_table = arrow::Table::Make(schema, arrays);
        ret[label] = v_table;
      }
    }
    return ret;
  }

 private:
  grape::CommSpec comm_spec_;
  const partitioner_t& partitioner_;
};

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

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_FRAGMENT_LOADER_UTILS_H_
