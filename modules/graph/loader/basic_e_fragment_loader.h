/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef MODULES_GRAPH_LOADER_BASIC_E_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_BASIC_E_FRAGMENT_LOADER_H_

#include "graph/loader/basic_ev_fragment_loader.h"

namespace vineyard {

template <typename T>
class OidSet {
  using oid_t = T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;

 public:
  boost::leaf::result<void> BatchInsert(
      const std::shared_ptr<arrow::Array>& arr) {
    if (vineyard::ConvertToArrowType<oid_t>::TypeValue() != arr->type()) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "OID_T is not same with arrow::Column(" +
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
    typename vineyard::ConvertToArrowType<oid_t>::BuilderType builder;

    for (auto& oid : oids) {
      builder.Append(oid);
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
class BasicEFragmentLoader {
  static constexpr int src_column = 0;
  static constexpr int dst_column = 1;

  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_t = OID_T;
  using vid_t = VID_T;
  using partitioner_t = PARTITIONER_T;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using internal_oid_t = typename InternalType<oid_t>::type;

 public:
  explicit BasicEFragmentLoader(Client& client,
                                const grape::CommSpec& comm_spec,
                                const PARTITIONER_T& partitioner,
                                bool directed = true, bool retain_oid = false,
                                bool generate_eid = false)
      : client_(client), comm_spec_(comm_spec), partitioner_(partitioner) {
    ev_fragment_loader_ =
        std::make_shared<BasicEVFragmentLoader<OID_T, VID_T, PARTITIONER_T>>(
            client_, comm_spec_, partitioner_, directed, retain_oid,
            generate_eid);
  }

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
      const std::string& edge_label, std::shared_ptr<arrow::Table> edge_table) {
    input_tables_.emplace_back(src_label, dst_label, edge_label, edge_table);
    return {};
  }

  boost::leaf::result<void> ConstructEdges() {
    std::set<std::string> local_vertex_labels;
    for (auto& tab : input_tables_) {
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

    label_id_t cur_label = 0;
    for (auto& vec : gathered_vertex_label_lists) {
      for (auto& label : vec) {
        if (vertex_label_to_index_.find(label) ==
            vertex_label_to_index_.end()) {
          vertex_label_to_index_[label] = cur_label;
          vertex_labels_.push_back(label);
          ++cur_label;
        }
      }
    }
    vertex_label_num_ = cur_label;

    BOOST_LEAF_CHECK(constructVertices());

    for (auto& tab : input_tables_) {
      BOOST_LEAF_CHECK(ev_fragment_loader_->AddEdgeTable(
          tab.src_label, tab.dst_label, tab.edge_label, tab.table));
    }

    input_tables_.clear();
    return ev_fragment_loader_->ConstructEdges();
  }

  boost::leaf::result<vineyard::ObjectID> ConstructFragment() {
    return ev_fragment_loader_->ConstructFragment();
  }

 private:
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

  boost::leaf::result<void> constructVertices() {
    std::vector<OidSet<oid_t>> oids(vertex_label_num_);
    for (auto& tab : input_tables_) {
      label_id_t src_label_id = vertex_label_to_index_.at(tab.src_label);
      label_id_t dst_label_id = vertex_label_to_index_.at(tab.dst_label);

      BOOST_LEAF_CHECK(
          oids[src_label_id].BatchInsert(tab.table->column(src_column)));
      BOOST_LEAF_CHECK(
          oids[dst_label_id].BatchInsert(tab.table->column(dst_column)));
    }

    std::vector<std::shared_ptr<arrow::Field>> schema_vector{
        arrow::field("id", vineyard::ConvertToArrowType<oid_t>::TypeValue())};
    auto schema = std::make_shared<arrow::Schema>(schema_vector);
    for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
      auto& oid_set = oids[v_label];
      {
        BOOST_LEAF_AUTO(oid_array, oid_set.ToArrowArray());
        std::vector<std::shared_ptr<arrow::Array>> arrays{oid_array};
        auto v_table = arrow::Table::Make(schema, arrays);

        BOOST_LEAF_AUTO(tmp_table, beta::ShufflePropertyVertexTable(
                                       comm_spec_, partitioner_, v_table));

        oid_set.Clear();
        BOOST_LEAF_CHECK(oid_set.BatchInsert(tmp_table->column(0)));
      }

      {
        BOOST_LEAF_AUTO(oid_array, oid_set.ToArrowArray());
        std::vector<std::shared_ptr<arrow::Array>> arrays{oid_array};
        auto v_table = arrow::Table::Make(schema, arrays);

        BOOST_LEAF_CHECK(ev_fragment_loader_->AddVertexTable(
            vertex_labels_[v_label], v_table));
      }
    }

    return ev_fragment_loader_->ConstructVertices();
  }

  Client& client_;

  std::vector<InputTable> input_tables_;

  std::vector<std::string> vertex_labels_;
  std::map<std::string, label_id_t> vertex_label_to_index_;
  label_id_t vertex_label_num_;

  std::shared_ptr<BasicEVFragmentLoader<oid_t, vid_t, partitioner_t>>
      ev_fragment_loader_;

  grape::CommSpec comm_spec_;
  const partitioner_t& partitioner_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_BASIC_E_FRAGMENT_LOADER_H_
