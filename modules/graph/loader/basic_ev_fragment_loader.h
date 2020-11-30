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

#ifndef MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_H_

#include "grape/worker/comm_spec.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/table_shuffler.h"
#include "graph/utils/table_shuffler_beta.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {

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
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using internal_oid_t = typename InternalType<oid_t>::type;

 public:
  explicit BasicEVFragmentLoader(Client& client,
                                 const grape::CommSpec& comm_spec,
                                 const PARTITIONER_T& partitioner,
                                 bool directed = true, bool retain_oid = false,
                                 bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        partitioner_(partitioner),
        directed_(directed),
        retain_oid_(retain_oid),
        generate_eid_(generate_eid) {}

  /**
   * @brief Add a loaded vertex table.
   *
   * @param label vertex label name.
   * @param vertex_table
   *  | id : OID_T | property_1 | ... | property_n |
   * @return
   */
  boost::leaf::result<void> AddVertexTable(
      const std::string& label, std::shared_ptr<arrow::Table> vertex_table) {
    if (input_vertex_tables_.find(label) == input_vertex_tables_.end()) {
      input_vertex_tables_[label] = vertex_table;
    } else {
      std::vector<std::shared_ptr<arrow::Table>> tables;
      tables.push_back(input_vertex_tables_.at(label));
      tables.push_back(vertex_table);
      input_vertex_tables_[label] = ConcatenateTables(tables);
    }

    return {};
  }

  boost::leaf::result<void> ConstructVertices() {
    std::vector<std::string> input_vertex_labels;
    for (auto& pair : input_vertex_tables_) {
      input_vertex_labels.push_back(pair.first);
    }
    std::vector<std::vector<std::string>> gathered_vertex_labels;
    GlobalAllGatherv(input_vertex_labels, gathered_vertex_labels, comm_spec_);
    label_id_t cur_label = 0;
    for (auto& vec : gathered_vertex_labels) {
      for (auto& label_name : vec) {
        if (vertex_label_to_index_.find(label_name) ==
            vertex_label_to_index_.end()) {
          vertex_label_to_index_[label_name] = cur_label;
          vertex_labels_.push_back(label_name);
          ++cur_label;
        }
      }
    }
    vertex_label_num_ = cur_label;

    ordered_vertex_tables_.clear();
    ordered_vertex_tables_.resize(vertex_label_num_, nullptr);

    for (auto& pair : input_vertex_tables_) {
      ordered_vertex_tables_[vertex_label_to_index_[pair.first]] = pair.second;
    }

    input_vertex_tables_.clear();

    output_vertex_tables_.resize(vertex_label_num_);

    std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_lists(
        vertex_label_num_);

    for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
      auto vertex_table = ordered_vertex_tables_[v_label];
      auto metadata = std::make_shared<arrow::KeyValueMetadata>();
      metadata->Append("label", vertex_labels_[v_label]);
      metadata->Append("label_id", std::to_string(v_label));
      metadata->Append("type", "VERTEX");
      auto shuffle_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        auto id_column_type = vertex_table->column(id_column)->type();

        if (!id_column_type->Equals(ConvertToArrowType<oid_t>::TypeValue())) {
          RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                          "OID_T is not same with arrow::Column(" +
                              std::to_string(id_column) + ")");
        }

        BOOST_LEAF_AUTO(tmp_table,
                        beta::ShufflePropertyVertexTable<partitioner_t>(
                            comm_spec_, partitioner_, vertex_table));

        auto local_oid_array = std::dynamic_pointer_cast<oid_array_t>(
            tmp_table->column(id_column)->chunk(0));

        VY_OK_OR_RAISE(FragmentAllGatherArray<oid_t>(
            comm_spec_, local_oid_array, oid_lists[v_label]));

        if (retain_oid_) {
          auto id_field = tmp_table->schema()->field(id_column);
          auto id_array = tmp_table->column(id_column);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
          CHECK_ARROW_ERROR(tmp_table->RemoveColumn(id_column, &tmp_table));
          CHECK_ARROW_ERROR(tmp_table->AddColumn(
              tmp_table->num_columns(), id_field, id_array, &tmp_table));
#else
          CHECK_ARROW_ERROR_AND_ASSIGN(tmp_table,
                                       tmp_table->RemoveColumn(id_column));
          CHECK_ARROW_ERROR_AND_ASSIGN(
              tmp_table, tmp_table->AddColumn(tmp_table->num_columns(),
                                              id_field, id_array));
#endif
        }

        return tmp_table;
      };
      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));
      output_vertex_tables_[v_label] = table->ReplaceSchemaMetadata(metadata);
    }

    BasicArrowVertexMapBuilder<internal_oid_t, vid_t> vm_builder(
        client_, comm_spec_.fnum(), vertex_label_num_, oid_lists);

    auto vm = vm_builder.Seal(client_);

    vm_ptr_ = std::dynamic_pointer_cast<ArrowVertexMap<internal_oid_t, vid_t>>(
        client_.GetObject(vm->id()));

    ordered_vertex_tables_.clear();
    return {};
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
    label_id_t src_label_id, dst_label_id;
    auto iter = vertex_label_to_index_.find(src_label);
    if (iter == vertex_label_to_index_.end()) {
      return boost::leaf::new_error(ErrorCode::kInvalidValueError,
                                    "Invalid src vertex label " + src_label);
    }
    src_label_id = iter->second;
    iter = vertex_label_to_index_.find(dst_label);
    if (iter == vertex_label_to_index_.end()) {
      return boost::leaf::new_error(ErrorCode::kInvalidValueError,
                                    "Invalid dst vertex label " + dst_label);
    }
    dst_label_id = iter->second;
    input_edge_tables_[edge_label].emplace_back(
        std::make_pair(src_label_id, dst_label_id), edge_table);

    return {};
  }

  boost::leaf::result<void> ConstructEdges() {
    vineyard::IdParser<vid_t> id_parser;
    std::shared_ptr<arrow::Field> src_gid_field =
        std::make_shared<arrow::Field>(
            "src", vineyard::ConvertToArrowType<vid_t>::TypeValue());
    std::shared_ptr<arrow::Field> dst_gid_field =
        std::make_shared<arrow::Field>(
            "dst", vineyard::ConvertToArrowType<vid_t>::TypeValue());

    id_parser.Init(comm_spec_.fnum(), vertex_label_num_);

    std::vector<std::string> input_edge_labels;
    for (auto& pair : input_edge_tables_) {
      input_edge_labels.push_back(pair.first);
    }
    std::vector<std::vector<std::string>> gathered_edge_labels;
    GlobalAllGatherv(input_edge_labels, gathered_edge_labels, comm_spec_);

    label_id_t cur_label = 0;
    for (auto& vec : gathered_edge_labels) {
      for (auto& label_name : vec) {
        if (edge_label_to_index_.find(label_name) ==
            edge_label_to_index_.end()) {
          edge_label_to_index_[label_name] = cur_label;
          edge_labels_.push_back(label_name);
          ++cur_label;
        }
      }
    }
    edge_label_num_ = cur_label;

    ordered_edge_tables_.clear();
    ordered_edge_tables_.resize(edge_label_num_);

    for (auto& pair : input_edge_tables_) {
      ordered_edge_tables_[edge_label_to_index_[pair.first]] =
          std::move(pair.second);
    }

    input_edge_tables_.clear();

    if (generate_eid_) {
      generateEdgeId(ordered_edge_tables_);
    }

    edge_relations_.resize(edge_label_num_);
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      std::vector<std::pair<label_id_t, label_id_t>> relations;
      auto& vec = ordered_edge_tables_[e_label];
      for (auto& pair : vec) {
        relations.push_back(pair.first);
      }
      std::vector<std::vector<std::pair<label_id_t, label_id_t>>>
          gathered_relations;
      GlobalAllGatherv(relations, gathered_relations, comm_spec_);
      for (auto& pair_vec : gathered_relations) {
        for (auto& pair : pair_vec) {
          edge_relations_[e_label].insert(pair);
        }
      }
    }

    output_edge_tables_.resize(edge_label_num_);
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      auto metadata = std::make_shared<arrow::KeyValueMetadata>();
      metadata->Append("label", edge_labels_[e_label]);
      metadata->Append("label_id", std::to_string(e_label));
      metadata->Append("type", "EDGE");
      auto shuffle_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        auto& edge_table_list = ordered_edge_tables_[e_label];
        std::vector<std::shared_ptr<arrow::Table>> processed_table_list(
            edge_table_list.size());
        for (size_t edge_table_index = 0;
             edge_table_index < edge_table_list.size(); ++edge_table_index) {
          label_id_t src_label = edge_table_list[edge_table_index].first.first;
          label_id_t dst_label = edge_table_list[edge_table_index].first.second;
          auto edge_table = edge_table_list[edge_table_index].second;

          auto src_column_type = edge_table->column(src_column)->type();
          auto dst_column_type = edge_table->column(dst_column)->type();

          if (!src_column_type->Equals(
                  vineyard::ConvertToArrowType<oid_t>::TypeValue())) {
            RETURN_GS_ERROR(
                ErrorCode::kInvalidValueError,
                "OID_T is not consistent with src id of edge table");
          }
          if (!dst_column_type->Equals(
                  vineyard::ConvertToArrowType<oid_t>::TypeValue())) {
            RETURN_GS_ERROR(
                ErrorCode::kInvalidValueError,
                "OID_T is not consistent with dst id of edge table");
          }

          BOOST_LEAF_AUTO(
              src_gid_array,
              parseOidChunkedArray(src_label, edge_table->column(src_column)));
          BOOST_LEAF_AUTO(
              dst_gid_array,
              parseOidChunkedArray(dst_label, edge_table->column(dst_column)));

          // replace oid columns with gid
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
          ARROW_OK_OR_RAISE(edge_table->SetColumn(src_column, src_gid_field,
                                                  src_gid_array, &edge_table));
          ARROW_OK_OR_RAISE(edge_table->SetColumn(dst_column, dst_gid_field,
                                                  dst_gid_array, &edge_table));
#else
          ARROW_OK_ASSIGN_OR_RAISE(
              edge_table,
              edge_table->SetColumn(src_column, src_gid_field, src_gid_array));
          ARROW_OK_ASSIGN_OR_RAISE(
              edge_table,
              edge_table->SetColumn(dst_column, dst_gid_field, dst_gid_array));
#endif
          processed_table_list[edge_table_index] = edge_table;
        }

        auto table = vineyard::ConcatenateTables(processed_table_list);

        BOOST_LEAF_AUTO(table_out, beta::ShufflePropertyEdgeTable<vid_t>(
                                       comm_spec_, id_parser, src_column,
                                       dst_column, table));

        return table_out;
      };

      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));
      output_edge_tables_[e_label] = table->ReplaceSchemaMetadata(metadata);
      ordered_edge_tables_[e_label].clear();
    }
    ordered_edge_tables_.clear();
    return {};
  }

  boost::leaf::result<vineyard::ObjectID> ConstructFragment() {
    BasicArrowFragmentBuilder<oid_t, vid_t> frag_builder(client_, vm_ptr_);

    PropertyGraphSchema schema;
    initSchema(schema);
    frag_builder.SetPropertyGraphSchema(std::move(schema));

    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();

    BOOST_LEAF_CHECK(frag_builder.Init(
        comm_spec_.fid(), comm_spec_.fnum(), std::move(output_vertex_tables_),
        std::move(output_edge_tables_), directed_, thread_num));

    auto frag = std::dynamic_pointer_cast<ArrowFragment<oid_t, vid_t>>(
        frag_builder.Seal(client_));

    VINEYARD_CHECK_OK(client_.Persist(frag->id()));
    return frag->id();
  }

 private:
  boost::leaf::result<std::shared_ptr<arrow::ChunkedArray>>
  parseOidChunkedArray(label_id_t label_id,
                       std::shared_ptr<arrow::ChunkedArray> oid_arrays_in) {
    size_t chunk_num = oid_arrays_in->num_chunks();
    std::vector<std::shared_ptr<arrow::Array>> chunks_out(chunk_num);

    ArrowVertexMap<internal_oid_t, vid_t>* vm = vm_ptr_.get();

#if 0
    for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
      std::shared_ptr<oid_array_t> oid_array =
          std::dynamic_pointer_cast<oid_array_t>(oid_arrays_in->chunk(chunk_i));
      typename vineyard::ConvertToArrowType<vid_t>::BuilderType builder;
      size_t size = oid_array->length();
      ARROW_OK_OR_RAISE(builder.Resize(size));

      for (size_t i = 0; i != size; ++i) {
        internal_oid_t oid = oid_array->GetView(i);
        fid_t fid = partitioner_.GetPartitionId(oid_t(oid));
        CHECK_OR_RAISE(vm->GetGid(fid, label_id, oid, builder[i]));
      }
      ARROW_OK_OR_RAISE(builder.Advance(size));
      ARROW_OK_OR_RAISE(builder.Finish(&chunks_out[chunk_i]));
    }
#else
    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();
    std::vector<std::thread> parse_threads(thread_num);

    std::atomic<size_t> cur(0);
    std::vector<arrow::Status> statuses(thread_num, arrow::Status::OK());
    for (int i = 0; i < thread_num; ++i) {
      parse_threads[i] = std::thread(
          [&](int tid) {
            while (true) {
              auto got = cur.fetch_add(1);
              if (got >= chunk_num) {
                break;
              }
              std::shared_ptr<oid_array_t> oid_array =
                  std::dynamic_pointer_cast<oid_array_t>(
                      oid_arrays_in->chunk(got));
              typename ConvertToArrowType<vid_t>::BuilderType builder;
              size_t size = oid_array->length();

              arrow::Status status = builder.Resize(size);
              if (!status.ok()) {
                statuses[tid] = status;
                return;
              }

              for (size_t k = 0; k != size; ++k) {
                internal_oid_t oid = oid_array->GetView(k);
                fid_t fid = partitioner_.GetPartitionId(oid_t(oid));
                if (!vm->GetGid(fid, label_id, oid, builder[k])) {
                  LOG(ERROR) << "Mapping vertex " << oid << " failed.";
                }
              }

              status = builder.Advance(size);
              if (!status.ok()) {
                statuses[tid] = status;
                return;
              }
              status = builder.Finish(&chunks_out[got]);
              if (!status.ok()) {
                statuses[tid] = status;
                return;
              }
            }
          },
          i);
    }
    for (auto& thrd : parse_threads) {
      thrd.join();
    }
    for (auto& status : statuses) {
      if (!status.ok()) {
        RETURN_GS_ERROR(ErrorCode::kArrowError, status.ToString());
      }
    }
#endif

    return std::make_shared<arrow::ChunkedArray>(chunks_out);
  }

  void initSchema(PropertyGraphSchema& schema) {
    schema.set_fnum(comm_spec_.fnum());
    for (label_id_t v_label = 0; v_label != vertex_label_num_; ++v_label) {
      std::string vertex_label = vertex_labels_[v_label];
      auto entry = schema.CreateEntry(vertex_label, "VERTEX");

      auto table = output_vertex_tables_[v_label];

      if (retain_oid_) {
        entry->AddPrimaryKeys(1, std::vector<std::string>{
                                     std::to_string(table->num_columns() - 1)});
      }

      for (int i = 0; i < table->num_columns(); ++i) {
        entry->AddProperty(table->schema()->field(i)->name(),
                           table->schema()->field(i)->type());
      }
    }
    for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
      std::string edge_label = edge_labels_[e_label];
      auto entry = schema.CreateEntry(edge_label, "EDGE");

      auto& relation_set = edge_relations_[e_label];
      for (auto& pair : relation_set) {
        std::string src_label = vertex_labels_[pair.first];
        std::string dst_label = vertex_labels_[pair.second];
        entry->AddRelation(src_label, dst_label);
      }

      auto table = output_edge_tables_[e_label];

      for (int i = 2; i < table->num_columns(); ++i) {
        entry->AddProperty(table->schema()->field(i)->name(),
                           table->schema()->field(i)->type());
      }
    }
  }

  boost::leaf::result<void> generateEdgeId(
      std::vector<std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                        std::shared_ptr<arrow::Table>>>>&
          edge_tables) {
    IdParser<uint64_t> eid_parser;
    label_id_t edge_label_num = edge_tables.size();
    eid_parser.Init(comm_spec_.fnum(), edge_label_num);
    for (label_id_t e_label = 0; e_label < edge_label_num; ++e_label) {
      auto& edge_table_list = edge_tables[e_label];
      uint64_t cur_id = eid_parser.GenerateId(comm_spec_.fid(), e_label, 0);
      for (size_t edge_table_index = 0;
           edge_table_index != edge_table_list.size(); ++edge_table_index) {
        auto& edge_table = edge_table_list[edge_table_index].second;
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        VY_OK_OR_RAISE(TableToRecordBatches(edge_table, &batches));
        std::vector<std::shared_ptr<arrow::Array>> generated_arrays;
        for (auto& rb : batches) {
          int64_t row_num = rb->num_rows();
          typename ConvertToArrowType<int64_t>::BuilderType builder;
          for (int64_t i = 0; i != row_num; ++i) {
            builder.Append(static_cast<int64_t>(cur_id));
            ++cur_id;
          }
          std::shared_ptr<arrow::Array> eid_array;
          builder.Finish(&eid_array);
          generated_arrays.push_back(eid_array);
        }
        std::shared_ptr<arrow::ChunkedArray> chunked_eid_array =
            std::make_shared<arrow::ChunkedArray>(
                generated_arrays, ConvertToArrowType<int64_t>::TypeValue());

        auto eid_field = std::make_shared<arrow::Field>(
            "eid", ConvertToArrowType<int64_t>::TypeValue());

#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
        CHECK_ARROW_ERROR(edge_table->AddColumn(
            edge_id_column, eid_field, chunked_eid_array, &edge_table));
#else
        CHECK_ARROW_ERROR_AND_ASSIGN(
            edge_table, edge_table->AddColumn(edge_id_column, eid_field,
                                              chunked_eid_array));
#endif
      }
    }

    return {};
  }

  Client& client_;

  label_id_t vertex_label_num_;
  label_id_t edge_label_num_;

  grape::CommSpec comm_spec_;
  const PARTITIONER_T& partitioner_;

  bool directed_;
  bool retain_oid_;
  bool generate_eid_;

  std::map<std::string, label_id_t> vertex_label_to_index_;
  std::vector<std::string> vertex_labels_;
  std::map<std::string, label_id_t> edge_label_to_index_;
  std::vector<std::string> edge_labels_;

  std::map<std::string, std::shared_ptr<arrow::Table>> input_vertex_tables_;
  std::map<std::string, std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                              std::shared_ptr<arrow::Table>>>>
      input_edge_tables_;

  std::vector<std::shared_ptr<arrow::Table>> ordered_vertex_tables_;
  std::vector<std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                    std::shared_ptr<arrow::Table>>>>
      ordered_edge_tables_;

  std::vector<std::shared_ptr<arrow::Table>> output_vertex_tables_;
  std::vector<std::shared_ptr<arrow::Table>> output_edge_tables_;
  std::vector<std::set<std::pair<label_id_t, label_id_t>>> edge_relations_;

  std::shared_ptr<ArrowVertexMap<internal_oid_t, vid_t>> vm_ptr_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_H_
