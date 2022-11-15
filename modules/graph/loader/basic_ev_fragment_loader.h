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

#include "common/util/static_if.h"
#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/table_shuffler.h"
#include "graph/utils/table_shuffler_beta.h"
#include "graph/vertex_map/arrow_local_vertex_map.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {

template <typename OID_T, typename VID_T, typename PARTITIONER_T,
          typename VERTEX_MAP_T =
              ArrowVertexMap<typename InternalType<OID_T>::type, VID_T>>
class BasicEVFragmentLoader {
  static constexpr int id_column = 0;
  static constexpr int src_column = 0;
  static constexpr int dst_column = 1;
  static constexpr int edge_id_column = 2;

  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_t = OID_T;
  using vid_t = VID_T;
  using partitioner_t = PARTITIONER_T;
  using vertex_map_t = VERTEX_MAP_T;
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
    auto id_column_type = vertex_table->column(id_column)->type();

    if (!id_column_type->Equals(ConvertToArrowType<oid_t>::TypeValue())) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "OID_T is not same with arrow::Column(" +
                          std::to_string(id_column) + ") for label " + label);
    }

    if (input_vertex_tables_.find(label) == input_vertex_tables_.end()) {
      vertex_labels_.push_back(label);
      input_vertex_tables_[label] = vertex_table;
    } else {
      std::vector<std::shared_ptr<arrow::Table>> tables;
      tables.push_back(input_vertex_tables_.at(label));
      tables.push_back(vertex_table);
      input_vertex_tables_[label] = ConcatenateTables(tables);
    }
    return {};
  }

  boost::leaf::result<void> ConstructVertices(
      ObjectID vm_id = InvalidObjectID()) {
    for (size_t i = 0; i < vertex_labels_.size(); ++i) {
      vertex_label_to_index_[vertex_labels_[i]] = i;
    }
    vertex_label_num_ = vertex_labels_.size();

    ordered_vertex_tables_.clear();
    ordered_vertex_tables_.resize(vertex_label_num_, nullptr);

    for (auto& pair : input_vertex_tables_) {
      ordered_vertex_tables_[vertex_label_to_index_[pair.first]] = pair.second;
    }

    input_vertex_tables_.clear();

    output_vertex_tables_.resize(vertex_label_num_);

    constructVerticesImpl(
        vm_id,
        std::integral_constant<bool,
                               is_local_vertex_map<vertex_map_t>::value>{});
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
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid src vertex label " + src_label);
    }
    src_label_id = iter->second;
    iter = vertex_label_to_index_.find(dst_label);
    if (iter == vertex_label_to_index_.end()) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid dst vertex label " + dst_label);
    }
    dst_label_id = iter->second;

    auto src_column_type = edge_table->column(src_column)->type();
    auto dst_column_type = edge_table->column(dst_column)->type();

    if (!src_column_type->Equals(
            vineyard::ConvertToArrowType<oid_t>::TypeValue())) {
      RETURN_GS_ERROR(
          ErrorCode::kInvalidValueError,
          "OID_T is not consistent with src id of edge table for label " +
              src_label);
    }
    if (!dst_column_type->Equals(
            vineyard::ConvertToArrowType<oid_t>::TypeValue())) {
      RETURN_GS_ERROR(
          ErrorCode::kInvalidValueError,
          "OID_T is not consistent with dst id of edge table for label " +
              dst_label);
    }

    input_edge_tables_[edge_label].emplace_back(
        std::make_pair(src_label_id, dst_label_id), edge_table);
    if (std::find(std::begin(edge_labels_), std::end(edge_labels_),
                  edge_label) == std::end(edge_labels_)) {
      edge_labels_.push_back(edge_label);
    }
    return {};
  }

  boost::leaf::result<std::shared_ptr<arrow::Table>> edgesId2Gid(
      std::shared_ptr<arrow::Table> edge_table, label_id_t src_label,
      label_id_t dst_label) {
    std::shared_ptr<arrow::Field> src_gid_field =
        std::make_shared<arrow::Field>(
            "src", vineyard::ConvertToArrowType<vid_t>::TypeValue());
    std::shared_ptr<arrow::Field> dst_gid_field =
        std::make_shared<arrow::Field>(
            "dst", vineyard::ConvertToArrowType<vid_t>::TypeValue());

    BOOST_LEAF_AUTO(
        src_gid_array,
        parseOidChunkedArray(src_label, edge_table->column(src_column)));
    BOOST_LEAF_AUTO(
        dst_gid_array,
        parseOidChunkedArray(dst_label, edge_table->column(dst_column)));

    // replace oid columns with gid
    ARROW_OK_ASSIGN_OR_RAISE(
        edge_table,
        edge_table->SetColumn(src_column, src_gid_field, src_gid_array));
    ARROW_OK_ASSIGN_OR_RAISE(
        edge_table,
        edge_table->SetColumn(dst_column, dst_gid_field, dst_gid_array));
    return edge_table;
  }

  boost::leaf::result<void> ConstructEdges(int label_offset = 0,
                                           int vertex_label_num = 0) {
    if (vertex_label_num == 0) {
      vertex_label_num = vertex_label_num_;
    }
    for (size_t i = 0; i < edge_labels_.size(); ++i) {
      edge_label_to_index_[edge_labels_[i]] = i;
    }
    edge_label_num_ = edge_labels_.size();

    ordered_edge_tables_.clear();
    ordered_edge_tables_.resize(edge_label_num_);

    for (auto& pair : input_edge_tables_) {
      ordered_edge_tables_[edge_label_to_index_[pair.first]] =
          std::move(pair.second);
    }
    input_edge_tables_.clear();

    if (generate_eid_) {
      generateEdgeId(ordered_edge_tables_, comm_spec_, label_offset);
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

    vineyard::IdParser<vid_t> id_parser;
    id_parser.Init(comm_spec_.fnum(), vertex_label_num);

    output_edge_tables_.resize(edge_label_num_);
    LOG(INFO) << "Shuffling property edge tables";
    static_if<is_local_vertex_map<vertex_map_t>::value>(
        [&]() -> boost::leaf::result<void> {
          // Shuffle property tables
          for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
            std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                  std::shared_ptr<arrow::Table>>>
                shuffled_tables;
            for (auto& item : ordered_edge_tables_[e_label]) {
              auto shuffle_procedure =
                  [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
                BOOST_LEAF_AUTO(
                    table_out,
                    beta::ShufflePropertyEdgeTableByPartition<partitioner_t>(
                        comm_spec_, partitioner_, src_column, dst_column,
                        item.second));
                return table_out;
              };
              BOOST_LEAF_AUTO(table,
                              sync_gs_error(comm_spec_, shuffle_procedure));
              shuffled_tables.emplace_back(item.first, table);
            }
            ordered_edge_tables_[e_label] = shuffled_tables;
          }

          // Finish the local vertex map with outer vertices
          LOG(INFO) << "Finalizing local vertex map";
          fid_t fnum = comm_spec_.fnum();
          std::vector<std::vector<std::unordered_set<oid_t>>> outer_vertex_oids(
              fnum);
          for (fid_t i = 0; i < fnum; i++) {
            outer_vertex_oids[i].resize(vertex_label_num_);
          }
          for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
            for (auto& item : ordered_edge_tables_[e_label]) {
              auto src_label = item.first.first;
              auto dst_label = item.first.second;
              auto table = item.second;
              std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
              TableToRecordBatches(table, &batches);
              for (size_t i = 0; i < batches.size(); ++i) {
                auto cur_batch = batches[i];
                int64_t row_num = cur_batch->num_rows();

                auto src_col = std::dynamic_pointer_cast<oid_array_t>(
                    cur_batch->column(src_column));
                auto dst_col = std::dynamic_pointer_cast<oid_array_t>(
                    cur_batch->column(dst_column));
                for (int64_t row_id = 0; row_id < row_num; ++row_id) {
                  internal_oid_t internal_src_oid = src_col->GetView(row_id);
                  internal_oid_t internal_dst_oid = dst_col->GetView(row_id);
                  oid_t src_oid = oid_t(internal_src_oid);
                  oid_t dst_oid = oid_t(internal_dst_oid);
                  grape::fid_t src_fid = partitioner_.GetPartitionId(src_oid);
                  grape::fid_t dst_fid = partitioner_.GetPartitionId(dst_oid);
                  if (src_fid != comm_spec_.fid()) {
                    outer_vertex_oids[src_fid][src_label].insert(src_oid);
                  }
                  if (dst_fid != comm_spec_.fid()) {
                    outer_vertex_oids[dst_fid][dst_label].insert(dst_oid);
                  }
                }
              }
            }
          }
          // copy set to vector
          std::vector<std::vector<std::vector<oid_t>>> outer_oid_vec(fnum);
          {
            for (fid_t i = 0; i < fnum; i++) {
              outer_oid_vec[i].resize(vertex_label_num_);
              for (label_id_t j = 0; j < vertex_label_num_; j++) {
                auto& set = outer_vertex_oids[i][j];
                outer_oid_vec[i][j] =
                    std::vector<oid_t>(set.begin(), set.end());
                set.clear();
              }
            }
            outer_vertex_oids.clear();
          }

          // Request and response the outer vertex oid index
          int worker_id = comm_spec_.worker_id();
          int worker_num = comm_spec_.worker_num();
          std::vector<std::vector<std::vector<vid_t>>> index_lists(
              comm_spec_.fnum());
          std::thread request_thread([&]() {
            for (int i = 1; i < worker_num; ++i) {
              int dst_worker_id = (worker_id + i) % worker_num;
              auto& oids =
                  outer_oid_vec[comm_spec_.WorkerToFrag(dst_worker_id)];
              grape::sync_comm::Send(oids, dst_worker_id, 0, comm_spec_.comm());
              grape::sync_comm::Recv(
                  index_lists[comm_spec_.WorkerToFrag(dst_worker_id)],
                  dst_worker_id, 1, comm_spec_.comm());
            }
          });
          std::thread response_thread([&]() {
            for (int i = 1; i < worker_num; ++i) {
              int src_worker_id = (worker_id + worker_num - i) % worker_num;
              std::vector<std::vector<oid_t>> oids;
              grape::sync_comm::Recv(oids, src_worker_id, 0, comm_spec_.comm());
              std::vector<std::vector<vid_t>> index_list;
              local_vm_builder_->GetIndexOfOids(oids, index_list);
              grape::sync_comm::Send(index_list, src_worker_id, 1,
                                     comm_spec_.comm());
            }
          });

          request_thread.join();
          response_thread.join();
          MPI_Barrier(comm_spec_.comm());
          // Construct the outer vertex map with response o2i
          local_vm_builder_->AddOuterVerticesMapping(outer_oid_vec,
                                                     index_lists);
          auto vm = local_vm_builder_->_Seal(client_);
          vm_ptr_ = std::dynamic_pointer_cast<vertex_map_t>(
              client_.GetObject(vm->id()));

          // Concatenate and add metadata to final edge tables
          LOG(INFO) << "Transforming ids of edge tables and concatenate them";
          for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
            std::vector<std::shared_ptr<arrow::Table>> processed_table_list;
            for (auto& item : ordered_edge_tables_[e_label]) {
              BOOST_LEAF_AUTO(table, edgesId2Gid(item.second, item.first.first,
                                                 item.first.second));
              processed_table_list.emplace_back(table);
            }
            auto table = vineyard::ConcatenateTables(processed_table_list);
            auto metadata = std::make_shared<arrow::KeyValueMetadata>();
            metadata->Append("label", edge_labels_[e_label]);
            metadata->Append("label_id", std::to_string(e_label));
            metadata->Append("type", "EDGE");
            output_edge_tables_[e_label] =
                table->ReplaceSchemaMetadata(metadata);
            ordered_edge_tables_[e_label].clear();
          }
          return {};
        })();
    static_if<!is_local_vertex_map<vertex_map_t>::value>(
        [&]() -> boost::leaf::result<void> {
          for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
            auto& edge_table_list = ordered_edge_tables_[e_label];
            LOG(INFO) << "Transforming ids of edge tables and concatenate them";
            auto shuffle_procedure =
                [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
              std::vector<std::shared_ptr<arrow::Table>> processed_table_list;
              for (auto& item : edge_table_list) {
                label_id_t src_label = item.first.first;
                label_id_t dst_label = item.first.second;
                BOOST_LEAF_AUTO(tmp_table,
                                edgesId2Gid(item.second, src_label, dst_label));
                processed_table_list.emplace_back(tmp_table);
              }
              auto table = vineyard::ConcatenateTables(processed_table_list);
              // Shuffle the edge table with gid
              BOOST_LEAF_AUTO(table_out, beta::ShufflePropertyEdgeTable<vid_t>(
                                             comm_spec_, id_parser, src_column,
                                             dst_column, table));
              return table_out;
            };
            BOOST_LEAF_AUTO(table,
                            sync_gs_error(comm_spec_, shuffle_procedure));

            auto metadata = std::make_shared<arrow::KeyValueMetadata>();
            metadata->Append("label", edge_labels_[e_label]);
            metadata->Append("label_id", std::to_string(e_label));
            metadata->Append("type", "EDGE");
            output_edge_tables_[e_label] =
                table->ReplaceSchemaMetadata(metadata);
            ordered_edge_tables_[e_label].clear();
          }
          return {};
        })();

    ordered_edge_tables_.clear();
    return {};
  }

  boost::leaf::result<ObjectID> AddVerticesToFragment(
      std::shared_ptr<ArrowFragment<oid_t, vid_t, vertex_map_t>> frag) {
    if (vineyard::is_local_vertex_map<vertex_map_t>::value) {
      RETURN_GS_ERROR(
          vineyard::ErrorCode::kUnsupportedOperationError,
          "Cannot only add vertices to fragment with local vertex map");
    }
    int pre_vlabel_num = frag->schema().all_vertex_label_num();
    std::map<label_id_t, std::shared_ptr<arrow::Table>> vertex_tables_map;
    for (size_t i = 0; i < output_vertex_tables_.size(); ++i) {
      vertex_tables_map[pre_vlabel_num + i] = output_vertex_tables_[i];
    }
    return frag->AddVertices(client_, std::move(vertex_tables_map),
                             vm_ptr_->id());
  }

  boost::leaf::result<ObjectID> AddEdgesToFragment(
      std::shared_ptr<ArrowFragment<oid_t, vid_t, vertex_map_t>> frag) {
    std::vector<std::set<std::pair<std::string, std::string>>> edge_relations(
        edge_label_num_);
    int pre_vlabel_num = frag->schema().all_vertex_label_num();
    int pre_elabel_num = frag->schema().all_edge_label_num();
    std::map<label_id_t, std::shared_ptr<arrow::Table>> edge_tables_map;
    for (size_t i = 0; i < output_edge_tables_.size(); ++i) {
      edge_tables_map[pre_elabel_num + i] = output_edge_tables_[i];
    }

    vertex_labels_.resize(pre_vlabel_num);
    for (auto& pair : vertex_label_to_index_) {
      vertex_labels_[pair.second] = pair.first;
    }
    for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
      for (auto& pair : edge_relations_[e_label]) {
        std::string src_label = vertex_labels_[pair.first];
        std::string dst_label = vertex_labels_[pair.second];
        edge_relations[e_label].insert({src_label, dst_label});
      }
    }
    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();
    return frag->AddEdges(client_, std::move(edge_tables_map), edge_relations,
                          thread_num);
  }

  boost::leaf::result<ObjectID> AddVerticesAndEdgesToFragment(
      std::shared_ptr<ArrowFragment<oid_t, vid_t, vertex_map_t>> frag) {
    if (output_vertex_tables_.empty()) {
      return AddEdgesToFragment(frag);
    }
    int pre_vlabel_num = frag->schema().all_vertex_label_num();
    int pre_elabel_num = frag->schema().all_edge_label_num();
    std::map<label_id_t, std::shared_ptr<arrow::Table>> vertex_tables_map;
    for (size_t i = 0; i < output_vertex_tables_.size(); ++i) {
      vertex_tables_map[pre_vlabel_num + i] = output_vertex_tables_[i];
    }

    std::map<label_id_t, std::shared_ptr<arrow::Table>> edge_tables_map;
    for (size_t i = 0; i < output_edge_tables_.size(); ++i) {
      edge_tables_map[pre_elabel_num + i] = output_edge_tables_[i];
    }

    // The size is all vertex label number.
    vertex_labels_.resize(pre_vlabel_num + output_vertex_tables_.size());
    for (auto& pair : vertex_label_to_index_) {
      vertex_labels_[pair.second] = pair.first;
    }
    std::vector<std::set<std::pair<std::string, std::string>>> edge_relations(
        edge_label_num_);
    for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
      for (auto& pair : edge_relations_[e_label]) {
        std::string src_label = vertex_labels_[pair.first];
        std::string dst_label = vertex_labels_[pair.second];
        edge_relations[e_label].insert({src_label, dst_label});
      }
    }
    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();
    return frag->AddVerticesAndEdges(client_, std::move(vertex_tables_map),
                                     std::move(edge_tables_map), vm_ptr_->id(),
                                     edge_relations, thread_num);
  }

  boost::leaf::result<ObjectID> ConstructFragment() {
    BasicArrowFragmentBuilder<oid_t, vid_t, vertex_map_t> frag_builder(client_,
                                                                       vm_ptr_);

    PropertyGraphSchema schema;
    BOOST_LEAF_CHECK(initSchema(schema));
    frag_builder.SetPropertyGraphSchema(std::move(schema));

    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();

    BOOST_LEAF_CHECK(frag_builder.Init(
        comm_spec_.fid(), comm_spec_.fnum(), std::move(output_vertex_tables_),
        std::move(output_edge_tables_), directed_, thread_num));

    auto frag =
        std::dynamic_pointer_cast<ArrowFragment<oid_t, vid_t, vertex_map_t>>(
            frag_builder.Seal(client_));

    VINEYARD_CHECK_OK(client_.Persist(frag->id()));
    return frag->id();
  }

  void set_vertex_label_to_index(std::map<std::string, label_id_t>&& in) {
    vertex_label_to_index_ = std::move(in);
  }

  std::map<std::string, label_id_t> get_vertex_label_to_index() {
    return vertex_label_to_index_;
  }

  void set_vm_ptr(std::shared_ptr<vertex_map_t> in) { vm_ptr_ = in; }

 private:
  boost::leaf::result<std::shared_ptr<arrow::ChunkedArray>>
  parseOidChunkedArray(label_id_t label_id,
                       std::shared_ptr<arrow::ChunkedArray> oid_arrays_in) {
    size_t chunk_num = oid_arrays_in->num_chunks();
    std::vector<std::shared_ptr<arrow::Array>> chunks_out(chunk_num);

    vertex_map_t* vm = vm_ptr_.get();

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

  boost::leaf::result<void> initSchema(PropertyGraphSchema& schema) {
    schema.set_fnum(comm_spec_.fnum());
    for (label_id_t v_label = 0; v_label != vertex_label_num_; ++v_label) {
      std::string vertex_label = vertex_labels_[v_label];
      auto entry = schema.CreateEntry(vertex_label, "VERTEX");

      auto table = output_vertex_tables_[v_label];

      if (retain_oid_) {
        int col_id = table->num_columns() - 1;
        entry->AddPrimaryKey(table->schema()->field(col_id)->name());
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
    std::string message;
    if (!schema.Validate(message)) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError, message);
    }
    return {};
  }

  boost::leaf::result<void> generateEdgeId(
      std::vector<std::vector<std::pair<std::pair<label_id_t, label_id_t>,
                                        std::shared_ptr<arrow::Table>>>>&
          edge_tables,
      grape::CommSpec& comm_spec, int label_offset) {
    IdParser<uint64_t> eid_parser;
    label_id_t edge_label_num = edge_tables.size();
    eid_parser.Init(comm_spec.fnum(), edge_label_num + label_offset);
    for (label_id_t e_label = 0; e_label < edge_label_num; ++e_label) {
      auto& edge_table_list = edge_tables[e_label];
      uint64_t cur_id =
          eid_parser.GenerateId(comm_spec.fid(), e_label + label_offset, 0);
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
            CHECK_ARROW_ERROR(builder.Append(static_cast<int64_t>(cur_id)));
            ++cur_id;
          }
          std::shared_ptr<arrow::Array> eid_array;
          CHECK_ARROW_ERROR(builder.Finish(&eid_array));
          generated_arrays.push_back(eid_array);
        }
        std::shared_ptr<arrow::ChunkedArray> chunked_eid_array =
            std::make_shared<arrow::ChunkedArray>(
                generated_arrays, ConvertToArrowType<int64_t>::TypeValue());

        auto eid_field = std::make_shared<arrow::Field>(
            "eid", ConvertToArrowType<int64_t>::TypeValue());

        CHECK_ARROW_ERROR_AND_ASSIGN(
            edge_table, edge_table->AddColumn(edge_id_column, eid_field,
                                              chunked_eid_array));
      }
    }

    return {};
  }

  // constructVertices implementation for ArrowVertexMap
  boost::leaf::result<void> constructVerticesImpl(ObjectID vm_id,
                                                  std::false_type) {
    std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_lists(
        vertex_label_num_);
    for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
      auto vertex_table = ordered_vertex_tables_[v_label];
      auto shuffle_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
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
          CHECK_ARROW_ERROR_AND_ASSIGN(tmp_table,
                                       tmp_table->RemoveColumn(id_column));
          CHECK_ARROW_ERROR_AND_ASSIGN(
              tmp_table, tmp_table->AddColumn(tmp_table->num_columns(),
                                              id_field, id_array));
        }

        return tmp_table;
      };
      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));

      auto metadata = std::make_shared<arrow::KeyValueMetadata>();
      metadata->Append("label", vertex_labels_[v_label]);
      metadata->Append("label_id", std::to_string(v_label));
      metadata->Append("type", "VERTEX");
      metadata->Append("retain_oid", std::to_string(retain_oid_));
      output_vertex_tables_[v_label] = table->ReplaceSchemaMetadata(metadata);
    }
    ObjectID new_vm_id = InvalidObjectID();
    if (vm_id == InvalidObjectID()) {
      BasicArrowVertexMapBuilder<internal_oid_t, vid_t> vm_builder(
          client_, comm_spec_.fnum(), vertex_label_num_, oid_lists);

      auto vm = vm_builder.Seal(client_);
      new_vm_id = vm->id();
    } else {
      auto old_vm_ptr =
          std::dynamic_pointer_cast<vertex_map_t>(client_.GetObject(vm_id));
      label_id_t pre_label_num = old_vm_ptr->label_num();
      std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>
          oid_lists_map;
      for (size_t i = 0; i < oid_lists.size(); ++i) {
        oid_lists_map[pre_label_num + i] = oid_lists[i];
      }
      // No new vertices.
      if (oid_lists_map.empty()) {
        new_vm_id = vm_id;
      } else {
        new_vm_id = old_vm_ptr->AddVertices(client_, oid_lists_map);
      }
    }
    vm_ptr_ =
        std::dynamic_pointer_cast<vertex_map_t>(client_.GetObject(new_vm_id));

    ordered_vertex_tables_.clear();
    return {};
  }

  // constructVertices implementation for ArrowLocalVertexMap
  boost::leaf::result<void> constructVerticesImpl(ObjectID vm_id,
                                                  std::true_type) {
    if (vm_id != InvalidObjectID()) {
      auto old_vm_ptr =
          std::dynamic_pointer_cast<vertex_map_t>(client_.GetObject(vm_id));
      label_id_t pre_label_num = old_vm_ptr->label_num();
      if (pre_label_num != 0) {
        std::string msg =
            "Not support adding vertex label to existing labels with local "
            "vertex map";
        RETURN_GS_ERROR(ErrorCode::kUnsupportedOperationError, msg);
      }
    }
    local_vm_builder_ =
        std::make_shared<ArrowLocalVertexMapBuilder<internal_oid_t, vid_t>>(
            client_, comm_spec_.fnum(), comm_spec_.fid(), vertex_label_num_);

    std::vector<std::shared_ptr<oid_array_t>> local_oid_array(
        vertex_label_num_);
    for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
      auto vertex_table = ordered_vertex_tables_[v_label];
      auto shuffle_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        BOOST_LEAF_AUTO(tmp_table,
                        beta::ShufflePropertyVertexTable<partitioner_t>(
                            comm_spec_, partitioner_, vertex_table));

        local_oid_array[v_label] = std::dynamic_pointer_cast<oid_array_t>(
            tmp_table->column(id_column)->chunk(0));

        // TODO: check about add a new label on old vertex map
        // local_vm_builder_->AddLocalVertices(v_label, local_oid_array);

        if (retain_oid_) {
          auto id_field = tmp_table->schema()->field(id_column);
          auto id_array = tmp_table->column(id_column);
          CHECK_ARROW_ERROR_AND_ASSIGN(tmp_table,
                                       tmp_table->RemoveColumn(id_column));
          CHECK_ARROW_ERROR_AND_ASSIGN(
              tmp_table, tmp_table->AddColumn(tmp_table->num_columns(),
                                              id_field, id_array));
        }

        return tmp_table;
      };
      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, shuffle_procedure));

      auto metadata = std::make_shared<arrow::KeyValueMetadata>();
      metadata->Append("label", vertex_labels_[v_label]);
      metadata->Append("label_id", std::to_string(v_label));
      metadata->Append("type", "VERTEX");
      metadata->Append("retain_oid", std::to_string(retain_oid_));
      output_vertex_tables_[v_label] = table->ReplaceSchemaMetadata(metadata);
    }
    local_vm_builder_->AddLocalVertices(comm_spec_, local_oid_array);
    local_oid_array.clear();
    ordered_vertex_tables_.clear();

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

  std::shared_ptr<vertex_map_t> vm_ptr_;

  std::shared_ptr<ArrowLocalVertexMapBuilder<internal_oid_t, vid_t>>
      local_vm_builder_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_BASIC_EV_FRAGMENT_LOADER_H_
