/** Copyright 2020 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VINEYARD_LOADER_ARROW_FRAGMENT_BUILDER_H_
#define VINEYARD_LOADER_ARROW_FRAGMENT_BUILDER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <numeric>

#include "boost/leaf/error.hpp"
#include "boost/leaf/result.hpp"
#include "grape/worker/comm_spec.h"
#include "vineyard/basic/ds/arrow_utils.h"
#include "vineyard/client/client.h"
#include "vineyard/common/util/functions.h"
#include "vineyard/graph/loader/fragment_loader_utils.h"
#include "vineyard/io/io/i_io_adaptor.h"
#include "vineyard/io/io/io_factory.h"
#include "vineyard/graph/utils/partitioner.h"

#include "gsf/graph_info.h"
#include "gsf/utils/trans.h"
#include "gsf/reader/chunk_info_reader.h"
#include "gsf/reader/arrow_chunk_reader.h"

namespace bl = boost::leaf;

std::shared_ptr<arrow::Table> CombineTables(const std::shared_ptr<arrow::Table> left, const std::shared_ptr<arrow::Table> right) {
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns = left->columns();
  const std::vector<std::shared_ptr<arrow::ChunkedArray>>& right_columns = right->columns();
  columns.insert(columns.end(), right_columns.begin(), right_columns.end());

  std::vector<std::shared_ptr<arrow::Field>> fields = left->fields();
  const std::vector<std::shared_ptr<arrow::Field>>& right_fields = right->fields();
  fields.insert(fields.end(), right_fields.begin(), right_fields.end());

  return arrow::Table::Make(arrow::schema(std::move(fields)), std::move(columns));
}

std::shared_ptr<arrow::Table> ConcatenateTablesColumnWise(const std::vector<std::shared_ptr<arrow::Table>> table_vec) {
  CHECK(!table_vec.empty());
  auto table = table_vec[0];
  for (int i = 1; i < table_vec.size(); ++i) {
    table = CombineTables(table, table_vec[i]);
  }
  return table;
}


namespace vineyard {
/**
 * @brief This builder can load a ArrowFragment from graph store format data source
 * @tparam OID_T OID type
 * @tparam VID_T VID type
 */
template <typename OID_T = gsf::IdType,
          typename VID_T = vineyard::property_graph_types::VID_TYPE>
class ArrowFragmentBuilder {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = vineyard::property_graph_types::LABEL_ID_TYPE;
  using internal_oid_t = typename vineyard::InternalType<oid_t>::type;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using vertex_map_t = vineyard::ArrowVertexMap<internal_oid_t, vid_t>;
  using oid_array_builder_t = typename vineyard::ConvertToArrowType<oid_t>::BuilderType;
  static constexpr const char* LABEL_TAG = "label";
  static constexpr const char* SRC_LABEL_TAG = "src_label";
  static constexpr const char* DST_LABEL_TAG = "dst_label";

  static constexpr int src_column = 0;
  static constexpr int dst_column = 1;

  using table_vec_t = std::vector<std::shared_ptr<arrow::Table>>;

 public:
  ArrowFragmentBuilder(vineyard::Client& client,
                      const grape::CommSpec& comm_spec,
                      std::shared_ptr<gsf::GraphInfo> graph_info,
                      bool directed = false)
      : client_(client),
        comm_spec_(comm_spec),
        graph_info_(graph_info),
        directed_(directed) {}

  ~ArrowFragmentBuilder() = default;

  bl::result<vineyard::ObjectID> LoadFragment() {
    distributeVertexChunks();

    BOOST_LEAF_CHECK(loadVertexTables());
    BOOST_LEAF_CHECK(loadEdgeTables());

    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-VERTEX-MAP-0";
    BOOST_LEAF_CHECK(constructVertexMap());
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-VERTEX-MAP-100";

    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-EDGE-0";
    BOOST_LEAF_CHECK(constructEdges());
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-EDGE-100";

    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-SEAL-0";
    return constructFragment();
  }

 private:
  void distributeVertexChunks() {
    for (const auto& vertex : graph_info_->GetAllVertexInfo()) {
      const auto& label = vertex.first;
      const auto& vertex_info = vertex.second;
      vertex_chunk_sizes_.push_back(vertex_info.GetChunkSize());
      auto maybe_reader = gsf::ConstructVertexPropertyChunkInfoReader(*(graph_info_.get()), label, vertex_info.GetPropertyGroups()[0]);
      CHECK(!maybe_reader.has_error());
      auto& reader = maybe_reader.value();
      label2vertex_chunk_num_[label] = reader.GetChunkNum();
      vertex_chunk_begin_of_frag_[label].resize(comm_spec_.fnum() + 1);
      for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
        vertex_chunk_begin_of_frag_[label][fid] = fid * (reader.GetChunkNum() / comm_spec_.fnum());
      }
      vertex_chunk_begin_of_frag_[label][comm_spec_.fnum()] = reader.GetChunkNum();
    }
  }

  bl::result<void> loadVertexTables() {
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-VERTEX-0";
      for (const auto& vertex : graph_info_->GetAllVertexInfo()) {
        const auto& label = vertex.first;
        const auto& vertex_info = vertex.second;
        int vertex_chunk_begin = vertex_chunk_begin_of_frag_[label][comm_spec_.worker_id()];
        int worker_vertex_chunk_num =
            vertex_chunk_begin_of_frag_[label][comm_spec_.fid() + 1] - vertex_chunk_begin_of_frag_[label][comm_spec_.fid()];

        table_vec_t pg_tables;
        for (const auto& pg : vertex_info.GetPropertyGroups()) {
          std::vector<table_vec_t> chunk_tables_2d(worker_vertex_chunk_num);
          std::vector<std::thread> threads(worker_vertex_chunk_num);
          for (int i = 0; i < worker_vertex_chunk_num; ++i) {
            threads[i] = std::thread([&](int tid) {
              auto maybe_reader = gsf::ConstructVertexPropertyArrowChunkReader(*(graph_info_.get()), label, pg);
              CHECK(!maybe_reader.has_error());
              auto& reader = maybe_reader.value();
              CHECK(reader.seek((tid + vertex_chunk_begin) * vertex_info.GetChunkSize()).ok());
              auto chunk_table = reader.GetChunk();
              CHECK(!chunk_table.has_error());
              chunk_tables_2d[tid].push_back(chunk_table.value());
            }, i);
          }
          for (auto& t : threads) {
            t.join();
          }
          table_vec_t chunk_tables(chunk_tables_2d.size());
          for (int i = 0; i < chunk_tables_2d.size(); ++i) {
            chunk_tables[i] = arrow::ConcatenateTables(chunk_tables_2d[i]).ValueOrDie();
          }
          auto pg_table = arrow::ConcatenateTables(chunk_tables);
          if (!pg_table.status().ok()) {
            LOG(ERROR) << "worker-" << comm_spec_.worker_id() << " Error: " << pg_table.status().message();
          }
          pg_tables.push_back(pg_table.ValueOrDie());
        }
        auto v_table = ConcatenateTablesColumnWise(pg_tables);

        auto metadata = std::make_shared<arrow::KeyValueMetadata>();
        metadata->Append("label", label);
        metadata->Append("label_id", std::to_string(vertex_labels_.size()));
        metadata->Append("type", "VERTEX");
        metadata->Append("retain_oid", std::to_string(false));
        vertex_tables_.push_back(v_table->ReplaceSchemaMetadata(metadata));
        vertex_labels_.push_back(label);
      }
      for (size_t i = 0; i < vertex_labels_.size(); ++i) {
        vertex_label_to_index_[vertex_labels_[i]] = i;
      }
      vertex_label_num_ = vertex_labels_.size();

      LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-VERTEX-100";
    return {};
  }

  bl::result<void> loadEdgeTables() {
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-EDGE-0";
    // read the adj list chunk tables
    for (const auto& edge : graph_info_->GetAllEdgeInfo()) {
      const auto& edge_info = edge.second;
      auto src_label = edge_info.GetSrcLabel();
      auto dst_label = edge_info.GetDstLabel();
      auto edge_label = edge_info.GetEdgeLabel();
      if (edge_info.ContainAdjList(gsf::AdjListType::ordered_by_source)) {
        auto maybe_offset_reader = gsf::ConstructAdjListOffsetArrowChunkReader(*(graph_info_.get()),
            src_label, edge_label, dst_label, gsf::AdjListType::ordered_by_source);
        CHECK(maybe_offset_reader.status().ok());
        auto& offset_reader = maybe_offset_reader.value();
        arrow::Int64Builder offset_array_builder;

        int vertex_chunk_begin = vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid()];
        int worker_vertex_chunk_num =
            vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid() + 1] - vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid()];

        std::vector<table_vec_t> edge_chunk_tables_2d(worker_vertex_chunk_num);
        std::vector<std::thread> threads(worker_vertex_chunk_num);
        for (int i = 0; i < worker_vertex_chunk_num; ++i) {
            threads[i] = std::thread([&](int tid) {
              auto expect = gsf::ConstructAdjListArrowChunkReader(*(graph_info_.get()),
                src_label, edge_label, dst_label, gsf::AdjListType::ordered_by_source);
              CHECK(!expect.has_error());
              auto& reader = expect.value();
              reader.ResetChunk(tid + vertex_chunk_begin);
              do {
                auto result = reader.GetChunk();
                CHECK(result.status().ok());
                edge_chunk_tables_2d[tid].push_back(result.value());
              } while (reader.next_chunk().ok());
            },i);
        }
        for (auto& t : threads) {
          t.join();
        }
        table_vec_t edge_chunk_tables(edge_chunk_tables_2d.size());
        for (int i = 0; i < edge_chunk_tables_2d.size(); ++i) {
          edge_chunk_tables[i] = arrow::ConcatenateTables(edge_chunk_tables_2d[i]).ValueOrDie();
        }
        auto maybe_adj_list_table = arrow::ConcatenateTables(edge_chunk_tables);
        if (!maybe_adj_list_table.status().ok()) {
          LOG(ERROR) << "worker-" << comm_spec_.worker_id() << " Error: " << maybe_adj_list_table.status().message();
        }
        auto adj_list_table = maybe_adj_list_table.ValueOrDie();
        auto offset_array = offset_array_builder.Finish().ValueOrDie();

        // process property chunks
        table_vec_t pg_tables;
        for (const auto& pg : edge_info.GetPropertyGroups(gsf::AdjListType::ordered_by_source)) {
          edge_chunk_tables_2d.clear();
          edge_chunk_tables_2d.resize(worker_vertex_chunk_num);
          threads.clear();
          threads.resize(worker_vertex_chunk_num);
          for (int i = 0; i < worker_vertex_chunk_num; ++i) {
            threads[i] = std::thread([&](int tid) {
              auto maybe_reader = gsf::ConstructAdjListPropertyArrowChunkReader(*(graph_info_.get()), edge_info.GetSrcLabel(), edge_info.GetEdgeLabel(),
                  edge_info.GetDstLabel(), pg, gsf::AdjListType::ordered_by_source);
              CHECK(!maybe_reader.has_error());
              auto& reader = maybe_reader.value();
              reader.ResetChunk(tid + vertex_chunk_begin);
              do {
                auto result = reader.GetChunk();
                CHECK(result.status().ok());
                edge_chunk_tables_2d[tid].push_back(result.value());
              } while (reader.next_chunk().ok());
            },i);
          }
          for (auto& t : threads) {
            t.join();
          }

          for (int i = 0; i < edge_chunk_tables_2d.size(); ++i) {
            edge_chunk_tables[i] = arrow::ConcatenateTables(edge_chunk_tables_2d[i]).ValueOrDie();
          }
          auto pg_table = arrow::ConcatenateTables(edge_chunk_tables);
          if(!pg_table.status().ok()) {
            LOG(ERROR) << "worker-" << comm_spec_.worker_id() << " Error: " << pg_table.status().message();
          }
          pg_tables.push_back(pg_table.ValueOrDie());
        }
        auto property_table = ConcatenateTablesColumnWise(pg_tables);

        auto metadata = std::make_shared<arrow::KeyValueMetadata>();
        metadata->Append("label", edge_label);
        metadata->Append("label_id", std::to_string(edge_labels_.size()));
        metadata->Append("type", "EDGE");

        // add triple relation
        auto iter = vertex_label_to_index_.find(src_label);
        if (iter == vertex_label_to_index_.end()) {
          RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid src vertex label " + src_label);
        }
        label_id_t src_label_id = iter->second;
        iter = vertex_label_to_index_.find(dst_label);
        if (iter == vertex_label_to_index_.end()) {
          RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Invalid dst vertex label " + dst_label);
        }
        label_id_t dst_label_id = iter->second;
        edge_relations_.emplace_back(src_label_id, dst_label_id);
        edge_tables_.push_back(std::make_tuple(adj_list_table, offset_array, property_table->ReplaceSchemaMetadata(metadata)));
        edge_labels_.push_back(edge_label);
      }
    }

    for (size_t i = 0; i < edge_labels_.size(); ++i) {
      edge_label_to_index_[edge_labels_[i]] = i;
    }
    edge_label_num_ = edge_labels_.size();
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-EDGE-100";
    return {};
  }

  boost::leaf::result<void> constructVertexMap() {
    std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays(
        vertex_label_num_);
    for (label_id_t v_label = 0; v_label < vertex_label_num_; ++v_label) {
      oid_arrays[v_label].resize(comm_spec_.fnum());
      std::vector<std::thread> thrds(comm_spec_.fnum());
      for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
        thrds[fid] = std::thread(
          [&](fid_t tid) {
            auto& label = vertex_labels_[v_label];
            int64_t vertex_id_begin = vertex_chunk_begin_of_frag_[label][tid] * vertex_chunk_sizes_[v_label];
            int64_t vertex_num = vertex_chunk_begin_of_frag_[label][tid + 1] * vertex_chunk_sizes_[v_label] - vertex_id_begin;
            std::vector<oid_t> oids(vertex_num);
            std::iota(oids.begin(), oids.end(), vertex_id_begin);

            arrow::Int64Builder oid_array_builder;
            oid_array_builder.Reset();
            oid_array_builder.Reserve(vertex_num);
            oid_array_builder.AppendValues(oids);
            oid_arrays[v_label][tid] = std::dynamic_pointer_cast<oid_array_t>(
                oid_array_builder.Finish().ValueOrDie());
          },
          fid);
      }

      for (auto& thrd : thrds) {
        thrd.join();
      }
    }

    ObjectID vm_id = InvalidObjectID();
    BasicArrowVertexMapBuilder<internal_oid_t, vid_t> vm_builder(
        client_, comm_spec_.fnum(), vertex_label_num_, oid_arrays);

    auto vm = vm_builder.Seal(client_);
    vm_id = vm->id();
    vm_ptr_ = std::dynamic_pointer_cast<ArrowVertexMap<internal_oid_t, vid_t>>(
        client_.GetObject(vm_id));
    return {};
  }

  boost::leaf::result<void> constructEdges() {
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      auto& item = edge_tables_[e_label];
      label_id_t src_label = edge_relations_[e_label].first;
      label_id_t dst_label = edge_relations_[e_label].second;
      auto adj_list_table = std::get<0>(item);
      BOOST_LEAF_AUTO(table,
                      edgesId2Gid(adj_list_table, src_label, dst_label));
      // replace the adj_list_table with the new one
      std::get<0>(item) = table;
    }
    return {};
  }

  boost::leaf::result<std::shared_ptr<arrow::Table>> edgesId2Gid(
      std::shared_ptr<arrow::Table> adj_list_table, label_id_t src_label,
      label_id_t dst_label) {
    std::shared_ptr<arrow::Field> src_gid_field =
        std::make_shared<arrow::Field>(
            "src", vineyard::ConvertToArrowType<vid_t>::TypeValue());
    std::shared_ptr<arrow::Field> dst_gid_field =
        std::make_shared<arrow::Field>(
            "dst", vineyard::ConvertToArrowType<vid_t>::TypeValue());
    auto src_column_type = adj_list_table->column(src_column)->type();
    auto dst_column_type = adj_list_table->column(dst_column)->type();

    BOOST_LEAF_AUTO(
        src_gid_array,
        parseOidChunkedArray(src_label, adj_list_table->column(src_column), true));
    BOOST_LEAF_AUTO(
        dst_gid_array,
        parseOidChunkedArray(dst_label, adj_list_table->column(dst_column), false));

    // replace oid columns with gid
    ARROW_OK_ASSIGN_OR_RAISE(
        adj_list_table,
        adj_list_table->SetColumn(src_column, src_gid_field, src_gid_array));
    ARROW_OK_ASSIGN_OR_RAISE(
        adj_list_table,
        adj_list_table->SetColumn(dst_column, dst_gid_field, dst_gid_array));
    return adj_list_table;
  }

  // parse oid to global id
  boost::leaf::result<std::shared_ptr<arrow::ChunkedArray>>
  parseOidChunkedArray(label_id_t label_id,
                       std::shared_ptr<arrow::ChunkedArray> oid_arrays_in,
                       bool is_src = true) {
    size_t chunk_num = oid_arrays_in->num_chunks();
    std::vector<std::shared_ptr<arrow::Array>> chunks_out(chunk_num);

    ArrowVertexMap<internal_oid_t, vid_t>* vm = vm_ptr_.get();

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

              if (is_src) {
                for (size_t k = 0; k != size; ++k) {
                  internal_oid_t oid = oid_array->GetView(k);
                  if (!vm->GetGid(comm_spec_.fid(), label_id, oid, builder[k])) {
                    LOG(ERROR) << "Mapping vertex " << oid << " failed.";
                  }
                }
              } else {
                for (size_t k = 0; k != size; ++k) {
                  internal_oid_t oid = oid_array->GetView(k);
                  if (!vm->GetGid(getFid(oid, label_id), label_id, oid, builder[k])) {
                    LOG(ERROR) << "Mapping vertex " << oid << " failed.";
                  }
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
    return std::make_shared<arrow::ChunkedArray>(chunks_out);
  }

  boost::leaf::result<ObjectID> constructFragment() {
    BasicArrowFragmentBuilder<oid_t, vid_t> frag_builder(client_, vm_ptr_);

    PropertyGraphSchema schema;
    BOOST_LEAF_CHECK(initSchema(schema));
    frag_builder.SetPropertyGraphSchema(std::move(schema));

    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();

    BOOST_LEAF_CHECK(frag_builder.Init(
        comm_spec_.fid(), comm_spec_.fnum(), std::move(vertex_tables_),
        std::move(edge_tables_), directed_, thread_num));

    auto frag = std::dynamic_pointer_cast<ArrowFragment<oid_t, vid_t>>(
        frag_builder.Seal(client_));

    VINEYARD_CHECK_OK(client_.Persist(frag->id()));
    return frag->id();
  }

  boost::leaf::result<void> initSchema(PropertyGraphSchema& schema) {
    schema.set_fnum(comm_spec_.fnum());
    for (label_id_t v_label = 0; v_label != vertex_label_num_; ++v_label) {
      std::string vertex_label = vertex_labels_[v_label];
      auto entry = schema.CreateEntry(vertex_label, "VERTEX");

      auto table = vertex_tables_[v_label];

      for (int i = 0; i < table->num_columns(); ++i) {
        entry->AddProperty(table->schema()->field(i)->name(),
                           table->schema()->field(i)->type());
      }
    }
    for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
      std::string edge_label = edge_labels_[e_label];
      auto entry = schema.CreateEntry(edge_label, "EDGE");

      auto& relation = edge_relations_[e_label];
      std::string src_label = vertex_labels_[relation.first];
      std::string dst_label = vertex_labels_[relation.second];
      entry->AddRelation(src_label, dst_label);

      auto table = std::get<2>(edge_tables_[e_label]);

      for (int i = 0; i < table->num_columns(); ++i) {
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

  fid_t getFid(internal_oid_t oid, label_id_t label_id) {
    auto chunk_index = oid / vertex_chunk_sizes_[label_id];
    auto& vertex_chunk_begins = vertex_chunk_begin_of_frag_[vertex_labels_[label_id]];
    // binary search;
    fid_t low = 0, high = comm_spec_.fnum();
    while (low <= high) {
      fid_t mid = (low + high) / 2;
      if (vertex_chunk_begins[mid] <= chunk_index &&
          vertex_chunk_begins[mid + 1] > chunk_index) {
        return mid;
      } else if (vertex_chunk_begins[mid] > chunk_index) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return low;
  }

 private:
  vineyard::Client& client_;
  grape::CommSpec comm_spec_;

  std::shared_ptr<gsf::GraphInfo> graph_info_;

  bool directed_;
  std::map<std::string, int> label2vertex_chunk_num_;
  std::vector<int64_t> vertex_chunk_sizes_;
  std::map<std::string, std::vector<int>> vertex_chunk_begin_of_frag_;

  // basic_fragment_builder
  label_id_t vertex_label_num_;
  std::map<std::string, label_id_t> vertex_label_to_index_;
  std::vector<std::string> vertex_labels_;
  std::vector<std::shared_ptr<arrow::Table>> vertex_tables_;

  label_id_t edge_label_num_;
  std::map<std::string, label_id_t> edge_label_to_index_;
  std::vector<std::string> edge_labels_;
  std::vector<std::pair<label_id_t, label_id_t>> edge_relations_;
  std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>> edge_tables_;

  std::shared_ptr<ArrowVertexMap<internal_oid_t, vid_t>> vm_ptr_;
};

}  // namespace vineyard

#endif  // ANALYTICAL_ENGINE_CORE_LOADER_ARROW_FRAGMENT_LOADER_H_
