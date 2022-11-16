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
#include "gsf/reader/chunk_info_reader.h"
#include "gsf/reader/arrow_chunk_reader.h"

namespace bl = boost::leaf;

std::shared_ptr<arrow::Table> ConcatenateTablesColumnWise(const std::vector<std::shared_ptr<arrow::Table>> table_vec) {
  CHECK(!table_vec.empty());
  auto table = table_vec[0];
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns = table->columns();
  std::vector<std::shared_ptr<arrow::Field>> fields = table->fields();
  for (int i = 1; i < table_vec.size(); ++i) {
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& right_columns = table_vec[i]->columns();
    columns.insert(columns.end(), right_columns.begin(), right_columns.end());

    const std::vector<std::shared_ptr<arrow::Field>>& right_fields = table_vec[i]->fields();
    fields.insert(fields.end(), right_fields.begin(), right_fields.end());
  }
  table = arrow::Table::Make(arrow::schema(std::move(fields)), std::move(columns));
  return table;
}

std::shared_ptr<arrow::Table> CreateEmptyTable() {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  return arrow::Table::Make(arrow::schema(std::move(fields)), std::move(columns));
};


namespace vineyard {

template <typename OID_T, typename VID_T>
class ArrowFragmentBaseBuilder;

template <typename OID_T, typename VID_T>
class GSFArrowFragmentBuilder;

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

  bl::result<void> TestReadChunks() {
    double t = vineyard::GetCurrentTime();
    BOOST_LEAF_CHECK(distributeVertexChunks());
    double t1 = vineyard::GetCurrentTime();
    BOOST_LEAF_CHECK(loadVertexTables());
    LOG(INFO) << "Frag-" << comm_spec_.fid() << "  loadVertexTables time: " << vineyard::GetCurrentTime() - t1;
    double t2 = vineyard::GetCurrentTime();
    BOOST_LEAF_CHECK(loadEdgeTables());
    LOG(INFO) << "Frag-" << comm_spec_.fid() << "  loadEdgeTables time: " << vineyard::GetCurrentTime() - t2;
    LOG(INFO) << "Frag-" << comm_spec_.fid() << " TestReadChunks time: " << vineyard::GetCurrentTime() - t;
    return {};
  }

  bl::result<vineyard::ObjectID> LoadFragment() {
    BOOST_LEAF_CHECK(distributeVertexChunks());

    double t = vineyard::GetCurrentTime();
    BOOST_LEAF_CHECK(loadVertexTables());
    LOG(INFO) << "Frag-" << comm_spec_.fid() << " Load vertex tables cost: " << vineyard::GetCurrentTime() - t << "s";
    BOOST_LEAF_CHECK(loadEdgeTables());

    // LOG_IF(INFO, comm_spec_.worker_id() == 0)
    LOG(INFO)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-VERTEX-MAP-0 " << comm_spec_.worker_id();
    BOOST_LEAF_CHECK(constructVertexMap());
    // LOG_IF(INFO, comm_spec_.worker_id() == 0)
    LOG(INFO)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-VERTEX-MAP-100 " << comm_spec_.worker_id();

    LOG_IF(INFO, comm_spec_.worker_id() == 0)
    // LOG(INFO)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-EDGE-0 " << comm_spec_.worker_id();
    BOOST_LEAF_CHECK(constructEdges());
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
    // LOG(INFO)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-EDGE-100 " << comm_spec_.worker_id();

    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-SEAL-0";
    return constructFragment();
  }

 private:
  bl::result<void> distributeVertexChunks() {
    for (const auto& vertex : graph_info_->GetAllVertexInfo()) {
      const auto& label = vertex.first;
      const auto& vertex_info = vertex.second;
      vertex_chunk_sizes_.push_back(vertex_info.GetChunkSize());
      auto maybe_reader = gsf::ConstructVertexPropertyChunkInfoReader(*(graph_info_.get()), label, vertex_info.GetPropertyGroups()[0]);
      if (maybe_reader.has_error()) {
        RETURN_GS_ERROR(ErrorCode::kIOError,
                    "Construct reader error: " + maybe_reader.status().message());
      }
      auto& reader = maybe_reader.value();
      vertex_chunk_begin_of_frag_[label].resize(comm_spec_.fnum() + 1);
      for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
        vertex_chunk_begin_of_frag_[label][fid] = static_cast<gsf::IdType>(fid) * (reader.GetChunkNum() / static_cast<gsf::IdType>(comm_spec_.fnum()));
      }
      vertex_chunk_begin_of_frag_[label][comm_spec_.fnum()] = reader.GetChunkNum();
      LOG(INFO) << "frag-" << comm_spec_.fid() << " get vertex " << label << " chunk from " << vertex_chunk_begin_of_frag_[label][comm_spec_.fid()] << " to " << vertex_chunk_begin_of_frag_[label][comm_spec_.fid() + 1];
    }
    return {};
  }

  bl::result<void> distributeVertexChunksWithEdgeInfo() {
    for (const auto& edge : graph_info_->GetAllEdgeInfo()) {
      const auto& edge_info = edge.second;
      auto label = edge_info.GetSrcLabel();
      std::string base_dir;
      auto fs = gsf::FileSystemFromUriOrPath(graph_info_->GetPrefix(), &base_dir).value();
      base_dir += "/" + edge_info.GetAdjListDirPath(gsf::AdjListType::ordered_by_dest);
      gsf::IdType vertex_chunk_num = fs->GetFileNumInDir(base_dir).value();
      LOG(INFO) << "vertex_chunk_num = " << vertex_chunk_num;
      vertex_chunk_begin_of_frag_[label].resize(comm_spec_.fnum() + 1);
      for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
        vertex_chunk_begin_of_frag_[label][fid] = static_cast<gsf::IdType>(fid) * (vertex_chunk_num / static_cast<gsf::IdType>(comm_spec_.fnum()));
      }
      vertex_chunk_begin_of_frag_[label][comm_spec_.fnum()] = vertex_chunk_num;
      LOG(INFO) << "frag-" << comm_spec_.fid() << " get vertex " << label << " chunk from " << vertex_chunk_begin_of_frag_[label][comm_spec_.fid()] << " to " << vertex_chunk_begin_of_frag_[label][comm_spec_.fid() + 1];
    }
    return {};
  }

  bl::result<void> loadVertexTables() {
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-VERTEX-0";
    for (const auto& vertex : graph_info_->GetAllVertexInfo()) {
      // LOG_IF(INFO, comm_spec_.worker_id() == 0)
      // LOG(INFO) << "Frag-" << comm_spec_.fid() << " Loading vertex: " << vertex.first;
      const auto& label = vertex.first;
      const auto& vertex_info = vertex.second;
      auto vertex_chunk_begin = vertex_chunk_begin_of_frag_[label][comm_spec_.fid()];
      auto worker_vertex_chunk_num =
          vertex_chunk_begin_of_frag_[label][comm_spec_.fid() + 1] - vertex_chunk_begin_of_frag_[label][comm_spec_.fid()];
      auto chunk_size = vertex_info.GetChunkSize();

      table_vec_t pg_tables;
      for (const auto& pg : vertex_info.GetPropertyGroups()) {
        int64_t thread_num =
            (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
             comm_spec_.local_num();
        table_vec_t vertex_chunk_tables(worker_vertex_chunk_num);
        std::vector<std::thread> threads(thread_num);
        std::atomic<size_t> cur_chunk_index(0);
        size_t chunk = (worker_vertex_chunk_num + thread_num - 1) / thread_num;
        for (int64_t i = 0; i < thread_num; ++i) {
          threads[i] = std::thread([&]() {
            auto maybe_reader = gsf::ConstructVertexPropertyArrowChunkReader(*(graph_info_.get()), label, pg);
            CHECK(!maybe_reader.has_error());
            auto& reader = maybe_reader.value();
            while (true) {
              size_t begin = cur_chunk_index.fetch_add(chunk);
              if (begin >= worker_vertex_chunk_num) {
                break;
              }
              int64_t end = std::min(static_cast<int64_t>(begin + chunk), worker_vertex_chunk_num);
              int64_t iter = begin;
              while (iter != end) {
                reader.seek((vertex_chunk_begin + iter) * chunk_size);
                auto chunk_table = reader.GetChunk();
                vertex_chunk_tables[iter] = chunk_table.value();
                ++iter;
              }
            }
          });
        }
        for (auto& t : threads) {
          t.join();
        }
        auto pg_table = arrow::ConcatenateTables(vertex_chunk_tables);
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

    // LOG(INFO) << "Frag-" << comm_spec_.fid() << " Loading vertex: " << vertex.first << " Finished.";
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
      << "PROGRESS--GRAPH-LOADING-READ-VERTEX-100";
    return {};
  }

  bl::result<void> loadEdgeTables() {
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-EDGE-0";
    // read the adj list chunk tables
    for (const auto& edge : graph_info_->GetAllEdgeInfo()) {
      // LOG_IF(INFO, comm_spec_.worker_id() == 0)
      // LOG(INFO) << "Frag-" << comm_spec_.fid() << " Loading edge: " << edge.first;
      const auto& edge_info = edge.second;
      if (edge_info.ContainAdjList(gsf::AdjListType::ordered_by_source)) {
        // double t = vineyard::GetCurrentTime();
        readEdgeTableOfLabel(edge_info, gsf::AdjListType::ordered_by_source);
        // LOG(INFO) << "Frag-" << comm_spec_.fid() << " Load CSR time: " << vineyard::GetCurrentTime() - t;
      }
      if (edge_info.ContainAdjList(gsf::AdjListType::ordered_by_dest)) {
        // double t = vineyard::GetCurrentTime();
        readEdgeTableOfLabel(edge_info, gsf::AdjListType::ordered_by_dest);
        // LOG(INFO) << "Frag-" << comm_spec_.fid() << " Load CSC time: " << vineyard::GetCurrentTime() - t;
      }

      // add edge relation
      auto src_label = edge_info.GetSrcLabel();
      auto dst_label = edge_info.GetDstLabel();
      auto edge_label = edge_info.GetEdgeLabel();
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
      auto it = std::find(edge_labels_.begin(), edge_labels_.end(), edge_label);
      if (it != edge_labels_.end()) {
        edge_relations_[it - edge_labels_.begin()].push_back(std::make_pair(src_label_id, dst_label_id));
      } else {
        edge_labels_.push_back(edge_label);
        edge_relations_.push_back({std::make_pair(src_label_id, dst_label_id)});
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

  bl::result<void> constructVertexMap() {
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
            std::vector<oid_t> oids(vertex_num, 0);
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

  bl::result<void> readEdgeTableOfLabel(const gsf::EdgeInfo& edge_info, gsf::AdjListType adj_list_type) {
    auto src_label = edge_info.GetSrcLabel();
    auto dst_label = edge_info.GetDstLabel();
    auto edge_label = edge_info.GetEdgeLabel();
    const auto& property_groups = edge_info.GetPropertyGroups(adj_list_type);
    std::string base_dir;
    auto fs = gsf::FileSystemFromUriOrPath(graph_info_->GetPrefix(), &base_dir).value();
    base_dir += "/" + edge_info.GetAdjListDirPath(adj_list_type);

    int64_t vertex_chunk_begin = 0;
    int64_t worker_vertex_chunk_num = 0;
    int64_t vertex_chunk_size;
    if (adj_list_type == gsf::AdjListType::ordered_by_source) {
      vertex_chunk_begin = vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid()];
      worker_vertex_chunk_num =
        vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid() + 1] - vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid()];
      vertex_chunk_size = edge_info.GetSrcChunkSize();
    } else {
      vertex_chunk_begin = vertex_chunk_begin_of_frag_[dst_label][comm_spec_.fid()];
      worker_vertex_chunk_num =
        vertex_chunk_begin_of_frag_[dst_label][comm_spec_.fid() + 1] - vertex_chunk_begin_of_frag_[dst_label][comm_spec_.fid()];
      vertex_chunk_size = edge_info.GetDstChunkSize();
    }
    std::vector<std::shared_ptr<arrow::Array>> offset_arrays(worker_vertex_chunk_num);

    int64_t thread_num =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();
    std::vector<std::thread> threads(thread_num);
    std::vector<gsf::IdType> edge_chunk_num_vec(worker_vertex_chunk_num + 1, 0);
    std::atomic<size_t> cur_chunk(0);
    size_t chunk = (worker_vertex_chunk_num + thread_num - 1) / thread_num;
    for (int64_t i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        auto maybe_offset_reader = gsf::ConstructAdjListOffsetArrowChunkReader(*(graph_info_.get()),
              src_label, edge_label, dst_label, adj_list_type);
        CHECK(maybe_offset_reader.status().ok());
        auto& offset_reader = maybe_offset_reader.value();
        while (true) {
          size_t begin = cur_chunk.fetch_add(chunk);
          if (begin >= worker_vertex_chunk_num) {
            break;
          }
          int64_t end = std::min(static_cast<int64_t>(begin + chunk), worker_vertex_chunk_num);
          int64_t iter = begin;
          while (iter != end) {
            int64_t vertex_chunk_id = iter + vertex_chunk_begin;
            std::string chunk_dir = base_dir + "/part" + std::to_string(vertex_chunk_id);
            auto num = fs->GetFileNumInDir(chunk_dir);
            // if (!num.status().ok()) {
            //   LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << num.status().message();
            // }
            edge_chunk_num_vec[iter] = num.value();
            auto st = offset_reader.seek(vertex_chunk_id * vertex_chunk_size);
            // if (!st.ok()) {
            //   LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << st.message();
            // }
            auto offset_result = offset_reader.GetChunk();
            // if (!offset_result.status().ok()) {
            //   LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << offset_result.status().message();
            // }
            offset_arrays[iter] = offset_result.value();
            ++iter;
          }
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
    for (int i = 1; i < edge_chunk_num_vec.size() - 1; ++i) {
      edge_chunk_num_vec[i] += edge_chunk_num_vec[i - 1];
    }
    for (size_t i = edge_chunk_num_vec.size() - 1; i > 0; --i) {
      edge_chunk_num_vec[i] = edge_chunk_num_vec[i - 1];
    }
    edge_chunk_num_vec[0] = 0;
    auto total_edge_chunk_num = edge_chunk_num_vec.back();
    /*
    if (adj_list_type == gsf::AdjListType::ordered_by_source) {
      LOG(INFO) <<"frag-" << comm_spec_.fid() << " CSR total_edge_chunk_num: " << total_edge_chunk_num;
    } else {
      LOG(INFO) <<"frag-" << comm_spec_.fid() << " CSC total_edge_chunk_num: " << total_edge_chunk_num;
    }
    */
    table_vec_t edge_chunk_tables(total_edge_chunk_num), edge_property_chunk_tables(total_edge_chunk_num);

    /*
        auto expect = gsf::ConstructAdjListArrowChunkReader(*(graph_info_.get()),
             src_label, edge_label, dst_label, adj_list_type);
        CHECK(!expect.has_error());
        auto& reader = expect.value();
        std::vector<gsf::AdjListPropertyArrowChunkReader> property_readers;
        for (const auto& pg : property_groups) {
          property_readers.emplace_back(
              edge_info, pg, adj_list_type, graph_info_->GetPrefix());
        }
    int edge_count = 0;
    for (int64_t i = 0; i < total_edge_chunk_num; ++i) {
          auto chunk_pair = getChunkPair(edge_chunk_num_vec, i);
          auto vertex_chunk_id = chunk_pair.first + vertex_chunk_begin;
          auto edge_chunk_index = chunk_pair.second;
          reader.seek_chunk_index(vertex_chunk_id, edge_chunk_index);
          auto result = reader.GetChunk();
          // if (!result.status().ok()) {
          //   LOG(ERROR) << "worker-" << comm_spec_.worker_id() << " Error: " << result.status().message();
          // }
          edge_chunk_tables[i] = result.value();
          edge_count += edge_chunk_tables[i]->num_rows();
          // property group chunks
          for (int j = 0; j < property_groups.size(); ++j) {
            auto& preader = property_readers[j];
            preader.seek_chunk_index(vertex_chunk_id, edge_chunk_index);
            auto presult = preader.GetChunk();
            // if (!result.status().ok()) {
            //   LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << result.status().message() <<" " << edge_chunk_num;
            // }
            edge_property_chunk_tables[i] = presult.value();
          }
    }
    */
    double t = vineyard::GetCurrentTime();
    std::atomic<size_t> cur(0);
    chunk = (total_edge_chunk_num + thread_num - 1) / thread_num;
    for (int64_t i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        auto expect = gsf::ConstructAdjListArrowChunkReader(*(graph_info_.get()),
             src_label, edge_label, dst_label, adj_list_type);
        CHECK(!expect.has_error());
        auto& reader = expect.value();
        std::vector<gsf::AdjListPropertyArrowChunkReader> property_readers;
        for (const auto& pg : property_groups) {
          property_readers.emplace_back(
              edge_info, pg, adj_list_type, graph_info_->GetPrefix());
        }
        while (true) {
          size_t begin = cur.fetch_add(chunk);
          if (begin >= total_edge_chunk_num) {
            break;
          }
          int64_t end = std::min(static_cast<int64_t>(begin + chunk), total_edge_chunk_num);
          int64_t iter = begin;
          while (iter != end) {
            auto chunk_pair = getChunkPair(edge_chunk_num_vec, iter);
            auto vertex_chunk_id = chunk_pair.first + vertex_chunk_begin;
            auto edge_chunk_index = chunk_pair.second;
            reader.seek_chunk_index(vertex_chunk_id, edge_chunk_index);
            auto result = reader.GetChunk();
            // if (!result.status().ok()) {
            //   LOG(ERROR) << "worker-" << comm_spec_.worker_id() << " Error: " << result.status().message();
            // }
            edge_chunk_tables[iter] = result.value();
            // property group chunks
            for (int j = 0; j < property_groups.size(); ++j) {
              auto& preader = property_readers[j];
              preader.seek_chunk_index(vertex_chunk_id, edge_chunk_index);
              auto presult = preader.GetChunk();
              // if (!result.status().ok()) {
              //   LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << result.status().message() <<" " << edge_chunk_num;
              // }
              edge_property_chunk_tables[iter] = presult.value();
            }
            ++iter;
          }
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
    /*
    if (adj_list_type == gsf::AdjListType::ordered_by_source) {
      LOG(INFO) <<"frag-" << comm_spec_.fid() << " CSR loading adj time: " << vineyard::GetCurrentTime() -t;
    } else {
      LOG(INFO) <<"frag-" << comm_spec_.fid() << " CSC loading adj time: " << vineyard::GetCurrentTime() -t;
    }
    */
    /*
    for (int64_t i = 0; i < worker_vertex_chunk_num; ++i) {
      int64_t edge_chunk_num = gsf::utils::GetEdgeChunkNumOfVertexChunk(edge_info, adj_list_type, i + vertex_chunk_begin, base_dir).value();
      if (edge_chunk_num == 0) {
        continue;
      }
      table_vec_t edge_chunk_tables_of_vertex(edge_chunk_num), edge_property_chunk_tables_of_vertex(edge_chunk_num);
      threads.resize(edge_chunk_num * 2);
      for (int64_t tid = 0; tid < edge_chunk_num * 2; ++tid) {
        threads[tid] = std::thread([&, tid]() {
          // get edge chunk table
          if (tid < edge_chunk_num) {
            auto expect = gsf::ConstructAdjListArrowChunkReader(*(graph_info_.get()),
                src_label, edge_label, dst_label, adj_list_type);
            CHECK(!expect.has_error());
            auto& reader = expect.value();
            reader.seek_chunk_index(i + vertex_chunk_begin, tid);
            auto result = reader.GetChunk();
            if (!result.status().ok()) {
              LOG(ERROR) << "worker-" << comm_spec_.worker_id() << " Error: " << result.status().message();
            }
            edge_chunk_tables_of_vertex[tid] = result.value();
          } else {
            table_vec_t property_chunk_tables;
            // get edge property chunk
            for (const auto& pg : property_groups) {
              auto maybe_reader = gsf::ConstructAdjListPropertyArrowChunkReader(*(graph_info_.get()), src_label, edge_label,
                  dst_label, pg, adj_list_type);
              CHECK(!maybe_reader.has_error());
              auto& reader = maybe_reader.value();
              reader.seek_chunk_index(i + vertex_chunk_begin, tid - edge_chunk_num);
              auto result = reader.GetChunk();
              if (!result.status().ok()) {
                LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << result.status().message() <<" " << edge_chunk_num;
              }
              property_chunk_tables.push_back(result.value());
            }
            if (!property_chunk_tables.empty()) {
              edge_property_chunk_tables_of_vertex[tid - edge_chunk_num] = ConcatenateTablesColumnWise(property_chunk_tables);
            } else {
              edge_property_chunk_tables_of_vertex[tid - edge_chunk_num] = CreateEmptyTable();
            }
          }
        });
      }
      for (auto& t : threads) {
        t.join();
      }
      edge_chunk_tables.push_back(arrow::ConcatenateTables(edge_chunk_tables_of_vertex).ValueOrDie());
      edge_property_chunk_tables.push_back(arrow::ConcatenateTables(edge_property_chunk_tables_of_vertex).ValueOrDie());
    }

    threads.clear();
    threads.resize(worker_vertex_chunk_num);
    for (int tid = 0; tid < worker_vertex_chunk_num; ++tid) {
      threads[tid] = std::thread([&, tid]() {
          // get offset array of vertex chunk tid + vertex_chunk_begin
          auto maybe_offset_reader = gsf::ConstructAdjListOffsetArrowChunkReader(*(graph_info_.get()),
              src_label, edge_label, dst_label, adj_list_type);
          CHECK(maybe_offset_reader.status().ok());
          auto& offset_reader = maybe_offset_reader.value();
          auto st = offset_reader.seek((tid + vertex_chunk_begin) * edge_info.GetSrcChunkSize());
          if (!st.ok()) {
            LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << st.message();
          }
          auto offset_result = offset_reader.GetChunk();
          if (!offset_result.status().ok()) {
            LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << offset_result.status().message();
          }
          offset_arrays[tid] = offset_result.value();
        });
    }
    for (auto& t : threads) {
      t.join();
    }
    */
    auto maybe_adj_list_table = arrow::ConcatenateTables(edge_chunk_tables);
    if (!maybe_adj_list_table.status().ok()) {
      LOG(ERROR) << "frag-" << comm_spec_.fid() << " Error: " << maybe_adj_list_table.status().message();
    }
    auto adj_list_table = maybe_adj_list_table.ValueOrDie();
    auto offset_chunked_array = arrow::ChunkedArray::Make(offset_arrays).ValueOrDie();
    auto offset_array = offset_chunked_array->chunk(0);
    std::shared_ptr<arrow::Table> property_table;
    if (property_groups.empty()) {
      property_table = CreateEmptyTable();;
    } else {
      property_table = arrow::ConcatenateTables(edge_property_chunk_tables).ValueOrDie();
    }

    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append("label", edge_label);
    metadata->Append("type", "EDGE");
    int label_id = 0;
    auto it = std::find(edge_labels_.begin(), edge_labels_.end(), edge_label);
    if (it != edge_labels_.end()) {
      label_id = it - edge_labels_.begin();
      metadata->Append("label_id", std::to_string(label_id));
      if (adj_list_type == gsf::AdjListType::ordered_by_source) {
        csr_edge_tables_[label_id].push_back(std::make_tuple(adj_list_table, offset_array, property_table->ReplaceSchemaMetadata(metadata)));
      } else if (adj_list_type == gsf::AdjListType::ordered_by_dest) {
        csc_edge_tables_[label_id].push_back(std::make_tuple(adj_list_table, offset_array, property_table->ReplaceSchemaMetadata(metadata)));
      }
    } else {
      label_id = edge_labels_.size();
      metadata->Append("label_id", std::to_string(label_id));
      if (adj_list_type == gsf::AdjListType::ordered_by_source) {
        csr_edge_tables_.push_back({std::make_tuple(adj_list_table, offset_array, property_table->ReplaceSchemaMetadata(metadata))});
      } else if (adj_list_type == gsf::AdjListType::ordered_by_dest) {
        csc_edge_tables_.push_back({std::make_tuple(adj_list_table, offset_array, property_table->ReplaceSchemaMetadata(metadata))});
      }
    }
    return {};
  }

  bl::result<void> constructEdges() {
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      for (int i = 0; i < edge_relations_[e_label].size(); ++i) {
        label_id_t src_label = edge_relations_[e_label][i].first;
        label_id_t dst_label = edge_relations_[e_label][i].second;

        auto& csr_item = csr_edge_tables_[e_label][i];
        auto adj_list_table = std::get<0>(csr_item);
        std::shared_ptr<arrow::Table> table;
        BOOST_LEAF_ASSIGN(table,
                      CSREdgesId2Gid(adj_list_table, src_label, dst_label));
        std::get<0>(csr_item) = table;

        auto& csc_item = csc_edge_tables_[e_label][i];
        adj_list_table = std::get<0>(csc_item);
        BOOST_LEAF_ASSIGN(table,
                      CSCEdgesId2Gid(adj_list_table, src_label, dst_label));
        std::get<0>(csc_item) = table;
      }
      if (edge_relations_[e_label].size() > 1) {
        table_vec_t adj_tables;
        table_vec_t property_tables;
        for (int i = 0; i < edge_relations_[e_label].size(); ++i) {
          adj_tables.push_back(std::get<0>(csr_edge_tables_[e_label][i]));
          property_tables.push_back(std::get<2>(csr_edge_tables_[e_label][i]));
        }
        post_csr_edge_tables_.push_back(std::make_tuple(arrow::ConcatenateTables(adj_tables).ValueOrDie(),
                                                        std::get<1>(csr_edge_tables_[e_label][0]),
                                                        arrow::ConcatenateTables(property_tables).ValueOrDie()));
        adj_tables.clear();
        property_tables.clear();
        for (int i = 0; i < edge_relations_[e_label].size(); ++i) {
          adj_tables.push_back(std::get<0>(csc_edge_tables_[e_label][i]));
          property_tables.push_back(std::get<2>(csc_edge_tables_[e_label][i]));
        }
        post_csc_edge_tables_.push_back(std::make_tuple(arrow::ConcatenateTables(adj_tables).ValueOrDie(),
                                                        std::get<1>(csc_edge_tables_[e_label][0]),
                                                        arrow::ConcatenateTables(property_tables).ValueOrDie()));
      } else {
        post_csr_edge_tables_.push_back(csr_edge_tables_[e_label][0]);
        post_csc_edge_tables_.push_back(csc_edge_tables_[e_label][0]);
      }
    }
    return {};
  }

  bl::result<std::shared_ptr<arrow::Table>> CSREdgesId2Gid(
      std::shared_ptr<arrow::Table> adj_list_table, label_id_t src_label,
      label_id_t dst_label) {
      std::shared_ptr<arrow::Field> src_gid_field =
        std::make_shared<arrow::Field>(
            "_graphArSrcIndex", vineyard::ConvertToArrowType<vid_t>::TypeValue());
      BOOST_LEAF_AUTO(
        src_gid_array,
        parseOidChunkedArray(src_label, adj_list_table->column(src_column), true));
      ARROW_OK_ASSIGN_OR_RAISE(
        adj_list_table,
        adj_list_table->SetColumn(src_column, src_gid_field, src_gid_array));
      std::shared_ptr<arrow::Field> dst_gid_field =
        std::make_shared<arrow::Field>(
            "_graphArDstIndex", vineyard::ConvertToArrowType<vid_t>::TypeValue());
      BOOST_LEAF_AUTO(
        dst_gid_array,
        parseOidChunkedArray(dst_label, adj_list_table->column(dst_column), false));

      // replace oid columns with gid
      ARROW_OK_ASSIGN_OR_RAISE(
        adj_list_table,
        adj_list_table->SetColumn(dst_column, dst_gid_field, dst_gid_array));
    return adj_list_table;
  }

  bl::result<std::shared_ptr<arrow::Table>> CSCEdgesId2Gid(
      std::shared_ptr<arrow::Table> adj_list_table, label_id_t src_label,
      label_id_t dst_label) {
      std::shared_ptr<arrow::Field> src_gid_field =
        std::make_shared<arrow::Field>(
            "_graphArSrcIndex", vineyard::ConvertToArrowType<vid_t>::TypeValue());
      BOOST_LEAF_AUTO(
        src_gid_array,
        parseOidChunkedArray(src_label, adj_list_table->column(src_column), false));
      ARROW_OK_ASSIGN_OR_RAISE(
        adj_list_table,
        adj_list_table->SetColumn(src_column, src_gid_field, src_gid_array));
      std::shared_ptr<arrow::Field> dst_gid_field =
        std::make_shared<arrow::Field>(
            "_graphArDstIndex", vineyard::ConvertToArrowType<vid_t>::TypeValue());
      BOOST_LEAF_AUTO(
        dst_gid_array,
        parseOidChunkedArray(dst_label, adj_list_table->column(dst_column), true));

      // replace oid columns with gid
      ARROW_OK_ASSIGN_OR_RAISE(
        adj_list_table,
        adj_list_table->SetColumn(dst_column, dst_gid_field, dst_gid_array));
    return adj_list_table;
  }

  // parse oid to global id
  bl::result<std::shared_ptr<arrow::ChunkedArray>>
  parseOidChunkedArray(label_id_t label_id,
                       std::shared_ptr<arrow::ChunkedArray> oid_arrays_in,
                       bool is_local = true) {
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

              if (is_local) {
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

  bl::result<ObjectID> constructFragment() {
    GSFArrowFragmentBuilder<oid_t, vid_t> frag_builder(client_, vm_ptr_, comm_spec_);

    PropertyGraphSchema schema;
    BOOST_LEAF_CHECK(initSchema(schema));
    frag_builder.SetPropertyGraphSchema(std::move(schema));

    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();

    std::vector<oid_t> start_ids(vertex_label_num_);
    for (label_id_t i = 0; i < vertex_label_num_; ++i) {
      start_ids[i] = vertex_chunk_begin_of_frag_[vertex_labels_[i]][comm_spec_.fid()] * vertex_chunk_sizes_[i];
    }
    LOG(INFO) << "Init builder";
    BOOST_LEAF_CHECK(frag_builder.Init(
        comm_spec_.fid(), comm_spec_.fnum(), std::move(vertex_tables_),
        std::move(post_csr_edge_tables_), std::move(post_csc_edge_tables_), std::move(start_ids), directed_, thread_num));

    LOG(INFO) << "Seal builder";
    auto frag = std::dynamic_pointer_cast<ArrowFragment<oid_t, vid_t>>(
        frag_builder.Seal(client_));
    LOG(INFO) << "Persist builder";

    VINEYARD_CHECK_OK(client_.Persist(frag->id()));
    return frag->id();
  }

  bl::result<void> initSchema(PropertyGraphSchema& schema) {
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

      for (auto& relation : edge_relations_[e_label]) {
        std::string src_label = vertex_labels_[relation.first];
        std::string dst_label = vertex_labels_[relation.second];
        entry->AddRelation(src_label, dst_label);
      }

      auto table = std::get<2>(post_csr_edge_tables_[e_label]);

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

  static std::pair<int64_t, int64_t> getChunkPair(const std::vector<gsf::IdType>& num_vec, int64_t got) {
    // binary search;
    int64_t low = 0, high = num_vec.size() - 1;
    while (low <= high) {
      int64_t mid = (low + high) / 2;
      if (num_vec[mid] <= got && num_vec[mid + 1] > got) {
        return std::make_pair(mid, got - num_vec[mid]);
      } else if (num_vec[mid] > got) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return std::make_pair(low, got - num_vec[low]);
  }

 private:
  vineyard::Client& client_;
  grape::CommSpec comm_spec_;

  std::shared_ptr<gsf::GraphInfo> graph_info_;

  bool directed_;
  std::vector<int64_t> vertex_chunk_sizes_;
  std::map<std::string, std::vector<int64_t>> vertex_chunk_begin_of_frag_;

  // basic_fragment_builder
  label_id_t vertex_label_num_;
  std::map<std::string, label_id_t> vertex_label_to_index_;
  std::vector<std::string> vertex_labels_;
  std::vector<std::shared_ptr<arrow::Table>> vertex_tables_;

  label_id_t edge_label_num_;
  std::map<std::string, label_id_t> edge_label_to_index_;
  std::vector<std::string> edge_labels_;
  std::vector<std::vector<std::pair<label_id_t, label_id_t>>> edge_relations_;
  std::vector<std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>>> csr_edge_tables_;
  std::vector<std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>>> csc_edge_tables_;

  std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>> post_csr_edge_tables_;
  std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>> post_csc_edge_tables_;

  std::shared_ptr<ArrowVertexMap<internal_oid_t, vid_t>> vm_ptr_;
};


template <typename OID_T, typename VID_T>
class GSFArrowFragmentBuilder
    : public ArrowFragmentBaseBuilder<OID_T, VID_T> {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using eid_t = property_graph_types::EID_TYPE;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using vertex_map_t = ArrowVertexMap<internal_oid_t, vid_t>;
  using nbr_unit_t = property_graph_utils::NbrUnit<vid_t, eid_t>;
  using vid_array_t = typename vineyard::ConvertToArrowType<vid_t>::ArrayType;

 public:
  explicit GSFArrowFragmentBuilder(vineyard::Client& client,
                                   std::shared_ptr<vertex_map_t> vm_ptr, const grape::CommSpec& comm_spec)
      : ArrowFragmentBaseBuilder<oid_t, vid_t>(client), vm_ptr_(vm_ptr), comm_spec_(comm_spec) {}

  vineyard::Status Build(vineyard::Client& client) override {
    ThreadGroup tg;
    {
      auto fn = [this](Client* client) {
        vineyard::ArrayBuilder<vid_t> ivnums_builder(*client, ivnums_);
        vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums_);
        vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums_);
        this->set_ivnums_(std::dynamic_pointer_cast<vineyard::Array<vid_t>>(
            ivnums_builder.Seal(*client)));
        this->set_ovnums_(std::dynamic_pointer_cast<vineyard::Array<vid_t>>(
            ovnums_builder.Seal(*client)));
        this->set_tvnums_(std::dynamic_pointer_cast<vineyard::Array<vid_t>>(
            tvnums_builder.Seal(*client)));
        return Status::OK();
      };

      tg.AddTask(fn, &client);
    }

    for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
      auto fn = [this, i](Client* client) {
        vineyard::TableBuilder vt(*client, vertex_tables_[i]);
        this->set_vertex_tables_(
            i, std::dynamic_pointer_cast<vineyard::Table>(vt.Seal(*client)));

        vineyard::NumericArrayBuilder<vid_t> ovgid_list_builder(
            *client, ovgid_lists_[i]);
        this->set_ovgid_lists_(
            i, std::dynamic_pointer_cast<vineyard::NumericArray<vid_t>>(
                   ovgid_list_builder.Seal(*client)));

        vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
            *client, std::move(ovg2l_maps_[i]));
        this->set_ovg2l_maps_(
            i, std::dynamic_pointer_cast<vineyard::Hashmap<vid_t, vid_t>>(
                   ovg2l_builder.Seal(*client)));
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }

    for (label_id_t i = 0; i < this->edge_label_num_; ++i) {
      auto fn = [this, i](Client* client) {
        vineyard::TableBuilder et(*client, edge_tables_[i]);
        this->set_edge_tables_(
            i, std::dynamic_pointer_cast<vineyard::Table>(et.Seal(*client)));
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }

    for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
      for (label_id_t j = 0; j < this->edge_label_num_; ++j) {
        auto fn = [this, i, j](Client* client) {
          if (this->directed_) {
            vineyard::FixedSizeBinaryArrayBuilder ie_builder(*client,
                                                             ie_lists_[i][j]);
            this->set_ie_lists_(i, j, ie_builder.Seal(*client));
          }
          {
            vineyard::FixedSizeBinaryArrayBuilder oe_builder(*client,
                                                             oe_lists_[i][j]);
            this->set_oe_lists_(i, j, oe_builder.Seal(*client));
          }
          if (this->directed_) {
            vineyard::NumericArrayBuilder<int64_t> ieo(*client,
                                                       ie_offsets_lists_[i][j]);
            this->set_ie_offsets_lists_(i, j, ieo.Seal(*client));
          }
          {
            vineyard::NumericArrayBuilder<int64_t> oeo(*client,
                                                       oe_offsets_lists_[i][j]);
            this->set_oe_offsets_lists_(i, j, oeo.Seal(*client));
          }
          return Status::OK();
        };
        tg.AddTask(fn, &client);
      }
    }

    tg.TakeResults();

    this->set_vm_ptr_(vm_ptr_);

    this->set_oid_type(type_name<oid_t>());
    this->set_vid_type(type_name<vid_t>());

    return vineyard::Status::OK();
  }

  boost::leaf::result<void> Init(
      fid_t fid, fid_t fnum,
      std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
      std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>>&& csr_edge_tables,
      std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>>&& csc_edge_tables,
      std::vector<int64_t>&& start_ids,
      bool directed = true, int concurrency = 1) {
    this->fid_ = fid;
    this->fnum_ = fnum;
    this->directed_ = directed;
    this->vertex_label_num_ = vertex_tables.size();
    this->edge_label_num_ = csr_edge_tables.size();
    this->start_ids_ = std::move(start_ids);

    vid_parser_.Init(this->fnum_, this->vertex_label_num_);

    BOOST_LEAF_CHECK(initVertices(std::move(vertex_tables)));
    BOOST_LEAF_CHECK(initEdges(std::move(csr_edge_tables), std::move(csc_edge_tables), concurrency));
    return {};
  }

  boost::leaf::result<void> SetPropertyGraphSchema(
      PropertyGraphSchema&& schema) {
    this->set_schema_json_(schema.ToJSON());
    return {};
  }

 private:
  // | prop_0 | prop_1 | ... |
  boost::leaf::result<void> initVertices(
      std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables) {
    assert(vertex_tables.size() ==
           static_cast<size_t>(this->vertex_label_num_));
    vertex_tables_.resize(this->vertex_label_num_);
    ivnums_.resize(this->vertex_label_num_);
    ovnums_.resize(this->vertex_label_num_);
    tvnums_.resize(this->vertex_label_num_);
    for (size_t i = 0; i < vertex_tables.size(); ++i) {
      ARROW_OK_ASSIGN_OR_RAISE(
          vertex_tables_[i],
          vertex_tables[i]->CombineChunks(arrow::default_memory_pool()));
      ivnums_[i] = vm_ptr_->GetInnerVertexSize(this->fid_, i);
    }
    return {};
  }

  // | src_id(generated) | dst_id(generated) | prop_0 | prop_1
  // | ... |
  boost::leaf::result<void> initEdges(
      std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>>&& csr_edge_tables,
      std::vector<std::tuple<std::shared_ptr<arrow::Table>, std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Table>>>&& csc_edge_tables,
      int concurrency) {
    LOG(INFO) << "edge label num: " << this->edge_label_num_ << " concurrency: " << concurrency;
    CHECK(csr_edge_tables.size() == static_cast<size_t>(this->edge_label_num_));
    CHECK(csc_edge_tables.size() == static_cast<size_t>(this->edge_label_num_));
    std::vector<std::shared_ptr<vid_array_t>> csr_edge_src, csr_edge_dst, csc_edge_src, csc_edge_dst;
    csr_edge_src.resize(this->edge_label_num_);
    csr_edge_dst.resize(this->edge_label_num_);
    csc_edge_src.resize(this->edge_label_num_);
    csc_edge_dst.resize(this->edge_label_num_);

    edge_tables_.resize(this->edge_label_num_);
    csr_offset_arrays_.resize(this->edge_label_num_);
    csc_offset_arrays_.resize(this->edge_label_num_);
    std::vector<std::vector<vid_t>> collected_ovgids(this->vertex_label_num_);

    double t = vineyard::GetCurrentTime();
    for (size_t i = 0; i < this->edge_label_num_; ++i) {
      std::vector<std::vector<vid_t>> ov_gids;
      std::shared_ptr<arrow::Table> combined_table;
      ARROW_OK_ASSIGN_OR_RAISE(
          combined_table,
          std::get<0>(csr_edge_tables[i])->CombineChunks(arrow::default_memory_pool()));
      std::get<0>(csr_edge_tables[i]) = combined_table;
      ARROW_OK_ASSIGN_OR_RAISE(
          combined_table,
          std::get<0>(csc_edge_tables[i])->CombineChunks(arrow::default_memory_pool()));
      std::get<0>(csc_edge_tables[i]) = combined_table;
      auto csr_adj_list_table = std::get<0>(csr_edge_tables[i]);
      auto csc_adj_list_table = std::get<0>(csc_edge_tables[i]);
      LOG_IF(INFO, comm_spec_.worker_id() == 0) <<
        "label-" << i << " csr table: " << csr_adj_list_table->num_rows() << " csc table: " << csc_adj_list_table->num_rows();
      // the outer vertices on exist in dst column
      // ov_gids.resize(csr_adj_list_table->column(1)->num_chunks());
      collect_outer_vertices<vid_t>(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 csr_adj_list_table->column(1)->chunk(0)),
                             this->fid_, collected_ovgids);
      // for (auto& ov_gid_vec : ov_gids) {
      //   collected_ovgids[i].insert(collected_ovgids[i].end(), ov_gid_vec.begin(),
      //                              ov_gid_vec.end());
      // }
      // ov_gids.clear();
      // ov_gids.resize(csc_adj_list_table->column(0)->num_chunks());
      collect_outer_vertices<vid_t>(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 csc_adj_list_table->column(0)->chunk(0)),
                             this->fid_, collected_ovgids);
      /*
      for (auto& ov_gid_vec : ov_gids) {
        collected_ovgids[i].insert(collected_ovgids[i].end(), ov_gid_vec.begin(),
                                   ov_gid_vec.end());
      }
      */
    }
    LOG(INFO)
      << " Collect outer vertices: " << vineyard::GetCurrentTime() - t;
    t = vineyard::GetCurrentTime();
    std::vector<vid_t> start_ids(this->vertex_label_num_);
    for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
      start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]);
    }
    generate_outer_vertices_map<vid_t>(collected_ovgids, start_ids,
                                       this->vertex_label_num_, ovg2l_maps_,
                                       ovgid_lists_);
    collected_ovgids.clear();
    for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
      ovnums_[i] = ovgid_lists_[i]->length();
      tvnums_[i] = ivnums_[i] + ovnums_[i];
    }
    LOG(INFO)
      << " Generate outer vertices map: " << vineyard::GetCurrentTime() - t;

    t = vineyard::GetCurrentTime();
    for (size_t i = 0; i < csr_edge_tables.size(); ++i) {
      auto adj_list_table = std::get<0>(csr_edge_tables[i]);
      /*
      std::shared_ptr<arrow::Table> adj_list_table;
      ARROW_OK_ASSIGN_OR_RAISE(
          adj_list_table,
          std::get<0>(csr_edge_tables[i])->CombineChunks(arrow::default_memory_pool()));
      */
      LOG_IF(INFO, this->fid_ == 0) <<
        "label-" << i << " csr table: " << adj_list_table->num_rows();
      BOOST_LEAF_CHECK(generate_local_id_list<vid_t>(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 adj_list_table->column(0)->chunk(0)),
                             this->fid_, ovg2l_maps_, concurrency, csr_edge_src[i]));
      BOOST_LEAF_CHECK(generate_local_id_list<vid_t>(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 adj_list_table->column(1)->chunk(0)),
                             this->fid_, ovg2l_maps_, concurrency, csr_edge_dst[i]));

      csr_offset_arrays_[i] = std::dynamic_pointer_cast<arrow::Int64Array>(std::get<1>(csr_edge_tables[i]));
      edge_tables_[i] = std::get<2>(csr_edge_tables[i]);
    }
    for (size_t i = 0; i < csc_edge_tables.size(); ++i) {
      std::shared_ptr<arrow::Table> adj_list_table = std::get<0>(csc_edge_tables[i]); ;
      /*
      ARROW_OK_ASSIGN_OR_RAISE(
          adj_list_table,
          std::get<0>(csc_edge_tables[i])->CombineChunks(arrow::default_memory_pool()));
      */
      LOG_IF(INFO, comm_spec_.worker_id() == 0) <<
        "label-" << i << " csc table: " << adj_list_table->num_rows();
      BOOST_LEAF_CHECK(generate_local_id_list<vid_t>(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 adj_list_table->column(0)->chunk(0)),
                             this->fid_, ovg2l_maps_, concurrency, csc_edge_src[i]));
      BOOST_LEAF_CHECK(generate_local_id_list<vid_t>(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 adj_list_table->column(1)->chunk(0)),
                             this->fid_, ovg2l_maps_, concurrency, csc_edge_dst[i]));

      csc_offset_arrays_[i] = std::dynamic_pointer_cast<arrow::Int64Array>(std::get<1>(csc_edge_tables[i]));
      // edge_tables_[i] = std::get<2>(csc_edge_tables[i]);
    }
    LOG(INFO)
      << " To local id: " << vineyard::GetCurrentTime() - t;

    t = vineyard::GetCurrentTime();
    oe_lists_.resize(this->vertex_label_num_);
    oe_offsets_lists_.resize(this->vertex_label_num_);
    if (this->directed_) {
      ie_lists_.resize(this->vertex_label_num_);
      ie_offsets_lists_.resize(this->vertex_label_num_);
    }

    for (label_id_t v_label = 0; v_label < this->vertex_label_num_; ++v_label) {
      oe_lists_[v_label].resize(this->edge_label_num_);
      oe_offsets_lists_[v_label].resize(this->edge_label_num_);
      if (this->directed_) {
        ie_lists_[v_label].resize(this->edge_label_num_);
        ie_offsets_lists_[v_label].resize(this->edge_label_num_);
      }
    }
    for (label_id_t e_label = 0; e_label < this->edge_label_num_; ++e_label) {
      std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_ie_lists(
          this->vertex_label_num_);
      std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>> sub_oe_lists(
          this->vertex_label_num_);
      std::vector<std::shared_ptr<arrow::Int64Array>> sub_ie_offset_lists(
          this->vertex_label_num_);
      std::vector<std::shared_ptr<arrow::Int64Array>> sub_oe_offset_lists(
          this->vertex_label_num_);
      generate_gsf_csr<vid_t, eid_t>(
        vid_parser_, csr_edge_src[e_label], csr_edge_dst[e_label], tvnums_,
            this->vertex_label_num_, concurrency, sub_oe_lists,
            sub_oe_offset_lists);
      generate_gsf_csr<vid_t, eid_t>(
        vid_parser_, csc_edge_dst[e_label], csc_edge_src[e_label], tvnums_,
            this->vertex_label_num_, concurrency, sub_ie_lists,
            sub_ie_offset_lists);

      for (label_id_t v_label = 0; v_label < this->vertex_label_num_;
           ++v_label) {
        if (this->directed_) {
          ie_lists_[v_label][e_label] = sub_ie_lists[v_label];
          ie_offsets_lists_[v_label][e_label] = sub_ie_offset_lists[v_label];
        }
        oe_lists_[v_label][e_label] = sub_oe_lists[v_label];
        oe_offsets_lists_[v_label][e_label] = sub_oe_offset_lists[v_label];
      }
    }
    LOG(INFO)
      << " Generate CSR: " << vineyard::GetCurrentTime() - t;
    return {};
  }

  std::vector<vid_t> ivnums_, ovnums_, tvnums_;

  std::vector<std::shared_ptr<arrow::Table>> vertex_tables_;
  std::vector<std::shared_ptr<vid_array_t>> ovgid_lists_;
  std::vector<typename ArrowFragment<OID_T, VID_T>::ovg2l_map_t> ovg2l_maps_;

  std::vector<std::shared_ptr<arrow::Table>> edge_tables_;
  std::vector<std::shared_ptr<arrow::Int64Array>> csr_offset_arrays_;
  std::vector<std::shared_ptr<arrow::Int64Array>> csc_offset_arrays_;

  std::vector<std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>>>
      ie_lists_, oe_lists_;
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>>
      ie_offsets_lists_, oe_offsets_lists_;
  std::vector<int64_t> start_ids_;

  std::shared_ptr<vertex_map_t> vm_ptr_;

  IdParser<vid_t> vid_parser_;
  grape::CommSpec comm_spec_;
};

}  // namespace vineyard

#endif  // ANALYTICAL_ENGINE_CORE_LOADER_ARROW_FRAGMENT_LOADER_H_
