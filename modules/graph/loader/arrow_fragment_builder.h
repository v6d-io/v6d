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

#include "boost/leaf/error.hpp"
#include "boost/leaf/result.hpp"
#include "grape/worker/comm_spec.h"
#include "vineyard/basic/ds/arrow_utils.h"
#include "vineyard/client/client.h"
#include "vineyard/common/util/functions.h"
#include "vineyard/graph/loader/basic_ev_fragment_builder.h"
#include "vineyard/graph/loader/fragment_loader_utils.h"
#include "vineyard/io/io/i_io_adaptor.h"
#include "vineyard/io/io/io_factory.h"
#include "vineyard/graph/utils/partitioner.h"

#include "gsf/graph_info.h"
#include "gsf/utils/trans.h"
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
  using range_t = std::pair<oid_t, oid_t>;
  static constexpr const char* LABEL_TAG = "label";
  static constexpr const char* SRC_LABEL_TAG = "src_label";
  static constexpr const char* DST_LABEL_TAG = "dst_label";

  const int id_column = 0;

  using partitioner_t = typename vineyard::SegmentedPartitioner<oid_t>;
  using table_vec_t = std::vector<std::shared_ptr<arrow::Table>>;
  using vertex_table_info_t =
      std::map<std::string, std::pair<std::vector<range_t>, std::shared_ptr<arrow::Table>>>;
  using edge_table_info_t = std::vector<vineyard::InputEdgeTable>;

 public:
  ArrowFragmentBuilder(vineyard::Client& client,
                      const grape::CommSpec& comm_spec,
                      std::shared_ptr<gsf::GraphInfo> graph_info,
                      bool directed = false)
      : client_(client),
        comm_spec_(comm_spec),
        graph_info_(graph_info),
        directed_(directed),
        generate_eid_(false) {}

  ~ArrowFragmentBuilder() = default;

  bl::result<std::pair<vertex_table_info_t, edge_table_info_t>>
  LoadVertexEdgeTables() {
    if (graph_info_) {
      std::stringstream labels;
      labels << "Loading ";
      if (graph_info_->GetAllVertexInfo().empty() && graph_info_->GetAllEdgeInfo().empty()) {
        labels << "empty graph";
      } else {
        for (auto it = graph_info_->GetAllVertexInfo().begin(); it != graph_info_->GetAllVertexInfo().end(); ++it) {
          if (it == graph_info_->GetAllVertexInfo().begin()) {
            labels << "vertex labeled ";  // prefix
          } else {
            labels << ", ";  // label seperator
          }
          labels << it->first;
        }

        if (!graph_info_->GetAllVertexInfo().empty()) {
          labels << " and ";
        }
        for (auto it = graph_info_->GetAllEdgeInfo().begin(); it != graph_info_->GetAllEdgeInfo().end(); ++it) {
          if (it == graph_info_->GetAllEdgeInfo().begin()) {
            labels << "edge labeled ";  // prefix
          } else {
            labels << ", ";  // label seperator
          }
          labels << it->first;
        }
      }
      LOG_IF(INFO, comm_spec_.worker_id() == 0)
          << "PROGRESS--GRAPH-LOADING-DESCRIPTION-" << labels.str();
    }
    BOOST_LEAF_AUTO(v_tables, LoadVertexTables());
    BOOST_LEAF_AUTO(e_tables, LoadEdgeTables());
    return std::make_pair(v_tables, e_tables);
  }

  bl::result<vertex_table_info_t> LoadVertexTables() {
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-VERTEX-0";
    vertex_table_info_t v_tables;
    if (graph_info_) {
      const auto& vertex_infos = graph_info_->GetAllVertexInfo();
      for (const auto& item : vertex_infos) {
        const auto& vertex_info = item.second;
        table_vec_t pg_tables;
        std::vector<range_t> id_ranges;
        int total_chunk_num = 0;
        std::vector<int> chunk_index;
        for (const auto& pg : vertex_info.GetPropertyGroups()) {
          auto maybe_reader = gsf::ConstructVertexPropertyArrowChunkReader(*(graph_info_.get()), vertex_info.GetLabel(), pg);
          CHECK(!maybe_reader.has_error());
          auto& reader = maybe_reader.value();
          table_vec_t chunk_tables;
          if (total_chunk_num == 0) {
            total_chunk_num = reader.GetChunkNum();
            // distribute vertex chunks
            for (int i = 0; i < total_chunk_num; ++i) {
              if (i % comm_spec_.worker_num() == comm_spec_.worker_id()) {
                chunk_index.push_back(i);
              }
            }
          }
          for (auto index : chunk_index) {
            CHECK(reader.seek(index * vertex_info.GetChunkSize()).ok());
            auto result = reader.GetChunk();
            chunk_tables.push_back(result.value());
            if (id_ranges.size() != chunk_index.size()) {
              auto id_range = reader.GetRange().value();
              id_ranges.push_back(id_range);
            }
          }
          auto pg_table = arrow::ConcatenateTables(chunk_tables);
          if (!pg_table.status().ok()) {
            LOG(ERROR) << "Error: " << pg_table.status().message();
          }
          pg_tables.push_back(pg_table.ValueOrDie());
        }
        auto v_table = ConcatenateTablesColumnWise(pg_tables);
        v_tables[vertex_info.GetLabel()] = std::make_pair(id_ranges, v_table);
      }
    }
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-VERTEX-100";
    return v_tables;
  }

  bl::result<edge_table_info_t> LoadEdgeTables() {
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-EDGE-0";
    edge_table_info_t e_tables;
    if (graph_info_) {
      // read the adj list chunk tables
      const auto& edge_infos = graph_info_->GetAllEdgeInfo();
      for (const auto& item : edge_infos) {
        const auto& edge_info = item.second;
        table_vec_t pg_tables;
        auto expect = gsf::ConstructAdjListArrowChunkReader(*(graph_info_.get()),
            edge_info.GetSrcLabel(), edge_info.GetEdgeLabel(), edge_info.GetDstLabel(),
            gsf::AdjListType::ordered_by_source);
        CHECK(!expect.has_error());
        auto& reader = expect.value();

        auto maybe_offset_reader = gsf::ConstructAdjListOffsetArrowChunkReader(*(graph_info_.get()),
            edge_info.GetSrcLabel(), edge_info.GetEdgeLabel(), edge_info.GetDstLabel(),
            gsf::AdjListType::ordered_by_source);
        CHECK(maybe_offset_reader.status().ok());

        auto& offset_reader = maybe_offset_reader.value();
        int vertex_chunk_num = reader.GetVertexChunkNum();
        // distribute vertex chunks
        std::vector<int> chunk_index;
        for (int i = 0; i < vertex_chunk_num; ++i) {
          if (i % comm_spec_.worker_num() == comm_spec_.worker_id()) {
            chunk_index.push_back(i);
          }
        }
        table_vec_t chunk_tables;
        oid_array_builder_t offset_array_builder;
        oid_t last_chunk_end = 0;
        for (auto index : chunk_index) {
          // table_vec_t adj_list_chunk_tables;
          reader.ResetChunk(index);
          do {
            auto result = reader.GetChunk();
            CHECK(result.status().ok());
            chunk_tables.push_back(result.value());
          } while (reader.next_chunk().ok());
          // auto chunk_table = arrow::ConcatenateTables(adj_list_chunk_tables);
          // CHECK(chunk_table.status().ok());
          // chunk_tables.push_back(chunk_table.ValueOrDie());

          CHECK(offset_reader.seek(index * edge_info.GetSrcChunkSize()).ok());
          auto offset_result = offset_reader.GetChunk();
          CHECK(offset_result.status().ok());
          auto offset_chunk_array = std::dynamic_pointer_cast<arrow::Int64Array>(offset_result.value());
          for (int64_t i = 0; i < offset_chunk_array->length() - 1; ++i) {
            offset_array_builder.Append(offset_chunk_array->Value(i) + last_chunk_end);
          }
          if (offset_chunk_array->length() > 0) {
            last_chunk_end += offset_chunk_array->Value(offset_chunk_array->length() - 1);
          }
        }
        auto maybe_adj_list_table = arrow::ConcatenateTables(chunk_tables);
        if (!maybe_adj_list_table.status().ok()) {
          LOG(ERROR) << "Error: " << maybe_adj_list_table.status().message();
        }
        auto adj_list_table = maybe_adj_list_table.ValueOrDie();
        auto offset_array = offset_array_builder.Finish().ValueOrDie();

        // process property chunks
        for (const auto& pg : edge_info.GetPropertyGroups(gsf::AdjListType::ordered_by_source)) {
          auto maybe_reader = gsf::ConstructAdjListPropertyArrowChunkReader(*(graph_info_.get()), edge_info.GetSrcLabel(), edge_info.GetEdgeLabel(),
            edge_info.GetDstLabel(), pg, gsf::AdjListType::ordered_by_source);
          CHECK(!maybe_reader.has_error());
          auto& reader = maybe_reader.value();
          table_vec_t pg_chunk_tables;
          for (auto index : chunk_index) {
            table_vec_t adj_list_chunk_tables;
            reader.ResetChunk(index);
            do {
              auto result = reader.GetChunk();
              CHECK(result.status().ok());
              // adj_list_chunk_tables.push_back(result.value());
              pg_chunk_tables.push_back(result.value());
            } while (reader.next_chunk().ok());
            // auto chunk_table = arrow::ConcatenateTables(adj_list_chunk_tables);
            // CHECK(chunk_table.status().ok());
            // chunk_tables.push_back(chunk_table.ValueOrDie());
          }
          auto pg_table = arrow::ConcatenateTables(chunk_tables);
          if(!pg_table.status().ok()) {
            LOG(ERROR) << "Error: " << pg_table.status().message();
          }
          pg_tables.push_back(pg_table.ValueOrDie());
        }
        auto property_table = ConcatenateTablesColumnWise(pg_tables);
        e_tables.emplace_back(edge_info.GetSrcLabel(), edge_info.GetDstLabel(), edge_info.GetEdgeLabel(), adj_list_table, offset_array, property_table);
      }
    }
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-READ-EDGE-100";
    return e_tables;
  }

  bl::result<vineyard::ObjectID> LoadFragment() {
    BOOST_LEAF_AUTO(v_e_tables, LoadVertexEdgeTables());
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-VERTEX-0";

    auto& vertex_tables_with_label = v_e_tables.first;
    auto& edge_tables_with_label = v_e_tables.second;
    std::vector<partitioner_t> partitioners(vertex_tables_with_label.size());
    std::vector<std::vector<std::vector<oid_t>>> oid_lists(vertex_tables_with_label.size());
    initPartitioners(partitioners, oid_lists, vertex_tables_with_label);

    std::shared_ptr<
        vineyard::BasicEVFragmentBuilder<OID_T, VID_T, partitioner_t>>
        basic_fragment_loader = std::make_shared<
            vineyard::BasicEVFragmentBuilder<OID_T, VID_T, partitioner_t>>(
            client_, comm_spec_, partitioners, oid_lists, directed_, false, generate_eid_);

    for (auto& pair : vertex_tables_with_label) {
      BOOST_LEAF_CHECK(
          basic_fragment_loader->AddVertexTable(pair.first, pair.second.second));
    }
    BOOST_LEAF_CHECK(basic_fragment_loader->ConstructVertices());
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-VERTEX-100";
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-EDGE-0";

    vertex_tables_with_label.clear();

    for (auto& table : edge_tables_with_label) {
      BOOST_LEAF_CHECK(basic_fragment_loader->AddEdgeTable(
          table.src_label, table.dst_label, table.edge_label, table.adj_list_table,
          table.offset_array, table.property_table));
    }
    edge_tables_with_label.clear();

    BOOST_LEAF_CHECK(basic_fragment_loader->ConstructEdges());
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-CONSTRUCT-EDGE-100";
    LOG_IF(INFO, comm_spec_.worker_id() == 0)
        << "PROGRESS--GRAPH-LOADING-SEAL-0";
    return basic_fragment_loader->ConstructFragment();
  }

 private:
  bl::result<void> initPartitioners(std::vector<partitioner_t>& partitioners,
      std::vector<std::vector<std::vector<oid_t>>>& oid_lists, const vertex_table_info_t& vertex_tables) {
    if (graph_info_ == nullptr) {
      RETURN_GS_ERROR(
          vineyard::ErrorCode::kInvalidOperationError,
          "Segmented partitioner is not supported when the v-file is "
          "not provided");
    }
      label_id_t vlabel = 0;
      for (auto& pair : vertex_tables) {
        oid_lists[vlabel].resize(comm_spec_.fnum());
        auto& range_table_pair = pair.second;
        for (auto& id_range : range_table_pair.first) {
          for (oid_t id = id_range.first; id < id_range.second; ++id) {
            oid_lists[vlabel][comm_spec_.WorkerToFrag(comm_spec_.worker_id())].push_back(id);
          }
        }
        // TODO: gather the range, not oid lists
        grape::sync_comm::AllGather(oid_lists[vlabel], comm_spec_.comm());
        partitioners[vlabel].Init(comm_spec_.fnum(), oid_lists[vlabel]);
        ++vlabel;
      }
    return {};
  }

  vineyard::Client& client_;
  grape::CommSpec comm_spec_;

  std::shared_ptr<gsf::GraphInfo> graph_info_;

  bool directed_;
  bool generate_eid_;

  std::function<void(vineyard::IIOAdaptor*)> io_deleter_ =
      [](vineyard::IIOAdaptor* adaptor) {
        VINEYARD_CHECK_OK(adaptor->Close());
        delete adaptor;
      };
};

}  // namespace vineyard

#endif  // ANALYTICAL_ENGINE_CORE_LOADER_ARROW_FRAGMENT_LOADER_H_
