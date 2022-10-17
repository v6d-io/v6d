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

#ifndef VINEYARD_LOADER_ARROW_FRAGMENT_WRITER_H_
#define VINEYARD_LOADER_ARROW_FRAGMENT_WRITER_H_

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
#include "gsf/writer/arrow_chunk_writer.h"


namespace vineyard {

struct WriterConfig {
  WriterConfig() = default;
  WriterConfig(const WriterConfig& other) {
    prefix = other.prefix;
    vertex_chunk_size = other.vertex_chunk_size;
    edge_chunk_size = other.edge_chunk_size;
    vertex_chunk_file_type = other.vertex_chunk_file_type;
    edge_chunk_file_type = other.edge_chunk_file_type;
    adj_list_type = other.adj_list_type;
    yaml_output_path = other.yaml_output_path;
  }

  std::string prefix;
  int vertex_chunk_size;
  int edge_chunk_size;
  gsf::FileType vertex_chunk_file_type;
  gsf::FileType edge_chunk_file_type;
  gsf::AdjListType adj_list_type;
  std::string yaml_output_path;
};

/**
 * @brief This builder can load a ArrowFragment from graph store format data source
 * @tparam OID_T OID type
 * @tparam VID_T VID type
 */
template <typename OID_T = gsf::IdType,
          typename VID_T = vineyard::property_graph_types::VID_TYPE>
class ArrowFragmentWriter {
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

 public:
  ArrowFragmentWriter(const std::shared_ptr<ArrowFragment<OID_T, VID_T>> frag,
                      const grape::CommSpec& comm_spec,
                      const std::string& graph_name,
                      const WriterConfig& config,
                      bool directed = false)
     : frag_(frag),
       comm_spec_(comm_spec),
       name_(graph_name),
       config_(config),
       directed_(directed),
       graph_info_(ConstructGraphInfo()) {}

  ~ArrowFragmentWriter() = default;

  std::shared_ptr<gsf::GraphInfo> ConstructGraphInfo() {
    auto graph_info = std::make_shared<gsf::GraphInfo>(name_, config_.prefix);
    auto& schema = frag_->schema();
    label_id_t prop_id;
    for (auto& entry : schema.vertex_entries()) {
      gsf::VertexInfo vertex_info(entry.label, config_.vertex_chunk_size, "vertex/");
      std::vector<gsf::Property> properties;
      for (auto& prop_def : entry.props_) {
        gsf::Property property = {prop_def.name, gsf::Type::arrow_data_type_to_type(prop_def.type), false};
        properties.push_back(property);
      }
      gsf::PropertyGroup group = {"properties", config_.vertex_chunk_file_type, properties};
      CHECK(vertex_info.AddPropertyGroup(group).ok());
      graph_info->AddVertex(vertex_info);
      LOG(INFO) <<"vertex info: " << vertex_info.Dump().value();
    }
    for (auto& entry : schema.edge_entries()) {
      auto& edge_label = entry.label;
      std::vector<gsf::Property> properties;
      for (auto& prop_def : entry.props_) {
        gsf::Property property = {prop_def.name, gsf::Type::arrow_data_type_to_type(prop_def.type), false};
        properties.push_back(property);
      }
      gsf::PropertyGroup group = {"properties", config_.edge_chunk_file_type, properties};
      for (auto& relation : entry.relations) {
        gsf::EdgeInfo edge_info(relation.first, edge_label, relation.second, config_.edge_chunk_size, config_.vertex_chunk_size, config_.vertex_chunk_size, directed_, "edge/");
        CHECK(edge_info.AddAdjList(config_.adj_list_type, config_.edge_chunk_file_type).ok());
        CHECK(edge_info.AddPropertyGroup(group, config_.adj_list_type).ok());
        graph_info->AddEdge(edge_info);
      }
    }
    return graph_info;
  }

  void Write() {
    if (graph_info_ == nullptr) {
      LOG(ERROR) << "Graph info is null";
      return;
    }
    WriteVertexChunks();
    WriteEdgeChunks();
  }

  void WriteVertexChunks() {
    if (graph_info_ == nullptr) {
      LOG(ERROR) << "Graph info is null";
      return;
    }
    auto& schema = frag_->schema();
    auto vm_ptr = frag_->GetVertexMap();
    for (label_id_t label = 0; label < frag_->vertex_label_num(); ++label) {
      // set the start chunk index
      gsf::IdType start_chunk_index = 0;
      for (fid_t fid = 0; fid < frag_->fid(); ++fid) {
        if (vm_ptr->GetInnerVertexSize(fid) % config_.vertex_chunk_size == 0 ) {
          start_chunk_index += vm_ptr->GetInnerVertexSize(fid) / config_.vertex_chunk_size;
        } else {
          start_chunk_index += vm_ptr->GetInnerVertexSize(fid) / config_.vertex_chunk_size + 1;
        }
      }

      auto vertex_table = frag_->vertex_data_table(label);
      LOG(INFO) << "frag-" << frag_->fid() <<" got vertex " <<vertex_table->num_rows() << " start id=" <<start_chunk_index;
      auto vertex_info = graph_info_->GetVertexInfo(schema.GetVertexLabelName(label));
      gsf::VertexPropertyWriter writer(vertex_info.value(), graph_info_->GetPrefix());
      auto st = writer.WriteTable(vertex_table, start_chunk_index);
      CHECK(st.ok());
    }
  }

  void WriteEdgeChunks() {
    if (graph_info_ == nullptr) {
      LOG(ERROR) << "Graph info is null";
      return;
    }

    auto& schema = frag_->schema();
    auto vm_ptr = frag_->GetVertexMap();
    vineyard::IdParser<vid_t> vid_parser;
    vid_parser.Init(frag_->fnum(), frag_->vertex_label_num());
    for (label_id_t label = 0; label < frag_->vertex_label_num(); ++label) {
      std::vector<gsf::IdType> start_chunk_indices(frag_->fnum(), 0);
      for (fid_t fid = 0; fid < frag_->fnum() - 1; ++fid) {
        if (vm_ptr->GetInnerVertexSize(fid) % config_.vertex_chunk_size == 0 ) {
          start_chunk_indices[fid + 1] += vm_ptr->GetInnerVertexSize(fid) / config_.vertex_chunk_size;
        } else {
          start_chunk_indices[fid + 1] += vm_ptr->GetInnerVertexSize(fid) / config_.vertex_chunk_size + 1;
        }
      }
      auto& start_chunk_index = start_chunk_indices[frag_->fid()];
      auto inner_vertices = frag_->InnerVertices(label);
      for (auto& pair : graph_info_->GetAllEdgeInfo()) {
        auto& edge_info = pair.second;
        auto edge_label = edge_info.GetEdgeLabel();
        LOG(INFO) <<"edge_info: " << edge_info.Dump().value();
        auto src_label = edge_info.GetSrcLabel();
        auto dst_label = edge_info.GetDstLabel();
        if (config_.adj_list_type == gsf::AdjListType::ordered_by_source && src_label == schema.GetVertexLabelName(label)) {
          gsf::EdgeChunkWriter writer(edge_info, graph_info_->GetPrefix(), config_.adj_list_type);
          int v_count = 0, e_count = 0;
          int v_chunk_index = start_chunk_index, e_chunk_index = 0;
          std::vector<std::shared_ptr<arrow::Array>> src_dst_columns(3);
          std::vector<std::shared_ptr<arrow::Array>> offset_columns(1);
          arrow::Int64Builder src_builder, dst_builder, offset_builder;
          arrow::Int64Builder prop_builder;
          int64_t offset = 0;
          offset_builder.Append(offset);
          for (auto& u : inner_vertices) {
            auto u_global_id = start_chunk_index * config_.vertex_chunk_size + vid_parser.GetOffset(u.GetValue());
            auto oe = frag_->GetOutgoingAdjList(u, schema.GetEdgeLabelId(edge_label));
            for (auto& e : oe) {
              auto v = e.neighbor();
              gsf::IdType v_global_id;
              if (frag_->IsInnerVertex(v)) {
                v_global_id = start_chunk_index * config_.vertex_chunk_size + vid_parser.GetOffset(v.GetValue());
              } else {
                auto gid = frag_->GetOuterVertexGid(v);
                auto v_fid = vid_parser.GetFid(gid);
                auto v_offset = vid_parser.GetOffset(gid);
                v_global_id = start_chunk_indices[v_fid] * config_.vertex_chunk_size + v_offset;
              }
              src_builder.Append(u_global_id);
              dst_builder.Append(v_global_id);
              prop_builder.Append(e.get_int(0));
              e_count++;
              if (e_count == config_.edge_chunk_size) {
                src_builder.Finish(&src_dst_columns[0]);
                dst_builder.Finish(&src_dst_columns[1]);
                prop_builder.Finish(&src_dst_columns[2]);
                // auto edge_table = arrow::Table::Make(arrow::schema({arrow::field("src", arrow::int64()), arrow::field("dst", arrow::int64())}), src_dst_columns);
                auto edge_table = arrow::Table::Make(arrow::schema({arrow::field("src", arrow::int64()), arrow::field("dst", arrow::int64()), arrow::field("weight", arrow::int64())}), src_dst_columns);
                auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
                CHECK(st.ok());
                src_builder.Reset();
                dst_builder.Reset();
                prop_builder.Reset();
                e_count = 0;
                e_chunk_index++;
              }
            }
            offset += oe.size();
            offset_builder.Append(offset);
            v_count++;
            // one chunk size.
            if (v_count == config_.vertex_chunk_size) {
              if (e_count != 0) {
                src_builder.Finish(&src_dst_columns[0]);
                dst_builder.Finish(&src_dst_columns[1]);
                prop_builder.Finish(&src_dst_columns[2]);
                // auto edge_table = arrow::Table::Make(arrow::schema({arrow::field("src", arrow::int64()), arrow::field("dst", arrow::int64())}), src_dst_columns);
                // auto st = writer.WriteAdjListChunk(edge_table, v_chunk_index, e_chunk_index);
                auto edge_table = arrow::Table::Make(arrow::schema({arrow::field("src", arrow::int64()), arrow::field("dst", arrow::int64()), arrow::field("weight", arrow::int64())}), src_dst_columns);
                auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
                if (!st.ok()) {
                  LOG(ERROR) << "write chunk error: " << st.message();
                }
                CHECK(st.ok());
                src_builder.Reset();
                dst_builder.Reset();
                prop_builder.Reset();
                e_count = 0;
                e_chunk_index = 0;
              }
              offset_builder.Finish(&offset_columns[0]);
              auto offset_table = arrow::Table::Make(arrow::schema({arrow::field("offset", arrow::int64())}), offset_columns);
              auto st = writer.WriteOffsetChunk(offset_table, v_chunk_index);
              CHECK(st.ok());
              offset_builder.Reset();
              v_chunk_index++;
              e_chunk_index = 0;
              v_count = 0;
            }
          }
          // maybe the last chunk not align to vertex chunk size
          if (v_count != 0) {
              if (e_count != 0) {
                src_builder.Finish(&src_dst_columns[0]);
                dst_builder.Finish(&src_dst_columns[1]);
                prop_builder.Finish(&src_dst_columns[2]);
                // auto edge_table = arrow::Table::Make(arrow::schema({arrow::field("src", arrow::int64()), arrow::field("dst", arrow::int64())}), src_dst_columns);
                // auto st = writer.WriteAdjListChunk(edge_table, v_chunk_index, e_chunk_index);
                auto edge_table = arrow::Table::Make(arrow::schema({arrow::field("src", arrow::int64()), arrow::field("dst", arrow::int64()), arrow::field("weight", arrow::int64())}), src_dst_columns);
                auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
                CHECK(st.ok());
                src_builder.Reset();
                dst_builder.Reset();
                prop_builder.Reset();
                e_count = 0;
                e_chunk_index = 0;
              }
              offset_builder.Finish(&offset_columns[0]);
              auto offset_table = arrow::Table::Make(arrow::schema({arrow::field("offset", arrow::int64())}), offset_columns);
              offset_builder.Reset();
              auto st = writer.WriteOffsetChunk(offset_table, v_chunk_index);
              CHECK(st.ok());
          }
        }
      }
    }
  }

 private:
  std::shared_ptr<ArrowFragment<OID_T, VID_T>> frag_;
  grape::CommSpec comm_spec_;
  std::string name_;
  WriterConfig config_;
  bool directed_;
  std::shared_ptr<gsf::GraphInfo> graph_info_;
};

}  // namespace vineyard

#endif  // ANALYTICAL_ENGINE_CORE_LOADER_ARROW_FRAGMENT_LOADER_H_
