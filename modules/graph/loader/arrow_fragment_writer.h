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

#include <cmath>
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
#include "vineyard/graph/loader/fragment_loader_utils.h"
#include "vineyard/io/io/i_io_adaptor.h"
#include "vineyard/io/io/io_factory.h"
#include "vineyard/graph/utils/partitioner.h"

#include "gsf/graph_info.h"
#include "gsf/writer/arrow_chunk_writer.h"

namespace bl = boost::leaf;

namespace vineyard {

struct WriterConfig {
  WriterConfig() = default;
  WriterConfig(const WriterConfig& other) {
    prefix = other.prefix;
    vertex_chunk_size = other.vertex_chunk_size;
    edge_chunk_size = other.edge_chunk_size;
    vertex_chunk_file_type = other.vertex_chunk_file_type;
    edge_chunk_file_type = other.edge_chunk_file_type;
    adj_list_types = other.adj_list_types;
    yaml_output_path = other.yaml_output_path;
  }

  std::string prefix;
  int64_t vertex_chunk_size;
  int64_t edge_chunk_size;
  gsf::FileType vertex_chunk_file_type;
  gsf::FileType edge_chunk_file_type;
  std::vector<gsf::AdjListType> adj_list_types;
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
  using eid_t = property_graph_types::EID_TYPE;
  using label_id_t = vineyard::property_graph_types::LABEL_ID_TYPE;
  using vertex_range_t = grape::VertexRange<vid_t>;
  using internal_oid_t = typename vineyard::InternalType<oid_t>::type;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using vertex_map_t = vineyard::ArrowVertexMap<internal_oid_t, vid_t>;
  using oid_array_builder_t = typename vineyard::ConvertToArrowType<oid_t>::BuilderType;
  using nbr_t = property_graph_utils::Nbr<vid_t, eid_t>;

 public:
  ArrowFragmentWriter(const std::shared_ptr<ArrowFragment<OID_T, VID_T>> frag,
                      const grape::CommSpec& comm_spec,
                      std::shared_ptr<gsf::GraphInfo> graph_info,
                      bool directed = false)
     : frag_(frag),
       comm_spec_(comm_spec),
       graph_info_(graph_info),
       directed_(directed) {}

  ~ArrowFragmentWriter() = default;

  /*
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
      if (!properties.empty()) {
        gsf::PropertyGroup group = {"", config_.vertex_chunk_file_type, properties};
        CHECK(vertex_info.AddPropertyGroup(group).ok());
      }
      graph_info->AddVertex(vertex_info);
    }
    for (auto& entry : schema.edge_entries()) {
      auto& edge_label = entry.label;
      std::vector<gsf::Property> properties;
      for (auto& prop_def : entry.props_) {
        gsf::Property property = {prop_def.name, gsf::Type::arrow_data_type_to_type(prop_def.type), false};
        properties.push_back(property);
      }
      gsf::PropertyGroup group = {"", config_.edge_chunk_file_type, properties};
      for (auto& relation : entry.relations) {
        gsf::EdgeInfo edge_info(relation.first, edge_label, relation.second, config_.edge_chunk_size, config_.vertex_chunk_size, config_.vertex_chunk_size, directed_, "edge/");
        for (auto& adj_list_type : config_.adj_list_types) {
          CHECK(edge_info.AddAdjList(adj_list_type, config_.edge_chunk_file_type).ok());
          if (!properties.empty()) {
            CHECK(edge_info.AddPropertyGroup(group, adj_list_type).ok());
          }
        }
        graph_info->AddEdge(edge_info);
      }
    }
    return graph_info;
  }
  */

  bl::result<void> Write() {
    if (graph_info_ == nullptr) {
      RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError,
                      "Graph info should not be null.");
    }
    BOOST_LEAF_CHECK(WriteVertexChunks());
    BOOST_LEAF_CHECK(WriteEdgeChunks());

    return {};
  }

  /*
  void SaveGraphInfo() {
    auto maybe_fs = gsf::FileSystemFromUriOrPath(config_.yaml_output_path);
    if (!maybe_fs.status().ok()) {
      LOG(ERROR) << "Failed to get file system from uri: " << maybe_fs.status().message();
    }
    auto fs = maybe_fs.value();
    for (auto& pair : graph_info_->GetAllVertexInfo()) {
      auto& label = pair.first;
      auto& vertex_info = pair.second;
      std::string vertex_info_path = config_.yaml_output_path + "/" + label + ".vertex.yml";
      auto st = vertex_info.Save(fs, vertex_info_path);
      if (!st.ok()) {
        LOG(ERROR) << "Save VertexInfo Error: "<< st.message();
      }
      graph_info_->AddVertexInfoPath(vertex_info.GetLabel() + ".vertex.yml");
    }
    for (auto& pair : graph_info_->GetAllEdgeInfo()) {
      auto& concat_label = pair.first;
      auto& edge_info = pair.second;
      std::string edge_info_path = config_.yaml_output_path + "/" + concat_label + ".edge.yml";
      auto st = edge_info.Save(fs, edge_info_path);
      if (!st.ok()) {
        LOG(ERROR) << "Save EdgeInfo Error: "<< st.message();
      }
      graph_info_->AddEdgeInfoPath(concat_label + ".edge.yml");
    }
    graph_info_->Save(fs, config_.yaml_output_path + "/" + name_ + ".graph.yml");
  }
  */

  bl::result<void> WriteVertexChunks() {
    if (graph_info_ == nullptr) {
      RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError,
                        "Graph info should not be null.");
    }
    auto& schema = frag_->schema();
    auto vm_ptr = frag_->GetVertexMap();
    for (auto& item : graph_info_->GetAllVertexInfo()) {
      auto& label = item.first;
      LOG(INFO) << "write out vertex label: " << label;
      auto& vertex_info = item.second;
      auto label_id = schema.GetVertexLabelId(label);
      if (label_id == -1) {
        RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError,
                        "The vertex label " + label + "is not exist in graph.");
      }

      // set the start chunk index of fragment
      gsf::IdType start_chunk_index = 0;
      for (fid_t fid = 0; fid < frag_->fid(); ++fid) {
        start_chunk_index += static_cast<gsf::IdType>(std::ceil(vm_ptr->GetInnerVertexSize(fid, label_id) / static_cast<double>(vertex_info.GetChunkSize())));
      }

      auto vertex_table = frag_->vertex_data_table(label_id);
      gsf::VertexPropertyWriter writer(vertex_info, graph_info_->GetPrefix());
      auto st = writer.WriteTable(vertex_table, start_chunk_index);
      if (!st.ok()) {
        RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError, st.message());
      }
    }

    return {};
  }

  bl::result<void> WriteEdgeChunks() {
    if (graph_info_ == nullptr) {
      RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError,
                        "Graph info should not be null.");
    }
    auto& schema = frag_->schema();
    auto vm_ptr = frag_->GetVertexMap();
    for (auto& item : graph_info_->GetAllEdgeInfo()) {
      auto& edge_info = item.second;
      auto label = edge_info.GetEdgeLabel();
      auto src_label = edge_info.GetSrcLabel();
      auto dst_label = edge_info.GetDstLabel();
      auto src_label_id = schema.GetVertexLabelId(src_label);
      auto dst_label_id = schema.GetVertexLabelId(dst_label);
      std::vector<gsf::IdType> src_start_chunk_indices(frag_->fnum(), 0);
      std::vector<gsf::IdType> dst_start_chunk_indices(frag_->fnum(), 0);
      for (fid_t fid = 0; fid < frag_->fnum() - 1; ++fid) {
        src_start_chunk_indices[fid + 1] = src_start_chunk_indices[fid] +
            static_cast<gsf::IdType>(std::ceil(vm_ptr->GetInnerVertexSize(fid, src_label_id) / static_cast<double>(edge_info.GetSrcChunkSize())));
        dst_start_chunk_indices[fid + 1] = dst_start_chunk_indices[fid] +
            static_cast<gsf::IdType>(std::ceil(vm_ptr->GetInnerVertexSize(fid, dst_label_id) / static_cast<double>(edge_info.GetDstChunkSize())));
      }
      if (edge_info.ContainAdjList(gsf::AdjListType::ordered_by_source)) {
        auto inner_vertices = frag_->InnerVertices(src_label_id);
        writeCSR(edge_info, src_start_chunk_indices, dst_start_chunk_indices, inner_vertices);
      }
      if (edge_info.ContainAdjList(gsf::AdjListType::ordered_by_dest)) {
        auto inner_vertices = frag_->InnerVertices(dst_label_id);
        writeCSC(edge_info, src_start_chunk_indices, dst_start_chunk_indices, inner_vertices);
      }
    }

    return {};
  }

 private:
  void writeCSR(const gsf::EdgeInfo& edge_info,
    const std::vector<gsf::IdType>& src_start_chunk_indices,
    const std::vector<gsf::IdType>& dst_start_chunk_indices,
    const vertex_range_t& vertices) {
    vineyard::IdParser<vid_t> vid_parser;
    vid_parser.Init(frag_->fnum(), frag_->vertex_label_num());
    auto edge_label = edge_info.GetEdgeLabel();
    auto src_label = edge_info.GetSrcLabel();
    auto dst_label = edge_info.GetDstLabel();
    auto edge_chunk_size = edge_info.GetChunkSize();
    auto src_chunk_size = edge_info.GetSrcChunkSize();
    auto dst_chunk_size = edge_info.GetDstChunkSize();
    auto src_start_chunk_index = src_start_chunk_indices[frag_->fid()];
    auto dst_start_chunk_index = dst_start_chunk_indices[frag_->fid()];
    auto& schema = frag_->schema();
    gsf::EdgeChunkWriter writer(edge_info, graph_info_->GetPrefix(), gsf::AdjListType::ordered_by_source);
    int64_t v_count = 0, e_count = 0;
    int64_t v_chunk_index = src_start_chunk_index, e_chunk_index = 0;
    label_id_t e_label_id = schema.GetEdgeLabelId(edge_label);
    label_id_t prop_num = frag_->edge_property_num(e_label_id);
    std::vector<std::shared_ptr<arrow::Array>> edge_columns(prop_num + 2);
    std::vector<std::shared_ptr<arrow::Array>> offset_columns(1);
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders(prop_num + 2);
    arrow::Int64Builder offset_builder;
    construct_arrow_array_builders(builders, prop_num, e_label_id);
    auto src_builder = std::dynamic_pointer_cast<arrow::Int64Builder>(builders[0]);
    auto dst_builder = std::dynamic_pointer_cast<arrow::Int64Builder>(builders[1]);
    int64_t offset = 0;
    offset_builder.Append(offset);
    std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field(gsf::EdgeChunkWriter::src_string, arrow::int64()), arrow::field(gsf::EdgeChunkWriter::dst_string, arrow::int64())};
    for (int64_t i = 0; i < prop_num; ++i) {
      auto prop_name = schema.GetEdgePropertyName(e_label_id, i);
      auto prop_type = schema.GetEdgePropertyType(e_label_id, i);
      fields.push_back(arrow::field(prop_name, prop_type));
    }
    std::shared_ptr<arrow::Schema> table_schema = arrow::schema(fields);
    auto dst_label_id = schema.GetVertexLabelId(dst_label);
    auto edge_label_id = schema.GetEdgeLabelId(edge_label);
    for (auto& u : vertices) {
      auto u_global_id = src_start_chunk_index * src_chunk_size + vid_parser.GetOffset(u.GetValue());
      auto oe = frag_->GetOutgoingAdjList(u, edge_label_id);
      for (auto& e : oe) {
        auto v = e.neighbor();
        if (vid_parser.GetLabelId(v.GetValue()) != dst_label_id) {
          continue;
        }
        gsf::IdType v_global_id;
        if (frag_->IsInnerVertex(v)) {
          v_global_id = dst_start_chunk_index * dst_chunk_size + vid_parser.GetOffset(v.GetValue());
        } else {
          auto gid = frag_->GetOuterVertexGid(v);
          auto v_fid = vid_parser.GetFid(gid);
          auto v_offset = vid_parser.GetOffset(gid);
          v_global_id = dst_start_chunk_indices[v_fid] * dst_chunk_size + v_offset;
        }
        src_builder->Append(u_global_id);
        dst_builder->Append(v_global_id);
        append_property_to_arrow_array_builders(builders, e, prop_num, e_label_id);
        e_count++;
        if (e_count == edge_chunk_size) {
          finish_arrow_array_builders(builders, edge_columns);
          auto edge_table = arrow::Table::Make(table_schema, edge_columns);
          auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
          if (!st.ok()) {
            LOG(ERROR) <<"write edge chunk failed: " << st.message();
          }
          CHECK(st.ok());
          reset_arrow_array_builders(builders);
          e_count = 0;
          e_chunk_index++;
        }
      }
      offset += oe.size();
      offset_builder.Append(offset);
      v_count++;
      // one chunk size.
      if (v_count == src_chunk_size) {
        if (e_count != 0) {
          finish_arrow_array_builders(builders, edge_columns);
          auto edge_table = arrow::Table::Make(table_schema, edge_columns);
          auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
          if (!st.ok()) {
            LOG(ERROR) << "write chunk error: " << st.message();
          }
          CHECK(st.ok());
          reset_arrow_array_builders(builders);
          e_count = 0;
          e_chunk_index = 0;
        }
        offset_builder.Finish(&offset_columns[0]);
        auto offset_table = arrow::Table::Make(arrow::schema({arrow::field(gsf::EdgeChunkWriter::offset_string, arrow::int64())}), offset_columns);
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
        finish_arrow_array_builders(builders, edge_columns);
        auto edge_table = arrow::Table::Make(table_schema, edge_columns);
        auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
        CHECK(st.ok());
        reset_arrow_array_builders(builders);
        e_count = 0;
        e_chunk_index = 0;
      }
      offset_builder.Finish(&offset_columns[0]);
      auto offset_table = arrow::Table::Make(arrow::schema({arrow::field(gsf::EdgeChunkWriter::offset_string, arrow::int64())}), offset_columns);
      offset_builder.Reset();
      auto st = writer.WriteOffsetChunk(offset_table, v_chunk_index);
      CHECK(st.ok());
    }
  }

  void writeCSC(const gsf::EdgeInfo& edge_info,
      const std::vector<gsf::IdType>& src_start_chunk_indices,
      const std::vector<gsf::IdType>& dst_start_chunk_indices,
      const vertex_range_t& vertices) {
    vineyard::IdParser<vid_t> vid_parser;
    vid_parser.Init(frag_->fnum(), frag_->vertex_label_num());
    int64_t total_edge_count = 0;
    auto edge_label = edge_info.GetEdgeLabel();
    auto src_label = edge_info.GetSrcLabel();
    auto dst_label = edge_info.GetDstLabel();
    auto edge_chunk_size = edge_info.GetChunkSize();
    auto src_chunk_size = edge_info.GetSrcChunkSize();
    auto dst_chunk_size = edge_info.GetDstChunkSize();
    auto src_start_chunk_index = src_start_chunk_indices[frag_->fid()];
    auto dst_start_chunk_index = dst_start_chunk_indices[frag_->fid()];
    auto& schema = frag_->schema();
    gsf::EdgeChunkWriter writer(edge_info, graph_info_->GetPrefix(), gsf::AdjListType::ordered_by_dest);
    int64_t v_count = 0, e_count = 0;
    int64_t v_chunk_index = dst_start_chunk_index, e_chunk_index = 0;
    label_id_t e_label_id = schema.GetEdgeLabelId(edge_label);
    label_id_t prop_num = frag_->edge_property_num(e_label_id);
    std::vector<std::shared_ptr<arrow::Array>> edge_columns(prop_num + 2);
    std::vector<std::shared_ptr<arrow::Array>> offset_columns(1);
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders(prop_num + 2);
    arrow::Int64Builder offset_builder;
    construct_arrow_array_builders(builders, prop_num, e_label_id);
    auto src_builder = std::dynamic_pointer_cast<arrow::Int64Builder>(builders[0]);
    auto dst_builder = std::dynamic_pointer_cast<arrow::Int64Builder>(builders[1]);
    int64_t offset = 0;
    offset_builder.Append(offset);
    std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field(gsf::EdgeChunkWriter::src_string, arrow::int64()), arrow::field(gsf::EdgeChunkWriter::dst_string, arrow::int64())};
    for (int64_t i = 0; i < prop_num; ++i) {
      auto prop_name = schema.GetEdgePropertyName(e_label_id, i);
      auto prop_type = schema.GetEdgePropertyType(e_label_id, i);
      fields.push_back(arrow::field(prop_name, prop_type));
    }
    std::shared_ptr<arrow::Schema> table_schema = arrow::schema(fields);
    auto src_label_id = schema.GetVertexLabelId(src_label);
    auto edge_label_id = schema.GetEdgeLabelId(edge_label);
    for (auto& u : vertices) {
      auto u_global_id = dst_start_chunk_index * dst_chunk_size + vid_parser.GetOffset(u.GetValue());
      auto ie = frag_->GetIncomingAdjList(u, edge_label_id);
      for (auto& e : ie) {
        auto v = e.neighbor();
        if (vid_parser.GetLabelId(v.GetValue()) != src_label_id) {
          continue;
        }
        total_edge_count++;
        gsf::IdType v_global_id;
        if (frag_->IsInnerVertex(v)) {
          v_global_id = src_start_chunk_index * src_chunk_size + vid_parser.GetOffset(v.GetValue());
        } else {
          auto gid = frag_->GetOuterVertexGid(v);
          auto v_fid = vid_parser.GetFid(gid);
          auto v_offset = vid_parser.GetOffset(gid);
          v_global_id = src_start_chunk_indices[v_fid] * src_chunk_size + v_offset;
        }
        src_builder->Append(v_global_id);
        dst_builder->Append(u_global_id);
        append_property_to_arrow_array_builders(builders, e, prop_num, e_label_id);
        e_count++;
        if (e_count == edge_chunk_size) {
          finish_arrow_array_builders(builders, edge_columns);
          auto edge_table = arrow::Table::Make(table_schema, edge_columns);
          auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
          CHECK(st.ok());
          reset_arrow_array_builders(builders);
          e_count = 0;
          e_chunk_index++;
        }
      }
      offset += ie.size();
      offset_builder.Append(offset);
      v_count++;
      // one chunk size.
      if (v_count == dst_chunk_size) {
        if (e_count != 0) {
          finish_arrow_array_builders(builders, edge_columns);
          auto edge_table = arrow::Table::Make(table_schema, edge_columns);
          auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
          if (!st.ok()) {
            LOG(ERROR) << "write chunk error: " << st.message();
          }
          CHECK(st.ok());
          reset_arrow_array_builders(builders);
          e_count = 0;
          e_chunk_index = 0;
        }
        offset_builder.Finish(&offset_columns[0]);
        auto offset_table = arrow::Table::Make(arrow::schema({arrow::field(gsf::EdgeChunkWriter::offset_string, arrow::int64())}), offset_columns);
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
        finish_arrow_array_builders(builders, edge_columns);
        auto edge_table = arrow::Table::Make(table_schema, edge_columns);
        auto st = writer.WriteChunk(edge_table, v_chunk_index, e_chunk_index);
        CHECK(st.ok());
        reset_arrow_array_builders(builders);
        e_count = 0;
        e_chunk_index = 0;
      }
      offset_builder.Finish(&offset_columns[0]);
      auto offset_table = arrow::Table::Make(arrow::schema({arrow::field(gsf::EdgeChunkWriter::offset_string, arrow::int64())}), offset_columns);
      offset_builder.Reset();
      auto st = writer.WriteOffsetChunk(offset_table, v_chunk_index);
      CHECK(st.ok());
    }
  }

  void construct_arrow_array_builders(std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders, int64_t prop_num, label_id_t edge_label) {
    builders.resize(prop_num + 2);
    builders[0] = std::make_shared<arrow::Int64Builder>();
    builders[1] = std::make_shared<arrow::Int64Builder>();
    auto& schema = frag_->schema();
    for (int64_t i = 0; i < prop_num; i++) {
      auto prop_type = schema.GetEdgePropertyType(edge_label, i);
      if (arrow::boolean()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::BooleanBuilder>();
      } else if (arrow::int16()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::Int16Builder>();
      } else if (arrow::int32()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::Int32Builder>();
      } else if (arrow::int64()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::Int64Builder>();
      } else if (arrow::float32()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::FloatBuilder>();
      } else if (arrow::float64()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::DoubleBuilder>();
      } else if (arrow::utf8()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::StringBuilder>();
      } else if (arrow::large_utf8()->Equals(prop_type)) {
        builders[i+2] = std::make_shared<arrow::LargeStringBuilder>();
      }
    }
  }

  void append_property_to_arrow_array_builders(std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders, const nbr_t& edge, int64_t prop_num, label_id_t edge_label) {
    auto& schema = frag_->schema();
    for (int64_t i = 0; i < prop_num; i++) {
      auto prop_type = schema.GetEdgePropertyType(edge_label, i);
      if (arrow::boolean()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::BooleanBuilder>(builders[i+2]);
        builder->Append(edge.template get_data<bool>(i));
      } else if (arrow::int16()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::Int16Builder>(builders[i+2]);
        builder->Append(edge.template get_data<int16_t>(i));
      } else if (arrow::int32()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::Int32Builder>(builders[i+2]);
        builder->Append(edge.template get_data<int32_t>(i));
      } else if (arrow::int64()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::Int64Builder>(builders[i+2]);
        builder->Append(edge.template get_data<int64_t>(i));
      } else if (arrow::float32()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::FloatBuilder>(builders[i+2]);
        builder->Append(edge.template get_data<float>(i));
      } else if (arrow::float64()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::DoubleBuilder>(builders[i+2]);
        builder->Append(edge.template get_data<double>(i));
      } else if (arrow::utf8()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::StringBuilder>(builders[i+2]);
        builder->Append(edge.template get_data<std::string>(i));
      } else if (arrow::large_utf8()->Equals(prop_type)) {
        auto builder = std::dynamic_pointer_cast<arrow::LargeStringBuilder>(builders[i+2]);
        builder->Append(edge.template get_data<std::string>(i));
      }
    }
  }

  void reset_arrow_array_builders(std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders) {
    for (auto& builder : builders) {
      builder->Reset();
    }
  }

  void finish_arrow_array_builders(std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders, std::vector<std::shared_ptr<arrow::Array>>& columns) {
    for (int64_t i = 0; i < builders.size(); i++) {
      builders[i]->Finish(&columns[i]);
    }
  }

 private:
  std::shared_ptr<ArrowFragment<OID_T, VID_T>> frag_;
  grape::CommSpec comm_spec_;
  std::shared_ptr<gsf::GraphInfo> graph_info_;
  bool directed_;
};

}  // namespace vineyard

#endif  // ANALYTICAL_ENGINE_CORE_LOADER_ARROW_FRAGMENT_LOADER_H_
