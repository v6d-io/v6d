/** Copyright 2020-2023 Alibaba Group Holding Limited.
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

#ifndef MODULES_GRAPH_WRITER_ARROW_FRAGMENT_WRITER_IMPL_H_
#define MODULES_GRAPH_WRITER_ARROW_FRAGMENT_WRITER_IMPL_H_

#ifdef ENABLE_GAR

#include <algorithm>
#include <cmath>
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
#include "gar/graph_info.h"
#include "gar/writer/arrow_chunk_writer.h"
#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "common/util/functions.h"
#include "graph/loader/fragment_loader_utils.h"
#include "graph/utils/partitioner.h"
#include "graph/utils/thread_group.h"
#include "graph/writer/arrow_fragment_writer.h"
#include "io/io/i_io_adaptor.h"
#include "io/io/io_factory.h"

namespace vineyard {

template <typename FRAG_T>
ArrowFragmentWriter<FRAG_T>::ArrowFragmentWriter(
    const std::shared_ptr<fragment_t>& frag, const grape::CommSpec& comm_spec,
    const std::string& graph_info_yaml)
    : frag_(frag), comm_spec_(comm_spec) {
  // Load graph info.
  auto maybe_graph_info = GraphArchive::GraphInfo::Load(graph_info_yaml);
  if (!maybe_graph_info.status().ok()) {
    LOG(ERROR) << "Failed to load graph info from " << graph_info_yaml;
  }
  graph_info_ = std::make_shared<GraphArchive::GraphInfo>(
      std::move(maybe_graph_info.value()));
}

template <typename FRAG_T>
boost::leaf::result<void> ArrowFragmentWriter<FRAG_T>::WriteFragment() {
  BOOST_LEAF_CHECK(WriteVertices());
  BOOST_LEAF_CHECK(WriteEdges());
  return {};
}

template <typename FRAG_T>
boost::leaf::result<void> ArrowFragmentWriter<FRAG_T>::WriteVertices() {
  for (auto& item : graph_info_->GetVertexInfos()) {
    std::string label = item.first;
    BOOST_LEAF_CHECK(WriteVertex(label));
  }
  return {};
}

template <typename FRAG_T>
boost::leaf::result<void> ArrowFragmentWriter<FRAG_T>::WriteVertex(
    const std::string& label) {
  auto maybe_vertex_info = graph_info_->GetVertexInfo(label);
  if (maybe_vertex_info.has_error()) {
    RETURN_GS_ERROR(ErrorCode::kGraphArError,
                    maybe_vertex_info.status().message());
  }
  auto& vertex_info = maybe_vertex_info.value();

  auto& schema = frag_->schema();
  auto label_id = schema.GetVertexLabelId(label);
  if (label_id == -1) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "The vertex label " + label + "is not exist in fragment.");
  }

  auto vm_ptr = frag_->GetVertexMap();
  // initialize the start vertex chunk index of fragment
  GraphArchive::IdType chunk_index_begin = 0;
  for (fid_t fid = 0; fid < frag_->fid(); ++fid) {
    chunk_index_begin += static_cast<GraphArchive::IdType>(
        std::ceil(vm_ptr->GetInnerVertexSize(fid, label_id) /
                  static_cast<double>(vertex_info.GetChunkSize())));
  }
  GraphArchive::VertexPropertyWriter writer(vertex_info,
                                            graph_info_->GetPrefix());
  // write vertex data start from chunk index begin
  auto vertex_table = frag_->vertex_data_table(label_id);
  auto st = writer.WriteTable(vertex_table, chunk_index_begin);
  if (!st.ok()) {
    RETURN_GS_ERROR(ErrorCode::kGraphArError, st.message());
  }

  if (comm_spec_.worker_id() == 0) {
    // write vertex number
    auto st = writer.WriteVerticesNum(vm_ptr->GetTotalNodesNum(label_id));
    if (!st.ok()) {
      RETURN_GS_ERROR(ErrorCode::kGraphArError, st.message());
    }
  }

  return {};
}

template <typename FRAG_T>
boost::leaf::result<void> ArrowFragmentWriter<FRAG_T>::WriteEdges() {
  for (auto& item : graph_info_->GetEdgeInfos()) {
    const auto src_label = item.second.GetSrcLabel();
    const auto edge_label = item.second.GetEdgeLabel();
    const auto dst_label = item.second.GetDstLabel();
    BOOST_LEAF_CHECK(WriteEdge(src_label, edge_label, dst_label));
  }
  return {};
}

template <typename FRAG_T>
boost::leaf::result<void> ArrowFragmentWriter<FRAG_T>::WriteEdge(
    const std::string& src_label, const std::string& edge_label,
    const std::string& dst_label) {
  auto maybe_edge_info =
      graph_info_->GetEdgeInfo(src_label, edge_label, dst_label);
  if (maybe_edge_info.has_error()) {
    RETURN_GS_ERROR(ErrorCode::kGraphArError,
                    maybe_edge_info.status().message());
  }
  auto& edge_info = maybe_edge_info.value();

  // check if the edge information is valid in fragment
  bool is_valid_edge = true;
  auto& schema = frag_->schema();
  auto edge_label_id = schema.GetEdgeLabelId(edge_label);
  auto src_label_id = schema.GetVertexLabelId(src_label);
  auto dst_label_id = schema.GetVertexLabelId(dst_label);
  if (src_label_id == -1 || edge_label_id == -1 || dst_label_id == -1) {
    is_valid_edge = false;
  }
  if (is_valid_edge) {
    is_valid_edge = false;
    auto& entry =
        schema.GetEntry(edge_label_id, PropertyGraphSchema::EDGE_TYPE_NAME);
    for (auto& relation : entry.relations) {
      if (relation.first == src_label && relation.second == dst_label) {
        is_valid_edge = true;
        break;
      }
    }
  }
  if (!is_valid_edge) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "The edge " + src_label + "_" + edge_label + "_" +
                        dst_label + "is not exist in fragment.");
  }

  auto vm_ptr = frag_->GetVertexMap();
  std::vector<GraphArchive::IdType> src_vertex_chunk_begin_indices(
      frag_->fnum(), 0);
  std::vector<GraphArchive::IdType> dst_vertex_chunk_begin_indices(
      frag_->fnum(), 0);
  for (fid_t fid = 0; fid < frag_->fnum() - 1; ++fid) {
    src_vertex_chunk_begin_indices[fid + 1] =
        src_vertex_chunk_begin_indices[fid] +
        static_cast<GraphArchive::IdType>(
            std::ceil(vm_ptr->GetInnerVertexSize(fid, src_label_id) /
                      static_cast<double>(edge_info.GetSrcChunkSize())));
    dst_vertex_chunk_begin_indices[fid + 1] =
        dst_vertex_chunk_begin_indices[fid] +
        static_cast<GraphArchive::IdType>(
            std::ceil(vm_ptr->GetInnerVertexSize(fid, dst_label_id) /
                      static_cast<double>(edge_info.GetDstChunkSize())));
  }
  if (edge_info.ContainAdjList(GraphArchive::AdjListType::ordered_by_source)) {
    auto inner_vertices = frag_->InnerVertices(src_label_id);
    writeEdgeImpl(edge_info, src_label_id, edge_label_id, dst_label_id,
                  src_vertex_chunk_begin_indices,
                  dst_vertex_chunk_begin_indices, inner_vertices,
                  GraphArchive::AdjListType::ordered_by_source);
  }
  if (edge_info.ContainAdjList(
          GraphArchive::AdjListType::unordered_by_source)) {
    auto inner_vertices = frag_->InnerVertices(src_label_id);
    writeEdgeImpl(edge_info, src_label_id, edge_label_id, dst_label_id,
                  src_vertex_chunk_begin_indices,
                  dst_vertex_chunk_begin_indices, inner_vertices,
                  GraphArchive::AdjListType::unordered_by_source);
  }
  if (edge_info.ContainAdjList(GraphArchive::AdjListType::ordered_by_dest)) {
    auto inner_vertices = frag_->InnerVertices(dst_label_id);
    writeEdgeImpl(edge_info, dst_label_id, edge_label_id, src_label_id,
                  dst_vertex_chunk_begin_indices,
                  src_vertex_chunk_begin_indices, inner_vertices,
                  GraphArchive::AdjListType::ordered_by_dest);
  }
  if (edge_info.ContainAdjList(GraphArchive::AdjListType::unordered_by_dest)) {
    auto inner_vertices = frag_->InnerVertices(dst_label_id);
    writeEdgeImpl(edge_info, dst_label_id, edge_label_id, src_label_id,
                  dst_vertex_chunk_begin_indices,
                  src_vertex_chunk_begin_indices, inner_vertices,
                  GraphArchive::AdjListType::unordered_by_dest);
  }

  return {};
}

template <typename FRAG_T>
boost::leaf::result<void> ArrowFragmentWriter<FRAG_T>::writeEdgeImpl(
    const GraphArchive::EdgeInfo& edge_info, label_id_t main_label_id,
    label_id_t edge_label_id, label_id_t another_label_id,
    const std::vector<GraphArchive::IdType>& main_start_chunk_indices,
    const std::vector<GraphArchive::IdType>& another_start_chunk_indices,
    const vertex_range_t& vertices, GraphArchive::AdjListType adj_list_type) {
  vineyard::IdParser<vid_t> vid_parser;
  vid_parser.Init(frag_->fnum(), frag_->vertex_label_num());
  size_t main_vertex_chunk_size = 0, another_vertex_chunk_size = 0;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  if (adj_list_type == GraphArchive::AdjListType::ordered_by_source ||
      adj_list_type == GraphArchive::AdjListType::unordered_by_source) {
    main_vertex_chunk_size = edge_info.GetSrcChunkSize();
    another_vertex_chunk_size = edge_info.GetDstChunkSize();
    fields = {
        arrow::field(GraphArchive::GeneralParams::kSrcIndexCol, arrow::int64()),
        arrow::field(GraphArchive::GeneralParams::kDstIndexCol,
                     arrow::int64())};
  } else {
    main_vertex_chunk_size = edge_info.GetDstChunkSize();
    another_vertex_chunk_size = edge_info.GetSrcChunkSize();
    fields = {
        arrow::field(GraphArchive::GeneralParams::kDstIndexCol, arrow::int64()),
        arrow::field(GraphArchive::GeneralParams::kSrcIndexCol,
                     arrow::int64())};
  }
  auto main_start_chunk_index = main_start_chunk_indices[frag_->fid()];
  auto another_start_chunk_index = another_start_chunk_indices[frag_->fid()];

  GraphArchive::EdgeChunkWriter writer(edge_info, graph_info_->GetPrefix(),
                                       adj_list_type);
  size_t vertex_chunk_num =
      std::ceil(vertices.size() / static_cast<double>(main_vertex_chunk_size));

  // collect properties
  auto& graph_schema = frag_->schema();
  std::set<label_id_t> properties;
  for (auto& pg : edge_info.GetPropertyGroups(adj_list_type).value()) {
    for (auto& property : pg.GetProperties()) {
      label_id_t property_label_id =
          graph_schema.GetEdgePropertyId(edge_label_id, property.name);
      if (property_label_id == -1) {
        RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                        "property " + property.name +
                            " not found in edge label " +
                            std::to_string(edge_label_id));
      }
      properties.insert(property_label_id);
      fields.push_back(arrow::field(
          property.name,
          graph_schema.GetEdgePropertyType(edge_label_id, property_label_id)));
    }
  }

  label_id_t column_num = 2 + properties.size();  // src, dst and properties
  auto fn = [&](const size_t index) -> Status {
    std::vector<std::shared_ptr<arrow::Array>> column_arrays(column_num);
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders(column_num);
    InitializeArrayArrayBuilders(builders, properties, edge_label_id,
                                 graph_schema);
    std::vector<std::shared_ptr<arrow::Array>> offset_columns(1);
    arrow::Int64Builder offset_builder(arrow::default_memory_pool());
    std::shared_ptr<arrow::Schema> table_schema = arrow::schema(fields);
    auto main_builder =
        std::dynamic_pointer_cast<arrow::Int64Builder>(builders[0]);
    auto another_builder =
        std::dynamic_pointer_cast<arrow::Int64Builder>(builders[1]);

    // reset the builders
    ResetArrowArrayBuilders(builders);
    offset_builder.Reset();

    int64_t vertex_chunk_index = main_start_chunk_index + index;
    auto cur_begin = vertices.begin() + index * main_vertex_chunk_size;
    auto cur_end = std::min(cur_begin + main_vertex_chunk_size, vertices.end());
    int64_t edge_offset = 0;
    int64_t v_global_index = 0, distance = 0;
    for (auto iter = cur_begin; iter != cur_end; ++iter) {
      auto& vertex = *iter;
      auto global_index =
          vertex_chunk_index * main_vertex_chunk_size + distance;
      ++distance;
      adj_list_t edges;
      if (adj_list_type == GraphArchive::AdjListType::ordered_by_source ||
          adj_list_type == GraphArchive::AdjListType::unordered_by_source) {
        edges = frag_->GetOutgoingAdjList(vertex, edge_label_id);
      } else {
        edges = frag_->GetIncomingAdjList(vertex, edge_label_id);
      }
      int64_t edge_cnt = 0;
      for (auto& e : edges) {
        auto v = e.neighbor();
        if (vid_parser.GetLabelId(v.GetValue()) != another_label_id) {
          continue;
        }
        if (frag_->IsInnerVertex(v)) {
          v_global_index =
              another_start_chunk_index * another_vertex_chunk_size +
              vid_parser.GetOffset(v.GetValue());
        } else {
          auto gid = frag_->GetOuterVertexGid(v);
          v_global_index = another_start_chunk_indices[vid_parser.GetFid(gid)] *
                               another_vertex_chunk_size +
                           vid_parser.GetOffset(gid);
        }
        RETURN_ON_ARROW_ERROR(main_builder->Append(global_index));
        RETURN_ON_ARROW_ERROR(another_builder->Append(v_global_index));
        if (!properties.empty()) {
          appendPropertiesToArrowArrayBuilders(e, properties, edge_label_id,
                                               graph_schema, builders);
        }
        ++edge_cnt;
      }
      if (adj_list_type == GraphArchive::AdjListType::ordered_by_source ||
          adj_list_type == GraphArchive::AdjListType::ordered_by_dest) {
        RETURN_ON_ARROW_ERROR(offset_builder.Append(edge_offset));
        edge_offset += edge_cnt;
      }
    }

    // write the adj list chunks
    FinishArrowArrayBuilders(builders, column_arrays);
    auto table = arrow::Table::Make(table_schema, column_arrays);
    auto s = writer.WriteTable(table, vertex_chunk_index);
    if (!s.ok()) {
      return Status::IOError(
          "GAR error: " + std::to_string(static_cast<int>(s.code())) + ", " +
          s.message());
    }

    // write the offset chunks
    if (adj_list_type == GraphArchive::AdjListType::ordered_by_source ||
        adj_list_type == GraphArchive::AdjListType::ordered_by_dest) {
      while (distance % main_vertex_chunk_size != 0) {
        RETURN_ON_ARROW_ERROR(offset_builder.Append(edge_offset));
        ++distance;
      }
      RETURN_ON_ARROW_ERROR(
          offset_builder.Append(edge_offset));  // append the last offset
      RETURN_ON_ARROW_ERROR(offset_builder.Finish(&offset_columns[0]));
      auto offset_table = arrow::Table::Make(
          arrow::schema({arrow::field(GraphArchive::GeneralParams::kOffsetCol,
                                      arrow::int64())}),
          offset_columns);
      auto st = writer.WriteOffsetChunk(offset_table, vertex_chunk_index);
      if (!st.ok()) {
        return Status::IOError(
            "GAR error: " + std::to_string(static_cast<int>(st.code())) + ", " +
            st.message());
      }
    }

    // write edge num of vertex chunk
    auto st = writer.WriteEdgesNum(vertex_chunk_index, edge_offset);
    if (!st.ok()) {
      return Status::IOError(
          "GAR error: " + std::to_string(static_cast<int>(st.code())) + ", " +
          st.message());
    }
    if (comm_spec_.worker_id() == 0) {
      // write vertex number
      auto st = writer.WriteVerticesNum(
          frag_->GetVertexMap()->GetTotalNodesNum(main_label_id));
      if (!st.ok()) {
        return Status::IOError(
            "GAR error: " + std::to_string(static_cast<int>(st.code())) + ", " +
            st.message());
      }
    }
    return Status::OK();
  };

  ThreadGroup tg(comm_spec_);
  for (size_t chunk_index = 0; chunk_index < vertex_chunk_num; ++chunk_index) {
    tg.AddTask(fn, chunk_index);
  }
  Status status;
  for (auto const& s : tg.TakeResults()) {
    status += s;
  }
  VY_OK_OR_RAISE(status);
  return {};
}

template <typename FRAG_T>
boost::leaf::result<void>
ArrowFragmentWriter<FRAG_T>::appendPropertiesToArrowArrayBuilders(
    const nbr_t& edge, const std::set<label_id_t>& property_ids,
    const label_id_t edge_label, const PropertyGraphSchema& graph_schema,
    std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders) {
  int col_id = 2;
  for (auto& pid : property_ids) {
    auto prop_type = graph_schema.GetEdgePropertyType(edge_label, pid);
    if (arrow::boolean()->Equals(prop_type)) {
      auto builder =
          std::dynamic_pointer_cast<arrow::BooleanBuilder>(builders[col_id]);
      ARROW_OK_OR_RAISE(builder->Append(edge.template get_data<bool>(pid)));
    } else if (arrow::int32()->Equals(prop_type)) {
      auto builder =
          std::dynamic_pointer_cast<arrow::Int32Builder>(builders[col_id]);
      ARROW_OK_OR_RAISE(builder->Append(edge.template get_data<int32_t>(pid)));
    } else if (arrow::int64()->Equals(prop_type)) {
      auto builder =
          std::dynamic_pointer_cast<arrow::Int64Builder>(builders[col_id]);
      ARROW_OK_OR_RAISE(builder->Append(edge.template get_data<int64_t>(pid)));
    } else if (arrow::float32()->Equals(prop_type)) {
      auto builder =
          std::dynamic_pointer_cast<arrow::FloatBuilder>(builders[col_id]);
      ARROW_OK_OR_RAISE(builder->Append(edge.template get_data<float>(pid)));
    } else if (arrow::float64()->Equals(prop_type)) {
      auto builder =
          std::dynamic_pointer_cast<arrow::DoubleBuilder>(builders[col_id]);
      ARROW_OK_OR_RAISE(builder->Append(edge.template get_data<double>(pid)));
    } else if (arrow::utf8()->Equals(prop_type)) {
      auto builder =
          std::dynamic_pointer_cast<arrow::StringBuilder>(builders[col_id]);
      ARROW_OK_OR_RAISE(
          builder->Append(edge.template get_data<std::string>(pid)));
    } else if (arrow::large_utf8()->Equals(prop_type)) {
      auto builder = std::dynamic_pointer_cast<arrow::LargeStringBuilder>(
          builders[col_id]);
      ARROW_OK_OR_RAISE(
          builder->Append(edge.template get_data<std::string>(pid)));
    }
    ++col_id;
  }
  return {};
}

}  // namespace vineyard

#endif  // ENABLE_GAR
#endif  // MODULES_GRAPH_WRITER_ARROW_FRAGMENT_WRITER_IMPL_H_
