/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

#ifndef MODULES_GRAPH_FRAGMENT_GAR_FRAGMENT_BUILDER_IMPL_H_
#define MODULES_GRAPH_FRAGMENT_GAR_FRAGMENT_BUILDER_IMPL_H_

#ifdef ENABLE_GAR

#include <memory>
#include <utility>
#include <vector>

#include "basic/utils.h"
#include "graph/fragment/gar_fragment_builder.h"
#include "graph/fragment/property_graph_utils_impl.h"

namespace vineyard {

// generate csr for
template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_csr_for_reused_edge_label(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> dst_chunks,
    std::vector<VID_T> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;

  int64_t num_chunks = src_chunks.size();
  std::vector<std::vector<int>> degree(vertex_label_num);
  std::vector<int64_t> actual_edge_num(vertex_label_num, 0);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    degree[v_label].resize(tvnums[v_label], 0);
  }

  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&degree, &parser, &src_chunks](int64_t chunk_index) {
        auto src_array = src_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();

        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          grape::atomic_add(
              degree[parser.GetLabelId(src_id)][parser.GetOffset(src_id)], 1);
        }
      },
      concurrency);

  std::vector<std::vector<int64_t>> offsets(vertex_label_num);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    auto tvnum = tvnums[v_label];
    auto& offset_vec = offsets[v_label];
    auto& degree_vec = degree[v_label];

    offset_vec.resize(tvnum + 1);
    offset_vec[0] = 0;

    if (tvnum > 0) {
      parallel_prefix_sum(degree_vec.data(), &offset_vec[1], tvnum,
                          concurrency);
    }
    // build the arrow's offset array
    std::shared_ptr<arrow::Buffer> offsets_buffer;
    ARROW_OK_ASSIGN_OR_RAISE(
        offsets_buffer, arrow::AllocateBuffer((tvnum + 1) * sizeof(int64_t),
                                              arrow::default_memory_pool()));
    memcpy(offsets_buffer->mutable_data(), offset_vec.data(),
           (tvnum + 1) * sizeof(int64_t));
    edge_offsets[v_label] = std::make_shared<arrow::Int64Array>(
        arrow::int64(), tvnum + 1, offsets_buffer, nullptr, 0, 0);
    actual_edge_num[v_label] = offset_vec[tvnum];
  }
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    edges[v_label] = std::make_shared<PodArrayBuilder<nbr_unit_t>>(
        client, actual_edge_num[v_label]);
  }

  std::vector<int64_t> chunk_offsets(num_chunks + 1, 0);
  for (int64_t i = 0; i < num_chunks; ++i) {
    chunk_offsets[i + 1] = chunk_offsets[i] + src_chunks[i]->length();
  }

  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&src_chunks, &dst_chunks, &parser, &edges, &offsets,
       &chunk_offsets](int64_t chunk_index) {
        auto src_array = src_chunks[chunk_index];
        auto dst_array = dst_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();
        const VID_T* dst_list_ptr = dst_array->raw_values();
        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          int v_label = parser.GetLabelId(src_id);
          int64_t v_offset = parser.GetOffset(src_id);
          int64_t adj_offset =
              __sync_fetch_and_add(&offsets[v_label][v_offset], 1);
          nbr_unit_t* ptr = edges[v_label]->MutablePointer(adj_offset);
          ptr->vid = dst_list_ptr[i];
          ptr->eid = static_cast<EID_T>(chunk_offsets[chunk_index] + i);
        }
        src_chunks[chunk_index].reset();
        dst_chunks[chunk_index].reset();
      },
      concurrency);
  return {};
}

template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_csr(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>>&& src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>>&& dst_chunks,
    std::shared_ptr<arrow::Int64Array>& offset_array, std::vector<VID_T> tvnums,
    int vertex_label_num, property_graph_types::LABEL_ID_TYPE vertex_label,
    int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets,
    int64_t start_offset) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;

  int64_t num_chunks = src_chunks.size();
  int64_t edge_num = offset_array->Value(offset_array->length() - 1);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    auto tvnum = tvnums[v_label];
    // build the arrow's offset array
    std::shared_ptr<arrow::Buffer> offsets_buffer;
    int64_t buffer_length = static_cast<int64_t>(tvnum + 1);
    ARROW_OK_ASSIGN_OR_RAISE(
        offsets_buffer, arrow::AllocateBuffer(buffer_length * sizeof(int64_t)));
    if (v_label == vertex_label) {
      int64_t array_length = offset_array->length();
      VINEYARD_ASSERT(array_length <= buffer_length,
                      "Invalid offset array: the offset array length is larger "
                      "than the tvnum + 1.");
      memcpy(offsets_buffer->mutable_data(),
             reinterpret_cast<const uint8_t*>(offset_array->raw_values()),
             array_length * sizeof(int64_t));
      // we do not store the edge offset of outer vertices, so fill edge_num
      // to the outer vertices offset
      std::fill_n(reinterpret_cast<int64_t*>(offsets_buffer->mutable_data() +
                                             array_length * sizeof(int64_t)),
                  buffer_length - array_length, edge_num);
      edges[v_label] =
          std::make_shared<PodArrayBuilder<nbr_unit_t>>(client, edge_num);
    } else {
      std::fill_n(reinterpret_cast<int64_t*>(offsets_buffer->mutable_data()),
                  buffer_length, 0);
      edges[v_label] = std::make_shared<PodArrayBuilder<nbr_unit_t>>(client, 0);
    }
    edge_offsets[v_label] = std::make_shared<arrow::Int64Array>(
        arrow::int64(), buffer_length, offsets_buffer, nullptr, 0, 0);
  }

  std::vector<int64_t> chunk_offsets(num_chunks + 1, 0);
  for (int64_t i = 0; i < num_chunks; ++i) {
    chunk_offsets[i + 1] = chunk_offsets[i] + src_chunks[i]->length();
  }

  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&src_chunks, &dst_chunks, &edges, &vertex_label, &chunk_offsets,
       &start_offset](int64_t chunk_index) {
        auto dst_array = dst_chunks[chunk_index];
        const VID_T* dst_list_ptr = dst_array->raw_values();
        for (int64_t i = 0; i < dst_array->length(); ++i) {
          int64_t adj_offset = chunk_offsets[chunk_index] + i;
          nbr_unit_t* ptr = edges[vertex_label]->MutablePointer(adj_offset);
          ptr->vid = dst_list_ptr[i];
          ptr->eid = static_cast<EID_T>(adj_offset + start_offset);
        }
        src_chunks[chunk_index].reset();
        dst_chunks[chunk_index].reset();
      },
      concurrency);

  return {};
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
vineyard::Status GARFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::Build(
    vineyard::Client& client) {
  ThreadGroup tg;
  {
    auto fn = [this](Client* client) -> Status {
      vineyard::ArrayBuilder<vid_t> ivnums_builder(*client, ivnums_);
      vineyard::ArrayBuilder<vid_t> ovnums_builder(*client, ovnums_);
      vineyard::ArrayBuilder<vid_t> tvnums_builder(*client, tvnums_);
      std::shared_ptr<Object> object;
      RETURN_ON_ERROR(ivnums_builder.Seal(*client, object));
      this->set_ivnums_(object);
      RETURN_ON_ERROR(ovnums_builder.Seal(*client, object));
      this->set_ovnums_(object);
      RETURN_ON_ERROR(tvnums_builder.Seal(*client, object));
      this->set_tvnums_(object);
      return Status::OK();
    };

    tg.AddTask(fn, &client);
  }

  Base::vertex_tables_.resize(this->vertex_label_num_);
  Base::ovgid_lists_.resize(this->vertex_label_num_);
  Base::ovg2l_maps_.resize(this->vertex_label_num_);

  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    auto fn = [this, i](Client* client) -> Status {
      this->set_vertex_tables_(i, std::make_shared<vineyard::TableBuilder>(
                                      *client, std::move(vertex_tables_[i]),
                                      true /* merge chunks */));

      vineyard::NumericArrayBuilder<vid_t> ovgid_list_builder(
          *client, std::move(ovgid_lists_[i]));
      std::shared_ptr<Object> ovgid_object;
      RETURN_ON_ERROR(ovgid_list_builder.Seal(*client, ovgid_object));
      this->set_ovgid_lists_(i, ovgid_object);

      vineyard::HashmapBuilder<vid_t, vid_t> ovg2l_builder(
          *client, std::move(ovg2l_maps_[i]));
      std::shared_ptr<Object> ovg2l_map_object;
      RETURN_ON_ERROR(ovg2l_builder.Seal(*client, ovg2l_map_object));
      this->set_ovg2l_maps_(i, ovg2l_map_object);
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  Base::edge_tables_.resize(this->edge_label_num_);
  for (label_id_t i = 0; i < this->edge_label_num_; ++i) {
    auto fn = [this, i](Client* client) -> Status {
      this->set_edge_tables_(
          i, std::make_shared<vineyard::TableBuilder>(
                 *client, std::move(edge_tables_[i]), true /* merge chunks */));
      return Status::OK();
    };
    tg.AddTask(fn, &client);
  }

  if (this->directed_) {
    Base::ie_lists_.resize(this->vertex_label_num_);
    Base::ie_offsets_lists_.resize(this->vertex_label_num_);
  }
  Base::oe_lists_.resize(this->vertex_label_num_);
  Base::oe_offsets_lists_.resize(this->vertex_label_num_);
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    if (this->directed_) {
      Base::ie_lists_[i].resize(this->edge_label_num_);
      Base::ie_offsets_lists_[i].resize(this->edge_label_num_);
    }
    Base::oe_lists_[i].resize(this->edge_label_num_);
    Base::oe_offsets_lists_[i].resize(this->edge_label_num_);
    for (label_id_t j = 0; j < this->edge_label_num_; ++j) {
      auto fn = [this, i, j](Client* client) -> Status {
        std::shared_ptr<Object> object;
        if (this->directed_) {
          RETURN_ON_ERROR(ie_lists_[i][j]->Seal(*client, object));
          this->set_ie_lists_(i, j, object);
          vineyard::NumericArrayBuilder<int64_t> ieo(
              *client, std::move(ie_offsets_lists_[i][j]));
          RETURN_ON_ERROR(ieo.Seal(*client, object));
          this->set_ie_offsets_lists_(i, j, object);
        }
        {
          RETURN_ON_ERROR(oe_lists_[i][j]->Seal(*client, object));
          this->set_oe_lists_(i, j, object);
          vineyard::NumericArrayBuilder<int64_t> oeo(
              *client, std::move(oe_offsets_lists_[i][j]));
          RETURN_ON_ERROR(oeo.Seal(*client, object));
          this->set_oe_offsets_lists_(i, j, object);
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

  VLOG(100) << "[frag-" << this->fid_
            << "] RSS after building into vineyard: " << get_rss_pretty()
            << ", peak: " << get_peak_rss_pretty();

  return Status::OK();
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<void> GARFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::Init(
    fid_t fid, fid_t fnum,
    std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
    std::vector<EdgeTableInfo>&& csr_edge_tables,
    std::vector<EdgeTableInfo>&& csc_edge_tables, bool directed,
    int concurrency) {
  this->fid_ = fid;
  this->fnum_ = fnum;
  this->directed_ = directed;
  this->vertex_label_num_ = vertex_tables.size();
  this->edge_label_num_ = csr_edge_tables.size();

  vid_parser_.Init(this->fnum_, this->vertex_label_num_);

  BOOST_LEAF_CHECK(initVertices(std::move(vertex_tables)));
  VLOG(100) << "[frag-" << this->fid_
            << "] RSS after constructing vertices: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  BOOST_LEAF_CHECK(initEdges(std::move(csr_edge_tables),
                             std::move(csc_edge_tables), concurrency));
  VLOG(100) << "[frag-" << this->fid_
            << "] RSS after constructing edges: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  return {};
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::SetPropertyGraphSchema(
    PropertyGraphSchema&& schema) {
  this->set_schema_json_(schema.ToJSON());
  return {};
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::initVertices(
    std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables) {
  vertex_tables_ = vertex_tables;
  ivnums_.resize(this->vertex_label_num_);
  ovnums_.resize(this->vertex_label_num_);
  tvnums_.resize(this->vertex_label_num_);
  for (size_t i = 0; i < vertex_tables_.size(); ++i) {
    ivnums_[i] = vm_ptr_->GetInnerVertexSize(this->fid_, i);
  }
  return {};
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentBuilder<OID_T, VID_T, VERTEX_MAP_T>::initEdges(
    std::vector<EdgeTableInfo>&& csr_edge_tables,
    std::vector<EdgeTableInfo>&& csc_edge_tables, int concurrency) {
  CHECK(csr_edge_tables.size() == static_cast<size_t>(this->edge_label_num_));
  CHECK(csc_edge_tables.size() == static_cast<size_t>(this->edge_label_num_));
  std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs(this->edge_label_num_),
      dsts(this->edge_label_num_);
  edge_tables_.resize(this->edge_label_num_);
  for (label_id_t label = 0; label < this->edge_label_num_; ++label) {
    srcs[label] = csc_edge_tables[label].adj_list_table->column(0);
    if (this->directed_) {
      dsts[label] = csr_edge_tables[label].adj_list_table->column(1);
      auto concat =
          arrow::ConcatenateTables({csr_edge_tables[label].property_table,
                                    csc_edge_tables[label].property_table});
      if (!concat.status().ok()) {
        RETURN_GS_ERROR(vineyard::ErrorCode::kArrowError,
                        concat.status().message());
      }
      edge_tables_[label] = std::move(concat).ValueOrDie();
    } else {
      edge_tables_[label] = csr_edge_tables[label].property_table;
    }
  }

  std::vector<vid_t> start_ids(this->vertex_label_num_);
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    start_ids[i] = vid_parser_.GenerateId(0, i, ivnums_[i]);
  }
  generate_outer_vertices_map<vid_t>(vid_parser_, this->fid_,
                                     this->vertex_label_num_, srcs, dsts,
                                     start_ids, ovg2l_maps_, ovgid_lists_);
  for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
    ovnums_[i] = ovgid_lists_[i]->length();
    tvnums_[i] = ivnums_[i] + ovnums_[i];
  }

  std::vector<std::vector<std::shared_ptr<vid_array_t>>> csr_edge_src,
      csr_edge_dst;
  std::vector<std::vector<std::shared_ptr<vid_array_t>>> csc_edge_src,
      csc_edge_dst;
  csr_edge_src.resize(this->edge_label_num_);
  csr_edge_dst.resize(this->edge_label_num_);
  csc_edge_src.resize(this->edge_label_num_);
  csc_edge_dst.resize(this->edge_label_num_);
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::MemoryPool> recorder;
  if (VLOG_IS_ON(1000)) {
    recorder = std::make_shared<arrow::LoggingMemoryPool>(pool);
    pool = recorder.get();
  }
  for (size_t i = 0; i < csr_edge_tables.size(); ++i) {
    auto csr_adj_list_table = std::move(csr_edge_tables[i].adj_list_table);
    generate_local_id_list(vid_parser_,
                           std::move(csr_adj_list_table->column(0)), this->fid_,
                           ovg2l_maps_, concurrency, csr_edge_src[i], pool);
    generate_local_id_list(vid_parser_,
                           std::move(csr_adj_list_table->column(1)), this->fid_,
                           ovg2l_maps_, concurrency, csr_edge_dst[i], pool);
    if (this->directed_) {
      auto csc_adj_list_table = std::move(csc_edge_tables[i].adj_list_table);
      generate_local_id_list(
          vid_parser_, std::move(csc_adj_list_table->column(0)), this->fid_,
          ovg2l_maps_, concurrency, csc_edge_src[i], pool);
      generate_local_id_list(
          vid_parser_, std::move(csc_adj_list_table->column(1)), this->fid_,
          ovg2l_maps_, concurrency, csc_edge_dst[i], pool);
    }
  }
  VLOG(100) << "[frag-" << this->fid_
            << "] RSS after generating local_id_list: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

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
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_ie_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>> sub_oe_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_ie_offset_lists(
        this->vertex_label_num_);
    std::vector<std::shared_ptr<arrow::Int64Array>> sub_oe_offset_lists(
        this->vertex_label_num_);
    if (csr_edge_tables[e_label].flag) {
      // reuse the offset array
      generate_csr_for_reused_edge_label<vid_t, eid_t>(
          client_, vid_parser_, std::move(csr_edge_src[e_label]),
          std::move(csr_edge_dst[e_label]), tvnums_, this->vertex_label_num_,
          concurrency, sub_oe_lists, sub_oe_offset_lists);
    } else {
      generate_csr<vid_t, eid_t>(
          client_, vid_parser_, std::move(csr_edge_src[e_label]),
          std::move(csr_edge_dst[e_label]), csr_edge_tables[e_label].offsets,
          tvnums_, this->vertex_label_num_,
          csr_edge_tables[e_label].vertex_label_id, concurrency, sub_oe_lists,
          sub_oe_offset_lists, 0);
    }
    if (this->directed_) {
      if (csc_edge_tables[e_label].flag) {
        // reuse the offset array
        generate_csr_for_reused_edge_label<vid_t, eid_t>(
            client_, vid_parser_, std::move(csc_edge_dst[e_label]),
            std::move(csc_edge_src[e_label]), tvnums_, this->vertex_label_num_,
            concurrency, sub_ie_lists, sub_ie_offset_lists);
      } else {
        generate_csr<vid_t, eid_t>(
            client_, vid_parser_, std::move(csc_edge_dst[e_label]),
            std::move(csc_edge_src[e_label]), csc_edge_tables[e_label].offsets,
            tvnums_, this->vertex_label_num_,
            csc_edge_tables[e_label].vertex_label_id, concurrency, sub_ie_lists,
            sub_ie_offset_lists,
            csr_edge_tables[e_label].property_table->num_rows());
      }
    }

    VLOG(100) << "[frag-" << this->fid_ << "] RSS after building the CSR ..."
              << get_rss_pretty() << ", peak = " << get_peak_rss_pretty();

    for (label_id_t v_label = 0; v_label < this->vertex_label_num_; ++v_label) {
      if (this->directed_) {
        ie_lists_[v_label][e_label] = sub_ie_lists[v_label];
        ie_offsets_lists_[v_label][e_label] = sub_ie_offset_lists[v_label];
      }
      oe_lists_[v_label][e_label] = sub_oe_lists[v_label];
      oe_offsets_lists_[v_label][e_label] = sub_oe_offset_lists[v_label];
    }
  }
  return {};
}

}  // namespace vineyard

#endif  // ENABLE_GAR
#endif  // MODULES_GRAPH_FRAGMENT_GAR_FRAGMENT_BUILDER_IMPL_H_
