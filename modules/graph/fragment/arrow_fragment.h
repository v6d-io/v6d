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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/fragment/fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/utils/vertex_array.h"

#include "basic/ds/arrow.h"
#include "basic/ds/arrow_utils.h"
#include "common/util/functions.h"
#include "common/util/typename.h"

#include "graph/fragment/arrow_fragment.vineyard.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"
#include "graph/utils/context_protocols.h"
#include "graph/utils/error.h"
#include "graph/utils/thread_group.h"
#include "graph/vertex_map/arrow_vertex_map.h"
#include "graph/vertex_map/arrow_vertex_map_builder.h"

namespace vineyard {

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
class ArrowFragmentBaseBuilder;

template <typename OID_T, typename VID_T,
          typename VERTEX_MAP_T =
              ArrowVertexMap<typename InternalType<OID_T>::type, VID_T>>
class BasicArrowFragmentBuilder
    : public ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T> {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using eid_t = property_graph_types::EID_TYPE;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using vertex_map_t = VERTEX_MAP_T;
  using nbr_unit_t = property_graph_utils::NbrUnit<vid_t, eid_t>;
  using vid_array_t = typename vineyard::ConvertToArrowType<vid_t>::ArrayType;

 public:
  explicit BasicArrowFragmentBuilder(vineyard::Client& client,
                                     std::shared_ptr<vertex_map_t> vm_ptr)
      : ArrowFragmentBaseBuilder<oid_t, vid_t, vertex_map_t>(client),
        vm_ptr_(vm_ptr) {}

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

    this->vertex_tables_.resize(this->vertex_label_num_);
    this->ovgid_lists_.resize(this->vertex_label_num_);
    this->ovg2l_maps_.resize(this->vertex_label_num_);
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

    this->edge_tables_.resize(this->edge_label_num_);
    for (label_id_t i = 0; i < this->edge_label_num_; ++i) {
      auto fn = [this, i](Client* client) {
        vineyard::TableBuilder et(*client, edge_tables_[i]);
        this->set_edge_tables_(
            i, std::dynamic_pointer_cast<vineyard::Table>(et.Seal(*client)));
        return Status::OK();
      };
      tg.AddTask(fn, &client);
    }

    if (this->directed_) {
      this->ie_lists_.resize(this->vertex_label_num_);
      this->ie_offsets_lists_.resize(this->vertex_label_num_);
    }
    this->oe_lists_.resize(this->vertex_label_num_);
    this->oe_offsets_lists_.resize(this->vertex_label_num_);
    for (label_id_t i = 0; i < this->vertex_label_num_; ++i) {
      if (this->directed_) {
        this->ie_lists_[i].resize(this->edge_label_num_);
        this->ie_offsets_lists_[i].resize(this->edge_label_num_);
      }
      this->oe_lists_[i].resize(this->edge_label_num_);
      this->oe_offsets_lists_[i].resize(this->edge_label_num_);
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

    return Status::OK();
  }

  boost::leaf::result<void> Init(
      fid_t fid, fid_t fnum,
      std::vector<std::shared_ptr<arrow::Table>>&& vertex_tables,
      std::vector<std::shared_ptr<arrow::Table>>&& edge_tables,
      bool directed = true, int concurrency = 1) {
    this->fid_ = fid;
    this->fnum_ = fnum;
    this->directed_ = directed;
    this->is_multigraph_ = false;
    this->vertex_label_num_ = vertex_tables.size();
    this->edge_label_num_ = edge_tables.size();

    vid_parser_.Init(this->fnum_, this->vertex_label_num_);

    BOOST_LEAF_CHECK(initVertices(std::move(vertex_tables)));
    BOOST_LEAF_CHECK(initEdges(std::move(edge_tables), concurrency));
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
      std::vector<std::shared_ptr<arrow::Table>>&& edge_tables,
      int concurrency) {
    assert(edge_tables.size() == static_cast<size_t>(this->edge_label_num_));
    std::vector<std::shared_ptr<vid_array_t>> edge_src, edge_dst;
    edge_src.resize(this->edge_label_num_);
    edge_dst.resize(this->edge_label_num_);

    edge_tables_.resize(this->edge_label_num_);
    std::vector<std::vector<vid_t>> collected_ovgids(this->vertex_label_num_);

    for (size_t i = 0; i < edge_tables.size(); ++i) {
      std::shared_ptr<arrow::Table> combined_table;
      ARROW_OK_ASSIGN_OR_RAISE(
          combined_table,
          edge_tables[i]->CombineChunks(arrow::default_memory_pool()));
      edge_tables[i].swap(combined_table);

      collect_outer_vertices(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 edge_tables[i]->column(0)->chunk(0)),
                             this->fid_, collected_ovgids);
      collect_outer_vertices(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 edge_tables[i]->column(1)->chunk(0)),
                             this->fid_, collected_ovgids);
    }
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

    for (size_t i = 0; i < edge_tables.size(); ++i) {
      generate_local_id_list(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 edge_tables[i]->column(0)->chunk(0)),
                             this->fid_, ovg2l_maps_, concurrency, edge_src[i]);
      generate_local_id_list(vid_parser_,
                             std::dynamic_pointer_cast<vid_array_t>(
                                 edge_tables[i]->column(1)->chunk(0)),
                             this->fid_, ovg2l_maps_, concurrency, edge_dst[i]);

      std::shared_ptr<arrow::Table> tmp_table0;
      ARROW_OK_ASSIGN_OR_RAISE(tmp_table0, edge_tables[i]->RemoveColumn(0));
      ARROW_OK_ASSIGN_OR_RAISE(edge_tables_[i], tmp_table0->RemoveColumn(0));

      edge_tables[i].reset();
    }

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
      if (this->directed_) {
        generate_directed_csr<vid_t, eid_t>(
            vid_parser_, edge_src[e_label], edge_dst[e_label], tvnums_,
            this->vertex_label_num_, concurrency, sub_oe_lists,
            sub_oe_offset_lists, this->is_multigraph_);
        generate_directed_csr<vid_t, eid_t>(
            vid_parser_, edge_dst[e_label], edge_src[e_label], tvnums_,
            this->vertex_label_num_, concurrency, sub_ie_lists,
            sub_ie_offset_lists, this->is_multigraph_);
      } else {
        generate_undirected_csr<vid_t, eid_t>(
            vid_parser_, edge_src[e_label], edge_dst[e_label], tvnums_,
            this->vertex_label_num_, concurrency, sub_oe_lists,
            sub_oe_offset_lists, this->is_multigraph_);
      }

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
    return {};
  }

  std::vector<vid_t> ivnums_, ovnums_, tvnums_;

  std::vector<std::shared_ptr<arrow::Table>> vertex_tables_;
  std::vector<std::shared_ptr<vid_array_t>> ovgid_lists_;
  std::vector<typename ArrowFragment<OID_T, VID_T>::ovg2l_map_t> ovg2l_maps_;

  std::vector<std::shared_ptr<arrow::Table>> edge_tables_;

  std::vector<std::vector<std::shared_ptr<arrow::FixedSizeBinaryArray>>>
      ie_lists_, oe_lists_;
  std::vector<std::vector<std::shared_ptr<arrow::Int64Array>>>
      ie_offsets_lists_, oe_offsets_lists_;

  std::shared_ptr<vertex_map_t> vm_ptr_;

  IdParser<vid_t> vid_parser_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_H_
