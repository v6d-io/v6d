#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H

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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_MOD_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_MOD_H_

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/fragment/fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/utils/vertex_array.h"

#include "client/ds/core_types.h"
#include "client/ds/object_meta.h"

#include "basic/ds/arrow.h"
#include "basic/ds/arrow_utils.h"
#include "common/util/typename.h"

#include "graph/fragment/arrow_fragment_base.h"
#include "graph/fragment/fragment_traits.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/vertex_map/arrow_local_vertex_map.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace gs {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T>
class ArrowProjectedFragment;

}  // namespace gs

namespace vineyard {

inline std::string generate_name_with_suffix(
    const std::string& prefix, property_graph_types::LABEL_ID_TYPE label) {
  return prefix + "_" + std::to_string(label);
}

inline std::string generate_name_with_suffix(
    const std::string& prefix, property_graph_types::LABEL_ID_TYPE v_label,
    property_graph_types::LABEL_ID_TYPE e_label) {
  return prefix + "_" + std::to_string(v_label) + "_" + std::to_string(e_label);
}

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
class ArrowFragmentBaseBuilder;

template <typename OID_T, typename VID_T,
          typename VERTEX_MAP_T =
              ArrowVertexMap<typename InternalType<OID_T>::type, VID_T>>
class __attribute__((annotate("vineyard"))) ArrowFragment
    : public ArrowFragmentBase,
      public vineyard::BareRegistered<
          ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>> {
 
  public:
    static std::unique_ptr<Object> Create() __attribute__((used)) {
        return std::static_pointer_cast<Object>(
            std::unique_ptr<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>>{
                new ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>()});
    }


  public:
    void Construct(const ObjectMeta& meta) override {
        std::string __type_name = type_name<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>>();
        VINEYARD_ASSERT(
            meta.GetTypeName() == __type_name,
            "Expect typename '" + __type_name + "', but got '" + meta.GetTypeName() + "'");
        this->meta_ = meta;
        this->id_ = meta.GetId();

        meta.GetKeyValue("fid_", this->fid_);
        meta.GetKeyValue("fnum_", this->fnum_);
        meta.GetKeyValue("directed_", this->directed_);
        meta.GetKeyValue("is_multigraph_", this->is_multigraph_);
        meta.GetKeyValue("vertex_label_num_", this->vertex_label_num_);
        meta.GetKeyValue("edge_label_num_", this->edge_label_num_);
        meta.GetKeyValue("oid_type", this->oid_type);
        meta.GetKeyValue("vid_type", this->vid_type);
        this->ivnums_.Construct(meta.GetMemberMeta("ivnums_"));
        this->ovnums_.Construct(meta.GetMemberMeta("ovnums_"));
        this->tvnums_.Construct(meta.GetMemberMeta("tvnums_"));
        for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__vertex_tables_-size"); ++__idx) {
            this->vertex_tables_.emplace_back(std::dynamic_pointer_cast<Table>(
                    meta.GetMember("__vertex_tables_-" + std::to_string(__idx))));
        }
        for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__ovgid_lists_-size"); ++__idx) {
            this->ovgid_lists_.emplace_back(std::dynamic_pointer_cast<ArrowFragment::vid_vineyard_array_t>(
                    meta.GetMember("__ovgid_lists_-" + std::to_string(__idx))));
        }
        for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__ovg2l_maps_-size"); ++__idx) {
            this->ovg2l_maps_.emplace_back(std::dynamic_pointer_cast<Hashmap<vid_t, vid_t>>(
                    meta.GetMember("__ovg2l_maps_-" + std::to_string(__idx))));
        }
        for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__edge_tables_-size"); ++__idx) {
            this->edge_tables_.emplace_back(std::dynamic_pointer_cast<Table>(
                    meta.GetMember("__edge_tables_-" + std::to_string(__idx))));
        }
        this->ie_lists_.resize(meta.GetKeyValue<size_t>("__ie_lists_-size"));
        for (size_t __idx = 0; __idx < this->ie_lists_.size(); ++__idx) {
            for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
                    "__ie_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
                this->ie_lists_[__idx].emplace_back(std::dynamic_pointer_cast<FixedSizeBinaryArray>(
                    meta.GetMember("__ie_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
            }
        }
        this->oe_lists_.resize(meta.GetKeyValue<size_t>("__oe_lists_-size"));
        for (size_t __idx = 0; __idx < this->oe_lists_.size(); ++__idx) {
            for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
                    "__oe_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
                this->oe_lists_[__idx].emplace_back(std::dynamic_pointer_cast<FixedSizeBinaryArray>(
                    meta.GetMember("__oe_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
            }
        }
        this->ie_offsets_lists_.resize(meta.GetKeyValue<size_t>("__ie_offsets_lists_-size"));
        for (size_t __idx = 0; __idx < this->ie_offsets_lists_.size(); ++__idx) {
            for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
                    "__ie_offsets_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
                this->ie_offsets_lists_[__idx].emplace_back(std::dynamic_pointer_cast<Int64Array>(
                    meta.GetMember("__ie_offsets_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
            }
        }
        this->oe_offsets_lists_.resize(meta.GetKeyValue<size_t>("__oe_offsets_lists_-size"));
        for (size_t __idx = 0; __idx < this->oe_offsets_lists_.size(); ++__idx) {
            for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
                    "__oe_offsets_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
                this->oe_offsets_lists_[__idx].emplace_back(std::dynamic_pointer_cast<Int64Array>(
                    meta.GetMember("__oe_offsets_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
            }
        }
        this->vm_ptr_ = std::dynamic_pointer_cast<ArrowFragment::vertex_map_t>(meta.GetMember("vm_ptr_"));
        meta.GetKeyValue("schema_json_", this->schema_json_);

        
        if (meta.IsLocal()) {
            this->PostConstruct(meta);
        }
    }

 private:
public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using eid_t = property_graph_types::EID_TYPE;
  using prop_id_t = property_graph_types::PROP_ID_TYPE;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using vertex_range_t = grape::VertexRange<vid_t>;
  using inner_vertices_t = vertex_range_t;
  using outer_vertices_t = vertex_range_t;
  using vertices_t = vertex_range_t;
  using nbr_t = property_graph_utils::Nbr<vid_t, eid_t>;
  using nbr_unit_t = property_graph_utils::NbrUnit<vid_t, eid_t>;
  using adj_list_t = property_graph_utils::AdjList<vid_t, eid_t>;
  using raw_adj_list_t = property_graph_utils::RawAdjList<vid_t, eid_t>;
  using vertex_map_t = VERTEX_MAP_T;
  using vertex_t = grape::Vertex<vid_t>;

  using ovg2l_map_t =
      ska::flat_hash_map<vid_t, vid_t, typename Hashmap<vid_t, vid_t>::KeyHash>;

  using vid_array_t = ArrowArrayType<vid_t>;
  using vid_vineyard_array_t = ArrowVineyardArrayType<vid_t>;
  using vid_vineyard_builder_t = ArrowVineyardBuilderType<vid_t>;
  using eid_array_t = ArrowArrayType<eid_t>;
  using eid_vineyard_array_t = ArrowVineyardArrayType<eid_t>;
  using eid_vineyard_builder_t = ArrowVineyardBuilderType<eid_t>;

  using vid_builder_t = ArrowBuilderType<vid_t>;

  template <typename DATA_T>
  using vertex_array_t = grape::VertexArray<vertices_t, DATA_T>;

  template <typename DATA_T>
  using inner_vertex_array_t = grape::VertexArray<inner_vertices_t, DATA_T>;

  template <typename DATA_T>
  using outer_vertex_array_t = grape::VertexArray<outer_vertices_t, DATA_T>;

  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

 public:
  ~ArrowFragment() = default;

  vineyard::ObjectID vertex_map_id() const override { return vm_ptr_->id(); }

  bool directed() const override { 
    // grin structure directed
    return directed_; 
  }

  bool is_multigraph() const override {
    // grin structure multigraph ++++
    return is_multigraph_; 
  }

  const std::string vid_typename() const override { 
    // grin structure vid type
    return vid_type; 
  }

  const std::string oid_typename() const override { return oid_type; }


  fid_t fid() const { 
    // grin partition get_partition
    return fid_; 
  }

  fid_t fnum() const { 
    // grin partition get_partition_number
    return fnum_; 
  }

  label_id_t vertex_label(const vertex_t& v) const {
    // grin label get vertex_label by vertex
    return vid_parser_.GetLabelId(v.GetValue());
  }

  int64_t vertex_offset(const vertex_t& v) const {
    // to remove ----
    return vid_parser_.GetOffset(v.GetValue());
  }

  label_id_t vertex_label_num() const { 
    // grin label get vertex label list size
    return schema_.vertex_label_num(); 
  }

  label_id_t edge_label_num() const { 
    // grin label get edge label list size
    return schema_.edge_label_num(); 
  }

  prop_id_t vertex_property_num(label_id_t label) const {
    // grin pg get_all_vertex_properties_from_label 
    std::string type = "VERTEX";
    return static_cast<prop_id_t>(schema_.GetEntry(label, type).property_num());
  }

  std::shared_ptr<arrow::DataType> vertex_property_type(label_id_t label,
                                                        prop_id_t prop) const {
    // grin property get_property_type
    return vertex_tables_[label]->schema()->field(prop)->type();
  }

  prop_id_t edge_property_num(label_id_t label) const {
    // grin pg get_all_edge_properties_from_label
    std::string type = "EDGE";
    return static_cast<prop_id_t>(schema_.GetEntry(label, type).property_num());
  }

  std::shared_ptr<arrow::DataType> edge_property_type(label_id_t label,
                                                      prop_id_t prop) const {
    // grin property get_property_type
    return edge_tables_[label]->schema()->field(prop)->type();
  }

  std::shared_ptr<arrow::Table> vertex_data_table(label_id_t i) const {
    // grin pg get_all_rows ++++???
    return vertex_tables_[i]->GetTable();
  }

  std::shared_ptr<arrow::Table> edge_data_table(label_id_t i) const {
    // grin pg get_all_rows ++++???
    return edge_tables_[i]->GetTable();
  }

  template <typename DATA_T>
  property_graph_utils::EdgeDataColumn<DATA_T, nbr_unit_t> edge_data_column(
      label_id_t label, prop_id_t prop) const {
    // get rid of this method and EdgeDataColumn structure
    // this structure actually serves to get a specific property of an edge
    // and it can be replaced by grin property get_edge_row
    if (edge_tables_[label]->num_rows() == 0) {
      return property_graph_utils::EdgeDataColumn<DATA_T, nbr_unit_t>();
    } else {
      // the finalized etables are guaranteed to have been concatenated
      return property_graph_utils::EdgeDataColumn<DATA_T, nbr_unit_t>(
          edge_tables_[label]->column(prop)->chunk(0));
    }
  }

  template <typename DATA_T>
  property_graph_utils::VertexDataColumn<DATA_T, vid_t> vertex_data_column(
      label_id_t label, prop_id_t prop) const {
    // Ditto. it can be replaced by grin property get_vertex_row && get_property_value_from_row
    if (vertex_tables_[label]->num_rows() == 0) {
      return property_graph_utils::VertexDataColumn<DATA_T, vid_t>(
          InnerVertices(label));
    } else {
      // the finalized vtables are guaranteed to have been concatenated
      return property_graph_utils::VertexDataColumn<DATA_T, vid_t>(
          InnerVertices(label), vertex_tables_[label]->column(prop)->chunk(0));
    }
  }

  vertex_range_t Vertices(label_id_t label_id) const {
    // continuous_vid_traits
    return vertex_range_t(
        vid_parser_.GenerateId(0, label_id, 0),
        vid_parser_.GenerateId(0, label_id, tvnums_[label_id]));
  }

  vertex_range_t InnerVertices(label_id_t label_id) const {
    // continuous_vid_traits
    return vertex_range_t(
        vid_parser_.GenerateId(0, label_id, 0),
        vid_parser_.GenerateId(0, label_id, ivnums_[label_id]));
  }

  vertex_range_t OuterVertices(label_id_t label_id) const {
    // continuous_vid_traits
    return vertex_range_t(
        vid_parser_.GenerateId(0, label_id, ivnums_[label_id]),
        vid_parser_.GenerateId(0, label_id, tvnums_[label_id]));
  }

  vertex_range_t InnerVerticesSlice(label_id_t label_id, vid_t start, vid_t end)
      const {
    // continuous_vid_traits
    CHECK(start <= end && start <= ivnums_[label_id]);
    if (end <= ivnums_[label_id]) {
      return vertex_range_t(vid_parser_.GenerateId(0, label_id, start),
                            vid_parser_.GenerateId(0, label_id, end));
    } else {
      return vertex_range_t(
          vid_parser_.GenerateId(0, label_id, start),
          vid_parser_.GenerateId(0, label_id, ivnums_[label_id]));
    }
  }

  inline vid_t GetVerticesNum(label_id_t label_id) const {
    // grin label get_vertex_num_by_label
    return tvnums_[label_id];
  }

  bool GetVertex(label_id_t label, const oid_t& oid, vertex_t& v) const {
    vid_t gid;
    if (vm_ptr_->GetGid(label, internal_oid_t(oid), gid)) {
      return (vid_parser_.GetFid(gid) == fid_) ? InnerVertexGid2Vertex(gid, v)
                                               : OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  oid_t GetId(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexId(v) : GetOuterVertexId(v);
  }

  internal_oid_t GetInternalId(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexInternalId(v)
                            : GetOuterVertexInternalId(v);
  }

  fid_t GetFragId(const vertex_t& u) const {
    return IsInnerVertex(u) ? fid_ : vid_parser_.GetFid(GetOuterVertexGid(u));
  }

  size_t GetTotalNodesNum() const { return vm_ptr_->GetTotalNodesNum(); }
  size_t GetTotalVerticesNum() const { return vm_ptr_->GetTotalNodesNum(); }
  size_t GetTotalVerticesNum(label_id_t label) const {
    return vm_ptr_->GetTotalNodesNum(label);
  }

  size_t GetEdgeNum() const { return directed_ ? oenum_ + ienum_ : oenum_; }

  size_t GetInEdgeNum() const { return ienum_; }

  size_t GetOutEdgeNum() const { return oenum_; }

  template <typename T>
  T GetData(const vertex_t& v, prop_id_t prop_id) const {
    // grin get vertex row && get_property_value_from_row
    return property_graph_utils::ValueGetter<T>::Value(
        vertex_tables_columns_[vid_parser_.GetLabelId(v.GetValue())][prop_id],
        vid_parser_.GetOffset(v.GetValue()));
  }

  bool HasChild(const vertex_t& v, label_id_t e_label) const {
    return GetLocalOutDegree(v, e_label) != 0;
  }

  bool HasParent(const vertex_t& v, label_id_t e_label) const {
    return GetLocalInDegree(v, e_label) != 0;
  }

  int GetLocalOutDegree(const vertex_t& v, label_id_t e_label) const {
    return GetOutgoingAdjList(v, e_label).Size();
  }

  int GetLocalInDegree(const vertex_t& v, label_id_t e_label) const {
    return GetIncomingAdjList(v, e_label).Size();
  }

  // FIXME: grape message buffer compatibility
  bool Gid2Vertex(const vid_t& gid, vertex_t& v) const {
    return (vid_parser_.GetFid(gid) == fid_) ? InnerVertexGid2Vertex(gid, v)
                                             : OuterVertexGid2Vertex(gid, v);
  }

  vid_t Vertex2Gid(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

  inline vid_t GetInnerVerticesNum(label_id_t label_id) const {
    return ivnums_[label_id];
  }

  inline vid_t GetOuterVerticesNum(label_id_t label_id) const {
    return ovnums_[label_id];
  }

  inline bool IsInnerVertex(const vertex_t& v) const {
    return vid_parser_.GetOffset(v.GetValue()) <
           static_cast<int64_t>(ivnums_[vid_parser_.GetLabelId(v.GetValue())]);
  }

  inline bool IsOuterVertex(const vertex_t& v) const {
    vid_t offset = vid_parser_.GetOffset(v.GetValue());
    label_id_t label = vid_parser_.GetLabelId(v.GetValue());
    return offset < tvnums_[label] && offset >= ivnums_[label];
  }

  bool GetInnerVertex(label_id_t label, const oid_t& oid, vertex_t& v) const {
    vid_t gid;
    if (vm_ptr_->GetGid(label, internal_oid_t(oid), gid)) {
      if (vid_parser_.GetFid(gid) == fid_) {
        v.SetValue(vid_parser_.GetLid(gid));
        return true;
      }
    }
    return false;
  }

  bool GetOuterVertex(label_id_t label, const oid_t& oid, vertex_t& v) const {
    vid_t gid;
    if (vm_ptr_->GetGid(label, internal_oid_t(oid), gid)) {
      return OuterVertexGid2Vertex(gid, v);
    }
    return false;
  }

  inline oid_t GetInnerVertexId(const vertex_t& v) const {
    return oid_t(GetInnerVertexInternalId(v));
  }

  inline internal_oid_t GetInnerVertexInternalId(const vertex_t& v) const {
    internal_oid_t internal_oid;
    vid_t gid =
        vid_parser_.GenerateId(fid_, vid_parser_.GetLabelId(v.GetValue()),
                               vid_parser_.GetOffset(v.GetValue()));
    CHECK(vm_ptr_->GetOid(gid, internal_oid));
    return internal_oid;
  }

  inline oid_t GetOuterVertexId(const vertex_t& v) const {
    return oid_t(GetOuterVertexInternalId(v));
  }

  inline internal_oid_t GetOuterVertexInternalId(const vertex_t& v) const {
    vid_t gid = GetOuterVertexGid(v);
    internal_oid_t internal_oid;
    CHECK(vm_ptr_->GetOid(gid, internal_oid));
    return internal_oid;
  }

  inline oid_t Gid2Oid(const vid_t& gid) const {
    internal_oid_t internal_oid;
    CHECK(vm_ptr_->GetOid(gid, internal_oid));
    return oid_t(internal_oid);
  }

  inline bool Oid2Gid(label_id_t label, const oid_t& oid, vid_t& gid) const {
    return vm_ptr_->GetGid(label, internal_oid_t(oid), gid);
  }

  inline bool Oid2Gid(label_id_t label, const oid_t& oid, vertex_t& v) const {
    vid_t gid;
    if (vm_ptr_->GetGid(label, internal_oid_t(oid), gid)) {
      v.SetValue(gid);
      return true;
    }
    return false;
  }

  inline bool InnerVertexGid2Vertex(const vid_t& gid, vertex_t& v) const {
    v.SetValue(vid_parser_.GetLid(gid));
    return true;
  }

  inline bool OuterVertexGid2Vertex(const vid_t& gid, vertex_t& v) const {
    auto map = ovg2l_maps_ptr_[vid_parser_.GetLabelId(gid)];
    auto iter = map->find(gid);
    if (iter != map->end()) {
      v.SetValue(iter->second);
      return true;
    } else {
      return false;
    }
  }

  inline vid_t GetOuterVertexGid(const vertex_t& v) const {
    label_id_t v_label = vid_parser_.GetLabelId(v.GetValue());
    return ovgid_lists_ptr_[v_label][vid_parser_.GetOffset(v.GetValue()) -
                                     static_cast<int64_t>(ivnums_[v_label])];
  }
  inline vid_t GetInnerVertexGid(const vertex_t& v) const {
    return vid_parser_.GenerateId(fid_, vid_parser_.GetLabelId(v.GetValue()),
                                  vid_parser_.GetOffset(v.GetValue()));
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v, label_id_t e_label)
      const {
    // grin vertexlist continous_vid_trait get_vertex_from_vid ++++
    vid_t vid = v.GetValue();
    label_id_t v_label = vid_parser_.GetLabelId(vid);
    int64_t v_offset = vid_parser_.GetOffset(vid);
    const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
    const nbr_unit_t* ie = ie_ptr_lists_[v_label][e_label];
    return adj_list_t(&ie[offset_array[v_offset]],
                      &ie[offset_array[v_offset + 1]],
                      flatten_edge_tables_columns_[e_label]);
  }

  inline raw_adj_list_t GetIncomingRawAdjList(const vertex_t& v,
                                              label_id_t e_label) const {
    vid_t vid = v.GetValue();
    label_id_t v_label = vid_parser_.GetLabelId(vid);
    int64_t v_offset = vid_parser_.GetOffset(vid);
    const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
    const nbr_unit_t* ie = ie_ptr_lists_[v_label][e_label];
    return raw_adj_list_t(&ie[offset_array[v_offset]],
                          &ie[offset_array[v_offset + 1]]);
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v, label_id_t e_label)
      const {
    vid_t vid = v.GetValue();
    label_id_t v_label = vid_parser_.GetLabelId(vid);
    int64_t v_offset = vid_parser_.GetOffset(vid);
    const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
    const nbr_unit_t* oe = oe_ptr_lists_[v_label][e_label];
    return adj_list_t(&oe[offset_array[v_offset]],
                      &oe[offset_array[v_offset + 1]],
                      flatten_edge_tables_columns_[e_label]);
  }

  inline raw_adj_list_t GetOutgoingRawAdjList(const vertex_t& v,
                                              label_id_t e_label) const {
    vid_t vid = v.GetValue();
    label_id_t v_label = vid_parser_.GetLabelId(vid);
    int64_t v_offset = vid_parser_.GetOffset(vid);
    const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
    const nbr_unit_t* oe = oe_ptr_lists_[v_label][e_label];
    return raw_adj_list_t(&oe[offset_array[v_offset]],
                          &oe[offset_array[v_offset + 1]]);
  }

  /**
   * N.B.: as an temporary solution, for POC of graph-learn, will be removed
   * later.
   */

  inline const int64_t* GetIncomingOffsetArray(label_id_t v_label,
                                               label_id_t e_label) const {
    return ie_offsets_ptr_lists_[v_label][e_label];
  }

  inline const int64_t* GetOutgoingOffsetArray(label_id_t v_label,
                                               label_id_t e_label) const {
    return oe_offsets_ptr_lists_[v_label][e_label];
  }

  inline int64_t GetIncomingOffsetLength(label_id_t v_label, label_id_t e_label)
      const {
    return ie_offsets_lists_[v_label][e_label]->length();
  }

  inline int64_t GetOutgoingOffsetLength(label_id_t v_label, label_id_t e_label)
      const {
    return oe_offsets_lists_[v_label][e_label]->length();
  }

  inline std::pair<int64_t, int64_t> GetOutgoingAdjOffsets(
      const vertex_t& v, label_id_t e_label) const {
    vid_t vid = v.GetValue();
    label_id_t v_label = vid_parser_.GetLabelId(vid);
    int64_t v_offset = vid_parser_.GetOffset(vid);
    const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
    return std::make_pair(offset_array[v_offset], offset_array[v_offset + 1]);
  }

  inline grape::DestList IEDests(const vertex_t& v, label_id_t e_label) const {
    int64_t offset = vid_parser_.GetOffset(v.GetValue());
    auto v_label = vertex_label(v);

    return grape::DestList(idoffset_[v_label][e_label][offset],
                           idoffset_[v_label][e_label][offset + 1]);
  }

  inline grape::DestList OEDests(const vertex_t& v, label_id_t e_label) const {
    int64_t offset = vid_parser_.GetOffset(v.GetValue());
    auto v_label = vertex_label(v);

    return grape::DestList(odoffset_[v_label][e_label][offset],
                           odoffset_[v_label][e_label][offset + 1]);
  }

  inline grape::DestList IOEDests(const vertex_t& v, label_id_t e_label) const {
    int64_t offset = vid_parser_.GetOffset(v.GetValue());
    auto v_label = vertex_label(v);

    return grape::DestList(iodoffset_[v_label][e_label][offset],
                           iodoffset_[v_label][e_label][offset + 1]);
  }

  std::shared_ptr<vertex_map_t> GetVertexMap() { return vm_ptr_; }

  const PropertyGraphSchema& schema() const override { return schema_; }

  void PrepareToRunApp(const grape::CommSpec& comm_spec,
                       grape::PrepareConf conf);
/* mutable functions are not supported in grin 0.1
  boost::leaf::result<ObjectID> AddVerticesAndEdges(
      Client & client,
      std::map<label_id_t, std::shared_ptr<arrow::Table>> && vertex_tables_map,
      std::map<label_id_t, std::shared_ptr<arrow::Table>> && edge_tables_map,
      ObjectID vm_id,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      int concurrency);

  boost::leaf::result<ObjectID> AddVertices(
      Client & client,
      std::map<label_id_t, std::shared_ptr<arrow::Table>> && vertex_tables_map,
      ObjectID vm_id);

  boost::leaf::result<ObjectID> AddEdges(
      Client & client,
      std::map<label_id_t, std::shared_ptr<arrow::Table>> && edge_tables_map,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      int concurrency);

  /// Add a set of new vertex labels and a set of new edge labels to graph.
  /// Vertex label id started from vertex_label_num_, and edge label id
  /// started from edge_label_num_.
  boost::leaf::result<ObjectID> AddNewVertexEdgeLabels(
      Client & client,
      std::vector<std::shared_ptr<arrow::Table>> && vertex_tables,
      std::vector<std::shared_ptr<arrow::Table>> && edge_tables, ObjectID vm_id,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      int concurrency);

  /// Add a set of new vertex labels to graph. Vertex label id started from
  /// vertex_label_num_.
  boost::leaf::result<ObjectID> AddNewVertexLabels(
      Client & client,
      std::vector<std::shared_ptr<arrow::Table>> && vertex_tables,
      ObjectID vm_id);

  /// Add a set of new edge labels to graph. Edge label id started from
  /// edge_label_num_.
  boost::leaf::result<ObjectID> AddNewEdgeLabels(
      Client & client,
      std::vector<std::shared_ptr<arrow::Table>> && edge_tables,
      const std::vector<std::set<std::pair<std::string, std::string>>>&
          edge_relations,
      int concurrency);

  boost::leaf::result<vineyard::ObjectID> AddVertexColumns(
      vineyard::Client & client,
      const std::map<
          label_id_t,
          std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>>
          columns,
      bool replace = false) override;

  boost::leaf::result<vineyard::ObjectID> AddVertexColumns(
      vineyard::Client & client,
      const std::map<label_id_t,
                     std::vector<std::pair<
                         std::string, std::shared_ptr<arrow::ChunkedArray>>>>
          columns,
      bool replace = false) override;

  template <typename ArrayType = arrow::Array>
  boost::leaf::result<vineyard::ObjectID> AddVertexColumnsImpl(
      vineyard::Client & client,
      const std::map<
          label_id_t,
          std::vector<std::pair<std::string, std::shared_ptr<ArrayType>>>>
          columns,
      bool replace = false);

  boost::leaf::result<vineyard::ObjectID> AddEdgeColumns(
      vineyard::Client & client,
      const std::map<
          label_id_t,
          std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>>
          columns,
      bool replace = false) override;

  boost::leaf::result<vineyard::ObjectID> AddEdgeColumns(
      vineyard::Client & client,
      const std::map<label_id_t,
                     std::vector<std::pair<
                         std::string, std::shared_ptr<arrow::ChunkedArray>>>>
          columns,
      bool replace = false) override;

  template <typename ArrayType = arrow::Array>
  boost::leaf::result<vineyard::ObjectID> AddEdgeColumnsImpl(
      vineyard::Client & client,
      const std::map<
          label_id_t,
          std::vector<std::pair<std::string, std::shared_ptr<ArrayType>>>>
          columns,
      bool replace = false);

  boost::leaf::result<vineyard::ObjectID> Project(
      vineyard::Client & client,
      std::map<label_id_t, std::vector<label_id_t>> vertices,
      std::map<label_id_t, std::vector<label_id_t>> edges);

  boost::leaf::result<vineyard::ObjectID> TransformDirection(
      vineyard::Client & client, int concurrency);

  boost::leaf::result<vineyard::ObjectID> ConsolidateVertexColumns(
      vineyard::Client & client, const label_id_t vlabel,
      std::vector<std::string> const& prop_names,
      std::string const& consolidate_name);

  boost::leaf::result<vineyard::ObjectID> ConsolidateVertexColumns(
      vineyard::Client & client, const label_id_t vlabel,
      std::vector<prop_id_t> const& props, std::string const& consolidate_name);

  boost::leaf::result<vineyard::ObjectID> ConsolidateEdgeColumns(
      vineyard::Client & client, const label_id_t elabel,
      std::vector<std::string> const& prop_names,
      std::string const& consolidate_name);

  boost::leaf::result<vineyard::ObjectID> ConsolidateEdgeColumns(
      vineyard::Client & client, const label_id_t elabel,
      std::vector<prop_id_t> const& props, std::string const& consolidate_name);
*/
  // we support projection by providing a "view" graph
  void* Project(
      vineyard::Client & client,
      std::map<label_id_t, std::vector<label_id_t>> vertices,
      std::map<label_id_t, std::vector<label_id_t>> edges);
    // use grin property "select"
 private:
  void initPointers();

  void initDestFidList(
      bool in_edge, bool out_edge,
      std::vector<std::vector<std::vector<fid_t>>>& fid_lists,
      std::vector<std::vector<std::vector<fid_t*>>>& fid_lists_offset);

  void directedCSR2Undirected(
      vineyard::Client & client,
      std::vector<std::vector<std::shared_ptr<PodArrayBuilder<nbr_unit_t>>>> &
          oe_lists,
      std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>> &
          oe_offsets_lists,
      int concurrency, bool& is_multigraph);

  __attribute__((annotate("shared"))) fid_t fid_, fnum_;
  __attribute__((annotate("shared"))) bool directed_;
  __attribute__((annotate("shared"))) bool is_multigraph_;
  __attribute__((annotate("shared"))) property_graph_types::LABEL_ID_TYPE vertex_label_num_;
  __attribute__((annotate("shared"))) property_graph_types::LABEL_ID_TYPE edge_label_num_;
  size_t oenum_, ienum_;  // FIXME: should be pre-computable

  __attribute__((annotate("shared"))) String oid_type, vid_type;

  __attribute__((annotate("shared"))) vineyard::Array<vid_t> ivnums_, ovnums_, tvnums_;

  __attribute__((annotate("shared"))) List<std::shared_ptr<Table>> vertex_tables_;
  std::vector<std::vector<const void*>> vertex_tables_columns_;

  __attribute__((annotate("shared"))) List<std::shared_ptr<vid_vineyard_array_t>> ovgid_lists_;
  std::vector<const vid_t*> ovgid_lists_ptr_;

  __attribute__((annotate("shared"))) List<std::shared_ptr<vineyard::Hashmap<vid_t, vid_t>>> ovg2l_maps_;
  std::vector<vineyard::Hashmap<vid_t, vid_t>*> ovg2l_maps_ptr_;

  __attribute__((annotate("shared"))) List<std::shared_ptr<Table>> edge_tables_;
  std::vector<std::vector<const void*>> edge_tables_columns_;
  std::vector<const void**> flatten_edge_tables_columns_;

  __attribute__((annotate("shared"))) List<List<std::shared_ptr<FixedSizeBinaryArray>>> ie_lists_,
      oe_lists_;
  std::vector<std::vector<const nbr_unit_t*>> ie_ptr_lists_, oe_ptr_lists_;
  __attribute__((annotate("shared"))) List<List<std::shared_ptr<Int64Array>>> ie_offsets_lists_,
      oe_offsets_lists_;
  std::vector<std::vector<const int64_t*>> ie_offsets_ptr_lists_,
      oe_offsets_ptr_lists_;

  std::vector<std::vector<std::vector<fid_t>>> idst_, odst_, iodst_;
  std::vector<std::vector<std::vector<fid_t*>>> idoffset_, odoffset_,
      iodoffset_;

  __attribute__((annotate("shared"))) std::shared_ptr<vertex_map_t> vm_ptr_;

  vineyard::IdParser<vid_t> vid_parser_;

  __attribute__((annotate("shared"))) json schema_json_;
  PropertyGraphSchema schema_;

  friend class ArrowFragmentBaseBuilder<OID_T, VID_T, VERTEX_MAP_T>;

  template <typename _OID_T, typename _VID_T, typename VDATA_T,
            typename EDATA_T, typename _VERTEX_MAP_T>
  friend class gs::ArrowProjectedFragment;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_MOD_H_

// vim: syntax=cpp

namespace vineyard {

template<typename OID_T, typename VID_T, typename VERTEX_MAP_T =
              ArrowVertexMap<typename InternalType<OID_T>::type, VID_T>>
class ArrowFragmentBaseBuilder: public ObjectBuilder {
  public:
    // using oid_t
    using oid_t = OID_T;
    // using vid_t
    using vid_t = VID_T;
    // using internal_oid_t
    using internal_oid_t = typename InternalType<oid_t>::type;
    // using eid_t
    using eid_t = property_graph_types::EID_TYPE;
    // using prop_id_t
    using prop_id_t = property_graph_types::PROP_ID_TYPE;
    // using label_id_t
    using label_id_t = property_graph_types::LABEL_ID_TYPE;
    // using vertex_range_t
    using vertex_range_t = grape::VertexRange<vid_t>;
    // using inner_vertices_t
    using inner_vertices_t = vertex_range_t;
    // using outer_vertices_t
    using outer_vertices_t = vertex_range_t;
    // using vertices_t
    using vertices_t = vertex_range_t;
    // using nbr_t
    using nbr_t = property_graph_utils::Nbr<vid_t, eid_t>;
    // using nbr_unit_t
    using nbr_unit_t = property_graph_utils::NbrUnit<vid_t, eid_t>;
    // using adj_list_t
    using adj_list_t = property_graph_utils::AdjList<vid_t, eid_t>;
    // using raw_adj_list_t
    using raw_adj_list_t = property_graph_utils::RawAdjList<vid_t, eid_t>;
    // using vertex_map_t
    using vertex_map_t = VERTEX_MAP_T;
    // using vertex_t
    using vertex_t = grape::Vertex<vid_t>;
    // using ovg2l_map_t
    using ovg2l_map_t =
      ska::flat_hash_map<vid_t, vid_t, typename Hashmap<vid_t, vid_t>::KeyHash>;
    // using vid_array_t
    using vid_array_t = ArrowArrayType<vid_t>;
    // using vid_vineyard_array_t
    using vid_vineyard_array_t = ArrowVineyardArrayType<vid_t>;
    // using vid_vineyard_builder_t
    using vid_vineyard_builder_t = ArrowVineyardBuilderType<vid_t>;
    // using eid_array_t
    using eid_array_t = ArrowArrayType<eid_t>;
    // using eid_vineyard_array_t
    using eid_vineyard_array_t = ArrowVineyardArrayType<eid_t>;
    // using eid_vineyard_builder_t
    using eid_vineyard_builder_t = ArrowVineyardBuilderType<eid_t>;
    // using vid_builder_t
    using vid_builder_t = ArrowBuilderType<vid_t>;

    explicit ArrowFragmentBaseBuilder(Client &client) {}

    explicit ArrowFragmentBaseBuilder(
            ArrowFragment<OID_T, VID_T, VERTEX_MAP_T> const &__value) {
        this->set_fid_(__value.fid_);
        this->set_fnum_(__value.fnum_);
        this->set_directed_(__value.directed_);
        this->set_is_multigraph_(__value.is_multigraph_);
        this->set_vertex_label_num_(__value.vertex_label_num_);
        this->set_edge_label_num_(__value.edge_label_num_);
        this->set_oid_type(__value.oid_type);
        this->set_vid_type(__value.vid_type);
        this->set_ivnums_(
            std::make_shared<typename std::decay<decltype(__value.ivnums_)>::type>(
                __value.ivnums_));
        this->set_ovnums_(
            std::make_shared<typename std::decay<decltype(__value.ovnums_)>::type>(
                __value.ovnums_));
        this->set_tvnums_(
            std::make_shared<typename std::decay<decltype(__value.tvnums_)>::type>(
                __value.tvnums_));
        for (auto const &__vertex_tables__item: __value.vertex_tables_) {
            this->add_vertex_tables_(__vertex_tables__item);
        }
        for (auto const &__ovgid_lists__item: __value.ovgid_lists_) {
            this->add_ovgid_lists_(__ovgid_lists__item);
        }
        for (auto const &__ovg2l_maps__item: __value.ovg2l_maps_) {
            this->add_ovg2l_maps_(__ovg2l_maps__item);
        }
        for (auto const &__edge_tables__item: __value.edge_tables_) {
            this->add_edge_tables_(__edge_tables__item);
        }
        this->ie_lists_.resize(__value.ie_lists_.size());
        for (size_t __idx = 0; __idx < __value.ie_lists_.size(); ++__idx) {
            this->ie_lists_[__idx].resize(__value.ie_lists_[__idx].size());
            for (size_t __idy = 0; __idy < __value.ie_lists_[__idx].size(); ++__idy) {
                this->ie_lists_[__idx][__idy] = __value.ie_lists_[__idx][__idy];
            }
        }
        this->oe_lists_.resize(__value.oe_lists_.size());
        for (size_t __idx = 0; __idx < __value.oe_lists_.size(); ++__idx) {
            this->oe_lists_[__idx].resize(__value.oe_lists_[__idx].size());
            for (size_t __idy = 0; __idy < __value.oe_lists_[__idx].size(); ++__idy) {
                this->oe_lists_[__idx][__idy] = __value.oe_lists_[__idx][__idy];
            }
        }
        this->ie_offsets_lists_.resize(__value.ie_offsets_lists_.size());
        for (size_t __idx = 0; __idx < __value.ie_offsets_lists_.size(); ++__idx) {
            this->ie_offsets_lists_[__idx].resize(__value.ie_offsets_lists_[__idx].size());
            for (size_t __idy = 0; __idy < __value.ie_offsets_lists_[__idx].size(); ++__idy) {
                this->ie_offsets_lists_[__idx][__idy] = __value.ie_offsets_lists_[__idx][__idy];
            }
        }
        this->oe_offsets_lists_.resize(__value.oe_offsets_lists_.size());
        for (size_t __idx = 0; __idx < __value.oe_offsets_lists_.size(); ++__idx) {
            this->oe_offsets_lists_[__idx].resize(__value.oe_offsets_lists_[__idx].size());
            for (size_t __idy = 0; __idy < __value.oe_offsets_lists_[__idx].size(); ++__idy) {
                this->oe_offsets_lists_[__idx][__idy] = __value.oe_offsets_lists_[__idx][__idy];
            }
        }
        this->set_vm_ptr_(__value.vm_ptr_);
        this->set_schema_json_(__value.schema_json_);
    }

    explicit ArrowFragmentBaseBuilder(
            std::shared_ptr<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>> const & __value):
        ArrowFragmentBaseBuilder(*__value) {
    }

    ObjectMeta &ValueMetaRef(std::shared_ptr<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>> &__value) {
        return __value->meta_;
    }

    std::shared_ptr<Object> _Seal(Client &client) override {
        // ensure the builder hasn't been sealed yet.
        ENSURE_NOT_SEALED(this);

        VINEYARD_CHECK_OK(this->Build(client));
        auto __value = std::make_shared<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>>();

        return this->_Seal(client, __value);
    }

    std::shared_ptr<Object> _Seal(Client &client, std::shared_ptr<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>> &__value) {
        size_t __value_nbytes = 0;

        __value->meta_.SetTypeName(type_name<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>>());
        if (std::is_base_of<GlobalObject, ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>>::value) {
            __value->meta_.SetGlobal(true);
        }

        __value->fid_ = fid_;
        __value->meta_.AddKeyValue("fid_", __value->fid_);

        __value->fnum_ = fnum_;
        __value->meta_.AddKeyValue("fnum_", __value->fnum_);

        __value->directed_ = directed_;
        __value->meta_.AddKeyValue("directed_", __value->directed_);

        __value->is_multigraph_ = is_multigraph_;
        __value->meta_.AddKeyValue("is_multigraph_", __value->is_multigraph_);

        __value->vertex_label_num_ = vertex_label_num_;
        __value->meta_.AddKeyValue("vertex_label_num_", __value->vertex_label_num_);

        __value->edge_label_num_ = edge_label_num_;
        __value->meta_.AddKeyValue("edge_label_num_", __value->edge_label_num_);

        __value->oid_type = oid_type;
        __value->meta_.AddKeyValue("oid_type", __value->oid_type);

        __value->vid_type = vid_type;
        __value->meta_.AddKeyValue("vid_type", __value->vid_type);

        // using __ivnums__value_type = typename vineyard::Array<vid_t>;
        using __ivnums__value_type = decltype(__value->ivnums_);
        auto __value_ivnums_ = std::dynamic_pointer_cast<__ivnums__value_type>(
            ivnums_->_Seal(client));
        __value->ivnums_ = *__value_ivnums_;
        __value->meta_.AddMember("ivnums_", __value->ivnums_);
        __value_nbytes += __value_ivnums_->nbytes();

        // using __ovnums__value_type = typename vineyard::Array<vid_t>;
        using __ovnums__value_type = decltype(__value->ovnums_);
        auto __value_ovnums_ = std::dynamic_pointer_cast<__ovnums__value_type>(
            ovnums_->_Seal(client));
        __value->ovnums_ = *__value_ovnums_;
        __value->meta_.AddMember("ovnums_", __value->ovnums_);
        __value_nbytes += __value_ovnums_->nbytes();

        // using __tvnums__value_type = typename vineyard::Array<vid_t>;
        using __tvnums__value_type = decltype(__value->tvnums_);
        auto __value_tvnums_ = std::dynamic_pointer_cast<__tvnums__value_type>(
            tvnums_->_Seal(client));
        __value->tvnums_ = *__value_tvnums_;
        __value->meta_.AddMember("tvnums_", __value->tvnums_);
        __value_nbytes += __value_tvnums_->nbytes();

        // using __vertex_tables__value_type = typename List<std::shared_ptr<Table>>::value_type::element_type;
        using __vertex_tables__value_type = typename decltype(__value->vertex_tables_)::value_type::element_type;

        size_t __vertex_tables__idx = 0;
        for (auto &__vertex_tables__value: vertex_tables_) {
            auto __value_vertex_tables_ = std::dynamic_pointer_cast<__vertex_tables__value_type>(
                __vertex_tables__value->_Seal(client));
            __value->vertex_tables_.emplace_back(__value_vertex_tables_);
            __value->meta_.AddMember("__vertex_tables_-" + std::to_string(__vertex_tables__idx),
                                     __value_vertex_tables_);
            __value_nbytes += __value_vertex_tables_->nbytes();
            __vertex_tables__idx += 1;
        }
        __value->meta_.AddKeyValue("__vertex_tables_-size", __value->vertex_tables_.size());

        // using __ovgid_lists__value_type = typename List<std::shared_ptr<vid_vineyard_array_t>>::value_type::element_type;
        using __ovgid_lists__value_type = typename decltype(__value->ovgid_lists_)::value_type::element_type;

        size_t __ovgid_lists__idx = 0;
        for (auto &__ovgid_lists__value: ovgid_lists_) {
            auto __value_ovgid_lists_ = std::dynamic_pointer_cast<__ovgid_lists__value_type>(
                __ovgid_lists__value->_Seal(client));
            __value->ovgid_lists_.emplace_back(__value_ovgid_lists_);
            __value->meta_.AddMember("__ovgid_lists_-" + std::to_string(__ovgid_lists__idx),
                                     __value_ovgid_lists_);
            __value_nbytes += __value_ovgid_lists_->nbytes();
            __ovgid_lists__idx += 1;
        }
        __value->meta_.AddKeyValue("__ovgid_lists_-size", __value->ovgid_lists_.size());

        // using __ovg2l_maps__value_type = typename List<std::shared_ptr<vineyard::Hashmap<vid_t, vid_t>>>::value_type::element_type;
        using __ovg2l_maps__value_type = typename decltype(__value->ovg2l_maps_)::value_type::element_type;

        size_t __ovg2l_maps__idx = 0;
        for (auto &__ovg2l_maps__value: ovg2l_maps_) {
            auto __value_ovg2l_maps_ = std::dynamic_pointer_cast<__ovg2l_maps__value_type>(
                __ovg2l_maps__value->_Seal(client));
            __value->ovg2l_maps_.emplace_back(__value_ovg2l_maps_);
            __value->meta_.AddMember("__ovg2l_maps_-" + std::to_string(__ovg2l_maps__idx),
                                     __value_ovg2l_maps_);
            __value_nbytes += __value_ovg2l_maps_->nbytes();
            __ovg2l_maps__idx += 1;
        }
        __value->meta_.AddKeyValue("__ovg2l_maps_-size", __value->ovg2l_maps_.size());

        // using __edge_tables__value_type = typename List<std::shared_ptr<Table>>::value_type::element_type;
        using __edge_tables__value_type = typename decltype(__value->edge_tables_)::value_type::element_type;

        size_t __edge_tables__idx = 0;
        for (auto &__edge_tables__value: edge_tables_) {
            auto __value_edge_tables_ = std::dynamic_pointer_cast<__edge_tables__value_type>(
                __edge_tables__value->_Seal(client));
            __value->edge_tables_.emplace_back(__value_edge_tables_);
            __value->meta_.AddMember("__edge_tables_-" + std::to_string(__edge_tables__idx),
                                     __value_edge_tables_);
            __value_nbytes += __value_edge_tables_->nbytes();
            __edge_tables__idx += 1;
        }
        __value->meta_.AddKeyValue("__edge_tables_-size", __value->edge_tables_.size());

        // using __ie_lists__value_type = typename List<List<std::shared_ptr<FixedSizeBinaryArray>>>::value_type::value_type::element_type;
        using __ie_lists__value_type = typename decltype(__value->ie_lists_)::value_type::value_type::element_type;

        size_t __ie_lists__idx = 0;
        __value->ie_lists_.resize(ie_lists_.size());
        for (auto &__ie_lists__value_vec: ie_lists_) {
            size_t __ie_lists__idy = 0;
            __value->meta_.AddKeyValue("__ie_lists_-" + std::to_string(__ie_lists__idx) + "-size", __ie_lists__value_vec.size());
            for (auto &__ie_lists__value: __ie_lists__value_vec) {
                auto __value_ie_lists_ = std::dynamic_pointer_cast<__ie_lists__value_type>(
                    __ie_lists__value->_Seal(client));
                __value->ie_lists_[__ie_lists__idx].emplace_back(__value_ie_lists_);
                __value->meta_.AddMember("__ie_lists_-" + std::to_string(__ie_lists__idx) + "-" + std::to_string(__ie_lists__idy),
                                         __value_ie_lists_);
                __value_nbytes += __value_ie_lists_->nbytes();
                __ie_lists__idy += 1;
            }
            __ie_lists__idx += 1;
        }
        __value->meta_.AddKeyValue("__ie_lists_-size", __value->ie_lists_.size());

        // using __oe_lists__value_type = typename List<List<std::shared_ptr<FixedSizeBinaryArray>>>::value_type::value_type::element_type;
        using __oe_lists__value_type = typename decltype(__value->oe_lists_)::value_type::value_type::element_type;

        size_t __oe_lists__idx = 0;
        __value->oe_lists_.resize(oe_lists_.size());
        for (auto &__oe_lists__value_vec: oe_lists_) {
            size_t __oe_lists__idy = 0;
            __value->meta_.AddKeyValue("__oe_lists_-" + std::to_string(__oe_lists__idx) + "-size", __oe_lists__value_vec.size());
            for (auto &__oe_lists__value: __oe_lists__value_vec) {
                auto __value_oe_lists_ = std::dynamic_pointer_cast<__oe_lists__value_type>(
                    __oe_lists__value->_Seal(client));
                __value->oe_lists_[__oe_lists__idx].emplace_back(__value_oe_lists_);
                __value->meta_.AddMember("__oe_lists_-" + std::to_string(__oe_lists__idx) + "-" + std::to_string(__oe_lists__idy),
                                         __value_oe_lists_);
                __value_nbytes += __value_oe_lists_->nbytes();
                __oe_lists__idy += 1;
            }
            __oe_lists__idx += 1;
        }
        __value->meta_.AddKeyValue("__oe_lists_-size", __value->oe_lists_.size());

        // using __ie_offsets_lists__value_type = typename List<List<std::shared_ptr<Int64Array>>>::value_type::value_type::element_type;
        using __ie_offsets_lists__value_type = typename decltype(__value->ie_offsets_lists_)::value_type::value_type::element_type;

        size_t __ie_offsets_lists__idx = 0;
        __value->ie_offsets_lists_.resize(ie_offsets_lists_.size());
        for (auto &__ie_offsets_lists__value_vec: ie_offsets_lists_) {
            size_t __ie_offsets_lists__idy = 0;
            __value->meta_.AddKeyValue("__ie_offsets_lists_-" + std::to_string(__ie_offsets_lists__idx) + "-size", __ie_offsets_lists__value_vec.size());
            for (auto &__ie_offsets_lists__value: __ie_offsets_lists__value_vec) {
                auto __value_ie_offsets_lists_ = std::dynamic_pointer_cast<__ie_offsets_lists__value_type>(
                    __ie_offsets_lists__value->_Seal(client));
                __value->ie_offsets_lists_[__ie_offsets_lists__idx].emplace_back(__value_ie_offsets_lists_);
                __value->meta_.AddMember("__ie_offsets_lists_-" + std::to_string(__ie_offsets_lists__idx) + "-" + std::to_string(__ie_offsets_lists__idy),
                                         __value_ie_offsets_lists_);
                __value_nbytes += __value_ie_offsets_lists_->nbytes();
                __ie_offsets_lists__idy += 1;
            }
            __ie_offsets_lists__idx += 1;
        }
        __value->meta_.AddKeyValue("__ie_offsets_lists_-size", __value->ie_offsets_lists_.size());

        // using __oe_offsets_lists__value_type = typename List<List<std::shared_ptr<Int64Array>>>::value_type::value_type::element_type;
        using __oe_offsets_lists__value_type = typename decltype(__value->oe_offsets_lists_)::value_type::value_type::element_type;

        size_t __oe_offsets_lists__idx = 0;
        __value->oe_offsets_lists_.resize(oe_offsets_lists_.size());
        for (auto &__oe_offsets_lists__value_vec: oe_offsets_lists_) {
            size_t __oe_offsets_lists__idy = 0;
            __value->meta_.AddKeyValue("__oe_offsets_lists_-" + std::to_string(__oe_offsets_lists__idx) + "-size", __oe_offsets_lists__value_vec.size());
            for (auto &__oe_offsets_lists__value: __oe_offsets_lists__value_vec) {
                auto __value_oe_offsets_lists_ = std::dynamic_pointer_cast<__oe_offsets_lists__value_type>(
                    __oe_offsets_lists__value->_Seal(client));
                __value->oe_offsets_lists_[__oe_offsets_lists__idx].emplace_back(__value_oe_offsets_lists_);
                __value->meta_.AddMember("__oe_offsets_lists_-" + std::to_string(__oe_offsets_lists__idx) + "-" + std::to_string(__oe_offsets_lists__idy),
                                         __value_oe_offsets_lists_);
                __value_nbytes += __value_oe_offsets_lists_->nbytes();
                __oe_offsets_lists__idy += 1;
            }
            __oe_offsets_lists__idx += 1;
        }
        __value->meta_.AddKeyValue("__oe_offsets_lists_-size", __value->oe_offsets_lists_.size());

        // using __vm_ptr__value_type = typename std::shared_ptr<vertex_map_t>::element_type;
        using __vm_ptr__value_type = typename decltype(__value->vm_ptr_)::element_type;
        auto __value_vm_ptr_ = std::dynamic_pointer_cast<__vm_ptr__value_type>(
            vm_ptr_->_Seal(client));
        __value->vm_ptr_ = __value_vm_ptr_;
        __value->meta_.AddMember("vm_ptr_", __value->vm_ptr_);
        __value_nbytes += __value_vm_ptr_->nbytes();

        __value->schema_json_ = schema_json_;
        __value->meta_.AddKeyValue("schema_json_", __value->schema_json_);

        __value->meta_.SetNBytes(__value_nbytes);

        VINEYARD_CHECK_OK(client.CreateMetaData(__value->meta_, __value->id_));

        // mark the builder as sealed
        this->set_sealed(true);

        
        // run `PostConstruct` to return a valid object
        __value->PostConstruct(__value->meta_);

        return std::static_pointer_cast<Object>(__value);
    }

    Status Build(Client &client) override {
        return Status::OK();
    }

  protected:
    vineyard::fid_t fid_;
    vineyard::fid_t fnum_;
    bool directed_;
    bool is_multigraph_;
    property_graph_types::LABEL_ID_TYPE vertex_label_num_;
    property_graph_types::LABEL_ID_TYPE edge_label_num_;
    vineyard::String oid_type;
    vineyard::String vid_type;
    std::shared_ptr<ObjectBase> ivnums_;
    std::shared_ptr<ObjectBase> ovnums_;
    std::shared_ptr<ObjectBase> tvnums_;
    std::vector<std::shared_ptr<ObjectBase>> vertex_tables_;
    std::vector<std::shared_ptr<ObjectBase>> ovgid_lists_;
    std::vector<std::shared_ptr<ObjectBase>> ovg2l_maps_;
    std::vector<std::shared_ptr<ObjectBase>> edge_tables_;
    std::vector<std::vector<std::shared_ptr<ObjectBase>>> ie_lists_;
    std::vector<std::vector<std::shared_ptr<ObjectBase>>> oe_lists_;
    std::vector<std::vector<std::shared_ptr<ObjectBase>>> ie_offsets_lists_;
    std::vector<std::vector<std::shared_ptr<ObjectBase>>> oe_offsets_lists_;
    std::shared_ptr<ObjectBase> vm_ptr_;
    vineyard::json schema_json_;

    void set_fid_(vineyard::fid_t const &fid__) {
        this->fid_ = fid__;
    }

    void set_fnum_(vineyard::fid_t const &fnum__) {
        this->fnum_ = fnum__;
    }

    void set_directed_(bool const &directed__) {
        this->directed_ = directed__;
    }

    void set_is_multigraph_(bool const &is_multigraph__) {
        this->is_multigraph_ = is_multigraph__;
    }

    void set_vertex_label_num_(property_graph_types::LABEL_ID_TYPE const &vertex_label_num__) {
        this->vertex_label_num_ = vertex_label_num__;
    }

    void set_edge_label_num_(property_graph_types::LABEL_ID_TYPE const &edge_label_num__) {
        this->edge_label_num_ = edge_label_num__;
    }

    void set_oid_type(vineyard::String const &oid_type_) {
        this->oid_type = oid_type_;
    }

    void set_vid_type(vineyard::String const &vid_type_) {
        this->vid_type = vid_type_;
    }

    void set_ivnums_(std::shared_ptr<ObjectBase> const & ivnums__) {
        this->ivnums_ = ivnums__;
    }

    void set_ovnums_(std::shared_ptr<ObjectBase> const & ovnums__) {
        this->ovnums_ = ovnums__;
    }

    void set_tvnums_(std::shared_ptr<ObjectBase> const & tvnums__) {
        this->tvnums_ = tvnums__;
    }

    void set_vertex_tables_(std::vector<std::shared_ptr<ObjectBase>> const &vertex_tables__) {
        this->vertex_tables_ = vertex_tables__;
    }
    void set_vertex_tables_(size_t const idx, std::shared_ptr<ObjectBase> const &vertex_tables__) {
        if (idx >= this->vertex_tables_.size()) {
            this->vertex_tables_.resize(idx + 1);
        }
        this->vertex_tables_[idx] = vertex_tables__;
    }
    void add_vertex_tables_(std::shared_ptr<ObjectBase> const &vertex_tables__) {
        this->vertex_tables_.emplace_back(vertex_tables__);
    }
    void remove_vertex_tables_(const size_t vertex_tables__index_) {
        this->vertex_tables_.erase(this->vertex_tables_.begin() + vertex_tables__index_);
    }

    void set_ovgid_lists_(std::vector<std::shared_ptr<ObjectBase>> const &ovgid_lists__) {
        this->ovgid_lists_ = ovgid_lists__;
    }
    void set_ovgid_lists_(size_t const idx, std::shared_ptr<ObjectBase> const &ovgid_lists__) {
        if (idx >= this->ovgid_lists_.size()) {
            this->ovgid_lists_.resize(idx + 1);
        }
        this->ovgid_lists_[idx] = ovgid_lists__;
    }
    void add_ovgid_lists_(std::shared_ptr<ObjectBase> const &ovgid_lists__) {
        this->ovgid_lists_.emplace_back(ovgid_lists__);
    }
    void remove_ovgid_lists_(const size_t ovgid_lists__index_) {
        this->ovgid_lists_.erase(this->ovgid_lists_.begin() + ovgid_lists__index_);
    }

    void set_ovg2l_maps_(std::vector<std::shared_ptr<ObjectBase>> const &ovg2l_maps__) {
        this->ovg2l_maps_ = ovg2l_maps__;
    }
    void set_ovg2l_maps_(size_t const idx, std::shared_ptr<ObjectBase> const &ovg2l_maps__) {
        if (idx >= this->ovg2l_maps_.size()) {
            this->ovg2l_maps_.resize(idx + 1);
        }
        this->ovg2l_maps_[idx] = ovg2l_maps__;
    }
    void add_ovg2l_maps_(std::shared_ptr<ObjectBase> const &ovg2l_maps__) {
        this->ovg2l_maps_.emplace_back(ovg2l_maps__);
    }
    void remove_ovg2l_maps_(const size_t ovg2l_maps__index_) {
        this->ovg2l_maps_.erase(this->ovg2l_maps_.begin() + ovg2l_maps__index_);
    }

    void set_edge_tables_(std::vector<std::shared_ptr<ObjectBase>> const &edge_tables__) {
        this->edge_tables_ = edge_tables__;
    }
    void set_edge_tables_(size_t const idx, std::shared_ptr<ObjectBase> const &edge_tables__) {
        if (idx >= this->edge_tables_.size()) {
            this->edge_tables_.resize(idx + 1);
        }
        this->edge_tables_[idx] = edge_tables__;
    }
    void add_edge_tables_(std::shared_ptr<ObjectBase> const &edge_tables__) {
        this->edge_tables_.emplace_back(edge_tables__);
    }
    void remove_edge_tables_(const size_t edge_tables__index_) {
        this->edge_tables_.erase(this->edge_tables_.begin() + edge_tables__index_);
    }

    void set_ie_lists_(std::vector<std::vector<std::shared_ptr<ObjectBase>>> const &ie_lists__) {
        this->ie_lists_ = ie_lists__;
    }
    void set_ie_lists_(size_t const idx, std::vector<std::shared_ptr<ObjectBase>> const &ie_lists__) {
        if (idx >= this->ie_lists_.size()) {
            this->ie_lists_.resize(idx + 1);
        }
        this->ie_lists_[idx] = ie_lists__;
    }
    void set_ie_lists_(size_t const idx, size_t const idy,
                          std::shared_ptr<ObjectBase> const &ie_lists__) {
        if (idx >= this->ie_lists_.size()) {
            this->ie_lists_.resize(idx + 1);
        }
        if (idy >= this->ie_lists_[idx].size()) {
            this->ie_lists_[idx].resize(idy + 1);
        }
        this->ie_lists_[idx][idy] = ie_lists__;
    }
    void add_ie_lists_(std::vector<std::shared_ptr<ObjectBase>> const &ie_lists__) {
        this->ie_lists_.emplace_back(ie_lists__);
    }
    void remove_ie_lists_(const size_t ie_lists__index_) {
        this->ie_lists_.erase(this->ie_lists_.begin() + ie_lists__index_);
    }
    void remove_ie_lists_(const size_t ie_lists__index_, const size_t ie_lists__inner_index_) {
        auto &ie_lists__inner_ = this->ie_lists_[ie_lists__index_];
        ie_lists__inner_.erase(ie_lists__inner_.begin() + ie_lists__inner_index_);
    }

    void set_oe_lists_(std::vector<std::vector<std::shared_ptr<ObjectBase>>> const &oe_lists__) {
        this->oe_lists_ = oe_lists__;
    }
    void set_oe_lists_(size_t const idx, std::vector<std::shared_ptr<ObjectBase>> const &oe_lists__) {
        if (idx >= this->oe_lists_.size()) {
            this->oe_lists_.resize(idx + 1);
        }
        this->oe_lists_[idx] = oe_lists__;
    }
    void set_oe_lists_(size_t const idx, size_t const idy,
                          std::shared_ptr<ObjectBase> const &oe_lists__) {
        if (idx >= this->oe_lists_.size()) {
            this->oe_lists_.resize(idx + 1);
        }
        if (idy >= this->oe_lists_[idx].size()) {
            this->oe_lists_[idx].resize(idy + 1);
        }
        this->oe_lists_[idx][idy] = oe_lists__;
    }
    void add_oe_lists_(std::vector<std::shared_ptr<ObjectBase>> const &oe_lists__) {
        this->oe_lists_.emplace_back(oe_lists__);
    }
    void remove_oe_lists_(const size_t oe_lists__index_) {
        this->oe_lists_.erase(this->oe_lists_.begin() + oe_lists__index_);
    }
    void remove_oe_lists_(const size_t oe_lists__index_, const size_t oe_lists__inner_index_) {
        auto &oe_lists__inner_ = this->oe_lists_[oe_lists__index_];
        oe_lists__inner_.erase(oe_lists__inner_.begin() + oe_lists__inner_index_);
    }

    void set_ie_offsets_lists_(std::vector<std::vector<std::shared_ptr<ObjectBase>>> const &ie_offsets_lists__) {
        this->ie_offsets_lists_ = ie_offsets_lists__;
    }
    void set_ie_offsets_lists_(size_t const idx, std::vector<std::shared_ptr<ObjectBase>> const &ie_offsets_lists__) {
        if (idx >= this->ie_offsets_lists_.size()) {
            this->ie_offsets_lists_.resize(idx + 1);
        }
        this->ie_offsets_lists_[idx] = ie_offsets_lists__;
    }
    void set_ie_offsets_lists_(size_t const idx, size_t const idy,
                          std::shared_ptr<ObjectBase> const &ie_offsets_lists__) {
        if (idx >= this->ie_offsets_lists_.size()) {
            this->ie_offsets_lists_.resize(idx + 1);
        }
        if (idy >= this->ie_offsets_lists_[idx].size()) {
            this->ie_offsets_lists_[idx].resize(idy + 1);
        }
        this->ie_offsets_lists_[idx][idy] = ie_offsets_lists__;
    }
    void add_ie_offsets_lists_(std::vector<std::shared_ptr<ObjectBase>> const &ie_offsets_lists__) {
        this->ie_offsets_lists_.emplace_back(ie_offsets_lists__);
    }
    void remove_ie_offsets_lists_(const size_t ie_offsets_lists__index_) {
        this->ie_offsets_lists_.erase(this->ie_offsets_lists_.begin() + ie_offsets_lists__index_);
    }
    void remove_ie_offsets_lists_(const size_t ie_offsets_lists__index_, const size_t ie_offsets_lists__inner_index_) {
        auto &ie_offsets_lists__inner_ = this->ie_offsets_lists_[ie_offsets_lists__index_];
        ie_offsets_lists__inner_.erase(ie_offsets_lists__inner_.begin() + ie_offsets_lists__inner_index_);
    }

    void set_oe_offsets_lists_(std::vector<std::vector<std::shared_ptr<ObjectBase>>> const &oe_offsets_lists__) {
        this->oe_offsets_lists_ = oe_offsets_lists__;
    }
    void set_oe_offsets_lists_(size_t const idx, std::vector<std::shared_ptr<ObjectBase>> const &oe_offsets_lists__) {
        if (idx >= this->oe_offsets_lists_.size()) {
            this->oe_offsets_lists_.resize(idx + 1);
        }
        this->oe_offsets_lists_[idx] = oe_offsets_lists__;
    }
    void set_oe_offsets_lists_(size_t const idx, size_t const idy,
                          std::shared_ptr<ObjectBase> const &oe_offsets_lists__) {
        if (idx >= this->oe_offsets_lists_.size()) {
            this->oe_offsets_lists_.resize(idx + 1);
        }
        if (idy >= this->oe_offsets_lists_[idx].size()) {
            this->oe_offsets_lists_[idx].resize(idy + 1);
        }
        this->oe_offsets_lists_[idx][idy] = oe_offsets_lists__;
    }
    void add_oe_offsets_lists_(std::vector<std::shared_ptr<ObjectBase>> const &oe_offsets_lists__) {
        this->oe_offsets_lists_.emplace_back(oe_offsets_lists__);
    }
    void remove_oe_offsets_lists_(const size_t oe_offsets_lists__index_) {
        this->oe_offsets_lists_.erase(this->oe_offsets_lists_.begin() + oe_offsets_lists__index_);
    }
    void remove_oe_offsets_lists_(const size_t oe_offsets_lists__index_, const size_t oe_offsets_lists__inner_index_) {
        auto &oe_offsets_lists__inner_ = this->oe_offsets_lists_[oe_offsets_lists__index_];
        oe_offsets_lists__inner_.erase(oe_offsets_lists__inner_.begin() + oe_offsets_lists__inner_index_);
    }

    void set_vm_ptr_(std::shared_ptr<ObjectBase> const & vm_ptr__) {
        this->vm_ptr_ = vm_ptr__;
    }

    void set_schema_json_(vineyard::json const &schema_json__) {
        this->schema_json_ = schema_json__;
    }

  private:
    friend class ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>;
};


}  // namespace vineyard



#endif // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H
