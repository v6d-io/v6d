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

extern "C" {
#include "grin/include/topology/structure.h"
#include "grin/include/topology/vertexlist.h"
#include "grin/include/topology/edgelist.h"
#include "grin/include/topology/adjacentlist.h"
#include "grin/include/partition/partition.h"
#include "grin/include/propertygraph/label.h"
#include "grin/include/propertygraph/property.h"
#include "grin/include/propertygraph/propertygraph.h"
}

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
 
//   public:
//     static std::unique_ptr<Object> Create() __attribute__((used)) {
//         return std::static_pointer_cast<Object>(
//             std::unique_ptr<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>>{
//                 new ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>()});
//     }


//   public:
//     void Construct(const ObjectMeta& meta) override {
//         std::string __type_name = type_name<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T>>();
//         VINEYARD_ASSERT(
//             meta.GetTypeName() == __type_name,
//             "Expect typename '" + __type_name + "', but got '" + meta.GetTypeName() + "'");
//         this->meta_ = meta;
//         this->id_ = meta.GetId();

//         meta.GetKeyValue("fid_", this->fid_);
//         meta.GetKeyValue("fnum_", this->fnum_);
//         meta.GetKeyValue("directed_", this->directed_);
//         meta.GetKeyValue("is_multigraph_", this->is_multigraph_);
//         meta.GetKeyValue("vertex_label_num_", this->vertex_label_num_);
//         meta.GetKeyValue("edge_label_num_", this->edge_label_num_);
//         meta.GetKeyValue("oid_type", this->oid_type);
//         meta.GetKeyValue("vid_type", this->vid_type);
//         this->ivnums_.Construct(meta.GetMemberMeta("ivnums_"));
//         this->ovnums_.Construct(meta.GetMemberMeta("ovnums_"));
//         this->tvnums_.Construct(meta.GetMemberMeta("tvnums_"));
//         for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__vertex_tables_-size"); ++__idx) {
//             this->vertex_tables_.emplace_back(std::dynamic_pointer_cast<Table>(
//                     meta.GetMember("__vertex_tables_-" + std::to_string(__idx))));
//         }
//         for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__ovgid_lists_-size"); ++__idx) {
//             this->ovgid_lists_.emplace_back(std::dynamic_pointer_cast<ArrowFragment::vid_vineyard_array_t>(
//                     meta.GetMember("__ovgid_lists_-" + std::to_string(__idx))));
//         }
//         for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__ovg2l_maps_-size"); ++__idx) {
//             this->ovg2l_maps_.emplace_back(std::dynamic_pointer_cast<Hashmap<vid_t, vid_t>>(
//                     meta.GetMember("__ovg2l_maps_-" + std::to_string(__idx))));
//         }
//         for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__edge_tables_-size"); ++__idx) {
//             this->edge_tables_.emplace_back(std::dynamic_pointer_cast<Table>(
//                     meta.GetMember("__edge_tables_-" + std::to_string(__idx))));
//         }
//         this->ie_lists_.resize(meta.GetKeyValue<size_t>("__ie_lists_-size"));
//         for (size_t __idx = 0; __idx < this->ie_lists_.size(); ++__idx) {
//             for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
//                     "__ie_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
//                 this->ie_lists_[__idx].emplace_back(std::dynamic_pointer_cast<FixedSizeBinaryArray>(
//                     meta.GetMember("__ie_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
//             }
//         }
//         this->oe_lists_.resize(meta.GetKeyValue<size_t>("__oe_lists_-size"));
//         for (size_t __idx = 0; __idx < this->oe_lists_.size(); ++__idx) {
//             for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
//                     "__oe_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
//                 this->oe_lists_[__idx].emplace_back(std::dynamic_pointer_cast<FixedSizeBinaryArray>(
//                     meta.GetMember("__oe_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
//             }
//         }
//         this->ie_offsets_lists_.resize(meta.GetKeyValue<size_t>("__ie_offsets_lists_-size"));
//         for (size_t __idx = 0; __idx < this->ie_offsets_lists_.size(); ++__idx) {
//             for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
//                     "__ie_offsets_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
//                 this->ie_offsets_lists_[__idx].emplace_back(std::dynamic_pointer_cast<Int64Array>(
//                     meta.GetMember("__ie_offsets_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
//             }
//         }
//         this->oe_offsets_lists_.resize(meta.GetKeyValue<size_t>("__oe_offsets_lists_-size"));
//         for (size_t __idx = 0; __idx < this->oe_offsets_lists_.size(); ++__idx) {
//             for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
//                     "__oe_offsets_lists_-" + std::to_string(__idx) + "-size"); ++__idy) {
//                 this->oe_offsets_lists_[__idx].emplace_back(std::dynamic_pointer_cast<Int64Array>(
//                     meta.GetMember("__oe_offsets_lists_-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
//             }
//         }
//         this->vm_ptr_ = std::dynamic_pointer_cast<ArrowFragment::vertex_map_t>(meta.GetMember("vm_ptr_"));
//         meta.GetKeyValue("schema_json_", this->schema_json_);

        
//         if (meta.IsLocal()) {
//             this->PostConstruct(meta);
//         }
//     }

//  private:
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
 // hide vertex_map
 // vineyard::ObjectID vertex_map_id() const override { return vm_ptr_->id(); }

  void init(void* partitioned_graph) {
    pg_ = partitioned_graph;
    assert(get_partition_list_size(pg_) == 1);
    auto pl = get_local_partitions(pg_);
    auto p = get_partition_from_list(pl, 0);
    g_ = get_local_graph_from_partition(pg_, p);

    directed_ = is_directed(g_);
    is_multigraph_ = is_multigraph(g_);
    fid_ = p;
    fnum_ = get_total_partitions_number(pg_);
  }

  bool directed() const override {
    return directed_;
  }

  bool is_multigraph() const override {
    return is_multigraph_; 
  }

  const std::string vid_typename() const override { 
    auto dt = get_vertex_id_data_type(g_);
    return GetDataTypeName(dt);
  }

  const std::string oid_typename() const override { 
    auto dt = DataTypeEnum<oid_t>::value();
    return GetDataTypeName(dt); 
  }

  fid_t fid() const { return fid_; }

  fid_t fnum() const { return fnum_; }

  void* vertex_label(const void* v) {
    return get_vertex_label(g_, v);
  }

  label_id_t vertex_label(const vertex_t& v) const {
    void* _v = get_vertex_from_id(v.GetValue());
    void* _label = vertex_label(_v);
    void* _label_id = get_label_id(g_, _label);
    return *(static_cast<label_id_t*>(_label_id))
  }

//   int64_t vertex_offset(const vertex_t& v) const {
//     // to remove ----
//     return vid_parser_.GetOffset(v.GetValue());
//   }

  label_id_t vertex_label_num() const { 
    void* vll = get_vertex_labels(g_);
    return get_vertex_label_list_size(vll);
  }

  label_id_t edge_label_num() const { 
    void* ell = get_edge_labels(g_);
    return get_edge_label_list_size(ell);
  }

  prop_id_t vertex_property_num(label_id_t label) const {
    void* _label = get_vertex_label_from_id((void*)(&label));
    void* vpl = get_all_vertex_properties_from_label(g_, _label);
    return get_property_list_size(vpl);
  }

  std::shared_ptr<arrow::DataType> vertex_property_type(label_id_t label,
                                                        prop_id_t prop) const {
    void* _label = get_vertex_label_from_id(label)
    void* _property = get_vertex_property_from_id(_label, prop);
    auto dt = get_property_type(_g, _property);
    return GetArrowDataType(dt);
  }

  prop_id_t edge_property_num(label_id_t label) const {
    void* _label = get_edge_label_from_id((void*)(&label));
    void* epl = get_all_edge_properties_from_label(g_, _label);
    return get_property_list_size(epl);
  }

  std::shared_ptr<arrow::DataType> edge_property_type(label_id_t label,
                                                      prop_id_t prop) const {
    void* _label = get_edge_label_from_id(label)
    void* _property = get_edge_property_from_id(_label, prop);
    auto dt = get_property_type(_g, _property);
    return GetArrowDataType(dt);
  }

//   std::shared_ptr<arrow::Table> vertex_data_table(label_id_t i) const {
//     // maybe we should get rid of the method
//     // there is no way we can provide the whole data with a C-style api
//     return vertex_tables_[i]->GetTable();
//   }

//   std::shared_ptr<arrow::Table> edge_data_table(label_id_t i) const {
//     // Ditto.
//     return edge_tables_[i]->GetTable();
//   }

//   template <typename DATA_T>
//   property_graph_utils::EdgeDataColumn<DATA_T, nbr_unit_t> edge_data_column(
//       label_id_t label, prop_id_t prop) const {
//     // get rid of this method and EdgeDataColumn structure
//     // this structure actually serves to get a specific property of an edge
//     // and it can be replaced by grin property get_edge_row
//     if (edge_tables_[label]->num_rows() == 0) {
//       return property_graph_utils::EdgeDataColumn<DATA_T, nbr_unit_t>();
//     } else {
//       // the finalized etables are guaranteed to have been concatenated
//       return property_graph_utils::EdgeDataColumn<DATA_T, nbr_unit_t>(
//           edge_tables_[label]->column(prop)->chunk(0));
//     }
//   }

//   template <typename DATA_T>
//   property_graph_utils::VertexDataColumn<DATA_T, vid_t> vertex_data_column(
//       label_id_t label, prop_id_t prop) const {
//     // Ditto. it can be replaced by grin property get_vertex_row && get_property_value_from_row
//     if (vertex_tables_[label]->num_rows() == 0) {
//       return property_graph_utils::VertexDataColumn<DATA_T, vid_t>(
//           InnerVertices(label));
//     } else {
//       // the finalized vtables are guaranteed to have been concatenated
//       return property_graph_utils::VertexDataColumn<DATA_T, vid_t>(
//           InnerVertices(label), vertex_tables_[label]->column(prop)->chunk(0));
//     }
//   }

  vertex_range_t Vertices(label_id_t label_id) const {
    //continuous vid trait
    void* _label = get_vertex_label_from_id((void*)(&label_id));
    void* vlh = get_vertex_list_by_label(pg_, fid_, _label);
    void* beginh = get_begin_vertex_id_from_list(vlh);
    VID_T* begin = static_cast<VID_T*>(beginh);
    void* endh = get_end_vertex_id_from_list(vlh);
    VID_T* end = static_cast<VID_T*>(endh);
    return vertex_range_t(*begin, *end);
  }

  vertex_range_t InnerVertices(label_id_t label_id) const {
    //continuous vid trait
    void* _label = get_vertex_label_from_id((void*)(&label_id));
    void* vlh = get_local_vertices_by_label(pg_, fid_, _label);
    void* beginh = get_begin_vertex_id_from_list(vlh);
    VID_T* begin = static_cast<VID_T*>(beginh);
    void* endh = get_end_vertex_id_from_list(vlh);
    VID_T* end = static_cast<VID_T*>(endh);
    return vertex_range_t(*begin, *end);
  }

  vertex_range_t OuterVertices(label_id_t label_id) const {
    //continuous vid trait
    void* _label = get_vertex_label_from_id((void*)(&label_id));
    void* vlh = get_remote_vertices_by_label(pg_, fid_, _label);
    void* beginh = get_begin_vertex_id_from_list(vlh);
    VID_T* begin = static_cast<VID_T*>(beginh);
    void* endh = get_end_vertex_id_from_list(vlh);
    VID_T* end = static_cast<VID_T*>(endh);
    return vertex_range_t(*begin, *end);
  }

  vertex_range_t InnerVerticesSlice(label_id_t label_id, vid_t start, vid_t end)
      const {
    // continuous_vid_traits
    vertex_range_t vr = InnerVertices(label_id);
    size_t _end = vr.size();
    CHECK(start <= end && start <= _end);
    if (end <= _end) {
      return vr.SetRange(start, end);
    } else {
      return vr.SetRange(start, _end);
    }
  }

  inline vid_t GetVerticesNum(label_id_t label_id) const {
    void* _label = get_vertex_label_from_id((void*)(&label_id));
    void* vlh = get_vertex_list_by_label(pg_, fid_, _label);
    return get_vertex_list_size(vlh);
  }

  bool GetVertex(label_id_t label, const oid_t& oid, vertex_t& v) const {
    void* _label = get_vertex_label_from_id((void*)(&label));
    void* _v = get_vertex_from_label_origin_id(_label, (void*)(&oid));
    if (_v == NULL_VERTEX) {
        return false;
    }
    void* _id = get_vertex_id(_v);
    v.SetValue(*(static_cast<vid_t*>(_id)));
    return true;
  }

  oid_t GetId(const vertex_t& v) const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    void* _id = get_vertex_origin_id(_v);
    return *(static_cast<oid_t*>(_id));
  }

//   internal_oid_t GetInternalId(const vertex_t& v) const {
//     return IsInnerVertex(v) ? GetInnerVertexInternalId(v)
//                             : GetOuterVertexInternalId(v);
//   }

  fid_t GetFragId(const vertex_t& u) const {
    auto rp = get_master_partition_for_vertex(pg_, fid_, (void*)(&u));
    if (rp == NULL_REMOTE_PARTITION) {
      return fid_;
    }
    return rp;
  }

  size_t GetTotalNodesNum() const {
    // secondary
    return GetTotalVerticesNum()
  }
  size_t GetTotalVerticesNum() const {
    void* vl = get_vertex_list(g_);
    return get_vertex_list_size(vl);
  }
  size_t GetTotalVerticesNum(label_id_t label) const {
    void* _label = get_vertex_label_from_id((void*)(&label));
    void* vl = get_vertex_list_by_label(_label);
    return get_vertex_list_size(vl);
  }

  // secondary
  size_t GetEdgeNum() const { return directed_ ? oenum_ + ienum_ : oenum_; }
  // secondary
  size_t GetInEdgeNum() const { return ienum_; }
  // secondary
  size_t GetOutEdgeNum() const { return oenum_; }

  template <typename T>
  T GetData(const vertex_t& v, prop_id_t prop_id) const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    void* _label = get_vertex_label(g_, _v);
    void* _property = get_vertex_property_from_id(_lable, (void*)(&prop_id));
    void* _pl = get_all_vertex_properties_from_label(g_, _label);
    void* _row = get_vertex_row(g_, _v, _pl);
    void* _value = get_property_value_from_row(_row, _property);
    return *(static_cast<T*>(_value));
  }

  bool HasChild(const vertex_t& v, label_id_t e_label) const {
    // secondary
    return GetLocalOutDegree(v, e_label) != 0;
  }

  bool HasParent(const vertex_t& v, label_id_t e_label) const {
    // secondary
    return GetLocalInDegree(v, e_label) != 0;
  }

  int GetLocalOutDegree(const vertex_t& v, label_id_t e_label) const {
    // secondary
    return GetOutgoingAdjList(v, e_label).Size();
  }

  int GetLocalInDegree(const vertex_t& v, label_id_t e_label) const {
    // secondary
    return GetIncomingAdjList(v, e_label).Size();
  }

  // FIXME: grape message buffer compatibility
  bool Gid2Vertex(const vid_t& gid, vertex_t& v) const {
    std::stringstream ss;
    ss << gid;
    void* vh = get_vertex_from_deserialization(pg_, fid_, ss.str().c_str());
    if (vh == NULL_VERTEX) {
      return false;
    }
    vertex_t* _v = static_cast<vertex_t*>(vh);
    v.SetValue(_v->GetValue());
    return true;
  }

  vid_t Vertex2Gid(const vertex_t& v) const {
    // secondary
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

  inline vid_t GetInnerVerticesNum(label_id_t label_id) const {
    void* _label = get_vertex_label_from_id((void*)(&label_id));
    void* vlh = get_local_vertices_by_label(pg_, fid_, _label);
    return get_vertex_list_size(vlh);  
  }

  inline vid_t GetOuterVerticesNum(label_id_t label_id) const {
    void* _label = get_vertex_label_from_id((void*)(&label_id));
    void* vlh = get_remote_vertices_by_label(pg_, fid_, _label);
    return get_vertex_list_size(vlh);
  }

  inline bool IsInnerVertex(const vertex_t& v) const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    return is_local_vertex(pg_, fid_, _v);
  }

  inline bool IsOuterVertex(const vertex_t& v) const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    return is_remote_vertex(pg_, fid_, _v);
  }

  bool GetInnerVertex(label_id_t label, const oid_t& oid, vertex_t& v) const {
    return GetVertex(label, oid, v);
    // vid_t gid;
    // if (vm_ptr_->GetGid(label, internal_oid_t(oid), gid)) {
    //   if (vid_parser_.GetFid(gid) == fid_) {
    //     v.SetValue(vid_parser_.GetLid(gid));
    //     return true;
    //   }
    // }
    // return false;
  }

  bool GetOuterVertex(label_id_t label, const oid_t& oid, vertex_t& v) const {
    return GetVertex(label, oid, v);
    // vid_t gid;
    // if (vm_ptr_->GetGid(label, internal_oid_t(oid), gid)) {
    //   return OuterVertexGid2Vertex(gid, v);
    // }
    // return false;
  }

  inline oid_t GetVertexOriginId(const vertex_t& v) const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    void* _id = get_vertex_origin_id(g_, _v);
    return *(static_cast<oid_t*>(_id));
  }

  inline oid_t GetInnerVertexId(const vertex_t& v) const {
    return GetVertexOriginId(v);
 //   return oid_t(GetInnerVertexInternalId(v));
  }

//   inline internal_oid_t GetInnerVertexInternalId(const vertex_t& v) const {
//     internal_oid_t internal_oid;
//     vid_t gid =
//         vid_parser_.GenerateId(fid_, vid_parser_.GetLabelId(v.GetValue()),
//                                vid_parser_.GetOffset(v.GetValue()));
//     CHECK(vm_ptr_->GetOid(gid, internal_oid));
//     return internal_oid;
//   }

  inline oid_t GetOuterVertexId(const vertex_t& v) const {
    return GetVertexOriginId(v);
 //   return oid_t(GetOuterVertexInternalId(v));
  }

//   inline internal_oid_t GetOuterVertexInternalId(const vertex_t& v) const {
//     vid_t gid = GetOuterVertexGid(v);
//     internal_oid_t internal_oid;
//     CHECK(vm_ptr_->GetOid(gid, internal_oid));
//     return internal_oid;
//   }

  inline oid_t Gid2Oid(const vid_t& gid) const {
    vertex_t v;
    Gid2Vertex(gid, v);
    return GetVertexOriginId(v);
    // internal_oid_t internal_oid;
    // CHECK(vm_ptr_->GetOid(gid, internal_oid));
    // return oid_t(internal_oid);
  }

  inline bool Oid2Gid(label_id_t label, const oid_t& oid, vid_t& gid) const {
    vertex_t v;
    if (!GetVertex(label, oid, v)) {
        return false;
    }
    gid = Vertex2Gid(v);
    return true;
 //   return vm_ptr_->GetGid(label, internal_oid_t(oid), gid);
  }

  inline bool Oid2Gid(label_id_t label, const oid_t& oid, vertex_t& v) const {
    vid_t gid;
//    if (vm_ptr_->GetGid(label, internal_oid_t(oid), gid)) {
    if (Oid2Gid(label, oid, gid)) {
      v.SetValue(gid);
      return true;
    }
    return false;
  }

  inline bool InnerVertexGid2Vertex(const vid_t& gid, vertex_t& v) const {
    return Gid2Vertex(gid, v);
    // v.SetValue(vid_parser_.GetLid(gid));
    // return true;
  }

  inline bool OuterVertexGid2Vertex(const vid_t& gid, vertex_t& v) const {
    return Gid2Vertex(gid, v);
    // auto map = ovg2l_maps_ptr_[vid_parser_.GetLabelId(gid)];
    // auto iter = map->find(gid);
    // if (iter != map->end()) {
    //   v.SetValue(iter->second);
    //   return true;
    // } else {
    //   return false;
    // }
  }

  inline vid_t GetOuterVertexGid(const vertex_t& v) const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    void* _mv = get_master_vertex_for_vertex(pg_, fid_, _v);
    void* _id = get_vertex_id(_mv);
    return *(static_cast<vid_t*>(_id));
    // label_id_t v_label = vid_parser_.GetLabelId(v.GetValue());
    // return ovgid_lists_ptr_[v_label][vid_parser_.GetOffset(v.GetValue()) -
    //                                  static_cast<int64_t>(ivnums_[v_label])];
  }
  inline vid_t GetInnerVertexGid(const vertex_t& v) const {
    std::stringstream ss(serialize_remote_vertex(pg_, (void*)(&v)));
    VID_T gid;
    ss >> gid;
    return gid;
    // return vid_parser_.GenerateId(fid_, vid_parser_.GetLabelId(v.GetValue()),
    //                               vid_parser_.GetOffset(v.GetValue()));
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v, label_id_t e_label)
      const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    void* _label = get_edge_label_from_id((void*)(&e_label));
    void* al = get_adjacent_list_by_edge_label(g_, Direction::IN, (void*)(&v), _label);
    return adj_list_t(g_, _label, al, get_adjacent_list_size(al));
    // // grin vertexlist continous_vid_trait get_vertex_from_vid ++++
    // vid_t vid = v.GetValue();
    // label_id_t v_label = vid_parser_.GetLabelId(vid);
    // int64_t v_offset = vid_parser_.GetOffset(vid);
    // const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
    // const nbr_unit_t* ie = ie_ptr_lists_[v_label][e_label];
    // return adj_list_t(&ie[offset_array[v_offset]],
    //                   &ie[offset_array[v_offset + 1]],
    //                   flatten_edge_tables_columns_[e_label]);
  }

//   inline raw_adj_list_t GetIncomingRawAdjList(const vertex_t& v,
//                                               label_id_t e_label) const {
//     vid_t vid = v.GetValue();
//     label_id_t v_label = vid_parser_.GetLabelId(vid);
//     int64_t v_offset = vid_parser_.GetOffset(vid);
//     const int64_t* offset_array = ie_offsets_ptr_lists_[v_label][e_label];
//     const nbr_unit_t* ie = ie_ptr_lists_[v_label][e_label];
//     return raw_adj_list_t(&ie[offset_array[v_offset]],
//                           &ie[offset_array[v_offset + 1]]);
//   }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v, label_id_t e_label)
      const {
    void* _v = get_vertex_from_id((void*)(&v.GetValue()));
    void* _label = get_edge_label_from_id((void*)(&e_label));
    void* al = get_adjacent_list_by_edge_label(g_, Direction::OUT, (void*)(&v), _label);
    return adj_list_t(g_, _label, al, get_adjacent_list_size(al));
    // vid_t vid = v.GetValue();
    // label_id_t v_label = vid_parser_.GetLabelId(vid);
    // int64_t v_offset = vid_parser_.GetOffset(vid);
    // const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
    // const nbr_unit_t* oe = oe_ptr_lists_[v_label][e_label];
    // return adj_list_t(&oe[offset_array[v_offset]],
    //                   &oe[offset_array[v_offset + 1]],
    //                   flatten_edge_tables_columns_[e_label]);
  }

//   inline raw_adj_list_t GetOutgoingRawAdjList(const vertex_t& v,
//                                               label_id_t e_label) const {
//     vid_t vid = v.GetValue();
//     label_id_t v_label = vid_parser_.GetLabelId(vid);
//     int64_t v_offset = vid_parser_.GetOffset(vid);
//     const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
//     const nbr_unit_t* oe = oe_ptr_lists_[v_label][e_label];
//     return raw_adj_list_t(&oe[offset_array[v_offset]],
//                           &oe[offset_array[v_offset + 1]]);
//   }

//   /**
//    * N.B.: as an temporary solution, for POC of graph-learn, will be removed
//    * later.
//    */

//   inline const int64_t* GetIncomingOffsetArray(label_id_t v_label,
//                                                label_id_t e_label) const {
//     return ie_offsets_ptr_lists_[v_label][e_label];
//   }

//   inline const int64_t* GetOutgoingOffsetArray(label_id_t v_label,
//                                                label_id_t e_label) const {
//     return oe_offsets_ptr_lists_[v_label][e_label];
//   }

//   inline int64_t GetIncomingOffsetLength(label_id_t v_label, label_id_t e_label)
//       const {
//     return ie_offsets_lists_[v_label][e_label]->length();
//   }

//   inline int64_t GetOutgoingOffsetLength(label_id_t v_label, label_id_t e_label)
//       const {
//     return oe_offsets_lists_[v_label][e_label]->length();
//   }

//   inline std::pair<int64_t, int64_t> GetOutgoingAdjOffsets(
//       const vertex_t& v, label_id_t e_label) const {
//     vid_t vid = v.GetValue();
//     label_id_t v_label = vid_parser_.GetLabelId(vid);
//     int64_t v_offset = vid_parser_.GetOffset(vid);
//     const int64_t* offset_array = oe_offsets_ptr_lists_[v_label][e_label];
//     return std::make_pair(offset_array[v_offset], offset_array[v_offset + 1]);
//   }

  inline grape::DestList IEDests(const vertex_t& v, label_id_t e_label) const {
    void* _v = get_vertex_from_id((void*)(v.GetValue()));
    void* _vertex_label = get_vertex_label(_v);
    int64_t offset = v.GetValue() - InnerVertices(_vertex_label).begin_value();

    return grape::DestList(idoffset_[v_label][e_label][offset],
                           idoffset_[v_label][e_label][offset + 1]);
  }

  inline grape::DestList OEDests(const vertex_t& v, label_id_t e_label) const {
    void* _v = get_vertex_from_id((void*)(v.GetValue()));
    void* _vertex_label = get_vertex_label(_v);
    int64_t offset = v.GetValue() - InnerVertices(_vertex_label).begin_value();

    return grape::DestList(odoffset_[v_label][e_label][offset],
                           odoffset_[v_label][e_label][offset + 1]);
  }

  inline grape::DestList IOEDests(const vertex_t& v, label_id_t e_label) const {
    void* _v = get_vertex_from_id((void*)(v.GetValue()));
    void* _vertex_label = get_vertex_label(_v);
    int64_t offset = v.GetValue() - InnerVertices(_vertex_label).begin_value();

    return grape::DestList(iodoffset_[v_label][e_label][offset],
                           iodoffset_[v_label][e_label][offset + 1]);
  }

//  std::shared_ptr<vertex_map_t> GetVertexMap() { return vm_ptr_; }

// const PropertyGraphSchema& schema() const override { return schema_; }

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
      std::map<label_id_t, std::vector<label_id_t>> edges) {
        void* new_g;
        for (auto& pair : vertices) {
            void* vertex_label = get_vetex_label_from_id((void*)(&pair.first));
            void* property_list = create_property_list();
            for (auto& prop_id: pair.second) {
                void* property = get_property_by_id(vertex_label, proper_id);
                insert_property_to_list(property_list, property);
            }
            new_g = select_vertex_properties_for_label(g_, propertylist, vertex_label);
        }
        // same for edge
      }
    // use grin property "select"
 private:
  void initPointers();

  void initDestFidList(
      bool in_edge, bool out_edge,
      std::vector<std::vector<std::vector<fid_t>>>& fid_lists,
      std::vector<std::vector<std::vector<fid_t*>>>& fid_lists_offset);

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

#endif // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H
