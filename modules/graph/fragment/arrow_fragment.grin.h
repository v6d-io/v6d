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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H

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

#include "graph/grin/include/topology/structure.h"
#include "graph/grin/include/topology/vertexlist.h"
#include "graph/grin/include/topology/adjacentlist.h"
#include "graph/grin/include/partition/partition.h"
#include "graph/grin/include/partition/topology.h"
#include "graph/grin/include/partition/reference.h"
#include "graph/grin/include/property/type.h"
#include "graph/grin/include/property/property.h"
#include "graph/grin/include/property/propertylist.h"
#include "graph/grin/include/property/propertytable.h"
#include "graph/grin/include/property/topology.h"
#include "graph/grin/include/index/order.h"

#include "graph/grin/src/predefine.h"

namespace vineyard {

std::shared_ptr<arrow::DataType> GetArrowDataType(GRIN_DATATYPE type) {
  switch (type) {
  case GRIN_DATATYPE::Undefined:
    return arrow::null();
  case GRIN_DATATYPE::Int32:
    return arrow::int32();
  case GRIN_DATATYPE::UInt32:
    return arrow::uint32();
  case GRIN_DATATYPE::Int64:
    return arrow::int64();
  case GRIN_DATATYPE::UInt64:
    return arrow::uint64();
  case GRIN_DATATYPE::Float:
    return arrow::float32();
  case GRIN_DATATYPE::Double:
    return arrow::float64();
  case GRIN_DATATYPE::String:
    return arrow::large_utf8();
  case GRIN_DATATYPE::Date32:
    return arrow::int32();
  case GRIN_DATATYPE::Date64:
    return arrow::int64();
  default:
    return arrow::null();
  }
}


struct GRIN_Nbr {
 public:
  GRIN_Nbr() : g_{nullptr}, al_(nullptr), cur_(0), ept_(nullptr) {}
  GRIN_Nbr(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t cur, GRIN_EDGE_PROPERTY_TABLE ept) 
    : g_{g}, al_(al), cur_(cur), ept_(ept) {}
  GRIN_Nbr(GRIN_Nbr& rhs) : g_{rhs.g_}, al_(rhs.al_), cur_(rhs.cur_), ept_(rhs.ept_) {}

  GRIN_Nbr& operator=(const GRIN_Nbr& rhs) {
    g_ = rhs.g_;
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    ept_ = rhs.ept_;
    return *this;
  }

  GRIN_Nbr& operator=(GRIN_Nbr&& rhs) {
    g_ = rhs.g_;
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    ept_ = rhs.ept_;
    return *this;
  }

  GRIN_VERTEX neighbor() {
    return grin_get_neighbor_from_adjacent_list(g_, al_, cur_);
  }

  GRIN_VERTEX get_neighbor() {
    return grin_get_neighbor_from_adjacent_list(g_, al_, cur_);
  }

  GRIN_EDGE get_edge() {
    return grin_get_edge_from_adjacent_list(g_, al_, cur_);
  }

  template <typename T>
  T get_data(GRIN_EDGE_PROPERTY prop) const {
    auto _e = grin_get_edge_from_adjacent_list(g_, al_, cur_);
    auto value = grin_get_value_from_edge_property_table(g_, ept_, _e, prop);
    return property_graph_utils::ValueGetter<T>::Value(value, 0);
  }

  std::string get_str(GRIN_EDGE_PROPERTY prop) const {
    auto _e = grin_get_edge_from_adjacent_list(g_, al_, cur_);
    auto value = grin_get_value_from_edge_property_table(g_, ept_, _e, prop);
    return property_graph_utils::ValueGetter<std::string>::Value(value, 0);
  }

  double get_double(GRIN_EDGE_PROPERTY prop) const {
    auto _e = grin_get_edge_from_adjacent_list(g_, al_, cur_);
    auto value = grin_get_value_from_edge_property_table(g_, ept_, _e, prop);
    return property_graph_utils::ValueGetter<double>::Value(value, 0);
  }

  int64_t get_int(GRIN_EDGE_PROPERTY prop) const {
    auto _e = grin_get_edge_from_adjacent_list(g_, al_, cur_);
    auto value = grin_get_value_from_edge_property_table(g_, ept_, _e, prop);
    return property_graph_utils::ValueGetter<int64_t>::Value(value, 0);
  }

  inline GRIN_Nbr& operator++() {
    cur_++;
    return *this;
  }

  inline GRIN_Nbr operator++(int) {
    cur_++;
    return *this;
  }

  inline GRIN_Nbr& operator--() {
    cur_--;
    return *this;
  }

  inline GRIN_Nbr operator--(int) {
    cur_--;
    return *this;
  }

  inline bool operator==(const GRIN_Nbr& rhs) const { 
    return al_ == rhs.al_ && cur_ == rhs.cur_; 
  }
  inline bool operator!=(const GRIN_Nbr& rhs) const { 
    return al_ != rhs.al_ || cur_ != rhs.cur_;
  }

  inline bool operator<(const GRIN_Nbr& rhs) const {
    return al_ == rhs.al_ && cur_ < rhs.cur_; 
  }

  inline GRIN_Nbr& operator*() { return *this; }

 private:
  GRIN_GRAPH g_;
  GRIN_ADJACENT_LIST al_;
  size_t cur_;
  GRIN_EDGE_PROPERTY_TABLE ept_;
};


class GRIN_AdjList {
 public:
  GRIN_AdjList(): g_(nullptr), adj_list_(nullptr), ept_(nullptr), begin_(0), end_(0) {}
  GRIN_AdjList(GRIN_GRAPH g, GRIN_ADJACENT_LIST adj_list, GRIN_EDGE_PROPERTY_TABLE ept, size_t begin, size_t end) 
    : g_{g}, adj_list_(adj_list), ept_(ept), begin_(begin), end_(end) {}

  inline GRIN_Nbr begin() const {
    return GRIN_Nbr(g_, adj_list_, begin_, ept_);
  }

  inline GRIN_Nbr end() const {
    return GRIN_Nbr(g_, adj_list_, end_, ept_);
  }

  inline size_t Size() const { return end_ - begin_; }

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return begin_ < end_; }

  size_t size() const { return end_ - begin_; }

 private:
  GRIN_GRAPH g_;
  GRIN_ADJACENT_LIST adj_list_;
  GRIN_EDGE_PROPERTY_TABLE ept_;
  size_t begin_;
  size_t end_;
};


class GRIN_VertexRange {
 public:
  GRIN_VertexRange() {}
  GRIN_VertexRange(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, const size_t begin, const size_t end)
      : g_(g), vl_(vl), begin_(begin), end_(end) {}
  GRIN_VertexRange(const GRIN_VertexRange& r) : g_(r.g_), vl_(r.vl_), begin_(r.begin_), end_(r.end_) {}

  class iterator {
    using reference_type = GRIN_VERTEX;

   private:
    GRIN_GRAPH g_;
    GRIN_VERTEX_LIST vl_;
    size_t cur_;

   public:
    iterator() noexcept : g_(nullptr), vl_(nullptr), cur_() {}
    explicit iterator(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t idx) noexcept : g_(g), vl_(vl), cur_(idx) {}

    reference_type operator*() noexcept { return grin_get_vertex_from_list(g_, vl_, cur_); }

    iterator& operator++() noexcept {
      ++cur_;
      return *this;
    }

    iterator operator++(int) noexcept {
      return iterator(g_, vl_, cur_ + 1);
    }

    iterator& operator--() noexcept {
      --cur_;
      return *this;
    }

    iterator operator--(int) noexcept {
      return iterator(g_, vl_, cur_--);
    }

    iterator operator+(size_t offset) const noexcept {
      return iterator(g_, vl_, cur_ + offset);
    }

    bool operator==(const iterator& rhs) const noexcept {
      return cur_ == rhs.cur_;
    }

    bool operator!=(const iterator& rhs) const noexcept {
      return cur_ != rhs.cur_;
    }

    bool operator<(const iterator& rhs) const noexcept {
      return vl_ == rhs.vl_ && cur_ < rhs.cur_;
    }
  };

  iterator begin() const { return iterator(g_, vl_, begin_); }

  iterator end() const { return iterator(g_, vl_, end_); }

  size_t size() const { return end_ - begin_; }

  void Swap(GRIN_VertexRange& rhs) {
    std::swap(begin_, rhs.begin_);
    std::swap(end_, rhs.end_);
  }

  void SetRange(const size_t begin, const size_t end) {
    begin_ = begin;
    end_ = end;
  }

  const size_t begin_value() const { return begin_; }

  const size_t end_value() const { return end_; }

 private:
  GRIN_GRAPH g_;
  GRIN_VERTEX_LIST vl_;
  size_t begin_;
  size_t end_;
};


class GRIN_ArrowFragment {
 public:
  using vertex_range_t = GRIN_VertexRange;
  using adj_list_t = GRIN_AdjList;

 public:
  ~GRIN_ArrowFragment() = default;

  void init(GRIN_PARTITIONED_GRAPH partitioned_graph, GRIN_PARTITION partition) {
    pg_ = partitioned_graph;
    partition_ = partition;
    g_ = grin_get_local_graph_from_partition(pg_, partition_);
  }

  bool directed() const {
    return grin_is_directed(g_);
  }

  bool multigraph() const {
    return grin_is_multigraph(g_);
  }

  const std::string oid_typename() const { 
    auto dt = grin_get_vertex_original_id_type(g_);
    return GetDataTypeName(dt);
  }

  size_t fnum() const { return grin_get_total_partitions_number(pg_); }

  GRIN_VERTEX_TYPE vertex_label(GRIN_VERTEX v) {
    return grin_get_vertex_type(g_, v);
  }

  size_t vertex_label_num() const { 
    auto vtl = grin_get_vertex_type_list(g_);
    return grin_get_vertex_type_list_size(g_, vtl);
  }

  size_t edge_label_num() const { 
    auto etl = grin_get_edge_type_list(g_);
    return grin_get_edge_type_list_size(g_, etl);
  }

  size_t vertex_property_num(GRIN_VERTEX_TYPE label) const {
    auto vpl = grin_get_vertex_property_list_by_type(g_, label);
    return grin_get_vertex_property_list_size(g_, vpl);
  }

  std::shared_ptr<arrow::DataType> vertex_property_type(GRIN_VERTEX_PROPERTY prop) const {
    auto dt = grin_get_vertex_property_data_type(g_, prop);
    return GetArrowDataType(dt);
  }
  
  size_t edge_property_num(GRIN_EDGE_TYPE label) const {
    auto epl = grin_get_edge_property_list_by_type(g_, label);
    return grin_get_edge_property_list_size(g_, epl);
  }

  std::shared_ptr<arrow::DataType> edge_property_type(GRIN_EDGE_PROPERTY prop) const {
    auto dt = grin_get_edge_property_data_type(g_, prop);
    return GetArrowDataType(dt);
  }

  vertex_range_t Vertices(GRIN_VERTEX_TYPE label) const {
    auto vl = grin_get_vertex_list(g_);
    auto vl1 = grin_filter_type_for_vertex_list(g_, label, vl);
    auto sz = grin_get_vertex_list_size(g_, vl1);
    return vertex_range_t(g_, vl1, 0, sz);
  }

  vertex_range_t InnerVertices(GRIN_VERTEX_TYPE label) const {
    auto vl = grin_get_vertex_list(g_);
    auto vl1 = grin_filter_type_for_vertex_list(g_, label, vl);
    auto vl2 = grin_filter_master_for_vertex_list(g_, vl1);
    auto sz = grin_get_vertex_list_size(g_, vl2);
    return vertex_range_t(g_, vl2, 0, sz);
  }

  vertex_range_t OuterVertices(GRIN_VERTEX_TYPE label) const {
    auto vl = grin_get_vertex_list(g_);
    auto vl1 = grin_filter_type_for_vertex_list(g_, label, vl);
    auto vl2 = grin_filter_mirror_for_vertex_list(g_, vl1);
    auto sz = grin_get_vertex_list_size(g_, vl2);
    return vertex_range_t(g_, vl2, 0, sz);
  }

  vertex_range_t InnerVerticesSlice(GRIN_VERTEX_TYPE label, size_t start, size_t end)
      const {
    auto vl = grin_get_vertex_list(g_);
    auto vl1 = grin_filter_type_for_vertex_list(g_, label, vl);
    auto vl2 = grin_filter_master_for_vertex_list(g_, vl1);
    auto _end = grin_get_vertex_list_size(g_, vl2);
    CHECK(start <= end && start <= _end);
    if (end <= _end) {
      return vertex_range_t(g_, vl2, start, end);
    } else {
      return vertex_range_t(g_, vl2, start, _end);
    }
  }

  inline size_t GetVerticesNum(GRIN_VERTEX_TYPE label) const {
    return grin_get_vertex_num_by_type(g_, label);
  }

  template <typename T>
  bool GetVertex(GRIN_VERTEX_TYPE label, T& oid, GRIN_VERTEX v) {
    if (GRIN_DATATYPE_ENUM<T>::value != grin_get_vertex_original_id_type(g_)) return false;
    v = grin_get_vertex_from_original_id_by_type(g_, label, (GRIN_VERTEX_ORIGINAL_ID)(&oid));
    return v != NULL;
  }

  template <typename T>
  bool GetId(GRIN_VERTEX v, T& oid) {
    if (GRIN_DATATYPE_ENUM<T>::value != grin_get_vertex_original_id_type(g_)) return false;
    auto _id = grin_get_vertex_original_id(g_, v);
    if (_id == NULL) return false;
    auto _oid = static_cast<T*>(_id);
    oid = *_oid;
    return true;
  }

#ifdef GRIN_NATURAL_PARTITION_ID_TRAIT
  GRIN_PARTITION_ID GetPartition(GRIN_VERTEX u) const {
    auto vref = grin_get_vertex_ref_for_vertex(g_, u);
    auto partition = grin_get_master_partition_from_vertex_ref(g_, vref); 
    return grin_get_partition_id(pg_, partition);
  }
#endif

  size_t GetTotalNodesNum() const {
    return GetTotalVerticesNum();
  }
  
  size_t GetTotalVerticesNum() const {
    return grin_get_vertex_num(g_);
  }

  size_t GetTotalVerticesNum(GRIN_VERTEX_TYPE label) const {
    return grin_get_vertex_num_by_type(g_, label);
  }

  size_t GetEdgeNum() const { return grin_get_edge_num(g_, GRIN_DIRECTION::BOTH); }
  size_t GetInEdgeNum() const { return grin_get_edge_num(g_, GRIN_DIRECTION::IN); }
  size_t GetOutEdgeNum() const { return grin_get_edge_num(g_, GRIN_DIRECTION::OUT); }

  template <typename T>
  bool GetData(GRIN_VERTEX v, GRIN_VERTEX_PROPERTY prop, T& value) const {
    if (GRIN_DATATYPE_ENUM<T>::value != grin_get_vertex_property_data_type(g_, prop)) return false;
    auto vtype = grin_get_vertex_type(g_, v);
    auto vpt = grin_get_vertex_property_table_by_type(g_, vtype);
    auto _value = grin_get_value_from_vertex_property_table(g_, vpt, v, prop);
    if (_value != NULL) {
      value = *(static_cast<T*>(_value));
      return true;
    }
    return false;
  }

  template <typename T>
  bool GetData(GRIN_VERTEX_PROPERTY_TABLE vpt, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY prop, T& value) const {
    if (GRIN_DATATYPE_ENUM<T>::value != grin_get_vertex_property_data_type(g_, prop)) return false;
    auto _value = grin_get_value_from_vertex_property_table(g_, vpt, v, prop);
    if (_value != NULL) {
      value = *(static_cast<T*>(_value));
      return true;
    }
    return false;
  }

  GRIN_VERTEX_PROPERTY_TABLE GetVertePropertyTable(GRIN_VERTEX_TYPE label) {
    return grin_get_vertex_property_table_by_type(g_, label);
  }

  bool HasChild(GRIN_VERTEX v, GRIN_EDGE_TYPE e_label) const {
    return GetLocalOutDegree(v, e_label) != 0;
  }

  bool HasParent(GRIN_VERTEX v, GRIN_EDGE_TYPE e_label) const {
    return GetLocalInDegree(v, e_label) != 0;
  }

  int GetLocalOutDegree(GRIN_VERTEX v, GRIN_EDGE_TYPE e_label) const {
    return GetOutgoingAdjList(v, e_label).Size();
  }

  int GetLocalInDegree(GRIN_VERTEX v, GRIN_EDGE_TYPE e_label) const {
    return GetIncomingAdjList(v, e_label).Size();
  }


  inline size_t GetInnerVerticesNum(GRIN_VERTEX_TYPE label) const {
    auto vl = grin_get_vertex_list(g_);
    auto vl1 = grin_filter_type_for_vertex_list(g_, label, vl);
    auto vl2 = grin_filter_master_for_vertex_list(g_, vl1);
    return grin_get_vertex_list_size(g_, vl2);
  }

  inline size_t GetOuterVerticesNum(GRIN_VERTEX_TYPE label) const {
    auto vl = grin_get_vertex_list(g_);
    auto vl1 = grin_filter_type_for_vertex_list(g_, label, vl);
    auto vl2 = grin_filter_mirror_for_vertex_list(g_, vl1);
    return grin_get_vertex_list_size(g_, vl2);
  }

  inline bool IsInnerVertex(GRIN_VERTEX v) const {
    return grin_is_master_vertex(g_, v);
  }

  inline bool IsOuterVertex(GRIN_VERTEX v) const {
    return grin_is_mirror_vertex(g_, v);
  }

  inline adj_list_t GetIncomingAdjList(GRIN_VERTEX v, GRIN_EDGE_TYPE e_label)
      const {
    auto al = grin_get_adjacent_list(g_, GRIN_DIRECTION::IN, v);
    auto al1 = grin_filter_edge_type_for_adjacent_list(g_, e_label, al);
    auto sz = grin_get_adjacent_list_size(g_, al1);
    auto ept = grin_get_edge_property_table_by_type(g_, e_label);
    return adj_list_t(g_, al1, ept, 0, sz);
  }

  inline adj_list_t GetOutgoingAdjList(GRIN_VERTEX v, GRIN_EDGE_TYPE e_label)
      const {
    auto al = grin_get_adjacent_list(g_, GRIN_DIRECTION::OUT, v);
    auto al1 = grin_filter_edge_type_for_adjacent_list(g_, e_label, al);
    auto sz = grin_get_adjacent_list_size(g_, al1);
    auto ept = grin_get_edge_property_table_by_type(g_, e_label);
    return adj_list_t(g_, al1, ept, 0, sz);
  }

  GRIN_GRAPH get_graph() { return g_; }

  // pos means the vertex position in the inner vertex list under vertex type 
  inline grape::DestList IEDests(GRIN_VERTEX_TYPE v_label, size_t pos, GRIN_EDGE_TYPE e_label) const {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
    auto vtype = static_cast<unsigned*>(v_label);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
    auto etype = static_cast<unsigned*>(e_label);
#endif

    return grape::DestList(idoffset_[*vtype][*etype][pos],
                           idoffset_[*vtype][*etype][pos + 1]);
  }

  inline grape::DestList OEDests(GRIN_VERTEX_TYPE v_label, size_t pos, GRIN_EDGE_TYPE e_label) const {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
    auto vtype = static_cast<unsigned*>(v_label);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
    auto etype = static_cast<unsigned*>(e_label);
#endif

    return grape::DestList(odoffset_[*vtype][*etype][pos],
                           odoffset_[*vtype][*etype][pos + 1]);
  }

  inline grape::DestList IOEDests(GRIN_VERTEX_TYPE v_label, size_t pos, GRIN_EDGE_TYPE e_label) const {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
    auto vtype = static_cast<unsigned*>(v_label);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
    auto etype = static_cast<unsigned*>(e_label);
#endif

    return grape::DestList(iodoffset_[*vtype][*etype][pos],
                           iodoffset_[*vtype][*etype][pos + 1]);
  }

  void initDestFidList(
      bool in_edge, bool out_edge,
      std::vector<std::vector<std::vector<fid_t>>>& fid_lists,
      std::vector<std::vector<std::vector<fid_t*>>>& fid_lists_offset) {
    auto vtl = grin_get_vertex_type_list(g_);
    auto vtl_sz = grin_get_vertex_type_list_size(g_, vtl);
    auto etl = grin_get_edge_type_list(g_);
    auto etl_sz = grin_get_edge_type_list_size(g_, etl);

    for (auto vti = 0; vti < vtl_sz; vti++) {
      auto vtype = grin_get_vertex_type_from_list(g_, vtl, vti);
      auto inner_vertices = InnerVertices(vtype);
      auto ivnum_ = inner_vertices.size();

      for (auto eti = 0; eti < etl_sz; eti++) {
        std::vector<int> id_num(ivnum_, 0);
        std::set<fid_t> dstset;

        auto etype = grin_get_edge_type_from_list(g_, etl, eti);
        auto v = inner_vertices.begin();
        auto& fid_list = fid_lists[vti][eti];
        auto& fid_list_offset = fid_lists_offset[vti][eti];

        if (!fid_list_offset.empty()) {
          return;
        }
        fid_list_offset.resize(ivnum_ + 1, NULL);
        for (auto i = 0; i < ivnum_; ++i) {
          dstset.clear();
          
          if (in_edge) {
            auto es = GetIncomingAdjList(*v, etype);
            for (auto& e : es) {
              auto vref = grin_get_vertex_ref_for_vertex(g_, e.neighbor());
              auto p = grin_get_master_partition_from_vertex_ref(g_, vref);

              if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
                auto f = static_cast<unsigned*>(p);
                dstset.insert(*f);
#else
                // todo 
#endif
              }
            }
          }
          if (out_edge) {
            auto es = GetOutgoingAdjList(*v, etype);
            for (auto& e : es) {
              auto vref = grin_get_vertex_ref_for_vertex(g_, e.neighbor());
              auto p = grin_get_master_partition_from_vertex_ref(g_, vref);

              if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
                auto f = static_cast<unsigned*>(p);
                dstset.insert(*f);
#else
                // todo 
#endif
              }
            }
          }
          id_num[i] = dstset.size();
          for (auto fid : dstset) {
            fid_list.push_back(fid);
          }
          ++v;
        }

        fid_list.shrink_to_fit();
        fid_list_offset[0] = fid_list.data();
        for (auto i = 0; i < ivnum_; ++i) {
          fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
        }
      }
    }
  }

  void PrepareToRunApp(const grape::CommSpec& comm_spec, grape::PrepareConf conf) {
    if (conf.message_strategy ==
        grape::MessageStrategy::kAlongEdgeToOuterVertex) {
      initDestFidList(true, true, iodst_, iodoffset_);
    } else if (conf.message_strategy ==
              grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      initDestFidList(true, false, idst_, idoffset_);
    } else if (conf.message_strategy ==
              grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initDestFidList(false, true, odst_, odoffset_);
    }
  }

 private:
  GRIN_PARTITIONED_GRAPH pg_;
  GRIN_GRAPH g_;
  GRIN_PARTITION partition_;
  std::vector<std::vector<std::vector<fid_t>>> idst_, odst_, iodst_;
  std::vector<std::vector<std::vector<fid_t*>>> idoffset_, odoffset_,
    iodoffset_;
};

}  // namespace vineyard

#endif // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H
