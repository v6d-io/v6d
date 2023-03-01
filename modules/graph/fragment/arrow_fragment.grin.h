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

extern "C" {
#include "graph/grin/include/topology/structure.h"
#include "graph/grin/include/topology/vertexlist.h"
#include "graph/grin/include/topology/adjacentlist.h"
#include "graph/grin/include/partition/partition.h"
#include "graph/grin/include/property/type.h"
#include "graph/grin/include/property/property.h"
#include "graph/grin/include/property/propertylist.h"
#include "graph/grin/include/property/propertytable.h"
}

#include "graph/grin/src/predefine.h"

namespace vineyard {

std::shared_ptr<arrow::DataType> GetArrowDataType(DataType type) {
  switch (type) {
  case DataType::Undefined:
    return arrow::null();
  case DataType::Int32:
    return arrow::int32();
  case DataType::UInt32:
    return arrow::uint32();
  case DataType::Int64:
    return arrow::int64();
  case DataType::UInt64:
    return arrow::uint64();
  case DataType::Float:
    return arrow::float32();
  case DataType::Double:
    return arrow::float64();
  case DataType::String:
    return arrow::large_utf8();
  case DataType::Date32:
    return arrow::int32();
  case DataType::Date64:
    return arrow::int64();
  default:
    return arrow::null();
  }
}


struct GRIN_Nbr {
 public:
  GRIN_Nbr() : al_(nullptr), cur_(0), ept_(nullptr) {}
  GRIN_Nbr(void* al, size_t cur, void* ept) 
    : al_(al), cur_(cur), ept_(ept) {}
  GRIN_Nbr(GRIN_Nbr& rhs) : al_(rhs.al_), cur_(rhs.cur_), ept_(rhs.ept_) {}

  GRIN_Nbr& operator=(const GRIN_Nbr& rhs) {
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    ept_ = rhs.ept_;
    return *this;
  }

  GRIN_Nbr& operator=(GRIN_Nbr&& rhs) {
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    ept_ = rhs.ept_;
    return *this;
  }

  void* neighbor() {
    return get_neighbor_from_adjacent_list(al_, cur_);
  }

  void* get_neighbor() {
    return get_neighbor_from_adjacent_list(al_, cur_);
  }

  void* get_edge() {
    return get_edge_from_adjacent_list(al_, cur_);
  }

  template <typename T>
  T get_data(void* prop) const {
    auto _e = get_edge_from_adjacent_list(al_, cur_);
    auto value = get_value_from_edge_property_table(ept_, _e, prop);
    return property_graph_utils::ValueGetter<T>::Value(value, 0);
  }

  std::string get_str(void* prop) const {
    auto _e = get_edge_from_adjacent_list(al_, cur_);
    auto value = get_value_from_edge_property_table(ept_, _e, prop);
    return property_graph_utils::ValueGetter<std::string>::Value(value, 0);
  }

  double get_double(void* prop) const {
    auto _e = get_edge_from_adjacent_list(al_, cur_);
    auto value = get_value_from_edge_property_table(ept_, _e, prop);
    return property_graph_utils::ValueGetter<double>::Value(value, 0);
  }

  int64_t get_int(void* prop) const {
    auto _e = get_edge_from_adjacent_list(al_, cur_);
    auto value = get_value_from_edge_property_table(ept_, _e, prop);
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
  void* al_;
  size_t cur_;
  void* ept_;
};


class GRIN_AdjList {
 public:
  GRIN_AdjList() : adj_list_(nullptr), ept_(nullptr), begin_(0), end_(0) {}
  GRIN_AdjList(void* adj_list, void* ept, size_t begin, size_t end) 
    : adj_list_(adj_list), ept_(ept), begin_(begin), end_(end) {}

  inline GRIN_Nbr begin() const {
    return GRIN_Nbr(adj_list_, begin_, ept_);
  }

  inline GRIN_Nbr end() const {
    return GRIN_Nbr(adj_list_, end_, ept_);
  }

  inline size_t Size() const { return end_ - begin_; }

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return begin_ < end_; }

  size_t size() const { return end_ - begin_; }

 private:
  void* adj_list_;
  void* ept_;
  size_t begin_;
  size_t end_;
};


class GRIN_VertexRange {
 public:
  GRIN_VertexRange() {}
  GRIN_VertexRange(void* vl, const size_t begin, const size_t end)
      : vl_(vl), begin_(begin), end_(end) {}
  GRIN_VertexRange(const GRIN_VertexRange& r) : vl_(r.vl_), begin_(r.begin_), end_(r.end_) {}

  class iterator {
    using reference_type = void*;

   private:
    void* vl_;
    size_t cur_;

   public:
    iterator() noexcept : vl_(nullptr), cur_() {}
    explicit iterator(void* vl, size_t idx) noexcept : vl_(vl), cur_(idx) {}

    reference_type operator*() noexcept { return get_vertex_from_list(vl_, cur_); }

    iterator& operator++() noexcept {
      ++cur_;
      return *this;
    }

    iterator operator++(int) noexcept {
      return iterator(vl_, cur_ + 1);
    }

    iterator& operator--() noexcept {
      --cur_;
      return *this;
    }

    iterator operator--(int) noexcept {
      return iterator(vl_, cur_--);
    }

    iterator operator+(size_t offset) const noexcept {
      return iterator(vl_, cur_ + offset);
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

  iterator begin() const { return iterator(vl_, begin_); }

  iterator end() const { return iterator(vl_, end_); }

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
  void* vl_;
  size_t begin_;
  size_t end_;
};


class GRIN_ArrowFragment {
 public:
  using vertex_range_t = GRIN_VertexRange;
  using adj_list_t = GRIN_AdjList;

 public:
  ~GRIN_ArrowFragment() = default;

  void init(void* partitioned_graph, size_t idx) {
    pg_ = partitioned_graph;
    auto pl = get_local_partition_list(pg_);
    assert(idx < get_partition_list_size(pl));
    partition_ = get_partition_from_list(pl, idx);
    g_ = get_local_graph_from_partition(pg_, partition_); 
  }

  bool directed() const {
    return is_directed(g_);
  }

  bool multigraph() const {
    return is_multigraph(g_);
  }

  const std::string oid_typename() const { 
    auto dt = get_vertex_original_id_type(g_);
    return GetDataTypeName(dt);
  }

  size_t fnum() const { return get_total_partitions_number(pg_); }

  void* vertex_label(void* v) {
    return get_vertex_type(g_, v);
  }

  size_t vertex_label_num() const { 
    auto vtl = get_vertex_type_list(g_);
    return get_vertex_type_list_size(vtl);
  }

  size_t edge_label_num() const { 
    auto etl = get_edge_type_list(g_);
    return get_edge_type_list_size(etl);
  }

  size_t vertex_property_num(void* label) const {
    auto vpl = get_vertex_property_list_by_type(g_, label);
    return get_vertex_property_list_size(vpl);
  }

  std::shared_ptr<arrow::DataType> vertex_property_type(void* prop) const {
    auto dt = get_vertex_property_data_type(g_, prop);
    return GetArrowDataType(dt);
  }
  
  size_t edge_property_num(void* label) const {
    void* epl = get_edge_property_list_by_type(g_, label);
    return get_edge_property_list_size(epl);
  }

  std::shared_ptr<arrow::DataType> edge_property_type(void* prop) const {
    auto dt = get_edge_property_data_type(g_, prop);
    return GetArrowDataType(dt);
  }

  vertex_range_t Vertices(void* label) const {
    auto vl = get_vertex_list_by_type(g_, label);
    auto sz = get_vertex_list_size(vl);
    return vertex_range_t(vl, 0, sz);
  }

  vertex_range_t InnerVertices(void* label) const {
    auto vl = get_master_vertices_by_type(g_, label);
    auto sz = get_vertex_list_size(vl);
    return vertex_range_t(vl, 0, sz);
  }

  vertex_range_t OuterVertices(void* label) const {
    auto vl = get_mirror_vertices_by_type(g_, label);
    auto sz = get_vertex_list_size(vl);
    return vertex_range_t(vl, 0, sz);
  }

  vertex_range_t InnerVerticesSlice(void* label, size_t start, size_t end)
      const {
    auto vl = get_master_vertices_by_type(g_, label);
    auto _end = get_vertex_list_size(vl);
    CHECK(start <= end && start <= _end);
    if (end <= _end) {
      return vertex_range_t(vl, start, end);
    } else {
      return vertex_range_t(vl, start, _end);
    }
  }

  inline size_t GetVerticesNum(void* label) const {
    auto vl = get_vertex_list_by_type(g_, label);
    return get_vertex_list_size(vl);
  }

  template <typename T>
  bool GetVertex(void* label, T& oid, void* v) {
    if (DataTypeEnum<T>::value != get_vertex_original_id_type(g_)) return false;
    v = get_vertex_from_original_id_by_type(g_, label, (void*)(&oid));
    return v != NULL;
  }

  template <typename T>
  bool GetId(void* v, T& oid) {
    if (DataTypeEnum<T>::value != get_vertex_original_id_type(g_)) return false;
    auto _id = get_vertex_original_id(g_, v);
    if (_id == NULL) return false;
    auto _oid = static_cast<T*>(_id);
    oid = *_oid;
    return true;
  }

  void* GetFragId(void* u) const {
    auto vref = get_vertex_ref_for_vertex(g_, partition_, u);
    return get_master_partition_from_vertex_ref(g_, vref); 
  }

  size_t GetTotalNodesNum() const {
    return GetTotalVerticesNum();
  }
  
  size_t GetTotalVerticesNum() const {
    return get_vertex_num(g_);
  }

  size_t GetTotalVerticesNum(void* label) const {
    return get_vertex_num_by_type(g_, label);
  }

  size_t GetEdgeNum() const { return get_edge_num(g_, Direction::BOTH); }
  size_t GetInEdgeNum() const { return get_edge_num(g_, Direction::IN); }
  size_t GetOutEdgeNum() const { return get_edge_num(g_, Direction::OUT); }

  template <typename T>
  bool GetData(void* v, void* prop, T& value) const {
    if (DataTypeEnum<T>::value != get_vertex_property_data_type(g_, prop)) return false;
    auto vtype = get_vertex_type(g_, v);
    auto vpt = get_vertex_property_table_by_type(g_, vtype);
    auto _value = get_value_from_vertex_property_table(vpt, v, prop);
    if (_value != NULL) {
      value = *(static_cast<T*>(_value));
      return true;
    }
    return false;
  }

  template <typename T>
  bool GetData(void* vpt, void* v, void* prop, T& value) const {
    if (DataTypeEnum<T>::value != get_vertex_property_data_type(g_, prop)) return false;
    auto _value = get_value_from_vertex_property_table(vpt, v, prop);
    if (_value != NULL) {
      value = *(static_cast<T*>(_value));
      return true;
    }
    return false;
  }

  void* GetVertePropertyTable(void* label) {
    return get_vertex_property_table_by_type(g_, label);
  }

  bool HasChild(void* v, void* e_label) const {
    return GetLocalOutDegree(v, e_label) != 0;
  }

  bool HasParent(void* v, void* e_label) const {
    return GetLocalInDegree(v, e_label) != 0;
  }

  int GetLocalOutDegree(void* v, void* e_label) const {
    return GetOutgoingAdjList(v, e_label).Size();
  }

  int GetLocalInDegree(void* v, void* e_label) const {
    return GetIncomingAdjList(v, e_label).Size();
  }


  inline size_t GetInnerVerticesNum(void* label) const {
    auto vl = get_master_vertices_by_type(g_, label);
    return get_vertex_list_size(vl);
  }

  inline size_t GetOuterVerticesNum(void* label) const {
    auto vl = get_mirror_vertices_by_type(g_, label);
    return get_vertex_list_size(vl);
  }

  inline bool IsInnerVertex(void* v) const {
    return is_master_vertex(g_, v);
  }

  inline bool IsOuterVertex(void* v) const {
    return is_mirror_vertex(g_, v);
  }

  inline adj_list_t GetIncomingAdjList(void* v, void* e_label)
      const {
    auto al = get_adjacent_list_by_edge_type(g_, Direction::IN, v, e_label);
    auto sz = get_adjacent_list_size(al);
    auto ept = get_edge_property_table_by_type(g_, e_label);
    return adj_list_t(al, ept, 0, sz);
  }

  inline adj_list_t GetOutgoingAdjList(void* v, void* e_label)
      const {
    auto al = get_adjacent_list_by_edge_type(g_, Direction::OUT, v, e_label);
    auto sz = get_adjacent_list_size(al);
    auto ept = get_edge_property_table_by_type(g_, e_label);
    return adj_list_t(al, ept, 0, sz);
  }

  void* get_graph() { return g_; }

 private:
  void* pg_;
  void* g_;
  void* partition_;
};

}  // namespace vineyard

#endif // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GRIN_H
