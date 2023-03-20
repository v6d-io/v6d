/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef GRIN_SRC_PREDEFINE_H_
#define GRIN_SRC_PREDEFINE_H_

#include "graph/grin/predefine.h"
#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/arrow_fragment_impl.h"

#include "client/client.h"
#include "arrow/api.h"

template <typename T>
struct GRIN_DATATYPE_ENUM {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Undefined;
};

template <>
struct GRIN_DATATYPE_ENUM<int32_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Int32;
};

template <>
struct GRIN_DATATYPE_ENUM<uint32_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::UInt32;
};

template <>
struct GRIN_DATATYPE_ENUM<int64_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Int64;
};

template <>
struct GRIN_DATATYPE_ENUM<uint64_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::UInt64;
};

template <>
struct GRIN_DATATYPE_ENUM<float> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Float;
};

template <>
struct GRIN_DATATYPE_ENUM<double> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Double;
};

template <>
struct GRIN_DATATYPE_ENUM<std::string> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::String;
};

template <>
struct GRIN_DATATYPE_ENUM<arrow::Date32Type> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Date32;
};

template <>
struct GRIN_DATATYPE_ENUM<arrow::Date64Type> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Date64;
};

GRIN_PARTITIONED_GRAPH get_partitioned_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id);
GRIN_GRAPH get_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id);
std::string GetDataTypeName(GRIN_DATATYPE);
GRIN_DATATYPE ArrowToDataType(std::shared_ptr<arrow::DataType>);

#define GRIN_OID_T int64_t
#define GRIN_VID_T uint64_t

/* The following data types shall be defined through typedef. */
typedef vineyard::ArrowFragment<GRIN_OID_T, GRIN_VID_T> GRIN_GRAPH_T;                      
typedef GRIN_GRAPH_T::vertex_t GRIN_VERTEX_T;     
struct GRIN_EDGE_T {
    GRIN_GRAPH_T::vid_t src;
    GRIN_GRAPH_T::vid_t dst;
    GRIN_DIRECTION dir;
    unsigned etype;
    GRIN_GRAPH_T::eid_t eid;
};                     

#ifdef GRIN_WITH_VERTEX_ORIGINAL_ID
typedef GRIN_GRAPH_T::oid_t VERTEX_ORIGINAL_ID_T;                   
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST
struct GRIN_VERTEX_LIST_T {
    unsigned type_begin;
    unsigned type_end;
    unsigned all_master_mirror;
    std::vector<unsigned> offsets;
    std::vector<GRIN_GRAPH_T::vertices_t> vrs;
};
void __grin_init_vertex_list(GRIN_GRAPH_T* g, GRIN_VERTEX_LIST_T* vl);
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST_ITERATOR
struct GRIN_VERTEX_LIST_ITERATOR_T {
    unsigned type_begin;
    unsigned type_end;
    unsigned all_master_mirror;
    unsigned type_current;
    unsigned current;
    GRIN_GRAPH_T::vertices_t vr;
};
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST
struct GRIN_ADJACENT_LIST_T {
    GRIN_GRAPH_T::vid_t vid;
    GRIN_DIRECTION dir;
    unsigned etype_begin;
    unsigned etype_end;
    std::vector<unsigned> offsets;
    std::vector<GRIN_GRAPH_T::raw_adj_list_t> data;
};
void __grin_init_adjacent_list(GRIN_GRAPH_T* g, GRIN_ADJACENT_LIST_T* al);
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
struct GRIN_ADJACENT_LIST_ITERATOR_T {
    GRIN_GRAPH_T::vid_t vid;
    GRIN_DIRECTION dir;
    unsigned etype_begin;
    unsigned etype_end;
    unsigned etype_current;
    unsigned current;
    GRIN_GRAPH_T::raw_adj_list_t data;
};  
#endif

#ifdef GRIN_ENABLE_GRAPH_PARTITION
typedef vineyard::ArrowFragmentGroup GRIN_PARTITIONED_GRAPH_T;
typedef unsigned GRIN_PARTITION_T;
typedef std::vector<unsigned> GRIN_PARTITION_LIST_T;
#endif

#ifdef GRIN_ENABLE_VERTEX_REF
typedef GRIN_GRAPH_T::vid_t GRIN_VERTEX_REF_T;
#endif

#ifdef GRIN_WITH_VERTEX_PROPERTY
typedef unsigned GRIN_VERTEX_TYPE_T;
typedef std::vector<unsigned> GRIN_VERTEX_TYPE_LIST_T;
typedef std::pair<unsigned, unsigned> GRIN_VERTEX_PROPERTY_T;
typedef std::vector<GRIN_VERTEX_PROPERTY_T> GRIN_VERTEX_PROPERTY_LIST_T;
struct GRIN_VERTEX_PROPERTY_TABLE_T {
    unsigned vtype;
    GRIN_GRAPH_T::vertices_t vertices;
};
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
typedef unsigned GRIN_EDGE_TYPE_T;
typedef std::vector<unsigned> GRIN_EDGE_TYPE_LIST_T;
typedef std::pair<unsigned, unsigned> GRIN_EDGE_PROPERTY_T;
typedef std::vector<GRIN_EDGE_PROPERTY_T> GRIN_EDGE_PROPERTY_LIST_T;
struct GRIN_EDGE_PROPERTY_TABLE_T {
    unsigned etype;
    unsigned num;
};
#endif

#if defined(GRIN_WITH_VERTEX_PROPERTY) || defined(GRIN_WITH_EDGE_PROPERTY)
typedef std::vector<const void*> GRIN_ROW_T;
#endif

#endif  // GRIN_SRC_PREDEFINE_H_
