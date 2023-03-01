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

#include "graph/grin/include/predefine.h"
#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/arrow_fragment_impl.h"

#include "client/client.h"
#include "arrow/api.h"

template <typename T>
struct DataTypeEnum {
  static constexpr DataType value = DataType::Undefined;
};

template <>
struct DataTypeEnum<int32_t> {
  static constexpr DataType value = DataType::Int32;
};

template <>
struct DataTypeEnum<uint32_t> {
  static constexpr DataType value = DataType::UInt32;
};

template <>
struct DataTypeEnum<int64_t> {
  static constexpr DataType value = DataType::Int64;
};

template <>
struct DataTypeEnum<uint64_t> {
  static constexpr DataType value = DataType::UInt64;
};

template <>
struct DataTypeEnum<float> {
  static constexpr DataType value = DataType::Float;
};

template <>
struct DataTypeEnum<double> {
  static constexpr DataType value = DataType::Double;
};

template <>
struct DataTypeEnum<std::string> {
  static constexpr DataType value = DataType::String;
};

template <>
struct DataTypeEnum<arrow::Date32Type> {
  static constexpr DataType value = DataType::Date32;
};

template <>
struct DataTypeEnum<arrow::Date64Type> {
  static constexpr DataType value = DataType::Date64;
};

PartitionedGraph get_partitioned_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id);
Graph get_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id);
std::string GetDataTypeName(DataType);
DataType ArrowToDataType(std::shared_ptr<arrow::DataType>);

#define G_OID_T int64_t
#define G_VID_T uint64_t

/* The following data types shall be defined through typedef. */
typedef vineyard::ArrowFragment<G_OID_T, G_VID_T> Graph_T;                      
typedef Graph_T::vertex_t Vertex_T;     
struct Edge_T {
    Vertex src;
    Vertex dst;
    Direction dir;
    unsigned etype;
    Graph_T::eid_t eid;
};                     

#ifdef WITH_VERTEX_ORIGINAL_ID
typedef Graph_T::oid_t OriginalID_T;                   
#endif

#ifdef ENABLE_VERTEX_LIST
typedef std::vector<Graph_T::vertices_t> VertexList_T;                 
#endif

#ifdef ENABLE_ADJACENT_LIST
struct AdjacentList_T {
    Vertex v;
    Direction dir;
    unsigned etype;
    std::vector<Graph_T::raw_adj_list_t> data;
};    
#endif

#ifdef ENABLE_GRAPH_PARTITION
typedef vineyard::ArrowFragmentGroup PartitionedGraph_T;
typedef unsigned Partition_T;
typedef std::vector<unsigned> PartitionList_T;
#endif

#ifdef ENABLE_VERTEX_REF
typedef Graph_T::vid_t VertexRef_T;
#endif

#ifdef WITH_VERTEX_PROPERTY
typedef unsigned VertexType_T;
typedef std::vector<unsigned> VertexTypeList_T;
typedef std::pair<unsigned, unsigned> VertexProperty_T;
typedef std::vector<VertexProperty_T> VertexPropertyList_T;
struct VertexPropertyTable_T {
    Graph_T* g;
    unsigned vtype;
    Graph_T::vertices_t vertices;
};
#endif

#ifdef WITH_EDGE_PROPERTY
typedef unsigned EdgeType_T;
typedef std::vector<unsigned> EdgeTypeList_T;
typedef std::pair<unsigned, unsigned> EdgeProperty_T;
typedef std::vector<EdgeProperty_T> EdgePropertyList_T;
struct EdgePropertyTable_T {
    Graph_T* g;
    unsigned etype;
    unsigned num;
};
#endif

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
typedef std::vector<const void*> Row_T;
#endif

#endif  // GRIN_SRC_PREDEFINE_H_
