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
#include "grin/include/predefine.h"
#include "modules/graph/fragment/arrow_fragment.h"
#include "modules/graph/fragment/arrow_fragment_group.h"

#include "arrow/api.h"


#ifndef GRIN_SRC_PREDEFINE_H_
#define GRIN_SRC_PREDEFINE_H_

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

std::string GetDataTypeName(DataType type) {
  switch (type) {
  case DataType::Int32:
    return "int32";
  case DataType::UInt32:
    return "uint32";
  case DataType::Int64:
    return "int64";
  case DataType::UInt64:
    return "uint64";
  case DataType::Float:
    return "float";
  case DataType::Double:
    return "double";
  case DataType::String:
    return "string";
  case DataType::Date32:
    return "date32";
  case DataType::Date64:
    return "date64";
  default:
    return "undefined";
  }
}

DataType ArrowToDataType(std::shared_ptr<arrow::DataType> type) {
  if (type == nullptr) {
    return DataType::Undefined;
  } else if (arrow::int32()->Equals(type)) {
    return DataType::Int32;
  } else if (arrow::int64()->Equals(type)) {
    return DataType::Int64;
  } else if (arrow::float32()->Equals(type)) {
    return DataType::Float;
  } else if (arrow::uint32()->Equals(type)) {
    return DataType::UInt32;
  } else if (arrow::uint64()->Equals(type)) {
    return DataType::UInt64;
  } else if (arrow::float64()->Equals(type)) {
    return DataType::Double;
  } else if (arrow::utf8()->Equals(type)) {
    return DataType::String;
  } else if (arrow::large_utf8()->Equals(type)) {
    return DataType::String;
  } 
  return DataType::Undefined;
}

#define G_OID_T int
#define G_VID_T unsigned

/* The following data types shall be defined through typedef. */
typedef vineyard::ArrowFragment<G_OID_T, G_VID_T> Graph_T;                      
typedef Graph_T::vertex_t Vertex_T;     
struct Edge_T {
    Vertex src;
    Vertex dst;
    Direction dir;
    unsigned elabel;
    Graph_T::eid_t eid;
};                     

#ifdef WITH_VERTEX_ORIGIN_ID
typedef Graph_T::oid_t OriginID_T;                   
#endif
#ifdef ENABLE_VERTEX_LIST
typedef std::vector<Graph_T::vertices_t> VertexList_T;                 
#endif
#ifdef CONTINUOUS_VERTEX_ID_TRAIT
typedef Graph_T::vid_t VertexID_T;                   
#endif
#ifdef ENABLE_ADJACENT_LIST
struct AdjacentList_T {
    Vertex v;
    Direction dir;
    unsigned elabel;
    std::vector<Graph_T::raw_adj_list_t> data;
};    
#endif

#ifdef ENABLE_GRAPH_PARTITION
typedef vineyard::ArrowFragmentGroup PartitionedGraph_T;
typedef unsigned Partition_T;
typedef std::vector<unsigned> PartitionList_T;
typedef Graph_T::vid_t VertexRef_T;
typedef Edge_T EdgeRef_T;
#endif

#ifdef WITH_VERTEX_LABEL
typedef unsigned VertexLabel_T;
typedef std::vector<unsigned> VertexLabelList_T;
#endif

#ifdef WITH_EDGE_LABEL
typedef unsigned EdgeLabel_T;
typedef std::vector<unsigned> EdgeLabelList_T;
#endif

#ifdef WITH_VERTEX_PROPERTY
typedef std::pair<unsigned, unsigned> VertexProperty_T;
typedef std::vector<VertexProperty_T> VertexPropertyList_T;
typedef std::pair<unsigned, Graph_T::vertices_t> VertexPropertyTable_T;
#endif

#ifdef WITH_EDGE_PROPERTY
typedef std::pair<unsigned, unsigned> EdgeProperty_T;
typedef std::vector<EdgeProperty_T> EdgePropertyList_T;
typedef std::pair<unsigned, unsigned> EdgePropertyTable_T;
#endif

#endif  // GRIN_SRC_PREDEFINE_H_
