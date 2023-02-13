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
#include "grape/fragment/immutable_edgecut_fragment.h"

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

/* The following data types shall be defined through typedef. */
// local graph
typedef grape::ImmutableEdgecutFragment<G_OID_T, G_VID_T, G_VDATA_T, G_EDATA_T>
    Graph_T;

// vertex
typedef Graph_T::vertex_t Vertex_T;
typedef Graph_T::vid_t VertexID_T;
typedef Graph_T::vdata_t VertexData_T;

// vertex list
#ifdef ENABLE_VERTEX_LIST
typedef Graph_T::vertex_range_t VertexList_T;
typedef Graph_T::vid_t VertexListIterator_T;
#endif

// indexed vertex list
#ifdef ENABLE_INDEXED_VERTEX_LIST
typedef Graph_T::vid_t VertexIndex_T;
#endif

// adjacent list
#ifdef ENABLE_ADJACENT_LIST
typedef Graph_T::adj_list_t AdjacentList_T;
typedef Graph_T::nbr_t AdjacentListIterator_T;
#endif

// edge
typedef Graph_T::edge_t Edge_T;
typedef Graph_T::edata_t EdgeData_T;

// partitioned graph
typedef Graph_T PartitionedGraph_T;

// remote vertex
typedef Vertex_T RemoteVertex_T;

// remote edge
typedef Edge_T RemoteEdge_T;

#endif  // GRIN_SRC_PREDEFINE_H_
