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
#ifndef GRIN_INCLUDE_PREDEFINE_H_
#define GRIN_INCLUDE_PREDEFINE_H_

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>

// The enum type for edge directions.
typedef enum {
  IN = 0,
  OUT = 1,
  BOTH = 2,
} Direction;

// The enum type for partition strategies.
typedef enum {
  VERTEX_CUT = 0,
  EDGE_CUT = 1,
  HYBRID = 2,
} PartitionStrategy;

// The enum type for replicate-edge strategies.
typedef enum {
  ALL = 0,
  PART = 1,
  NONE = 2,
} ReplicateEdgeStrategy;

// The enum type for vertex/edge data type.
typedef enum {
  Undefined = 0,
  Int32 = 1,
  UInt32 = 2,
  Int64 = 3,
  UInt64 = 4,
  Float = 5,
  Double = 6,
  String = 7,
  Date32 = 8,
  Date64 = 9,
} DataType;

/* The following macros are defined as the features of the storage. */
// Note: mutable graph is currently NOT supported in grin-libgrape-lite
// #define MUTABLE_GRAPH                // Graph is mutable
#define WITH_VERTEX_ORIGIN_ID        // There is origin id for vertex semantic
#define WITH_VERTEX_DATA             // There is data on vertex.
#define WITH_EDGE_DATA               // There is data on edge, e.g. weight.
#define ENABLE_VERTEX_LIST           // Enable the vertex list structure.
#define CONTINUOUS_VID_TRAIT         // Enable continous index on vertext list.
#define ENABLE_ADJACENT_LIST         // Enable the adjacent list structure.
// Note: edge_list is only used in vertex_cut fragment
#define ENABLE_EDGE_LIST          // Enable the edge list structure.

// The partition strategy.
#define PARTITION_STRATEGY EDGE_CUT
// There are all/part edges on local vertices.
#define EDGES_ON_LOCAL_VERTEX ALL
// There are all/part/none edges on non-local vertices.
#define EDGES_ON_NON_LOCAL_VERTEX NONE
// The direction of edges on local vertices.
#define EDGES_ON_LOCAL_VERTEX_DIRECTION BOTH

// propertygraph
#define WITH_VERTEX_LABEL
#define WITH_EDGE_LABEL
#define WITH_VERTEX_PROPERTY
#define WITH_EDGE_PROPERTY
#define COLUMN_STORE
#define CONTINIOUS_VERTEX_LABEL_ID_TRAIT
#define CONTINIOUS_EDGE_LABEL_ID_TRAIT


/* The followings macros are defined as invalid value. */
#define NULL_TYPE NULL              // Null type (null data type)
#define NULL_GRAPH NULL             // Null graph handle (invalid return value).
#define NULL_VERTEX NULL            // Null vertex handle (invalid return value).
#define NULL_EDGE NULL              // Null edge handle (invalid return value).
#define NULL_PARTITION UINT_MAX	    // Null partition handle (invalid return value).
#define NULL_LIST NULL              // Null list of any kind.
#define NULL_REMOTE_PARTITION UINT_MAX  // same as partition.
#define NULL_REMOTE_VERTEX NULL     // same as vertex.
#define NULL_REMOTE_EDGE NULL       // same as edge.

/* The following data types shall be defined through typedef. */
// local graph
typedef void* Graph;

// vertex
typedef void* Vertex;
typedef void* VertexID;

// vertex origin id
#ifdef WITH_VERTEX_ORIGIN_ID
typedef void* OriginID;
#endif

// vertex data
#ifdef WITH_VERTEX_DATA
typedef void* VertexData;
#endif

// vertex list
#ifdef ENABLE_VERTEX_LIST
typedef void* VertexList;
typedef void* VertexListIterator;
#endif

// adjacent list
#ifdef ENABLE_ADJACENT_LIST
typedef void* AdjacentList;
typedef void* AdjacentListIterator;
#endif

// edge
typedef void* Edge;

// edge data
#ifdef WITH_EDGE_DATA
typedef void* EdgeData;
#endif

// partitioned graph
typedef void* PartitionedGraph;

// partition and partition list
typedef unsigned Partition;
typedef Partition* PartitionList;

// remote partition and remote partition list
typedef Partition RemotePartition;
typedef PartitionList RemotePartitionList;

// remote vertex and remote vertex list
typedef Vertex RemoteVertex;
typedef RemoteVertex* RemoteVertexList;

// remote edge
typedef Edge RemoteEdge;

#ifdef WITH_VERTEX_LABEL
typedef void* VertexLabelID;
typedef void* VertexLabel;
typedef void* VertexLabelList;
#endif

#ifdef WITH_EDGE_LABEL
typedef void* EdgeLabelID;
typedef void* EdgeLabel;
typedef void* EdgeLabelList;
#endif

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
typedef void* PropertyID;
typedef void* Property;
typedef void* PropertyList;
typedef void* Row;
typedef void* RowList;
#endif

#endif  // GRIN_INCLUDE_PREDEFINE_H_
