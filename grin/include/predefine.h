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
// #define MUTABLE_GRAPH                  // Graph is mutable, NOT supported in v6d
#define WITH_VERTEX_ORIGIN_ID          // There is origin id for vertex semantic
// #define WITH_VERTEX_DATA               // There is data on vertex.
// #define WITH_EDGE_DATA                 // There is data on edge, e.g. weight.
#define ENABLE_VERTEX_LIST             // Enable the vertex list structure.
#define CONTINUOUS_VERTEX_ID_TRAIT     // Enable continous vertex id for vertex list.
#define ENABLE_ADJACENT_LIST           // Enable the adjacent list structure.
#define ENABLE_EDGE_LIST               // Enable the edge list structure.

#define PARTITION_STRATEGY EDGE_CUT    // The partition strategy.
#define EDGES_ON_LOCAL_VERTEX ALL      // There are all/part edges on local vertices.
#define EDGES_ON_NON_LOCAL_VERTEX NONE // There are all/part/none edges on non-local vertices.
#define EDGES_ON_LOCAL_VERTEX_DIRECTION BOTH  // The direction of edges on local vertices.

// propertygraph
#define WITH_VERTEX_LABEL                 // There are labels on vertices.
#define WITH_VERTEX_PROPERTY              // There is any property on vertices.
// #define WITH_VERTEX_PRIMARTY_KEYS         // There are primary keys for vertex.
#define NATURAL_VERTEX_LABEL_ID_TRAIT     // Vertex label has natural continuous id from 0.
#define WITH_EDGE_LABEL                   // There are labels for edges.
#define WITH_EDGE_PROPERTY                // There is any property for edges.
#define NATURAL_EDGE_LABEL_ID_TRAIT       // Edge label has natural continuous id from 0.
#define COLUMN_STORE                      // Column-oriented storage for properties.
#define NATURAL_PROPERTY_ID_TRAIT         // Property has natural continuous id from 0.

#define PROPERTY_ON_NON_LOCAL_VERTEX NONE // There are properties on non-local vertices.
#define PROPERTY_ON_NON_LOCAL_EDGE NONE   // There are properties on non-local edges.
// #define ENABLE_PREDICATE                  // Enable predicates for vertices and edges.


/* The followings macros are defined as invalid value. */
#define NULL_TYPE NULL                  // Null type (null data type)
#define NULL_GRAPH NULL                 // Null graph handle (invalid return value).
#define NULL_VERTEX NULL                // Null vertex handle (invalid return value).
#define NULL_EDGE NULL                  // Null edge handle (invalid return value).
#define NULL_PARTITION UINT_MAX	        // Null partition handle (invalid return value).
#define NULL_LIST NULL                  // Null list of any kind.
#define NULL_REMOTE_PARTITION UINT_MAX  // same as partition.
#define NULL_REMOTE_VERTEX NULL         // same as vertex.
#define NULL_REMOTE_EDGE NULL           // same as edge.
#define NULL_VERTEX_LABEL               // Null vertex label handle (invalid return value).
#define NULL_EDGE_LABEL                 // Null vertex label handle (invalid return value).
#define NULL_PROPERTY                   // Null property handle (invalid return value).
#define NULL_ROW                        // Null row handle (invalid return value).

/* The following data types shall be defined through typedef. */
// local graph
typedef void* Graph;

// vertex
typedef void* Vertex;

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

// vertex id
#ifdef CONTINUOUS_VERTEX_ID_TRAIT
typedef void* VertexID;
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

// edge list
#ifdef ENABLE_EDGE_LIST
typedef void* EdgeList;
typedef void* EdgeListIterator;
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
typedef void* VertexLabel;
typedef void* VertexLabelList;
#ifdef NATURAL_VERTEX_LABEL_ID_TRAIT
typedef void* VertexLabelID;
#endif
#endif

#ifdef WITH_EDGE_LABEL
typedef void* EdgeLabel;
typedef void* EdgeLabelList;
#ifdef NATURAL_EDGE_LABEL_ID_TRAIT
typedef void* EdgeLabelID;
#endif
#endif

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
typedef void* Property;
typedef void* PropertyList;
typedef void* PropertyListIterator;
#ifdef NATURAL_PROPERTY_ID_TRAIT
typedef void* PropertyID;
#endif
typedef void* Row;
typedef void* RowList;
typedef void* RowListIterator;
#endif

#endif  // GRIN_INCLUDE_PREDEFINE_H_
