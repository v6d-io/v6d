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

#include <stdbool.h>
#include <stddef.h>

// The enum type for edge directions.
typedef enum {
  IN = 0,
  OUT = 1,
  BOTH = 2,
} Direction;

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
#define WITH_VERTEX_ORIGIN_ID             // There is origin id for vertex semantic
// #define WITH_VERTEX_DATA                   // There is data on vertex.
// #define WITH_EDGE_DATA                     // There is data on edge, e.g. weight.
#define ENABLE_VERTEX_LIST                // Enable the vertex list structure.
// #define ENABLE_VERTEX_LIST_ITERATOR        // There is vertex list iterator for unknown size list.
#define CONTINUOUS_VERTEX_ID_TRAIT        // Enable continous vertex id for vertex list.
#define ENABLE_EDGE_LIST                  // Enable the edge list structure.
// #define ENABLE_EDGE_LIST_ITERATOR          // There is edge list iterator for unknown size list.
#define ENABLE_ADJACENT_LIST              // Enable the adjacent list structure.
// #define ENABLE_ADJACENT_LIST_ITERATOR      // There is adjacent list iterator for unknown size list.

// partitioned graph
#define ENABLE_GRAPH_PARTITION            // Enable partitioned graph.
#define NATURAL_PARTITION_ID_TRAIT        // Partition has natural continuous id from 0. 
// #define ENABLE_VALID_VERTEX_REF_LIST       // There is valid vertex ref list for vertex
// #define ENABLE_VALID_VERTEX_REF_LIST_ITERATOR // There is valid vertex ref list iterator for vertex
// #define ENABLE_VALID_EDGE_REF_LIST         // There is valid edge ref list for edge
// #define ENABLE_VALID_EDGE_REF_LIST_ITERATOR   // There is valid edge ref list iterator for vertex


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
#define ENABLE_ROW_LIST                   // Enable row list to access property value 


// predicate
// #define ENABLE_PREDICATE                  // Enable predicates for vertices and edges.


/* The followings macros are defined as invalid value. */
#define NULL_TYPE NULL                    // Null type (null data type)
#define NULL_GRAPH NULL                   // Null graph handle (invalid return value).
#define NULL_VERTEX NULL                  // Null vertex handle (invalid return value).
#define NULL_EDGE NULL                    // Null edge handle (invalid return value).
#define NULL_LIST NULL                    // Null list of any kind (invalid return value).
#define NULL_PARTITION NULL	              // Null partition handle (invalid return value).
#define NULL_VERTEX_REF NULL              // Null vertex ref (invalid return value).
#define NULL_EDGE_REF NULL                // Null edge ref (invalid return value).
#define NULL_VERTEX_LABEL NULL            // Null vertex label handle (invalid return value).
#define NULL_EDGE_LABEL NULL              // Null edge label handle (invalid return value).
#define NULL_PROPERTY NULL                // Null property handle (invalid return value).
#define NULL_ROW NULL                     // Null row handle (invalid return value).

/* The following data types shall be defined through typedef. */
typedef void* Graph;                      
typedef void* Vertex;                     
typedef void* Edge;                       

#ifdef WITH_VERTEX_ORIGIN_ID
typedef void* OriginID;                   
#endif
#ifdef WITH_VERTEX_DATA
typedef void* VertexData;                 
#endif
#ifdef ENABLE_VERTEX_LIST
typedef void* VertexList;                 
#endif
#ifdef ENABLE_VERTEX_LIST_ITERATOR
typedef void* VertexListIterator;         
#endif
#ifdef CONTINUOUS_VERTEX_ID_TRAIT
typedef void* VertexID;                   
#endif
#ifdef ENABLE_ADJACENT_LIST
typedef void* AdjacentList;               
#endif
#ifdef ENABLE_ADJACENT_LIST_ITERATOR
typedef void* AdjacentListIterator;       
#endif
#ifdef WITH_EDGE_DATA
typedef void* EdgeData;                   
#endif
#ifdef ENABLE_EDGE_LIST
typedef void* EdgeList;                   
#endif
#ifdef ENABLE_EDGE_LIST_ITERATOR
typedef void* EdgeListIterator;           
#endif

#ifdef ENABLE_GRAPH_PARTITION
typedef void* PartitionedGraph;
typedef void* Partition;
typedef void* PartitionList;
#ifdef NATURAL_PARTITION_ID_TRAIT
typedef unsigned PartitionID;
#endif
typedef void* VertexRef;
typedef void* EdgeRef;
#endif

#ifdef WITH_VERTEX_LABEL
typedef void* VertexLabel;
typedef void* VertexLabelList;
#ifdef NATURAL_VERTEX_LABEL_ID_TRAIT
typedef unsigned VertexLabelID;
#endif
#endif

#ifdef WITH_EDGE_LABEL
typedef void* EdgeLabel;
typedef void* EdgeLabelList;
#ifdef NATURAL_EDGE_LABEL_ID_TRAIT
typedef unsigned EdgeLabelID;
#endif
#endif

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
typedef void* Property;
typedef void* PropertyList;
#ifdef NATURAL_PROPERTY_ID_TRAIT
typedef unsigned PropertyID;
#endif
typedef void* Row;
typedef void* RowList;
#endif

#endif  // GRIN_INCLUDE_PREDEFINE_H_
