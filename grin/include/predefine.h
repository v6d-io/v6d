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

/**
 * @file predefine.h
 * @brief Pre-defined macros for storage features.
 * Undefine the macros of features that the storage does NOT support,
 * so that APIs under unsupported features will NOT be available to 
 * the callers to avoid ambiguity.
*/

#ifndef GRIN_INCLUDE_PREDEFINE_H_
#define GRIN_INCLUDE_PREDEFINE_H_

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>


/// Enumerates the directions of edges with respect to a certain vertex
typedef enum {
  IN = 0,     ///< incoming
  OUT = 1,    ///< outgoing
  BOTH = 2,   ///< incoming & outgoing
} Direction;

/// Enumerates the datatype supported in the storage
typedef enum {
  Undefined = 0,  ///< other unknown types
  Int32 = 1,      ///< int
  UInt32 = 2,     ///< unsigned int 
  Int64 = 3,      ///< long int
  UInt64 = 4,     ///< unsigned long int
  Float = 5,      ///< float
  Double = 6,     ///< double
  String = 7,     ///< string
  Date32 = 8,     ///< short date
  Date64 = 9,     ///< long date
} DataType;

/** @name TopologyMacros
 * @brief Macros for basic graph topology features
 */
///@{
/** @ingroup TopologyMacros 
 * @brief There is original ID for a vertex.
 * This facilitates queries starting from a specific vertex,
 * since one can get the vertex handler directly using its original ID.
 */
#define WITH_VERTEX_ORIGINAL_ID

/** @ingroup TopologyMacros 
 * @brief There is data on vertex. E.g., the PageRank value of a vertex.
 */
#define WITH_VERTEX_DATA

/** @ingroup TopologyMacros
 * @brief There is data on edge. E.g., the weight of an edge.
*/
#define WITH_EDGE_DATA

/** @ingroup TopologyMacros
 * @brief Enable the vertex list structure. 
 * The vertex list related APIs follow the design principle of GRIN List.
*/
#define ENABLE_VERTEX_LIST

/** @ingroup TopologyMacros
 * @brief Enable the vertex list iterator. 
 * The vertex list iterator related APIs follow the design principle of GRIN Iterator.
*/
#define ENABLE_VERTEX_LIST_ITERATOR

/** @ingroup TopologyMacros
 * @brief Enable the edge list structure. 
 * The edge list related APIs follow the design principle of GRIN List.
*/
#define ENABLE_EDGE_LIST

/** @ingroup TopologyMacros
 * @brief Enable the edge list iterator. 
 * The edge list iterator related APIs follow the design principle of GRIN Iterator.
*/
#define ENABLE_EDGE_LIST_ITERATOR

/** @ingroup TopologyMacros
 * @brief Enable the adjacent list structure. 
 * The adjacent list related APIs follow the design principle of GRIN List.
*/
#define ENABLE_ADJACENT_LIST

/** @ingroup TopologyMacros
 * @brief Enable the adjacent list iterator. 
 * The adjacent list iterator related APIs follow the design principle of GRIN Iterator.
*/
#define ENABLE_ADJACENT_LIST_ITERATOR

#undef WITH_VERTEX_DATA
#undef WITH_EDGE_DATA
#undef ENABLE_VERTEX_LIST_ITERATOR
#undef ENABLE_EDGE_LIST
#undef ENABLE_EDGE_LIST_ITERATOR
#undef ENABLE_ADJACENT_LIST_ITERATOR
///@}


/** @name PartitionMacros
 * @brief Macros for partitioned graph features
 */
///@{
/** @ingroup PartitionMacros
 * @brief Enable partitioned graph. A partitioned graph usually contains
 * several fragments (i.e., local graphs) that are distributedly stored 
 * in a cluster. In GRIN, Graph represents to a single fragment that can
 * be locally accessed.
 */
#define ENABLE_GRAPH_PARTITION

/** @ingroup PartitionMacros
 * @brief The storage provides natural number IDs for partitions.
 * It follows the design principle of natural number ID trait in GRIN.
*/
#define NATURAL_PARTITION_ID_TRAIT

/** @ingroup PartitionMacros
 * @brief The storage provides reference of vertex that can be
 * recognized in other partitions where the vertex also appears.
*/
#define ENABLE_VERTEX_REF

/** @ingroup PartitionMacros
 * @brief The storage provides reference of edge that can be
 * recognized in other partitions where the edge also appears.
*/
#define ENABLE_EDGE_REF

#ifndef ENABLE_GRAPH_PARTITION
#undef NATURAL_PARTITION_ID_TRAIT
#endif

#undef ENABLE_EDGE_REF
///@}


/** @name PropertyMacros
 * @brief Macros for property graph features
 */
///@{
/** @ingroup PropertyMacros
 * @brief There are property names for properties. The relationship between property
 * name and properties is one-to-many, because properties bound to different vertex/edge
 * types are distinguished even they may share the same property name. Please refer to
 * the design principle of Property for details.
*/
#define WITH_PROPERTY_NAME

/** @ingroup PropertyMacros
 * @brief There are properties bound to vertices. When vertices are typed, vertex
 * properties are bound to vertex types, according to the definition of vertex type.
*/
#define WITH_VERTEX_PROPERTY

/** @ingroup PropertyMacros
 * @brief There are primary keys for vertices. Vertex primary keys is
 * a set of vertex properties whose values can distinguish vertices. When vertices are
 * typed, each vertex type has its own primary keys which distinguishes the vertices of
 * that type. 
 * 
 * With primary keys, one can get the vertex from the graph or a certain type
 * by providing the values of the primary keys. The macro is unset if WITH_VERTEX_PROPERTY
 * is NOT defined, in which case, one can use WITH_VERTEX_ORIGINAL_ID when vertices have
 * no properties.
*/
#define WITH_VERTEX_PRIMARTY_KEYS

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for vertex types.
 * It follows the design principle of natural ID trait in GRIN.
*/
#define NATURAL_VERTEX_TYPE_ID_TRAIT

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for properties bound to
 * a certain vertex type.
 * It follows the design principle of natural ID trait in GRIN.
*/
#define NATURAL_VERTEX_PROPERTY_ID_TRAIT


#define WITH_EDGE_PROPERTY                // There is any property for edges.
#define WITH_EDGE_PRIMARTY_KEYS           // There is cross-type property name.
#define NATURAL_EDGE_TYPE_ID_TRAIT       // Edge type has natural continuous id from 0.
#define NATURAL_EDGE_PROPERTY_ID_TRAIT    // Edge property has natural continuous id from 0.


/** @ingroup PropertyMacros
 * @brief The storage uses column store for properties.
 * This enables efficient property selections for vertices and edges.
*/
#define COLUMN_STORE_TRAIT

#if !defined(WITH_VERTEX_PROPERTY) && !defined(WITH_EDGE_PROPERTY)
#undef WITH_PROPERTY_NAME
#endif

#ifndef WITH_VERTEX_PROPERTY
#undef WITH_VERTEX_PRIMARTY_KEYS
#undef NATURAL_VERTEX_TYPE_ID_TRAIT
#undef NATURAL_VERTEX_PROPERTY_ID_TRAIT
#endif

#ifndef WITH_EDGE_PROPERTY
#undef WITH_EDGE_PRIMARTY_KEYS
#undef NATURAL_EDGE_TYPE_ID_TRAIT
#undef NATURAL_EDGE_PROPERTY_ID_TRAIT
#endif

#undef WITH_VERTEX_PRIMARTY_KEYS

#undef WITH_LABEL
///@}

/** @name PredicateMacros
 * @brief Macros for predicate features
 */
///@{
/** @ingroup PredicateMacros
 * @brief Enable predicates on graph. 
*/
#define ENABLE_PREDICATE
#undef ENABLE_PREDICATE
///@}

/** @name IndexMacros
 * @brief Macros for index features
 */
///@{
/** @ingroup IndexMacros
 * @brief Enable vertex label on graph. 
*/
#define WITH_VERTEX_LABEL

/** @ingroup IndexMacros
 * @brief Enable edge label on graph. 
*/
#define WITH_EDGE_LABEL

#undef WITH_VERTEX_LABEL
#undef WITH_EDGE_LABEL
///@}

/** @name NullValues
 * Macros for Null(invalid) values
 */
///@{
/** @brief Null type (undefined data type) */
#define NULL_TYPE Undefined
/** @brief Null graph (invalid return value) */
#define NULL_GRAPH NULL
/** @brief Non-existing vertex (invalid return value) */
#define NULL_VERTEX NULL
/** @brief Non-existing edge (invalid return value) */
#define NULL_EDGE NULL
/** @brief Null list of any kind (invalid return value) */
#define NULL_LIST NULL
/** @brief Non-existing partition (invalid return value) */
#define NULL_PARTITION NULL
/** @brief Null vertex reference (invalid return value) */
#define NULL_VERTEX_REF NULL
/** @brief Null edge reference (invalid return value) */
#define NULL_EDGE_REF NULL
/** @brief Non-existing vertex type (invalid return value) */
#define NULL_VERTEX_TYPE NULL
/** @brief Non-existing edge type (invalid return value) */
#define NULL_EDGE_TYPE NULL
/** @brief Non-existing property (invalid return value) */
#define NULL_PROPERTY NULL
/** @brief Null row (invalid return value) */
#define NULL_ROW NULL
/** @brief Null natural id of any kind (invalid return value) */
#define NULL_NATURAL_ID UINT_MAX
///@}


/* Define the handlers using typedef */
typedef void* Graph;                      
typedef void* Vertex;                     
typedef void* Edge;                       

#ifdef WITH_VERTEX_ORIGINAL_ID
typedef void* OriginalID;                   
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
#endif

#ifdef NATURAL_PARTITION_ID_TRAIT
typedef unsigned PartitionID;
#endif

#ifdef ENABLE_VERTEX_REF
typedef void* VertexRef;
#endif

#ifdef ENABLE_EDGE_REF
typedef void* EdgeRef;
#endif


#ifdef WITH_VERTEX_PROPERTY
typedef void* VertexType;
typedef void* VertexTypeList;
typedef void* VertexProperty;
typedef void* VertexPropertyList;
typedef void* VertexPropertyTable;
#endif

#ifdef NATURAL_VERTEX_TYPE_ID_TRAIT
typedef unsigned VertexTypeID;
#endif

#ifdef NATURAL_VERTEX_PROPERTY_ID_TRAIT
typedef unsigned VertexPropertyID;
#endif

#ifdef WITH_EDGE_PROPERTY
typedef void* EdgeType;
typedef void* EdgeTypeList;
typedef void* EdgeProperty;
typedef void* EdgePropertyList;
typedef void* EdgePropertyTable;
#endif

#ifdef NATURAL_EDGE_TYPE_ID_TRAIT
typedef unsigned EdgeTypeID;
#endif

#ifdef NATURAL_EDGE_PROPERTY_ID_TRAIT
typedef unsigned EdgePropertyID;
#endif

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
typedef void* Row;
#endif

#ifdef WITH_LABEL
typedef void* Label
typedef void* LabelList
#endif

#endif  // GRIN_INCLUDE_PREDEFINE_H_
