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
} GRIN_DIRECTION;

/// Enumerates the datatype supported in the storage
typedef enum {
  Undefined = 0,      ///< other unknown types
  Int32 = 1,          ///< int
  UInt32 = 2,         ///< unsigned int 
  Int64 = 3,          ///< long int
  UInt64 = 4,         ///< unsigned long int
  Float = 5,          ///< float
  Double = 6,         ///< double
  String = 7,         ///< string
  Date32 = 8,         ///< short date
  Date64 = 9,         ///< long date
} GRIN_DATATYPE;

/** @name TopologyMacros
 * @brief Macros for basic graph topology features
 */
///@{
/** @ingroup TopologyMacros 
 * @brief There is original ID for a vertex.
 * This facilitates queries starting from a specific vertex,
 * since one can get the vertex handler directly using its original ID.
 */
#define GRIN_WITH_VERTEX_ORIGINAL_ID

/** @ingroup TopologyMacros 
 * @brief There is data on vertex. E.g., the PageRank value of a vertex.
 */
#define GRIN_WITH_VERTEX_DATA

/** @ingroup TopologyMacros
 * @brief There is data on edge. E.g., the weight of an edge.
*/
#define GRIN_WITH_EDGE_DATA

/** @ingroup TopologyMacros
 * @brief Enable the vertex list structure. 
 * The vertex list related APIs follow the design principle of GRIN List.
*/
#define GRIN_ENABLE_VERTEX_LIST

/** @ingroup TopologyMacros
 * @brief Enable the vertex list iterator. 
 * The vertex list iterator related APIs follow the design principle of GRIN Iterator.
*/
#define GRIN_ENABLE_VERTEX_LIST_ITERATOR

/** @ingroup TopologyMacros
 * @brief Enable the edge list structure. 
 * The edge list related APIs follow the design principle of GRIN List.
*/
#define GRIN_ENABLE_EDGE_LIST

/** @ingroup TopologyMacros
 * @brief Enable the edge list iterator. 
 * The edge list iterator related APIs follow the design principle of GRIN Iterator.
*/
#define GRIN_ENABLE_EDGE_LIST_ITERATOR

/** @ingroup TopologyMacros
 * @brief Enable the adjacent list structure. 
 * The adjacent list related APIs follow the design principle of GRIN List.
*/
#define GRIN_ENABLE_ADJACENT_LIST

/** @ingroup TopologyMacros
 * @brief Enable the adjacent list iterator. 
 * The adjacent list iterator related APIs follow the design principle of GRIN Iterator.
*/
#define GRIN_ENABLE_ADJACENT_LIST_ITERATOR

/**
 * 
*/
#define GRIN_GRANULA_ENALBE_VERTEX_LIST_BY_TYPE

#define GRIN_GRANULA_ENABLE_ADJACENT_LIST_BY_PARTITION

#define GRIN_GRANULA_ENABLE_ADJACENT_LIST_BY_TYPE

#define GRIN_GRANULA_ENABLE_ADJACENT_LIST_BY_PARTITION_TYPE

#ifndef GRIN_DOXYGEN_SKIP
#undef GRIN_WITH_VERTEX_DATA
#undef GRIN_WITH_EDGE_DATA
#undef GRIN_ENABLE_VERTEX_LIST_ITERATOR
#undef GRIN_ENABLE_EDGE_LIST
#undef GRIN_ENABLE_EDGE_LIST_ITERATOR
#undef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
#endif
///@}


/** @name PartitionMacros
 * @brief Macros for partitioned graph features
 */
///@{
/** @ingroup PartitionMacros
 * @brief Enable partitioned graph. A partitioned graph usually contains
 * several fragments (i.e., local graphs) that are distributedly stored 
 * in a cluster. In GRIN, GRIN_GRAPH represents to a single fragment that can
 * be locally accessed.
 */
#define GRIN_ENABLE_GRAPH_PARTITION

/** @ingroup PartitionMacros
 * @brief The storage provides natural number IDs for partitions.
 * It follows the design principle of natural number ID trait in GRIN.
*/
#define GRIN_NATURAL_PARTITION_ID_TRAIT

/** @ingroup PartitionMacros
 * @brief The storage provides reference of vertex that can be
 * recognized in other partitions where the vertex also appears.
*/
#define GRIN_ENABLE_VERTEX_REF

/** @ingroup PartitionMacros
 * @brief The storage provides reference of edge that can be
 * recognized in other partitions where the edge also appears.
*/
#define GRIN_ENABLE_EDGE_REF

#ifndef GRIN_ENABLE_GRAPH_PARTITION
#undef GRIN_NATURAL_PARTITION_ID_TRAIT
#endif

#ifndef GRIN_DOXYGEN_SKIP
#undef GRIN_ENABLE_EDGE_REF
#endif
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
#define GRIN_WITH_PROPERTY_NAME

/** @ingroup PropertyMacros
 * @brief There are properties bound to vertices. When vertices are typed, vertex
 * properties are bound to vertex types, according to the definition of vertex type.
*/
#define GRIN_WITH_VERTEX_PROPERTY

/** @ingroup PropertyMacros
 * @brief There are primary keys for vertices. Vertex primary keys is
 * a set of vertex properties whose values can distinguish vertices. When vertices are
 * typed, each vertex type has its own primary keys which distinguishes the vertices of
 * that type. 
 * 
 * With primary keys, one can get the vertex from the graph or a certain type
 * by providing the values of the primary keys. The macro is unset if GRIN_WITH_VERTEX_PROPERTY
 * is NOT defined, in which case, one can use GRIN_WITH_VERTEX_ORIGINAL_ID when vertices have
 * no properties.
*/
#define GRIN_WITH_VERTEX_PRIMARY_KEYS

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for vertex types.
 * It follows the design principle of natural ID trait in GRIN.
*/
#define GRIN_NATURAL_VERTEX_TYPE_ID_TRAIT

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for properties bound to
 * a certain vertex type.
 * It follows the design principle of natural ID trait in GRIN.
*/
#define GRIN_NATURAL_VERTEX_PROPERTY_ID_TRAIT


#define GRIN_WITH_EDGE_PROPERTY                // There is any property for edges.
#define GRIN_WITH_EDGE_PRIMARY_KEYS           // There is cross-type property name.
#define GRIN_NATURAL_EDGE_TYPE_ID_TRAIT       // Edge type has natural continuous id from 0.
#define GRIN_NATURAL_EDGE_PROPERTY_ID_TRAIT    // Edge property has natural continuous id from 0.


/** @ingroup PropertyMacros
 * @brief The storage uses column store for properties.
 * This enables efficient property selections for vertices and edges.
*/
#define GRIN_COLUMN_STORE_TRAIT

#if !defined(GRIN_WITH_VERTEX_PROPERTY) && !defined(GRIN_WITH_EDGE_PROPERTY)
#undef GRIN_WITH_PROPERTY_NAME
#endif

#ifndef GRIN_WITH_VERTEX_PROPERTY
#undef GRIN_WITH_VERTEX_PRIMARY_KEYS
#undef GRIN_NATURAL_VERTEX_TYPE_ID_TRAIT
#undef GRIN_NATURAL_VERTEX_PROPERTY_ID_TRAIT
#endif

#ifndef GRIN_WITH_EDGE_PROPERTY
#undef GRIN_WITH_EDGE_PRIMARY_KEYS
#undef GRIN_NATURAL_EDGE_TYPE_ID_TRAIT
#undef GRIN_NATURAL_EDGE_PROPERTY_ID_TRAIT
#endif

#ifndef GRIN_DOXYGEN_SKIP
#undef GRIN_WITH_VERTEX_PRIMARY_KEYS
#undef GRIN_WITH_EDGE_PRIMARY_KEYS
#endif
///@}

/** @name PredicateMacros
 * @brief Macros for predicate features
 */
///@{
/** @ingroup PredicateMacros
 * @brief Enable vertex ordering predicate
*/
#define GRIN_PREDICATE_ENABLE_VERTEX_ORDERING
///@}

/** @name IndexMacros
 * @brief Macros for index features
 */
///@{
/** @ingroup IndexMacros
 * @brief Enable vertex label on graph. 
*/
#define GRIN_WITH_VERTEX_LABEL

/** @ingroup IndexMacros
 * @brief Enable edge label on graph. 
*/
#define GRIN_WITH_EDGE_LABEL

#ifndef GRIN_DOXYGEN_SKIP 
#undef GRIN_WITH_VERTEX_LABEL
#undef GRIN_WITH_EDGE_LABEL
#endif
///@}

/** @name NullValues
 * Macros for Null(invalid) values
 */
///@{
/** @brief Null data type (undefined data type) */
#define GRIN_NULL_DATATYPE Undefined
/** @brief Null graph (invalid return value) */
#define GRIN_NULL_GRAPH NULL
/** @brief Non-existing vertex (invalid return value) */
#define GRIN_NULL_VERTEX NULL
/** @brief Non-existing edge (invalid return value) */
#define GRIN_NULL_EDGE NULL
/** @brief Null list of any kind (invalid return value) */
#define GRIN_NULL_LIST NULL
/** @brief Non-existing partition (invalid return value) */
#define GRIN_NULL_PARTITION NULL
/** @brief Null vertex reference (invalid return value) */
#define GRIN_NULL_VERTEX_REF NULL
/** @brief Null edge reference (invalid return value) */
#define GRIN_NULL_EDGE_REF NULL
/** @brief Non-existing vertex type (invalid return value) */
#define GRIN_NULL_VERTEX_TYPE NULL
/** @brief Non-existing edge type (invalid return value) */
#define GRIN_NULL_EDGE_TYPE NULL
/** @brief Non-existing vertex property (invalid return value) */
#define GRIN_NULL_VERTEX_PROPERTY NULL
/** @brief Non-existing vertex property (invalid return value) */
#define GRIN_NULL_EDGE_PROPERTY NULL
/** @brief Null row (invalid return value) */
#define GRIN_NULL_ROW NULL
/** @brief Null natural id of any kind (invalid return value) */
#define GRIN_NULL_NATURAL_ID UINT_MAX
///@}


/* Define the handlers using typedef */
typedef void* GRIN_GRAPH;                      
typedef void* GRIN_VERTEX;                     
typedef void* GRIN_EDGE;                       

#ifdef GRIN_WITH_VERTEX_ORIGINAL_ID
typedef void* GRIN_VERTEX_ORIGINAL_ID;                   
#endif

#ifdef GRIN_WITH_VERTEX_DATA
typedef void* GRIN_VERTEX_DATA;                 
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST
typedef void* GRIN_VERTEX_LIST;                 
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST_ITERATOR
typedef void* GRIN_VERTEX_LIST_ITERATOR;         
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST
typedef void* GRIN_ADJACENT_LIST;               
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
typedef void* GRIN_ADJACENT_LIST_ITERATOR;       
#endif

#ifdef GRIN_WITH_EDGE_DATA
typedef void* GRIN_EDGE_DATA;                   
#endif

#ifdef GRIN_ENABLE_EDGE_LIST
typedef void* GRIN_EDGE_LIST;                   
#endif

#ifdef GRIN_ENABLE_EDGE_LIST_ITERATOR
typedef void* GRIN_EDGE_LIST_ITERATOR;           
#endif

#ifdef GRIN_ENABLE_GRAPH_PARTITION
typedef void* GRIN_PARTITIONED_GRAPH;
typedef void* GRIN_PARTITION;
typedef void* GRIN_PARTITION_LIST;
#endif

#ifdef GRIN_NATURAL_PARTITION_ID_TRAIT
typedef unsigned GRIN_PARTITION_ID;
#endif

#ifdef GRIN_ENABLE_VERTEX_REF
typedef void* GRIN_VERTEX_REF;
#endif

#ifdef GRIN_ENABLE_EDGE_REF
typedef void* GRIN_EDGE_REF;
#endif


#ifdef GRIN_WITH_VERTEX_PROPERTY
typedef void* GRIN_VERTEX_TYPE;
typedef void* GRIN_VERTEX_TYPE_LIST;
typedef void* GRIN_VERTEX_PROPERTY;
typedef void* GRIN_VERTEX_PROPERTY_LIST;
typedef void* GRIN_VERTEX_PROPERTY_TABLE;
#endif

#ifdef GRIN_NATURAL_VERTEX_TYPE_ID_TRAIT
typedef unsigned GRIN_VERTEX_TYPE_ID;
#endif

#ifdef GRIN_NATURAL_VERTEX_PROPERTY_ID_TRAIT
typedef unsigned GRIN_VERTEX_PROPERTY_ID;
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
typedef void* GRIN_EDGE_TYPE;
typedef void* GRIN_EDGE_TYPE_LIST;
typedef void* GRIN_EDGE_PROPERTY;
typedef void* GRIN_EDGE_PROPERTY_LIST;
typedef void* GRIN_EDGE_PROPERTY_TABLE;
#endif

#ifdef GRIN_NATURAL_EDGE_TYPE_ID_TRAIT
typedef unsigned GRIN_EDGE_TYPE_ID;
#endif

#ifdef GRIN_NATURAL_EDGE_PROPERTY_ID_TRAIT
typedef unsigned GRIN_EDGE_PROPERTY_ID;
#endif

#if defined(GRIN_WITH_VERTEX_PROPERTY) || defined(GRIN_WITH_EDGE_PROPERTY)
typedef void* GRIN_ROW;
#endif

#if defined(GRIN_WITH_VERTEX_LABEL) || defined(GRIN_WITH_EDGE_LABEL)
typedef void* GRIN_LABEL;
typedef void* GRIN_LABEL_LIST;
#endif

#endif  // GRIN_INCLUDE_PREDEFINE_H_
