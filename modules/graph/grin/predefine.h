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
 * The macros are divided into several sections such as topology, partition,
 * and so on. 
 * In each section, the first part lists all available macros, and undefines
 * all GRIN_ASSUME_ macros by default.
 * After that is the MOST IMPORTANT part for storage implementors, i.e., the StorageSpecific area.
 * Storage implementors should turn ON/OFF the macros in this area based the features of the storage.
 * The final part is the rule part to handle dependencies between macros which should not be edited.
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

/* Section 1: Toplogy */

/** @name TopologyMacros
 * @brief Macros for basic graph topology features
 */
///@{
/** @ingroup TopologyMacros 
 * @brief The storage only support directed graphs.
 */
#define GRIN_ASSUME_GRAPH_DIRECTED

/** @ingroup TopologyMacros 
 * @brief The storage only support undirected graphs.
 */
#define GRIN_ASSUME_GRAPH_UNDIRECTED

/** @ingroup TopologyMacros 
 * @brief The storage only support graphs with single
 * edge between a pair of vertices.
 */
#define GRIN_ASSUME_GRAPH_SINGLE_EDGE

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
 * The vertex list related APIs follow the design of GRIN List.
*/
#define GRIN_ENABLE_VERTEX_LIST

/** @ingroup TopologyMacros
 * @brief Enable the vertex list array-style retrieval. 
 * The vertex list related APIs follow the design of GRIN List.
*/
#define GRIN_ENABLE_VERTEX_LIST_ARRAY

/** @ingroup TopologyMacros
 * @brief Enable the vertex list iterator. 
 * The vertex list iterator related APIs follow the design of GRIN Iterator.
*/
#define GRIN_ENABLE_VERTEX_LIST_ITERATOR

/** @ingroup TopologyMacros
 * @brief Enable the edge list structure. 
 * The edge list related APIs follow the design of GRIN List.
*/
#define GRIN_ENABLE_EDGE_LIST

/** @ingroup TopologyMacros
 * @brief Enable the edge list array-style retrieval. 
 * The edge list related APIs follow the design of GRIN List.
*/
#define GRIN_ENABLE_EDGE_LIST_ARRAY

/** @ingroup TopologyMacros
 * @brief Enable the edge list iterator. 
 * The edge list iterator related APIs follow the design of GRIN Iterator.
*/
#define GRIN_ENABLE_EDGE_LIST_ITERATOR

/** @ingroup TopologyMacros
 * @brief Enable the adjacent list structure. 
 * The adjacent list related APIs follow the design of GRIN List.
*/
#define GRIN_ENABLE_ADJACENT_LIST

/** @ingroup TopologyMacros
 * @brief Enable the adjacent list array-style retrieval. 
 * The adjacent list related APIs follow the design of GRIN List.
*/
#define GRIN_ENABLE_ADJACENT_LIST_ARRAY

/** @ingroup TopologyMacros
 * @brief Enable the adjacent list iterator. 
 * The adjacent list iterator related APIs follow the design of GRIN Iterator.
*/
#define GRIN_ENABLE_ADJACENT_LIST_ITERATOR
///@}


#ifndef GRIN_DOXYGEN_SKIP
/* StorageSpecific */

/* Disable the unsupported features */
#undef GRIN_ASSUME_GRAPH_DIRECTED
#undef GRIN_ASSUME_GRAPH_UNDIRECTED
#undef GRIN_ASSUME_GRAPH_SINGLE_EDGE
#undef GRIN_WITH_VERTEX_DATA
#undef GRIN_WITH_EDGE_DATA
#undef GRIN_ENABLE_EDGE_LIST
#undef GRIN_ENABLE_EDGE_LIST_ITERATOR
/* End of Disable */

/* Enable the supported features */
/* End of Enable */

/* End of StorageSpecific */

#ifndef GRIN_ENABLE_VERTEX_LIST
#undef GRIN_ENABLE_VERTEX_LIST_ARRAY
#undef GRIN_ENABLE_VERTEX_LIST_ITERATOR
#endif

#ifndef GRIN_ENABLE_EDGE_LIST
#undef GRIN_ENABLE_EDGE_LIST_ARRAY
#undef GRIN_ENABLE_EDGE_LIST_ITERATOR
#endif

#ifndef GRIN_ENABLE_ADJACENT_LIST
#undef GRIN_ENABLE_ADJACENT_LIST_ARRAY
#undef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
#endif

#endif  // GRIN_DOXYGEN_SKIP
/* End of Section 1 */

/* Section 2. Partition */

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
 * It follows the design of natural number ID trait in GRIN.
*/
#define GRIN_TRAIT_NATURAL_ID_FOR_PARTITION

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
///@}



/** @name PartitionStrategyMacros
 * @brief Macros to define partition strategy assumptions, a partition strategy
 * can be seen as a combination of detail partition assumptions which are defined after
 * the strategies. Please refer to the documents for strategy details.
*/
///@{
/** @ingroup PartitionStrategyMacros
 * @brief The storage ONLY uses edge-cut partition strategy. This means the 
 * storage's entire partition strategy complies with edge-cut strategy 
 * definition in GRIN.
*/
#define GRIN_ASSUME_EDGE_CUT_PARTITION

/** @ingroup PartitionStrategyMacros
 * @brief The storage ONLY uses vertex-cut partition strategy. This means the 
 * storage's entire partition strategy complies with vertex-cut strategy 
 * definition in GRIN.
*/
#define GRIN_ASSUME_VERTEX_CUT_PARTITION
///@}

/** @name PartitionAssumptionMacros
 * @brief Macros to define detailed partition assumptions with respect to the
 * concept of local complete. Please refer to the documents for the meaning of
 * local complete.
*/
///@{
/** @ingroup PartitionAssumptionMacros
 * @brief Assume the vertex data are local complete for all the vertices,
 * thus there is no need to fetch vertex data from other partitions.
*/
#define GRIN_ASSUME_ALL_VERTEX_DATA_LOCAL_COMPLETE

/** @ingroup PartitionAssumptionMacros
 * @brief Assume the vertex data are local complete for master vertices,
 * and the vertex data of a mirror vertex can be fetched from its master partition.
*/
#define GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE

/** @ingroup PartitionAssumptionMacros
 * @brief Assume the edge data are local complete for all the edges,
 * thus there is no need to fetch edge data from other partitions.
*/
#define GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE

/** @ingroup PartitionAssumptionMacros
 * @brief Assume the edge data are local complete for master edges,
 * and the edge data of a mirror edge can be fetched from its master partition.
*/
#define GRIN_ASSUME_MASTER_EDGE_DATA_LOCAL_COMPLETE

/** @ingroup PartitionAssumptionMacros
 * @brief Assume neighbors of a vertex is always local complete for all vertices. 
*/
#define GRIN_ASSUME_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE

/** @ingroup PartitionAssumptionMacros
 * @brief Assume neighbors of a vertex is always local complete for master vertices. 
*/
#define GRIN_ASSUME_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
///@}

/** @name TraitMirrorPartitionMacros
 * @brief Macros for storage that provides the partition list where the mirror
 * vertices are located. This trait is usually enabled by storages using vertex-cut
 * partition strategy.
*/
///@{
/** @ingroup TraitMirrorPartitionMacros
 * @brief The storage provides the partition list where the mirror
 * vertices are located of a local master vertex.
*/
#define GRIN_TRAIT_MASTER_VERTEX_MIRROR_PARTITION_LIST

/** @ingroup TraitMirrorPartitionMacros
 * @brief The storage provides the partition list where the mirror
 * vertices are located of a local mirror vertex
*/
#define GRIN_TRAIT_MIRROR_VERTEX_MIRROR_PARTITION_LIST

/** @ingroup TraitMirrorPartitionMacros
 * @brief The storage provides the partition list where the mirror
 * edges are located of a local master edge
*/
#define GRIN_TRAIT_MASTER_EDGE_MIRROR_PARTITION_LIST

/** @ingroup TraitMirrorPartitionMacros
 * @brief The storage provides the partition list where the mirror
 * edges are located of a local mirror edge
*/
#define GRIN_TRAIT_MIRROR_EDGE_MIRROR_PARTITION_LIST
///@}

/** @name TraitFilterMacros
 * @brief Macros for storage that provides filtering ability of partitions for structures
 * like vertex list or adjacent list. This trait is usually enabled for efficient graph traversal.
*/
///@{
/** @ingroup TraitFilterMacros
 * @brief The storage provides a filtering predicate of master vertices
 * for vertex list iterator. That means, the caller can use the predicate
 * to make a master-only vertex list iterator from the original iterator.
*/
#define GRIN_TRAIT_FILTER_MASTER_FOR_VERTEX_LIST

/** @ingroup TraitFilterMacros
 * @brief The storage provides a filtering predicate of single partition vertices
 * for vertex list iterator. That means, the caller can use the predicate
 * to make a single-partition vertex list iterator from the original iterator.
*/
#define GRIN_TRAIT_FILTER_PARTITION_FOR_VERTEX_LIST

/** @ingroup TraitFilterMacros
 * @brief The storage provides a filtering predicate of master edges
 * for edge list iterator. That means, the caller can use the predicate
 * to make a master-only edge list iterator from the original iterator.
*/
#define GRIN_TRAIT_FILTER_MASTER_FOR_EDGE_LIST

/** @ingroup TraitFilterMacros
 * @brief The storage provides a filtering predicate of single partition edges
 * for edge list iterator. That means, the caller can use the predicate
 * to make a single-partition edge list iterator from the original iterator.
*/
#define GRIN_TRAIT_FILTER_PARTITION_FOR_EDGE_LIST

/** @ingroup TraitFilterMacros
 * @brief The storage provides a filtering predicate of master neighbors
 * for adjacent list iterator. That means, the caller can use the predicate
 * to make a master-only adjacent list iterator from the original iterator.
*/
#define GRIN_TRAIT_FILTER_MASTER_NEIGHBOR_FOR_ADJACENT_LIST

/** @ingroup TraitFilterMacros
 * @brief The storage provides a filtering predicate of single-partition vertices
 * for adjacent list iterator. That means, the caller can use the predicate
 * to make a single-partition adjacent list iterator from the original iterator.
*/
#define GRIN_TRAIT_FILTER_NEIGHBOR_PARTITION_FOR_ADJACENT_LIST
///@}

#ifndef GRIN_DOXYGEN_SKIP 
// disable GRIN_ASSUME by default
#undef GRIN_ASSUME_EDGE_CUT_PARTITION
#undef GRIN_ASSUME_VERTEX_CUT_PARTITION
#undef GRIN_ASSUME_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE

/* StorageSpecific */

/* Disable the unsupported features */
#undef GRIN_ASSUME_VERTEX_CUT_PARTITION
#undef GRIN_ENABLE_EDGE_REF
#undef GRIN_TRAIT_MASTER_VERTEX_MIRROR_PARTITION_LIST
#undef GRIN_TRAIT_MIRROR_VERTEX_MIRROR_PARTITION_LIST
#undef GRIN_TRAIT_FILTER_PARTITION_FOR_VERTEX_LIST
#undef GRIN_TRAIT_FILTER_MASTER_NEIGHBOR_FOR_ADJACENT_LIST
#undef GRIN_TRAIT_FILTER_NEIGHBOR_PARTITION_FOR_ADJACENT_LIST
/* End of Disable */

/* Enable the supported features */
#define GRIN_ASSUME_EDGE_CUT_PARTITION  
/* End of Enable */

/* End of StorageSpecific */

#ifdef GRIN_ASSUME_EDGE_CUT_PARTITION
#define GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#define GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE
#define GRIN_ASSUME_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_VERTEX_CUT_PARTITION
#define GRIN_ASSUME_ALL_VERTEX_DATA_LOCAL_COMPLETE
#define GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE
#define GRIN_TRAIT_MASTER_VERTEX_MIRROR_PARTITION_LIST
#endif

#ifndef GRIN_ENABLE_GRAPH_PARTITION
#undef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
#undef GRIN_ENABLE_VERTEX_REF
#undef GRIN_ENABLE_EDGE_REF
#undef GRIN_ASSUME_EDGE_CUT_PARTITION
#undef GRIN_ASSUME_VERTEX_CUT_PARTITION
#endif

#ifndef GRIN_ENABLE_VERTEX_REF  // enable vertex pref is the prerequisite
#undef GRIN_TRAIT_MASTER_VERTEX_MIRROR_PARTITION_LIST
#undef GRIN_TRAIT_MIRROR_VERTEX_MIRROR_PARTITION_LIST
#undef GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_TRAIT_FILTER_MASTER_FOR_VERTEX_LIST
#endif

#ifndef GRIN_ENABLE_EDGE_REF  // enable edge pref is the prerequisite
#undef GRIN_TRAIT_MASTER_EDGE_MIRROR_PARTITION_LIST
#undef GRIN_TRAIT_MIRROR_EDGE_MIRROR_PARTITION_LIST
#undef GRIN_ASSUME_MASTER_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_TRAIT_FILTER_MASTER_FOR_EDGE_LIST
#endif

#ifndef GRIN_WITH_VERTEX_DATA  // enable vertex data is the prerequisite
#undef GRIN_ASSUME_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#endif

#ifndef GRIN_WITH_EDGE_DATA  // enable edge data is the prerequisite
#undef GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_EDGE_DATA_LOCAL_COMPLETE
#endif

#ifndef GRIN_ENABLE_VERTEX_LIST  // enable vertex list iterator is the prerequisite
#undef GRIN_TRAIT_FILTER_MASTER_FOR_VERTEX_LIST
#undef GRIN_TRAIT_FILTER_PARTITION_FOR_VERTEX_LIST
#endif

#ifndef GRIN_ENABLE_EDGE_LIST  // enable edge list iterator is the prerequisite
#undef GRIN_TRAIT_FILTER_MASTER_FOR_EDGE_LIST
#undef GRIN_TRAIT_FILTER_PARTITION_FOR_EDGE_LIST
#endif

#ifndef GRIN_ENABLE_ADJACENT_LIST  // enable adjacent list iterator is the prerequisite
#undef GRIN_TRAIT_FILTER_MASTER_NEIGHBOR_FOR_ADJACENT_LIST
#undef GRIN_TRAIT_FILTER_NEIGHBOR_PARTITION_FOR_ADJACENT_LIST
#endif

#ifdef GRIN_ASSUME_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_EDGE_DATA_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#endif
#endif // GRIN_DOXY_SKIP
/* End of Section 2 */

/* Section 3. Property */

/** @name PropertyMacros
 * @brief Macros for basic property graph features
 */
///@{
/** @ingroup PropertyMacros
 * @brief Enable the pure data structure Row, which is used in primary keys and tables.
*/
#define GRIN_ENABLE_ROW

/** @ingroup PropertyMacros
 * @brief There are properties bound to vertices. When vertices are typed, vertex
 * properties are bound to vertex types, according to the definition of vertex type.
*/
#define GRIN_WITH_VERTEX_PROPERTY

/** @ingroup PropertyMacros
 * @brief There are property names for vertex properties. The relationship between property
 * name and properties is one-to-many, because properties bound to different vertex/edge
 * types are distinguished even they may share the same property name. Please refer to
 * the design of Property for details.
*/
#define GRIN_WITH_VERTEX_PROPERTY_NAME

/** @ingroup PropertyMacros
 * @brief There are unique names for each vertex type.
*/
#define GRIN_WITH_VERTEX_TYPE_NAME

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for vertex types.
 * It follows the design of natural ID trait in GRIN.
*/
#define GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE

/** @ingroup PropertyMacros
 * @brief Enable the vertex property table structure, from where the value of property
 * can be fetched using vertex as row index and property as column index.
*/
#define GRIN_ENABLE_VERTEX_PROPERTY_TABLE

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
#define GRIN_ENABLE_VERTEX_PRIMARY_KEYS

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for properties bound to
 * a certain vertex type.
 * It follows the design of natural ID trait in GRIN.
*/
#define GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY

/** @ingroup PropertyMacros
 * @brief Assume the original id is ONLY unique under each vertex type. This means
 * to get a vertex from the original id, the caller must also provide the vertex type.
*/
#define GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID


/** @ingroup PropertyMacros
 * @brief There are properties bound to edges. When edges are typed, edge
 * properties are bound to edge types, according to the definition of edge type.
*/
#define GRIN_WITH_EDGE_PROPERTY

/** @ingroup PropertyMacros
 * @brief There are property names for edge properties. The relationship between property
 * name and properties is one-to-many, because properties bound to different vertex/edge
 * types are distinguished even they may share the same property name. Please refer to
 * the design of Property for details.
*/
#define GRIN_WITH_EDGE_PROPERTY_NAME

/** @ingroup PropertyMacros
 * @brief There are unique names for each edge type.
*/
#define GRIN_WITH_EDGE_TYPE_NAME

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for edge types.
 * It follows the design of natural ID trait in GRIN.
*/
#define GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE

/** @ingroup PropertyMacros
 * @brief Enable the edge property table structure, from where the value of property
 * can be fetched using edge as row index and property as column index.
*/
#define GRIN_ENABLE_EDGE_PROPERTY_TABLE

/** @ingroup PropertyMacros
 * @brief There are primary keys for edges. Edge primary keys is
 * a set of edge properties whose values can distinguish edges. When edges are
 * typed, each edge type has its own primary keys which distinguishes the edges of
 * that type. 
 * 
 * With primary keys, one can get the edge from the graph or a certain type
 * by providing the values of the primary keys. The macro is unset if GRIN_WITH_EDGE_PROPERTY
 * is NOT defined, in which case, one can use GRIN_WITH_EDGE_ORIGINAL_ID when edges have
 * no properties.
*/
#define GRIN_ENABLE_EDGE_PRIMARY_KEYS

/** @ingroup PropertyMacros
 * @brief The storage provides natural number IDs for properties bound to
 * a certain edge type.
 * It follows the design of natural ID trait in GRIN.
*/
#define GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
///@}

/** @name TraitFilterTypeMacros
 * @brief Macros of traits to filter vertex/edge type for
 * structures like vertex list and adjacent list.
 */
///@{
/** @ingroup TraitFilterTypeMacros
 * @brief The storage provides a filtering predicate of single-type vertices
 * for vertex list iterator. That means, the caller can use the predicate
 * to make a vertex list iterator for a certain type of vertices from the 
 * original iterator.
*/
#define GRIN_TRAIT_FILTER_TYPE_FOR_VERTEX_LIST

/** @ingroup TraitFilterTypeMacros
 * @brief The storage provides a filtering predicate of single-type edges
 * for edge list iterator. That means, the caller can use the predicate
 * to make an edge list iterator for a certain type of edges from the 
 * original iterator.
*/
#define GRIN_TRAIT_FILTER_TYPE_FOR_EDGE_LIST

/** @ingroup TraitFilterTypeMacros
 * @brief The storage provides a filtering predicate of single-type neighbors
 * for adjacent list iterator. That means, the caller can use the predicate
 * to make an adjacent list iterator of neighbors with a certain type from 
 * the original iterator.
*/
#define GRIN_TRAIT_FILTER_NEIGHBOR_TYPE_FOR_ADJACENT_LIST

/** @ingroup TraitFilterTypeMacros
 * @brief The storage provides a filtering predicate of single-type edges
 * for adjacent list iterator. That means, the caller can use the predicate
 * to make an adjacent list iterator of edges with a certain type from 
 * the original iterator.
*/
#define GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
///@}


/** @name PropetyAssumptionMacros
 * @brief Macros of assumptions for property local complete, and particularly define
 * the by type local complete assumptions for hybrid partiton strategy.
 */
///@{
/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a vertex is always local complete for all vertices. 
*/
#define GRIN_ASSUME_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a vertex is ONLY local complete for master vertices. 
*/
#define GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a vertex is local complete for all vertices with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a vertex is local complete for master vertices with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume vertex data is local complete for all vertices with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume vertex data is local complete for master vertices with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a edge is always local complete for all edges. 
*/
#define GRIN_ASSUME_ALL_EDGE_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a edge is ONLY local complete for master edges. 
*/
#define GRIN_ASSUME_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a edge is local complete for all edges with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_ALL_EDGE_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume property values of a edge is local complete for master edges with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume edge data is local complete for all edges with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume edge data is local complete for master edges with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume vertex neighbor is local complete for all vertices with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief Assume vertex data is local complete for master vertices with a certain type. 
*/
#define GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE

/** @ingroup PropetyAssumptionMacros
 * @brief The storage uses column store for properties.
 * This enables efficient property selections for vertices and edges.
*/
#define GRIN_ASSUME_COLUMN_STORE
///@}

#ifndef GRIN_DOXYGEN_SKIP  
// disable GRIN_ASSUME by default
#undef GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID
#undef GRIN_ASSUME_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_COLUMN_STORE

/* StorageSpecific */

/* Disable the unsupported features */
#undef GRIN_ENABLE_VERTEX_PRIMARY_KEYS
#undef GRIN_ENABLE_EDGE_PRIMARY_KEYS
#undef GRIN_TRAIT_FILTER_NEIGHBOR_TYPE_FOR_ADJACENT_LIST
/* End of Disable */

/* Enable the supported features */
#define GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID
#define GRIN_ASSUME_COLUMN_STORE
/* End of Enable */

/* End of StorageSpecific */

#ifdef GRIN_ASSUME_EDGE_CUT_PARTITION
#define GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#define GRIN_ASSUME_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_VERTEX_CUT_PARTITION
#define GRIN_ASSUME_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#define GRIN_ASSUME_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#endif

#ifndef GRIN_ENABLE_ROW
#undef GRIN_ENABLE_VERTEX_PRIMARY_KEYS
#undef GRIN_ENABLE_EDGE_PRIMARY_KEYS
#endif

#ifndef GRIN_WITH_VERTEX_PROPERTY
#undef GRIN_WITH_VERTEX_PROPERTY_NAME
#undef GRIN_WITH_VERTEX_TYPE_NAME
#undef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
#undef GRIN_ENABLE_VERTEX_PROPERTY_TABLE
#undef GRIN_ENABLE_VERTEX_PRIMARY_KEYS
#undef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY
#undef GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID
#undef GRIN_ASSUME_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#endif

#ifndef GRIN_WITH_EDGE_PROPERTY
#undef GRIN_WITH_EDGE_PROPERTY_NAME
#undef GRIN_WITH_EDGE_TYPE_NAME
#undef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
#undef GRIN_ENABLE_EDGE_PROPERTY_TABLE
#undef GRIN_ENABLE_EDGE_PRIMARY_KEYS
#undef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
#undef GRIN_ASSUME_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE
#endif

#ifndef GRIN_WITH_VERTEX_ORIGINAL_ID
#undef GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID
#endif

#ifndef GRIN_WITH_VERTEX_DATA
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#endif

#ifndef GRIN_WITH_EDGE_DATA
#undef GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE
#endif

#ifndef GRIN_ENABLE_VERTEX_LIST  // enable vertex list iterator is the prerequisite
#undef GRIN_TRAIT_FILTER_TYPE_FOR_VERTEX_LIST
#endif

#ifndef GRIN_ENABLE_EDGE_LIST  // enable edge list iterator is the prerequisite
#undef GRIN_TRAIT_FILTER_TYPE_FOR_EDGE_LIST
#endif

#ifndef GRIN_ENABLE_ADJACENT_LIST // enable adjacent list iterator is the prerequisite
#undef GRIN_TRAIT_FILTER_NEIGHBOR_TYPE_FOR_ADJACENT_LIST
#undef GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
#endif

// assumption on vertex property
#ifdef GRIN_ASSUME_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
#endif

// assumption on vertex data
#ifdef GRIN_ASSUME_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE
#endif

// assumption on edge property
#ifdef GRIN_ASSUME_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
#endif

// assumption on edge data
#ifdef GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_MASTER_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE
#endif

// assumption on vertex neighbor
#ifdef GRIN_ASSUME_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#undef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
#endif
#endif // GRIN_DOXY_SKIP
/* End of Section 3 */

/* Section 4. Index */
/** @name IndexLabelMacros
 * @brief Macros for label features
 */
///@{
/** @ingroup IndexLabelMacros
 * @brief Enable vertex label on graph. 
*/
#define GRIN_WITH_VERTEX_LABEL

/** @ingroup IndexLabelMacros
 * @brief Enable edge label on graph. 
*/
#define GRIN_WITH_EDGE_LABEL
///@}

/** @name IndexOrderMacros
 * @brief Macros for ordering features.
 * Please refer to the order section in the documents for details.
 */
///@{
/** @ingroup IndexOrderMacros
 * @brief assume all vertex list are sorted.
 * We will expend the assumption to support master/mirror or
 * by type in the future if needed.
*/
#define GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
///@}

#ifndef GRIN_DOXYGEN_SKIP
// disable assumption by default
#undef GRIN_ASSUME_ALL_VERTEX_LIST_SORTED

/* StorageSpecific */

/* Disable the unsupported features */
#undef GRIN_WITH_VERTEX_LABEL
#undef GRIN_WITH_EDGE_LABEL
/* End of Disable */

/* Enable the supported features */
#define GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
/* End of Enable */

/* End of StorageSpecific */
#endif  // GRIN_DOXYGEN_SKIP
/* End of Section 4 */

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
/** @brief Null list iterator of any kind (invalid return value) */
#define GRIN_NULL_LIST_ITERATOR NULL
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
/** @brief Null size (invalid return value) */
#define GRIN_NULL_SIZE UINT_MAX
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

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
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

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
typedef unsigned GRIN_VERTEX_TYPE_ID;
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY
typedef unsigned GRIN_VERTEX_PROPERTY_ID;
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
typedef void* GRIN_EDGE_TYPE;
typedef void* GRIN_EDGE_TYPE_LIST;
typedef void* GRIN_EDGE_PROPERTY;
typedef void* GRIN_EDGE_PROPERTY_LIST;
typedef void* GRIN_EDGE_PROPERTY_TABLE;
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
typedef unsigned GRIN_EDGE_TYPE_ID;
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
typedef unsigned GRIN_EDGE_PROPERTY_ID;
#endif

#ifdef GRIN_ENABLE_ROW
typedef void* GRIN_ROW;
#endif

#if defined(GRIN_WITH_VERTEX_LABEL) || defined(GRIN_WITH_EDGE_LABEL)
typedef void* GRIN_LABEL;
typedef void* GRIN_LABEL_LIST;
#endif

#endif  // GRIN_INCLUDE_PREDEFINE_H_
