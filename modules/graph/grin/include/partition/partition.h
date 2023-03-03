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
 @file partition.h
 @brief Define the partition related APIs
*/

#ifndef GRIN_INCLUDE_PARTITION_PARTITION_H_
#define GRIN_INCLUDE_PARTITION_PARTITION_H_

#include "../predefine.h"

#ifdef GRIN_ENABLE_GRAPH_PARTITION
size_t grin_get_total_partitions_number(GRIN_PARTITIONED_GRAPH);

GRIN_PARTITION_LIST grin_get_local_partition_list(GRIN_PARTITIONED_GRAPH);

void grin_destroy_partition_list(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION_LIST);

GRIN_PARTITION_LIST grin_create_partition_list(GRIN_PARTITIONED_GRAPH);

bool grin_insert_partition_to_list(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION_LIST, GRIN_PARTITION);

size_t grin_get_partition_list_size(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION_LIST);

GRIN_PARTITION grin_get_partition_from_list(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION_LIST, size_t);

bool grin_equal_partition(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION, GRIN_PARTITION);

void grin_destroy_partition(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION);

void* grin_get_partition_info(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION);

GRIN_GRAPH grin_get_local_graph_from_partition(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION);
#endif

#ifdef GRIN_NATURAL_PARTITION_ID_TRAIT
GRIN_PARTITION grin_get_partition_from_id(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION_ID);

GRIN_PARTITION_ID grin_get_partition_id(GRIN_PARTITIONED_GRAPH, GRIN_PARTITION);
#endif

// master & mirror vertices for vertexcut partition
// while they refer to inner & outer vertices in edgecut partition
#if defined(GRIN_ENABLE_GRAPH_PARTITION) && defined(GRIN_ENABLE_VERTEX_LIST)
GRIN_VERTEX_LIST grin_get_master_vertices(GRIN_GRAPH);

GRIN_VERTEX_LIST grin_get_mirror_vertices(GRIN_GRAPH);

GRIN_VERTEX_LIST grin_get_mirror_vertices_by_partition(GRIN_GRAPH, GRIN_PARTITION);

#ifdef GRIN_WITH_VERTEX_PROPERTY
GRIN_VERTEX_LIST grin_get_master_vertices_by_type(GRIN_GRAPH, GRIN_VERTEX_TYPE);

GRIN_VERTEX_LIST grin_get_mirror_vertices_by_type(GRIN_GRAPH, GRIN_VERTEX_TYPE);

GRIN_VERTEX_LIST grin_get_mirror_vertices_by_type_partition(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_PARTITION);
#endif
#endif

#if defined(GRIN_ENABLE_GRAPH_PARTITION) && defined(GRIN_ENABLE_ADJACENT_LIST)
/**
 * @brief get the adjacentlist of a vertex where the neigbors are only master vertices
 * @param GRIN_GRAPH the graph
 * @param GRIN_DIRECTION incoming or outgoing
 * @param GRIN_VERTEX the vertex
 */
GRIN_ADJACENT_LIST grin_get_adjacent_master_list(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);

/**
 * @brief get the adjacentlist of a vertex where the neigbors are only mirror vertices
 * @param GRIN_GRAPH the graph
 * @param GRIN_DIRECTION incoming or outgoing
 * @param GRIN_VERTEX the vertex
 */
GRIN_ADJACENT_LIST grin_get_adjacent_mirror_list(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);

/**
 * @brief get the adjacentlist of a vertex where the neigbors are only mirror vertices
 * whose master vertices are in a specific partition
 * @param GRIN_GRAPH the graph
 * @param GRIN_DIRECTION incoming or outgoing
 * @param GRIN_PARTITION the specific partition
 * @param GRIN_VERTEX the vertex
 */
GRIN_ADJACENT_LIST grin_get_adjacent_mirror_list_by_partition(GRIN_GRAPH, GRIN_DIRECTION,
                                                              GRIN_PARTITION, GRIN_VERTEX);
#endif


// Vertex ref refers to the same vertex referred in other partitions,
// while edge ref is likewise. Both can be serialized to const char* for
// message transporting and deserialized on the other end.
#ifdef GRIN_ENABLE_VERTEX_REF
GRIN_VERTEX_REF grin_get_vertex_ref_for_vertex(GRIN_GRAPH, GRIN_VERTEX);

/**
 * @brief get the local vertex from the vertex ref
 * if the vertex ref is not regconized, a null vertex is returned
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_REF the vertex ref
 */
GRIN_VERTEX grin_get_vertex_from_vertex_ref(GRIN_GRAPH, GRIN_VERTEX_REF);

/**
 * @brief get the master partition of a vertex ref.
 * Some storage can still provide the master partition of the vertex ref,
 * even if the vertex ref can NOT be recognized locally.
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_REF the vertex ref
 */
bool grin_is_master_vertex(GRIN_GRAPH, GRIN_VERTEX);

bool grin_is_mirror_vertex(GRIN_GRAPH, GRIN_VERTEX);

GRIN_PARTITION grin_get_master_partition_from_vertex_ref(GRIN_GRAPH, GRIN_VERTEX_REF);

const char* grin_serialize_vertex_ref(GRIN_GRAPH, GRIN_VERTEX_REF);

GRIN_VERTEX_REF grin_deserialize_to_vertex_ref(GRIN_GRAPH, const char*);
#endif

#ifdef GRIN_ENABLE_EDGE_REF
GRIN_EDGE_REF grin_get_edge_ref_for_edge(GRIN_GRAPH, GRIN_PARTITION, GRIN_EDGE);

GRIN_EDGE grin_get_edge_from_edge_ref(GRIN_GRAPH, GRIN_EDGE_REF);

Partition grin_get_master_partition_from_edge_ref(GRIN_GRAPH, GRIN_EDGE_REF);

const char* grin_serialize_edge_ref(GRIN_GRAPH, GRIN_EDGE_REF);

GRIN_EDGE_REF grin_deserialize_to_edge_ref(GRIN_GRAPH, const char*);
#endif

// The concept of local_complete refers to whether we can get complete data or properties
// locally in the partition. It is orthogonal to the concept of master/mirror which
// is mainly designed for data aggregation. In some extremely cases, master vertices
// may NOT contain all the data or properties locally.
bool grin_is_vertex_neighbor_local_complete(GRIN_GRAPH, GRIN_VERTEX);

/**
 * @brief get the partitions whose combination can provide the complete
 * neighbors of a vertex.
 * @param GRIN_GRAPH the graph
 * @param Vertex the vertex
 */
GRIN_PARTITION_LIST grin_vertex_neighbor_complete_partitions(GRIN_GRAPH, GRIN_VERTEX);

#ifdef GRIN_WITH_VERTEX_DATA
bool grin_is_vertex_data_local_complete(GRIN_GRAPH, Vertex);

GRIN_PARTITION_LIST grin_vertex_data_complete_partitions(GRIN_GRAPH, Vertex);
#endif

#ifdef GRIN_WITH_VERTEX_PROPERTY
bool grin_is_vertex_property_local_complete(GRIN_GRAPH, GRIN_VERTEX);

GRIN_PARTITION_LIST grin_vertex_property_complete_partitions(GRIN_GRAPH, GRIN_VERTEX);
#endif

#ifdef GRIN_WITH_EDGE_DATA
bool grin_is_edge_data_local_complete(GRIN_GRAPH, GRIN_EDGE);

GRIN_PARTITION_LIST grin_edge_data_complete_partitions(GRIN_GRAPH, GRIN_EDGE);
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
bool grin_is_edge_property_local_complete(GRIN_GRAPH, GRIN_EDGE);

GRIN_PARTITION_LIST grin_edge_data_complete_partitions(GRIN_GRAPH, GRIN_EDGE);
#endif

#endif  // GRIN_INCLUDE_PARTITION_PARTITION_H_
