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

#ifdef ENABLE_GRAPH_PARTITION
size_t get_total_partitions_number(PartitionedGraph);

PartitionList get_local_partition_list(PartitionedGraph);

void destroy_partition_list(PartitionList);

PartitionList create_partition_list();

bool insert_partition_to_list(PartitionList, Partition);

size_t get_partition_list_size(PartitionList);

Partition get_partition_from_list(PartitionList, size_t);

void destroy_partition(Partition);

void* get_partition_info(PartitionedGraph, Partition);

Graph get_local_graph_from_partition(PartitionedGraph, Partition);
#endif

#ifdef NATURAL_PARTITION_ID_TRAIT
Partition get_partition_from_id(PartitionID);

PartitionID get_partition_id(Partition);
#endif

// master & mirror vertices for vertexcut partition
// while they refer to inner & outer vertices in edgecut partition
#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_VERTEX_LIST)
VertexList get_master_vertices(Graph);

VertexList get_mirror_vertices(Graph);

VertexList get_mirror_vertices_by_partition(Graph, Partition);

#ifdef WITH_VERTEX_PROPERTY
VertexList get_master_vertices_by_type(Graph, VertexType);

VertexList get_mirror_vertices_by_type(Graph, VertexType);

VertexList get_mirror_vertices_by_type_partition(Graph, VertexType, Partition);
#endif
#endif

#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_ADJACENT_LIST)
/**
 * @brief get the adjacentlist of a vertex where the neigbors are only master vertices
 * @param Graph the graph
 * @param Direction incoming or outgoing
 * @param Vertex the vertex
 */
AdjacentList get_adjacent_master_list(Graph, Direction, Vertex);

/**
 * @brief get the adjacentlist of a vertex where the neigbors are only mirror vertices
 * @param Graph the graph
 * @param Direction incoming or outgoing
 * @param Vertex the vertex
 */
AdjacentList get_adjacent_mirror_list(Graph, Direction, Vertex);

/**
 * @brief get the adjacentlist of a vertex where the neigbors are only mirror vertices
 * whose master vertices are in a specific partition
 * @param Graph the graph
 * @param Direction incoming or outgoing
 * @param Paritition the specific partition
 * @param Vertex the vertex
 */
AdjacentList get_adjacent_mirror_list_by_partition(Graph, Direction,
                                                   Partition, Vertex);
#endif


// Vertex ref refers to the same vertex referred in other partitions,
// while edge ref is likewise. Both can be serialized to const char* for
// message transporting and deserialized on the other end.
#ifdef ENABLE_VERTEX_REF
VertexRef get_vertex_ref_for_vertex(Graph, Partition, Vertex);

/**
 * @brief get the local vertex from the vertex ref
 * if the vertex ref is not regconized, a null vertex is returned
 * @param Graph the graph
 * @param VertexRef the vertex ref
 */
Vertex get_vertex_from_vertex_ref(Graph, VertexRef);

/**
 * @brief get the master partition of a vertex ref.
 * Some storage can still provide the master partition of the vertex ref,
 * even if the vertex ref can NOT be recognized locally.
 * @param Graph the graph
 * @param VertexRef the vertex ref
 */
bool is_master_vertex(Graph, Vertex);

bool is_mirror_vertex(Graph, Vertex);

Partition get_master_partition_from_vertex_ref(Graph, VertexRef);

const char* serialize_vertex_ref(Graph, VertexRef);

VertexRef deserialize_to_vertex_ref(Graph, const char*);
#endif

#ifdef ENABLE_EDGE_REF
EdgeRef get_edge_ref_for_edge(Graph, Partition, Edge);

Edge get_edge_from_edge_ref(Graph, EdgeRef);

Partition get_master_partition_from_edge_ref(Graph, EdgeRef);

const char* serialize_edge_ref(Graph, EdgeRef);

EdgeRef deserialize_to_edge_ref(Graph, const char*);
#endif

// The concept of local_complete refers to whether we can get complete data or properties
// locally in the partition. It is orthogonal to the concept of master/mirror which
// is mainly designed for data aggregation. In some extremely cases, master vertices
// may NOT contain all the data or properties locally.
bool is_vertex_neighbor_local_complete(Graph, Vertex);

/**
 * @brief get the partitions whose combination can provide the complete
 * neighbors of a vertex.
 * @param Graph the graph
 * @param Vertex the vertex
 */
PartitionList vertex_neighbor_complete_partitions(Graph, Vertex);

#ifdef WITH_VERTEX_DATA
bool is_vertex_data_local_complete(Graph, Vertex);

PartitionList vertex_data_complete_partitions(Graph, Vertex);
#endif

#ifdef WITH_VERTEX_PROPERTY
bool is_vertex_property_local_complete(Graph, Vertex);

PartitionList vertex_property_complete_partitions(Graph, Vertex);
#endif

#ifdef WITH_EDGE_DATA
bool is_edge_data_local_complete(Graph, Edge);

PartitionList edge_data_complete_partitions(Graph, Edge);
#endif

#ifdef WITH_EDGE_PROPERTY
bool is_edge_property_local_complete(Graph, Edge);

PartitionList edge_data_complete_partitions(Graph, Edge);
#endif

#endif  // GRIN_INCLUDE_PARTITION_PARTITION_H_
