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

#ifndef GRIN_INCLUDE_PARTITION_PARTITION_H_
#define GRIN_INCLUDE_PARTITION_PARTITION_H_

#include "../predefine.h"

**** add enable_vertex/edge_ref macros ****

#ifdef ENABLE_GRAPH_PARTITION
size_t get_total_partitions_number(const PartitionedGraph);

PartitionList get_local_partition_list(const PartitionedGraph);

void destroy_partition_list(PartitionList);

PartitionList create_partition_list();

bool insert_partition_to_list(PartitionList, const Partition);

size_t get_partition_list_size(const PartitionList);

Partition get_partition_from_list(const PartitionList, const size_t);

#ifdef NATURAL_PARTITION_ID_TRAIT
Partition get_partition_from_id(const PartitionID);

PartitionID get_partition_id(const Partition);
#endif

void* get_partition_info(const PartitionedGraph, const Partition);

Graph get_local_graph_from_partition(const PartitionedGraph, const Partition);


// master & mirror vertices for vertexcut partition
// while they refer to inner & outer vertices in edgecut partition
#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_VERTEX_LIST)
VertexList get_master_vertices(const Graph);

VertexList get_mirror_vertices(const Graph);

VertexList get_mirror_vertices_by_partition(const Graph, const Partition);

#ifdef WITH_VERTEX_PROPERTY
VertexList get_master_vertices_by_type(const Graph, const VertexType);

VertexList get_mirror_vertices_by_type(const Graph, const VertexType);

VertexList get_mirror_vertices_by_type_partition(const Graph, const VertexType, const Partition);
#endif
#endif

#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_ADJACENT_LIST)
AdjacentList get_adjacent_master_list(const Graph, const Direction, const Vertex);

AdjacentList get_adjacent_mirror_list(const Graph, const Direction, const Vertex);

AdjacentList get_adjacent_mirror_list_by_partition(const Graph, const Direction,
                                                   const Partition, const Vertex);
#endif


// Vertex ref refers to the same vertex referred in other partitions,
// while edge ref is likewise. Both can be serialized to char* for
// message transporting and deserialized on the other end.
VertexRef get_vertex_ref_for_vertex(const Graph, const Partition, const Vertex);

Vertex get_vertex_from_vertex_ref(const Graph, const VertexRef);

Partition get_master_partition_from_vertex_ref(const Graph, const VertexRef);

char* serialize_vertex_ref(const Graph, const VertexRef);

VertexRef deserialize_to_vertex_ref(const Graph, const char*);

EdgeRef get_edge_ref_for_edge(const Graph, const Partition, const Edge);

Edge get_edge_from_edge_ref(const Graph, const EdgeRef);

Partition get_master_partition_from_edge_ref(const Graph, const EdgeRef);

char* serialize_edge_ref(const Graph, const EdgeRef);

EdgeRef deserialize_to_edge_ref(const Graph, const char*);

// The concept of local_complete refers to whether we can get complete data or properties
// locally in the partition. It is orthogonal to the concept of master/mirror which
// is mainly designed for data aggregation. In some extremely cases, master vertices
// may NOT contain all the data or properties locally.
bool is_vertex_neighbor_local_complete(const Graph, const Vertex);

PartitionList vertex_neighbor_complete_partitions(const Graph, const Vertex);

#ifdef WITH_VERTEX_DATA
bool is_vertex_data_local_complete(const Graph, const Vertex);

PartitionList vertex_data_complete_partitions(const Graph, const Vertex);
#endif

#ifdef WITH_VERTEX_PROPERTY
bool is_vertex_property_local_complete(const Graph, const Vertex);

PartitionList vertex_property_complete_partitions(const Graph, const Vertex);
#endif

#ifdef WITH_EDGE_DATA
bool is_edge_data_local_complete(const Graph, const Edge);

PartitionList edge_data_complete_partitions(const Graph, const Edge);
#endif

#ifdef WITH_EDGE_PROPERTY
bool is_edge_property_local_complete(const Graph, const Edge);

PartitionList edge_data_complete_partitions(const Graph, const Edge);
#endif

#endif

#endif  // GRIN_INCLUDE_PARTITION_PARTITION_H_
