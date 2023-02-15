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

#ifdef ENABLE_GRAPH_PARTITION
// basic partition informations
size_t get_total_partitions_number(const PartitionedGraph);

size_t get_total_vertices_number(const PartitionedGraph);

size_t get_total_edges_number(const PartitionedGraph);


// partition list
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

#ifdef WITH_VERTEX_LABEL
VertexList get_master_vertices_by_label(const Graph, const VertexLabel);

VertexList get_mirror_vertices_by_label(const Graph, const VertexLabel);

VertexList get_mirror_vertices_by_label_partition(const Graph, const VertexLabel, const Partition);
#endif
#endif

#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_ADJACENT_LIST)
AdjacentList get_adjacent_master_list(const Graph, const Direction, const Vertex);

AdjacentList get_adjacent_mirror_list(const Graph, const Direction, const Vertex);

AdjacentList get_adjacent_mirror_list_by_partition(const Graph, const Direction,
                                                   const Partition, const Vertex);
#endif


// Vertex ref refers to the same vertex referred in other partitions,
// while edge ref is likewise. And both can be serialized to char* for
// message transporting nd deserialized on the other end.
VertexRef get_vertex_ref_for_vertex(const Graph, const Partition, const Vertex);

VertexRef get_master_vertex_ref_for_vertex(const Graph, const Vertex);

char* serialize_vertex_ref(const Graph, const VertexRef);

Vertex get_vertex_from_deserialization(const Graph, const Partition, const char*);

EdgeRef get_edge_ref_for_edge(const Graph, const Partition, const Edge);

EdgeRef get_master_edge_ref_for_edge(const Graph, const Edge);

char* serialize_edge_ref(const Graph, const EdgeRef);

Edge get_edge_from_deserialization(const Graph, const Partition, const char*);


// The concept of local_complete refers to whether we can get complete data or properties
// locally in the partition. It is orthogonal to the concept of master/mirror which
// is mainly designed for data aggregation. In some extremely cases, master vertices
// may NOT contain all the data or properties locally.
bool is_vertex_data_local_complete(const Graph, const Vertex);

bool is_vertex_property_local_complete(const Graph, const Vertex);

bool is_edge_data_local_complete(const Graph, const Edge);

bool is_edge_property_local_complete(const Graph, const Edge);

// use valid vertex refs of vertex v (i.e., vertex v refered in other partitions)
// to help aggregate data/property when v is NOT local_complete
#ifdef ENABLE_VALID_VERTEX_REF_LIST
VertexRefList get_all_valid_vertex_ref_list_for_vertex(const Graph, const Vertex);

void destroy_vertex_ref_list(VertexRefList);

size_t get_vertex_ref_list_size(const VertexRefList);

VertexRef get_vertex_ref_from_list(const VertexRefList, size_t);
#endif

#ifdef ENABLE_VALID_VERTEX_REF_LIST_ITERATOR
VertexRefListIterator get_all_valid_vertex_ref_list_begin_for_vertex(const Graph, const Vertex);

bool get_next_vertex_ref_list_iter(VertexRefListIterator);

VertexRef get_vertex_ref_from_iter(VertexRefListIterator);
#endif

#ifdef ENABLE_VALID_EDGE_REF_LIST
EdgeRefList get_all_valid_edge_ref_list_for_edge(const Graph, const Edge);

void destroy_edge_ref_list(EdgeRefList);

size_t get_edge_ref_list_size(const EdgeRefList);

EdgeRef get_edge_ref_from_list(const EdgeRefList, size_t);
#endif

#ifdef ENABLE_VALID_EDGE_REF_LIST_ITERATOR
EdgeRefListIterator get_all_valid_edge_ref_list_begin_for_edge(const Graph, const Edge);

bool get_next_edge_ref_list_iter(EdgeRefListIterator);

EdgeRef get_edge_ref_from_iter(EdgeRefListIterator);
#endif

#endif

#endif  // GRIN_INCLUDE_PARTITION_PARTITION_H_
