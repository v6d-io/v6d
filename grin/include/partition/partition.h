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

#ifdef PARTITION_STRATEGY

// basic partition informations
size_t get_total_partitions_number(const PartitionedGraph);

size_t get_total_vertices_number(const PartitionedGraph);

PartitionList get_local_partitions(const PartitionedGraph);

void destroy_partition_list(PartitionList);

size_t get_partition_list_size(const PartitionList);

Partition get_partition_from_list(const PartitionList, const size_t);

#ifdef MUTABLE_GRAPH
PartitionList create_partition_list();
bool insert_partition_to_list(PartitionList, const Partition);
#endif

void* get_partition_info(const Partition);

Graph get_local_graph_from_partition(const PartitionedGraph, const Partition);

// serialization & deserialization
char* serialize_remote_partition(const PartitionedGraph, const RemotePartition);

char* serialize_remote_vertex(const PartitionedGraph, const RemoteVertex);

char* serialize_remote_vertex_with_data(const PartitionedGraph,
                                        const RemoteVertex, const VertexData);

char* serialize_remote_edge(const PartitionedGraph, const RemoteEdge);

Partition get_partition_from_deserialization(const PartitionedGraph,
                                             const char*);

Vertex get_vertex_from_deserialization(const PartitionedGraph, const Partition,
                                       const char*);

Vertex get_vertex_from_deserialization_with_data(const PartitionedGraph,
                                                 const Partition, const char*,
                                                 VertexData);

Edge get_edge_from_deserialization(const PartitionedGraph, const Partition,
                                   const char*);

// For local vertex: could get its properties locally, but not sure to have all
// its edges locally, which depends on the partition strategy; every vertex
// could be local in 1~n partitions.
bool is_local_vertex(const PartitionedGraph, const Partition, Vertex);

// For local edge: could get its properties locally;
// every edge could be local in 1/2/n partitions
bool is_local_edge(const PartitionedGraph, const Partition, const Edge);

// For a non-local vertex/edge, get its master partition (a remote partition);
// also, we can get its master vertex/edge (a remote vertex/edge);
// for a local vertex/edge, an invalid value would be returned
RemotePartition get_master_partition_for_vertex(const PartitionedGraph,
                                                const Partition, Vertex);

RemotePartition get_master_partition_for_edge(const PartitionedGraph,
                                              const Partition, const Edge);

RemoteVertex get_master_vertex_for_vertex(const PartitionedGraph,
                                          const Partition, Vertex);

RemoteEdge get_master_edge_for_edge(const PartitionedGraph, const Partition,
                                    const Edge);

// get the partitions in which a vertex exists
RemotePartitionList get_remote_partition_list_for_vertex(const PartitionedGraph,
                                                         const Partition,
                                                         Vertex);

void destroy_remote_partition_list(RemotePartitionList);

size_t get_remote_partition_list_size(const RemotePartitionList);

RemotePartition get_remote_partition_from_list(const RemotePartitionList,
                                               const size_t);

#ifdef MUTABLE_GRAPH
RemotePartitionList create_remote_partition_list();
bool insert_remote_partition_to_list(RemotePartitionList,
                                     const RemotePartition);
#endif

// get the replicas of a vertex
RemoteVertexList get_all_replicas_for_vertex(const PartitionedGraph,
                                             const Partition, Vertex);

void destroy_remote_vertex_list(RemoteVertexList);

size_t get_remote_vertex_list_size(const RemoteVertexList);

RemoteVertex get_remote_vertex_from_list(const RemoteVertexList, const size_t);

#ifdef MUTABLE_GRAPH
RemoteVertexList create_remote_vertex_list();
bool insert_remote_vertex_to_list(RemoteVertexList, const RemoteVertex);
#endif
#endif

#if defined(PARTITION_STRATEGY) && defined(ENABLE_VERTEX_LIST)
VertexList get_local_vertices(const PartitionedGraph, const Partition);

VertexList get_remote_vertices(const PartitionedGraph, const Partition);

VertexList get_remote_vertices_by_partition(const PartitionedGraph,
                                            const RemotePartition);
#endif

#if defined(PARTITION_STRATEGY) && defined(ENABLE_ADJACENT_LIST)
AdjacentList get_local_adjacent_list(const PartitionedGraph, const Direction,
                                     const Partition, Vertex);

AdjacentList get_remote_adjacent_list(const PartitionedGraph, const Direction,
                                      const Partition, Vertex);

AdjacentList get_remote_adjacent_list_by_partition(const PartitionedGraph,
                                                   const Direction,
                                                   const Partition,
                                                   Vertex);
#endif

#endif  // GRIN_INCLUDE_PARTITION_PARTITION_H_
