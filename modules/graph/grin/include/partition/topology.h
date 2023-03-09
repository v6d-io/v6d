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
 @file partition/topology.h
 @brief Define the topoloy related APIs under partitioned graph
*/

#ifndef GRIN_INCLUDE_PARTITION_TOPOLOGY_H_
#define GRIN_INCLUDE_PARTITION_TOPOLOGY_H_

#include "../predefine.h"

#if defined(GRIN_WITH_VERTEX_DATA) && \
    !defined(GRIN_ASSUME_ALL_VERTEX_DATA_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_MASTER_VERTEX_DATA_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE)
bool grin_is_vertex_data_local_complete(GRIN_GRAPH, GRIN_VERTEX);

GRIN_PARTITION_LIST grin_vertex_data_complete_partitions(GRIN_GRAPH, GRIN_VERTEX);
#endif

#if defined(GRIN_WITH_EDGE_DATA) && \
    !defined(GRIN_ASSUME_ALL_EDGE_DATA_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_MASTER_EDGE_DATA_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE) 
bool grin_is_edge_data_local_complete(GRIN_GRAPH, GRIN_EDGE);

GRIN_PARTITION_LIST grin_edge_data_complete_partitions(GRIN_GRAPH, GRIN_EDGE);
#endif


#ifdef GRIN_TRAIT_FILTER_MASTER_FOR_VERTEX_LIST
GRIN_VERTEX_LIST grin_filter_master_for_vertex_list(GRIN_GRAPH, GRIN_VERTEX_LIST);

GRIN_VERTEX_LIST grin_filter_mirror_for_vertex_list(GRIN_GRAPH, GRIN_VERTEX_LIST);
#endif


#ifdef GRIN_TRAIT_FILTER_PARTITION_FOR_VERTEX_LIST
GRIN_VERTEX_LIST grin_filter_partition_for_vertex_list(GRIN_GRAPH, GRIN_PARTITION, GRIN_VERTEX_LIST);
#endif



#ifdef GRIN_TRAIT_FILTER_MASTER_FOR_EDGE_LIST
GRIN_EDGE_LIST grin_filter_master_for_edge_list(GRIN_GRAPH, GRIN_EDGE_LIST);

GRIN_EDGE_LIST grin_filter_mirror_for_edge_list(GRIN_GRAPH, GRIN_EDGE_LIST);
#endif


#ifdef GRIN_TRAIT_FILTER_PARTITION_FOR_EDGE_LIST
GRIN_EDGE_LIST grin_filter_partition_for_edge_list(GRIN_GRAPH, GRIN_PARTITION, GRIN_EDGE_LIST);
#endif


#if defined(GRIN_ENABLE_ADJACENT_LIST) && \
    !defined(GRIN_ASSUME_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE) 
// The concept of local_complete refers to whether we can get complete data or properties
// locally in the partition. It is orthogonal to the concept of master/mirror which
// is mainly designed for data aggregation. In some extremely cases, master vertices
// may NOT contain all the data or properties locally.
bool grin_is_vertex_neighbor_local_complete(GRIN_GRAPH, GRIN_VERTEX);

/**
 * @brief get the partitions whose combination can provide the complete
 * neighbors of a vertex.
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX the vertex
 */
GRIN_PARTITION_LIST grin_vertex_neighbor_complete_partitions(GRIN_GRAPH, GRIN_VERTEX);
#endif


#ifdef GRIN_TRAIT_FILTER_MASTER_NEIGHBOR_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_filter_master_neighbor_for_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST);

GRIN_ADJACENT_LIST grin_filter_mirror_neighbor_for_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST);
#endif

#ifdef GRIN_TRAIT_FILTER_NEIGHBOR_PARTITION_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_filter_neighbor_partition_for_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST);
#endif


#endif // GRIN_INCLUDE_PARTITION_TOPOLOGY_H_