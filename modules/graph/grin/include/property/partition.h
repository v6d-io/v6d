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
 @file property/partition.h
 @brief Define the partition related APIs under property graph
*/

#ifndef GRIN_INCLUDE_PROPERTY_PARTITION_H_
#define GRIN_INCLUDE_PROPERTY_PARTITION_H_

#include "../predefine.h"


#if defined(GRIN_WITH_VERTEX_PROPERTY) && \
    !defined(GRIN_ASSUME_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE)
bool grin_is_vertex_property_local_complete(GRIN_GRAPH, GRIN_VERTEX);

GRIN_PARTITION_LIST grin_vertex_property_complete_partitions(GRIN_GRAPH, GRIN_VERTEX);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
GRIN_VERTEX_TYPE_LIST grin_get_master_vertex_property_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
GRIN_VERTEX_TYPE_LIST grin_get_all_vertex_property_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_DATA_LOCAL_COMPLETE
GRIN_VERTEX_TYPE_LIST grin_get_master_vertex_data_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_DATA_LOCAL_COMPLETE
GRIN_VERTEX_TYPE_LIST grin_get_all_vertex_data_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_NEIGHBOR_LOCAL_COMPLETE
GRIN_VERTEX_TYPE_LIST grin_get_master_vertex_neighbor_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_VERTEX_NEIGHBOR_LOCAL_COMPLETE
GRIN_VERTEX_TYPE_LIST grin_get_all_vertex_neighbor_local_complete_types(GRIN_GRAPH);
#endif

#if defined(GRIN_WITH_EDGE_PROPERTY) && \
    !defined(GRIN_ASSUME_ALL_EDGE_PROPERTY_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_ALL_EDGE_PROPERTY_LOCAL_COMPLETE) && \
    !defined(GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE)
bool grin_is_edge_property_local_complete(GRIN_GRAPH, GRIN_EDGE);

GRIN_PARTITION_LIST grin_edge_property_complete_partitions(GRIN_GRAPH, GRIN_EDGE);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_PROPERTY_LOCAL_COMPLETE
GRIN_EDGE_TYPE_LIST grin_get_master_edge_property_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_EDGE_PROPERTY_LOCAL_COMPLETE
GRIN_EDGE_TYPE_LIST grin_get_all_edge_property_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_DATA_LOCAL_COMPLETE
GRIN_EDGE_TYPE_LIST grin_get_master_edge_data_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_EDGE_DATA_LOCAL_COMPLETE
GRIN_EDGE_TYPE_LIST grin_get_all_edge_data_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_MASTER_EDGE_NEIGHBOR_LOCAL_COMPLETE
GRIN_EDGE_TYPE_LIST grin_get_master_edge_neighbor_local_complete_types(GRIN_GRAPH);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_ALL_EDGE_NEIGHBOR_LOCAL_COMPLETE
GRIN_EDGE_TYPE_LIST grin_get_all_edge_neighbor_local_complete_types(GRIN_GRAPH);
#endif


#endif // GRIN_INCLUDE_PROPERTY_PARTITION_H_