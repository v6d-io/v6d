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
 @file property/topology.h
 @brief Define the topology related APIs under property graph
*/

#ifndef GRIN_INCLUDE_PROPERTY_TOPOLOGY_H_
#define GRIN_INCLUDE_PROPERTY_TOPOLOGY_H_

#include "../predefine.h"

#ifdef GRIN_WITH_VERTEX_PROPERTY
size_t grin_get_vertex_num_by_type(GRIN_GRAPH, GRIN_VERTEX_TYPE);
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
size_t grin_get_edge_num_by_type(GRIN_GRAPH, GRIN_DIRECTION, GRIN_EDGE_TYPE);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID
GRIN_VERTEX grin_get_vertex_from_original_id_by_type(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_VERTEX_ORIGINAL_ID);
#endif

#ifdef GRIN_TRAIT_FILTER_TYPE_FOR_VERTEX_LIST
GRIN_VERTEX_LIST grin_filter_type_for_vertex_list(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_VERTEX_LIST);
#endif

#ifdef GRIN_TRAIT_FILTER_TYPE_FOR_EDGE_LIST
GRIN_EDGE_LIST grin_filter_type_for_edge_list(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_EDGE_LIST);
#endif

#ifdef GRIN_TRAIT_FILTER_NEIGHBOR_TYPE_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_filter_neighbor_type_for_adjacent_list(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_ADJACENT_LIST);
#endif

#ifdef GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_filter_edge_type_for_adjacent_list(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_ADJACENT_LIST);
#endif


#endif // GRIN_INCLUDE_PROPERTY_TOPOLOGY_H_