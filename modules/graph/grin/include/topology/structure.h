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

#ifndef GRIN_INCLUDE_TOPOLOGY_STRUCTURE_H_
#define GRIN_INCLUDE_TOPOLOGY_STRUCTURE_H_

#include "../predefine.h"

// Graph 
bool grin_is_directed(GRIN_GRAPH);

bool grin_is_multigraph(GRIN_GRAPH);

size_t grin_get_vertex_num(GRIN_GRAPH);

size_t grin_get_edge_num(GRIN_GRAPH, GRIN_DIRECTION);


// Vertex
void grin_destroy_vertex(GRIN_GRAPH, GRIN_VERTEX);

bool grin_equal_vertex(GRIN_GRAPH, GRIN_VERTEX, GRIN_VERTEX);

#ifdef GRIN_WITH_VERTEX_ORIGINAL_ID
void grin_destroy_vertex_original_id(GRIN_GRAPH, GRIN_VERTEX_ORIGINAL_ID);

GRIN_DATATYPE grin_get_vertex_original_id_type(GRIN_GRAPH);

GRIN_VERTEX_ORIGINAL_ID grin_get_vertex_original_id(GRIN_GRAPH, GRIN_VERTEX);
#endif

#if defined(GRIN_WITH_VERTEX_ORIGINAL_ID) && !defined(GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID)
GRIN_VERTEX grin_get_vertex_from_original_id(GRIN_GRAPH, GRIN_VERTEX_ORIGINAL_ID);
#endif

#ifdef GRIN_WITH_VERTEX_DATA
GRIN_DATATYPE grin_get_vertex_data_type(GRIN_GRAPH, GRIN_VERTEX);

GRIN_VERTEX_DATA grin_get_vertex_data_value(GRIN_GRAPH, GRIN_VERTEX);

void grin_destroy_vertex_data(GRIN_GRAPH, GRIN_VERTEX_DATA);
#endif

// Edge
void grin_destroy_edge(GRIN_GRAPH, GRIN_EDGE);

GRIN_VERTEX grin_get_edge_src(GRIN_GRAPH, GRIN_EDGE);

GRIN_VERTEX grin_get_edge_dst(GRIN_GRAPH, GRIN_EDGE);

#ifdef GRIN_WITH_EDGE_DATA
GRIN_DATATYPE grin_get_edge_data_type(GRIN_GRAPH, GRIN_EDGE);

GRIN_EDGE_DATA grin_get_edge_data_value(GRIN_GRAPH, GRIN_EDGE);

void grin_destroy_edge_data(GRIN_GRAPH, GRIN_EDGE_DATA);
#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_STRUCTURE_H_
