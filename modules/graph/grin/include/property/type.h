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
 @file type.h
 @brief Define the vertex/edge type related APIs
*/

#ifndef GRIN_INCLUDE_PROPERTY_TYPE_H_
#define GRIN_INCLUDE_PROPERTY_TYPE_H_

#include "../predefine.h"

#ifdef GRIN_WITH_VERTEX_PROPERTY
// Vertex type
bool grin_equal_vertex_type(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_VERTEX_TYPE);

GRIN_VERTEX_TYPE grin_get_vertex_type(GRIN_GRAPH, GRIN_VERTEX);

// Vertex type list
GRIN_VERTEX_TYPE_LIST grin_get_vertex_type_list(GRIN_GRAPH);

void grin_destroy_vertex_type_list(GRIN_GRAPH, GRIN_VERTEX_TYPE_LIST);

GRIN_VERTEX_TYPE_LIST grin_create_vertex_type_list(GRIN_GRAPH);

bool grin_insert_vertex_type_to_list(GRIN_GRAPH, GRIN_VERTEX_TYPE_LIST, GRIN_VERTEX_TYPE);

size_t grin_get_vertex_type_list_size(GRIN_GRAPH, GRIN_VERTEX_TYPE_LIST);

GRIN_VERTEX_TYPE grin_get_vertex_type_from_list(GRIN_GRAPH, GRIN_VERTEX_TYPE_LIST, size_t);
#endif

#ifdef GRIN_WITH_VERTEX_TYPE_NAME
const char* grin_get_vertex_type_name(GRIN_GRAPH, GRIN_VERTEX_TYPE);

GRIN_VERTEX_TYPE grin_get_vertex_type_by_name(GRIN_GRAPH, const char*);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
GRIN_VERTEX_TYPE_ID grin_get_vertex_type_id(GRIN_GRAPH, GRIN_VERTEX_TYPE);

GRIN_VERTEX_TYPE grin_get_vertex_type_from_id(GRIN_GRAPH, GRIN_VERTEX_TYPE_ID);
#endif


#ifdef GRIN_WITH_EDGE_PROPERTY
// Edge type
bool grin_equal_edge_type(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_EDGE_TYPE);

GRIN_EDGE_TYPE grin_get_edge_type(GRIN_GRAPH, GRIN_EDGE);

// Edge type list
GRIN_EDGE_TYPE_LIST grin_get_edge_type_list(GRIN_GRAPH);

void grin_destroy_edge_type_list(GRIN_GRAPH, GRIN_EDGE_TYPE_LIST);

GRIN_EDGE_TYPE_LIST grin_create_edge_type_list(GRIN_GRAPH);

bool grin_insert_edge_type_to_list(GRIN_GRAPH, GRIN_EDGE_TYPE_LIST, GRIN_EDGE_TYPE);

size_t grin_get_edge_type_list_size(GRIN_GRAPH, GRIN_EDGE_TYPE_LIST);

GRIN_EDGE_TYPE grin_get_edge_type_from_list(GRIN_GRAPH, GRIN_EDGE_TYPE_LIST, size_t);
#endif

#ifdef GRIN_WITH_EDGE_TYPE_NAME
const char* grin_get_edge_type_name(GRIN_GRAPH, GRIN_EDGE_TYPE);

GRIN_EDGE_TYPE grin_get_edge_type_by_name(GRIN_GRAPH, const char*);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
GRIN_EDGE_TYPE_ID grin_get_edge_type_id(GRIN_GRAPH, GRIN_EDGE_TYPE);

GRIN_EDGE_TYPE grin_get_edge_type_from_id(GRIN_GRAPH, GRIN_EDGE_TYPE_ID);
#endif

/** @name VertexEdgeTypeRelation
 * GRIN assumes the relation between edge type and pairs of vertex types is many-to-many.
 * Thus GRIN returns the pairs of vertex types related to an edge type as a pair of vertex type
 * lists of the same size, and the src/dst vertex types are aligned with their positions in the lists.
 */
///@{
#if defined(GRIN_WITH_VERTEX_PROPERTY) && defined(GRIN_WITH_EDGE_PROPERTY)
/** @brief  the src vertex type list */
GRIN_VERTEX_TYPE_LIST grin_get_src_types_from_edge_type(GRIN_GRAPH, GRIN_EDGE_TYPE);

/** @brief get the dst vertex type list */
GRIN_VERTEX_TYPE_LIST grin_get_dst_types_from_edge_type(GRIN_GRAPH, GRIN_EDGE_TYPE);

/** @brief get the edge type list related to a given pair of vertex types */
GRIN_EDGE_TYPE_LIST grin_get_edge_types_from_vertex_type_pair(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_VERTEX_TYPE);
#endif
///@}

#endif  // GRIN_INCLUDE_PROPERTY_TYPE_H_