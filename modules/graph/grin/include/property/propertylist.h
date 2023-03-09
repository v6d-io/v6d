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
 @file propertylist.h
 @brief Define the property list related and graph projection APIs
*/

#ifndef GRIN_INCLUDE_PROPERTY_PROPERTY_LIST_H_
#define GRIN_INCLUDE_PROPERTY_PROPERTY_LIST_H_

#include "../predefine.h"

#ifdef GRIN_WITH_VERTEX_PROPERTY
GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_property_list_by_type(GRIN_GRAPH, GRIN_VERTEX_TYPE);

size_t grin_get_vertex_property_list_size(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_LIST);

GRIN_VERTEX_PROPERTY grin_get_vertex_property_from_list(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_LIST, size_t);

GRIN_VERTEX_PROPERTY_LIST grin_create_vertex_property_list(GRIN_GRAPH);

void grin_destroy_vertex_property_list(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_LIST);

bool grin_insert_vertex_property_to_list(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_LIST, GRIN_VERTEX_PROPERTY);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY
GRIN_VERTEX_PROPERTY grin_get_vertex_property_from_id(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_VERTEX_PROPERTY_ID);

GRIN_VERTEX_PROPERTY_ID grin_get_vertex_property_id(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_VERTEX_PROPERTY);
#endif


#ifdef GRIN_WITH_EDGE_PROPERTY
GRIN_EDGE_PROPERTY_LIST grin_get_edge_property_list_by_type(GRIN_GRAPH, GRIN_EDGE_TYPE);

size_t grin_get_edge_property_list_size(GRIN_GRAPH, GRIN_EDGE_PROPERTY_LIST);

GRIN_EDGE_PROPERTY grin_get_edge_property_from_list(GRIN_GRAPH, GRIN_EDGE_PROPERTY_LIST, size_t);

GRIN_EDGE_PROPERTY_LIST grin_create_edge_property_list(GRIN_GRAPH);

void grin_destroy_edge_property_list(GRIN_GRAPH, GRIN_EDGE_PROPERTY_LIST);

bool grin_insert_edge_property_to_list(GRIN_GRAPH, GRIN_EDGE_PROPERTY_LIST, GRIN_EDGE_PROPERTY);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
GRIN_EDGE_PROPERTY grin_get_edge_property_from_id(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_EDGE_PROPERTY_ID);

GRIN_EDGE_PROPERTY_ID grin_get_edge_property_id(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_EDGE_PROPERTY);
#endif


/** @name Graph Projection
 * GRIN_GRAPH projection mainly works to shrink the properties into a subset
 * in need to improve the retrieval efficiency. Note that only the vertex/edge
 * type with at least one property left in the vertex/edge property list will
 * be kept after the projection.
 * 
 * The projection only works on column store systems.
 */
///@{
#if defined(GRIN_WITH_VERTEX_PROPERTY) && defined(GRIN_ASSUME_COLUMN_STORE)
/** @brief project vertex properties */
GRIN_GRAPH grin_select_vertex_properties(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_LIST);
#endif

#if defined(GRIN_WITH_EDGE_PROPERTY) && defined(GRIN_ASSUME_COLUMN_STORE)
/** @brief project edge properties */
GRIN_GRAPH grin_select_edge_properteis(GRIN_GRAPH, GRIN_EDGE_PROPERTY_LIST);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_LIST_H_