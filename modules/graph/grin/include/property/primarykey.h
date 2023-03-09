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
 @file primarykey.h
 @brief Define the primary key related APIs
*/

#ifndef GRIN_INCLUDE_PROPERTY_PRIMARY_KEY_H_
#define GRIN_INCLUDE_PROPERTY_PRIMARY_KEY_H_

#include "../predefine.h"

#ifdef GRIN_ENABLE_VERTEX_PRIMARY_KEYS
/** 
 * @brief get the vertex types with primary keys
 * @param GRIN_GRAPH the graph
*/
GRIN_VERTEX_TYPE_LIST grin_get_vertex_types_with_primary_keys(GRIN_GRAPH);

/** 
 * @brief get the primary keys (property list) of a specific vertex type
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_TYPE the vertex type
*/
GRIN_VERTEX_PROPERTY_LIST grin_get_primary_keys_by_vertex_type(GRIN_GRAPH, GRIN_VERTEX_TYPE);

/** 
 * @brief get the vertex with the given primary keys
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_PROPERTY_LIST the primary keys
 * @param GRIN_ROW the values of primary keys
*/
GRIN_VERTEX grin_get_vertex_by_primay_keys(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_LIST, GRIN_ROW);
#endif

#ifdef GRIN_WITH_EDGE_PRIMARY_KEYS
/** 
 * @brief get the edge types with primary keys
 * @param GRIN_GRAPH the graph
*/
GRIN_EDGE_TYPE_LIST grin_get_edge_types_with_primary_keys(GRIN_GRAPH);

/** 
 * @brief get the primary keys (property list) of a specific edge type
 * @param GRIN_GRAPH the graph
 * @param GRIN_EDGE_TYPE the edge type
*/
GRIN_EDGE_PROPERTY_LIST grin_get_primary_keys_by_edge_type(GRIN_GRAPH, GRIN_EDGE_TYPE);

/** 
 * @brief get the edge with the given primary keys
 * @param GRIN_GRAPH the graph
 * @param GRIN_EDGE_PROPERTY_LIST the primary keys
 * @param GRIN_ROW the values of primary keys
*/
GRIN_EDGE grin_get_edge_by_primay_keys(GRIN_GRAPH, GRIN_EDGE_PROPERTY_LIST, GRIN_ROW);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PRIMARY_KEY_H_