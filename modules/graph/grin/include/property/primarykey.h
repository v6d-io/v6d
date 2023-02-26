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

#ifdef WITH_VERTEX_PRIMARY_KEYS
/** 
 * @brief get the vertex types with primary keys
 * @param Graph the graph
*/
VertexTypeList get_vertex_types_with_primary_keys(const Graph);

/** 
 * @brief get the primary keys (property list) of a specific vertex type
 * @param Graph the graph
 * @param VertexType the vertex type
*/
VertexPropertyList get_primary_keys_by_vertex_type(const Graph, const VertexType);

/** 
 * @brief get the vertex with the given primary keys
 * @param Graph the graph
 * @param VertexPropertyList the primary keys
 * @param Row the values of primary keys
*/
Vertex get_vertex_by_primay_keys(const Graph, const VertexPropertyList, const Row);
#endif

#ifdef WITH_EDGE_PRIMARY_KEYS
/** 
 * @brief get the edge types with primary keys
 * @param Graph the graph
*/
EdgeTypeList get_edge_types_with_primary_keys(const Graph);

/** 
 * @brief get the primary keys (property list) of a specific edge type
 * @param Graph the graph
 * @param EdgeType the edge type
*/
EdgePropertyList get_primary_keys_by_edge_type(const Graph, const EdgeType);

/** 
 * @brief get the edge with the given primary keys
 * @param Graph the graph
 * @param EdgePropertyList the primary keys
 * @param Row the values of primary keys
*/
Edge get_edge_by_primay_keys(const Graph, const EdgePropertyList, const Row);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PRIMARY_KEY_H_