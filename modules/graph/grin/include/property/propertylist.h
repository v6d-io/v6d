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

#ifdef WITH_VERTEX_PROPERTY
VertexPropertyList get_vertex_property_list_by_type(Graph, VertexType);

size_t get_vertex_property_list_size(VertexPropertyList);

VertexProperty get_vertex_property_from_list(VertexPropertyList, size_t);

VertexPropertyList create_vertex_property_list();

void destroy_vertex_property_list(VertexPropertyList);

bool insert_vertex_property_to_list(VertexPropertyList, VertexProperty);
#endif

#ifdef NATURAL_VERTEX_PROPERTY_ID_TRAIT
VertexProperty get_vertex_property_from_id(VertexType, VertexPropertyID);

VertexPropertyID get_vertex_property_id(VertexType, VertexProperty);
#endif


#ifdef WITH_EDGE_PROPERTY
EdgePropertyList get_edge_property_list_by_type(Graph, EdgeType);

size_t get_edge_property_list_size(EdgePropertyList);

EdgeProperty get_edge_property_from_list(EdgePropertyList, size_t);

EdgePropertyList create_edge_property_list();

void destroy_edge_property_list(EdgePropertyList);

bool insert_edge_property_to_list(EdgePropertyList, EdgeProperty);
#endif

#ifdef NATURAL_EDGE_PROPERTY_ID_TRAIT
EdgeProperty get_edge_property_from_id(EdgeType, EdgePropertyID);

EdgePropertyID get_edge_property_id(EdgeType, EdgeProperty);
#endif


/** @name GraphProjection
 * Graph projection mainly works to shrink the properties into a subset
 * in need to improve the retrieval efficiency. Note that only the vertex/edge
 * type with at least one property left in the vertex/edge property list will
 * be kept after the projection.
 * 
 * The projection only works on column store systems.
 */
///@{
#if defined(WITH_VERTEX_PROPERTY) && defined(COLUMN_STORE_TRAIT)
/** @brief project vertex properties */
Graph select_vertex_properties(Graph, VertexPropertyList);
#endif

#if defined(WITH_EDGE_PROPERTY) && defined(COLUMN_STORE_TRAIT)
/** @brief project edge properties */
Graph select_edge_properteis(Graph, EdgePropertyList);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_LIST_H_