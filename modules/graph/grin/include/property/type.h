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

#ifdef WITH_VERTEX_PROPERTY
// Vertex type
bool equal_vertex_type(VertexType, VertexType);

VertexType get_vertex_type(Graph, Vertex);

const char* get_vertex_type_name(Graph, VertexType);

VertexType get_vertex_type_by_name(Graph, const char*);

// Vertex type list
VertexTypeList get_vertex_type_list(Graph);

void destroy_vertex_type_list(VertexTypeList);

VertexTypeList create_vertex_type_list();

bool insert_vertex_type_to_list(VertexTypeList, VertexType);

size_t get_vertex_type_list_size(VertexTypeList);

VertexType get_vertex_type_from_list(VertexTypeList, size_t);
#endif


#ifdef NATURAL_VERTEX_TYPE_ID_TRAIT
VertexTypeID get_vertex_type_id(VertexType);

VertexType get_vertex_type_from_id(VertexTypeID);
#endif


#ifdef WITH_EDGE_PROPERTY
// Edge type
bool equal_edge_type(EdgeType, EdgeType);

EdgeType get_edge_type(Graph, Edge);

const char* get_edge_type_name(Graph, EdgeType);

EdgeType get_edge_type_by_name(Graph, const char*);

// Edge type list
EdgeTypeList get_edge_type_list(Graph);

void destroy_edge_type_list(EdgeTypeList);

EdgeTypeList create_edge_type_list();

bool insert_edge_type_to_list(EdgeTypeList, EdgeType);

size_t get_edge_type_list_size(EdgeTypeList);

EdgeType get_edge_type_from_list(EdgeTypeList, size_t);
#endif

#ifdef NATURAL_EDGE_TYPE_ID_TRAIT
EdgeTypeID get_edge_type_id(EdgeType);

EdgeType get_edge_type_from_id(EdgeTypeID);
#endif

/** @name VertexEdgeTypeRelation
 * GRIN assumes the relation between edge type and pairs of vertex types is many-to-many.
 * Thus GRIN returns the pairs of vertex types related to an edge type as a pair of vertex type
 * lists of the same size, and the src/dst vertex types are aligned with their positions in the lists.
 */
///@{
#if defined(WITH_VERTEX_PROPERTY) && defined(WITH_EDGE_PROPERTY)
/** @brief  the src vertex type list */
VertexTypeList get_src_types_from_edge_type(Graph, EdgeType);

/** @brief get the dst vertex type list */
VertexTypeList get_dst_types_from_edge_type(Graph, EdgeType);

/** @brief get the edge type list related to a given pair of vertex types */
EdgeTypeList get_edge_types_from_vertex_type_pair(Graph, VertexType, VertexType);
#endif
///@}

#endif  // GRIN_INCLUDE_PROPERTY_TYPE_H_