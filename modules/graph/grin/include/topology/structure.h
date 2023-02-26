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
bool is_directed(const Graph);

bool is_multigraph(const Graph);

size_t get_vertex_num(const Graph);

#ifdef WITH_VERTEX_PROPERTY
size_t get_vertex_num_by_type(const Graph, const VertexType);
#endif

size_t get_edge_num(const Graph);

#ifdef WITH_EDGE_PROPERTY
size_t get_edge_num_by_type(const Graph, const EdgeType);
#endif


// Vertex
void destroy_vertex(Vertex);

#ifdef WITH_VERTEX_DATA
DataType get_vertex_data_type(const Graph, const Vertex);

VertexData get_vertex_data_value(const Graph, const Vertex);

void destroy_vertex_data(VertexData);
#endif

#ifdef WITH_VERTEX_ORIGINAL_ID
Vertex get_vertex_from_original_id(const Graph, const OriginalID);

OriginalID get_vertex_original_id(const Graph, const Vertex);

void destroy_vertex_original_id(OriginalID); 
#endif

#if defined(WITH_VERTEX_ORIGINAL_ID) && defined(WITH_VERTEX_PROPERTY)
Vertex get_vertex_from_original_id_by_type(const Graph, const VertexType, const OriginalID);
#endif


// Edge
void destroy_edge(Edge);

Vertex get_edge_src(const Graph, const Edge);

Vertex get_edge_dst(const Graph, const Edge);

#ifdef WITH_EDGE_DATA
DataType get_edge_data_type(const Graph, const Edge);

EdgeData get_edge_data_value(const Graph, const Edge);

void destroy_edge_data(EdgeData);
#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_STRUCTURE_H_
