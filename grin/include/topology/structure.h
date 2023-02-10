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

bool is_directed(const Graph);

size_t get_vertex_num(const Graph);

size_t get_edge_num(const Graph);

void destroy_vertex(Vertex);

DataType get_vertex_id_data_type(const Graph);

VertexID get_vertex_id(const Vertex);

#ifdef WITH_VERTEX_DATA
DataType get_vertex_data_type(const Graph, const Vertex);

VertexData get_vertex_data_value(const Graph, const Vertex);

void destroy_vertex_data(VertexData);

#ifdef MUTABLE_GRAPH
void set_vertex_data_value(Graph, Vertex, const VertexData);
#endif
#endif

void destroy_edge(Edge);

Vertex get_edge_src(const Graph, const Edge);

Vertex get_edge_dst(const Graph, const Edge);

#ifdef WITH_EDGE_DATA
DataType get_edge_data_type(const Graph, const Edge);

EdgeData get_edge_data_value(const Graph, const Edge);

void destroy_edge_data(EdgeData);

#ifdef MUTABLE_GRAPH
void set_edge_data_value(Graph, Edge, const EdgeData);
#endif
#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_STRUCTURE_H_
