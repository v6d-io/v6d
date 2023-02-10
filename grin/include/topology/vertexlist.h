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

#ifndef GRIN_INCLUDE_TOPOLOGY_VERTEXLIST_H_
#define GRIN_INCLUDE_TOPOLOGY_VERTEXLIST_H_

#include "../predefine.h"

#ifdef ENABLE_VERTEX_LIST

VertexList get_vertex_list(const Graph);

void destroy_vertex_list(Graph);

size_t get_vertex_list_size(const VertexList);

VertexListIterator get_vertex_list_begin(const VertexList);

VertexListIterator get_next_vertex_iter(const VertexList, VertexListIterator);

bool has_next_vertex_iter(const VertexList, const VertexListIterator);

Vertex get_vertex_from_iter(const VertexList, const VertexListIterator);

#ifdef CONTINUOUS_VID_TRAIT
VertexID get_begin_vertex_id_from_list(const VertexList);

VertexID get_end_vertex_id_from_list(const VertexList);
#endif

#ifdef MUTABLE_GRAPH
VertexList create_vertex_list();
bool insert_vertex_to_list(VertexList, const Vertex);
#endif
#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_VERTEXLIST_H_
