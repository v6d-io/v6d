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

#ifdef WITH_VERTEX_LABEL
VertexList get_vertex_list_by_label(const Graph, const VertexLabel);
#endif

void destroy_vertex_list(VertexList);

VertexList create_vertex_list();

bool insert_vertex_to_list(VertexList, const Vertex);

size_t get_vertex_list_size(const VertexList);

Vertex get_vertex_from_list(const VertexList, const size_t);

#ifdef ENABLE_VERTEX_LIST_ITERATOR
VertexListIterator get_vertex_list_begin(const Graph);

#ifdef WITH_VERTEX_LABEL
VertexListIterator get_vertex_list_begin_by_label(const Graph, const VertexLabel);
#endif

bool get_next_vertex_list_iter(VertexListIterator);

Vertex get_vertex_from_iter(VertexListIterator);
#endif

#ifdef CONTINUOUS_VERTEX_ID_TRAIT
VertexID get_begin_vertex_id_from_list(const VertexList);

VertexID get_end_vertex_id_from_list(const VertexList);

DataType get_vertex_id_data_type(const Graph);

VertexID get_vertex_id(const Vertex);

Vertex get_vertex_from_id(const VertexID);

void destroy_vertex_id(VertexID);
#endif

#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_VERTEXLIST_H_
