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

// This header file is not available for libgrape-lite.

#ifndef GRIN_INCLUDE_PROPERTY_LABEL_H_
#define GRIN_INCLUDE_PROPERTY_LABEL_H_

#include "../predefine.h"

#ifdef WITH_VERTEX_LABEL
// Vertex label
VertexLabel get_vertex_label(const Graph, const Vertex);

char* get_vertex_label_name(const Graph, const VertexLabel);

#ifdef NATURAL_VERTEX_LABEL_ID_TRAIT
VertexLabelID get_vertex_label_id(const VertexLabel);

VertexLabel get_vertex_label_from_id(const VertexLabelID);
#endif

// Vertex label list
VertexLabelList get_vertex_label_list(const Graph);

void destroy_vertex_label_list(VertexLabelList);

VertexLabelList create_vertex_label_list();

bool insert_vertex_label_to_list(VertexLabelList, const VertexLabel);

size_t get_vertex_label_list_size(const VertexLabelList);

VertexLabel get_vertex_label_from_list(const VertexLabelList, const size_t);
#endif

#ifdef WITH_EDGE_LABEL
// Edge label
EdgeLabel get_edge_label(const Graph, const Edge);

char* get_edge_label_name(const Graph, const EdgeLabel);

#ifdef NATURAL_EDGE_LABEL_ID_TRAIT
EdgeLabelID get_edge_label_id(const EdgeLabel);

EdgeLabel get_edge_label_from_id(const EdgeLabelID);
#endif

// Edge label list
EdgeLabelList get_edge_label_list(const Graph);

void destroy_edge_label_list(EdgeLabelList);

EdgeLabelList create_edge_label_list();

bool insert_edge_label_to_list(EdgeLabelList, const EdgeLabel);

size_t get_edge_label_list_size(const EdgeLabelList);

EdgeLabel get_edge_label_from_list(const EdgeLabelList, const size_t);
#endif


#if defined(WITH_VERTEX_LABEL) && defined(WITH_EDGE_LABEL)
VertexLabel get_src_label_from_edge_label(const Graph, const EdgeLabel);

VertexLabel get_dst_label_from_edge_label(const Graph, const EdgeLabel);
#endif

#if defined(WITH_VERTEX_LABEL) && defined(ENABLE_VERTEX_LIST)
VertexList get_vertex_list_by_label(const Graph, const VertexLabel);
#endif

#if defined(WITH_EDGE_LABEL) && defined(ENABLE_EDGE_LIST)
EdgeList get_edge_list_by_label(const Graph, const EdgeLabel);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_LABEL_H_