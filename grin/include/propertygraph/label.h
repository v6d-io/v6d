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

#ifndef GRIN_PROPERTY_GRAPH_LABEL_H_
#define GRIN_PROPERTY_GRAPH_LABEL_H_

#include "../predefine.h"

#ifdef WITH_VERTEX_LABEL
VertexLabelList get_vertex_labels(const Graph);

size_t get_vertex_label_list_size(const VertexLabelList);

VertexLabel get_vertex_label_from_list(const VertexLabelList, const size_t);

VertexLabelList create_vertex_label_list();

void destroy_vertex_label_list(VertexLabelList);

bool insert_vertex_label_to_list(VertexLabelList, const VertexLabel);

VertexLabel get_vertex_label(const Graph, const Vertex);

char* get_vertex_label_name(const Graph, const VertexLabel);
#ifdef CONTINIOUS_VERTEX_LABEL_ID_TRAIT
VertexLabelID get_vertex_label_list_begin(const VertexLabelList);
VertexLabelID get_vertex_label_list_end(const VertexLabelList);
VertexLabelID get_vertex_label_id(const VertexLabel);
VertexLabel get_vertex_label_from_id(const VertexLabelID);
#endif
#endif

#ifdef WITH_EDGE_LABEL
EdgeLabelList get_edge_labels(const Graph);

size_t get_edge_label_list_size(const EdgeLabelList);

EdgeLabel get_edge_label_from_list(const EdgeLabelList, const size_t);

EdgeLabelList create_edge_label_list();

void destroy_edge_label_list(EdgeLabelList);

bool insert_edge_label_to_list(EdgeLabelList, const EdgeLabel);

EdgeLabel get_edge_label(const Graph, const Edge);

char* get_edge_label_name(const Graph, const EdgeLabel);
#ifdef CONTINIOUS_EDGE_LABEL_ID_TRAIT
EdgeLabelID get_edge_label_list_begin(const EdgeLabelList);
EdgeLabelID get_edge_label_list_end(const EdgeLabelList);
EdgeLabelID get_edge_label_id(const EdgeLabel);
EdgeLabel get_edge_label_from_id(const EdgeLabelID);
#endif
#endif

#if defined(WITH_VERTEX_LABEL) && defined(WITH_EDGE_LABEL)
VertexLabel get_src_label_from_edge_label(const Graph, const EdgeLabel);

VertexLabel get_dst_label_from_edge_label(const Graph, const EdgeLabel);
#endif

#endif  // GRIN_PROPERTY_GRAPH_LABEL_H_