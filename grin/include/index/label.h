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
 @file label.h
 @brief Define the label related APIs
*/

#ifndef GRIN_INCLUDE_INDEX_LABEL_H_
#define GRIN_INCLUDE_INDEX_LABEL_H_

#include "../predefine.h"

#if defined(WITH_VERTEX_LABEL) || defined(WITH_EDGE_LABEL)
Label get_label_by_name(const Graph, const char*);

char* get_label_name(const Graph, const Label);

void destroy_label(Label);

void destroy_label_list(LabelList);

size_t get_label_list_size(const LabelList);

Label get_label_from_list(const LabelList, const size_t);
#endif

#ifdef WITH_VERTEX_LABEL
/** 
 * @brief assign a label to a vertex
 * @param Graph the graph
 * @param Label the label
 * @param Vertex the vertex
 * @return whether succeed
*/
bool assign_label_to_vertex(const Graph, const Label, Vertex);

/** 
 * @brief get the label list of a vertex
 * @param Graph the graph
 * @param Vertex the vertex
*/
LabelList get_vertex_label_list(const Graph, const Vertex);

/** 
 * @brief get the vertex list by label
 * @param Graph the graph
 * @param Label the label
*/
VertexList get_vertex_list_by_label(const Graph, const Label);

/** 
 * @brief filtering an existing vertex list by label
 * @param VertexList the existing vertex list
 * @param Label the label
*/
VertexList filter_vertex_list_by_label(const VertexList, const Label);
#endif

#ifdef WITH_EDGE_LABEL
/** 
 * @brief assign a label to a edge
 * @param Graph the graph
 * @param Label the label
 * @param Edge the edge
 * @return whether succeed
*/
bool assign_label_to_edge(const Graph, const Label, Edge);

/** 
 * @brief get the label list of a edge
 * @param Graph the graph
 * @param Edge the edge
*/
LabelList get_edge_label_list(const Graph, const Edge);

/** 
 * @brief get the edge list by label
 * @param Graph the graph
 * @param Label the label
*/
EdgeList get_edge_list_by_label(const Graph, const Label);

/** 
 * @brief filtering an existing edge list by label
 * @param EdgeList the existing edge list
 * @param Label the label
*/
EdgeList filter_edge_list_by_label(const EdgeList, const Label);
#endif

#endif // GRIN_INCLUDE_INDEX_LABEL_H_