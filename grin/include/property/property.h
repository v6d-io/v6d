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

#ifndef GRIN_INCLUDE_PROPERTY_PROPERTY_H_
#define GRIN_INCLUDE_PROPERTY_PROPERTY_H_

#include "../predefine.h"

// Vertex Property
#ifdef WITH_VERTEX_PROPERTY
void destroy_vertex_property(VertexProperty);

DataType get_vertex_property_type(VertexProperty);

// Vertex Property Name
#ifdef WITH_VERTEX_PROPERTY_NAME
char* get_vertex_property_name(const Graph, const VertexProperty);

VertexProperty get_vertex_property_by_name(const Graph, const VertexLabel, const char*);
#endif

// Vertex Property Table
void destroy_vertex_property_table(VertexPropertyTable);

void* get_value_from_vertex_property_table(const VertexPropertyTable, const Vertex, const VertexProperty);

VertexPropertyTable get_vertex_property_table_by_label(const Graph, const VertexLabel);

#ifdef COLUMN_STORE
VertexPropertyTable get_vertex_property_table_for_property(const Graph, const VertexProperty);
#else
VertexPropertyTable get_vertex_property_table_for_vertex(const Graph, const Vertex);
#endif

#endif


// Edge Property
#ifdef WITH_EDGE_PROPERTY
void destroy_edge_property(EdgeProperty);

DataType get_edge_property_type(EdgeProperty);

// Edge Property Name
#ifdef WITH_EDGE_PROPERTY_NAME
char* get_edge_property_name(const Graph, const EdgeProperty);

EdgeProperty get_edge_property_by_name(const Graph, const EdgeLabel, const char*);
#endif

// Edge Property Table
void destroy_edge_property_table(EdgePropertyTable);

void* get_value_from_edge_property_table(const EdgePropertyTable, const Edge, const EdgeProperty);

EdgePropertyTable get_edge_property_table_by_label(const Graph, const EdgeLabel);

#ifdef COLUMN_STORE
EdgePropertyTable get_edge_property_table_for_property(const Graph, const EdgeProperty);
#else
EdgePropertyTable get_edge_property_table_for_edge(const Graph, const Edge);
#endif

#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_H_