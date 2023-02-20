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

#ifdef WITH_VERTEX_PROPERTY
void destroy_vertex_property(VertexProperty);

#ifdef WITH_VERTEX_PROPERTY_NAME
char* get_vertex_property_name(const Graph, const VertexProperty);

#ifdef COLUMN_STORE
VertexColumn get_vertex_column_by_name(const Graph, char* name);
#endif
#endif

DataType get_vertex_property_type(const Graph, const VertexProperty);

#ifdef COLUMN_STORE
void destroy_vertex_column(VertexColumn);
#ifdef ENABLE_VERTEX_LIST
VertexColumn get_vertex_column_by_list(const Graph, const VertexList, const VertexProperty);
#endif
#ifdef WITH_VERTEX_LABEL
VertexColumn get_vertex_column_by_label(const Graph, const VertexLabel, const VertexProperty);
#endif
void* get_value_from_vertex_column(const VertexColumn, const Vertex);
#else
void destroy_vertex_row(VertexRow);
VertexRow get_vertex_row_by_list(const Graph, const Vertex, const VertexPropertyList);
#ifdef WITH_VERTEX_LABEL
VertexRow get_vertex_row_by_label(const Graph, const Vertex, const VertexLabel);
#endif
#endif

#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_H_