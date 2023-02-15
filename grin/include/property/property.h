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

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
#ifdef WITH_PROPERTY_NAME
char* get_property_name(const Graph, const Property);

Property get_property_by_name(const Graph, char* name);
#endif

DataType get_property_type(const Graph, const Property);

void* get_property_value_from_row(const Row, const Property);

size_t get_row_list_size(const RowList);

Row get_row_from_list(const RowList, size_t);

void* get_property_value_from_row(const Row, const Property);

#ifdef WITH_VERTEX_PROPERTY
RowList get_vertex_row_list(const Graph, const VertexList, const PropertyList);

Row get_vertex_row_from_list(const RowList, const Vertex);
#endif

#ifdef WITH_EDGE_PROPERTY
RowList get_edge_row_list(const Graph, const EdgeList, const PropertyList);

Row get_edge_row_from_list(const RowList, const Edge);
#endif

#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_H_