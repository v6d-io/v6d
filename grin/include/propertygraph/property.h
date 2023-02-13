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

#ifndef GRIN_PROPERTY_GRAPH_PROPERTY_H_
#define GRIN_PROPERTY_GRAPH_PROPERTY_H_

#include "grin/predefine.h"

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
char* get_property_name(const Graph, const Property);

#ifdef WITH_PROPERTY_NAME
Property get_property_by_name(const Graph, char* name);
#endif

DataType get_property_type(const Graph, const Property);

size_t get_property_list_size(const PropertyList);

Property get_property_from_list(const PropertyList, const size_t);

PropertyList create_property_list();

void destroy_property_list(PropertyList);

bool insert_property_to_list(PropertyList, const Property);

void* get_property_value_from_row(const Row, const Property);

size_t get_row_list_size(const RowList);

RowListIterator get_row_list_begin(RowList);

RowListIterator get_next_row_iter(RowList, const RowListIterator);

bool has_next_row_iter(RowList, const RowListIterator);

Row get_row_from_iter(RowList, const RowListIterator);

RowList create_row_list();

void destroy_row_list(RowList);

bool insert_row_to_list(RowList, const Row);
#endif

#ifdef WITH_VERTEX_PROPERTY
void* get_vertex_property_value(const Graph, const Vertex, const Property);

PropertyList get_all_vertex_properties(const Graph);

Row get_vertex_row(const Graph, const Vertex, const PropertyList);

RowList get_vertex_row_in_batch(const Graph, const VertexList,
                                const PropertyList);
#endif

#ifdef WITH_EDGE_PROPERTY
void* get_edge_property_value(const Graph, const Edge, const Property);

PropertyList get_all_edge_properties(const Graph);

Row get_edge_row(const Graph, const Edge, const PropertyList);

RowList get_edge_row_in_batch(const Graph, const EdgeList, const PropertyList);
#endif

#if defined(WITH_VERTEX_PROPERTY) && defined(COLUMN_STORE)
Graph select_vertex_properties(const Graph, const PropertyList);
#endif

#if defined(WITH_EDGE_PROPERTY) && defined(COLUMN_STORE)
Graph select_edge_properteis(const Graph, const PropertyList);
#endif

#endif  // GRIN_PROPERTY_GRAPH_PROPERTY_H_