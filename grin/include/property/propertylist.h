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

#ifndef GRIN_INCLUDE_PROPERTY_PROPERTY_LIST_H_
#define GRIN_INCLUDE_PROPERTY_PROPERTY_LIST_H_

#include "../predefine.h"

// Property list
#ifdef WITH_VERTEX_PROPERTY
PropertyList get_all_vertex_properties(const Graph);
#endif

#ifdef WITH_EDGE_PROPERTY
PropertyList get_all_edge_properties(const Graph);
#endif

size_t get_property_list_size(const PropertyList);

Property get_property_from_list(const PropertyList, const size_t);

PropertyList create_property_list();

void destroy_property_list(PropertyList);

bool insert_property_to_list(PropertyList, const Property);


#if defined(WITH_VERTEX_LABEL) && defined(WITH_VERTEX_PROPERTY)
PropertyList get_all_vertex_properties_by_label(const Graph, const VertexLabel);

#ifdef NATURAL_PROPERTY_ID_TRAIT
Property get_vertex_property_from_id(const VertexLabel, const PropertyID);

PropertyID get_vertex_property_id(const VertexLabel, const Property);
#endif
#endif

#if defined(WITH_EDGE_LABEL) && defined(WITH_EDGE_PROPERTY)
PropertyList get_all_edge_properties_by_label(const Graph, const EdgeLabel);

#ifdef NATURAL_PROPERTY_ID_TRAIT
Property get_edge_property_from_id(const EdgeLabel, const PropertyID);

PropertyID get_edge_property_id(const EdgeLabel, const Property);
#endif
#endif

#if defined(WITH_VERTEX_LABEL) && defined(WITH_VERTEX_PRIMARTY_KEYS)
PropertyList get_primary_keys_by_label(const Graph, const VertexLabel);

Vertex get_vertex_by_primary_keys_and_label(const Graph, const Row,
                                              const VertexLabel);
#endif

// graph projection
#if defined(WITH_VERTEX_PROPERTY) && defined(COLUMN_STORE)
Graph select_vertex_properties(const Graph, const PropertyList);
#endif

#if defined(WITH_EDGE_PROPERTY) && defined(COLUMN_STORE)
Graph select_edge_properteis(const Graph, const PropertyList);
#endif

#if defined(WITH_VERTEX_LABEL) && defined(WITH_VERTEX_PROPERTY) && \
    defined(COLUMN_STORE)
Graph select_vertex_properties_for_label(const Graph, const VertexLabel,
                                         const PropertyList);
#endif

#if defined(WITH_EDGE_LABEL) && defined(WITH_EDGE_PROPERTY) && \
    defined(COLUMN_STORE)
Graph select_edge_properties_for_label(const Graph, const EdgeLabel,
                                       const PropertyList);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_LIST_H_