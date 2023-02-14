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

#ifndef GRIN_PROPERTY_GRAPH_PROPERTY_GRAPH_H_
#define GRIN_PROPERTY_GRAPH_PROPERTY_GRAPH_H_

#include "../predefine.h"

#if defined(WITH_VERTEX_LABEL) && defined(ENABLE_VERTEX_LIST)
VertexList get_vertex_list_from_label(const Graph, const VertexLabel);
#endif

#if defined(WITH_EDGE_LABEL) && defined(ENABLE_EDGE_LIST)
EdgeList get_edge_list_from_label(const Graph, const EdgeLabel);
#endif

#if defined(WITH_VERTEX_LABEL) && defined(WITH_VERTEX_PROPERTY)
PropertyList get_all_vertex_properties_from_label(const Graph,
                                                  const VertexLabel);
#ifdef NATURAL_PROPERTY_ID_TRAIT
Property get_vertex_property_from_id(const VertexLabel, const PropertyID);
#endif
#endif

#if defined(WITH_EDGE_LABEL) && defined(WITH_EDGE_PROPERTY)
PropertyList get_all_edge_properties_from_label(const Graph, const EdgeLabel);
#ifdef NATURAL_PROPERTY_ID_TRAIT
Property get_edge_property_from_id(const EdgeLabel, const PropertyID);
#endif
#endif

#if defined(WITH_VERTEX_LABEL) && defined(WITH_VERTEX_PROPERTY) && \
    defined(COLUMN_STORE)
Graph select_vertex_properties_for_label(const Graph, const PropertyList,
                                         const VertexLabel);
#endif

#if defined(WITH_EDGE_LABEL) && defined(WITH_EDGE_PROPERTY) && \
    defined(COLUMN_STORE)
Graph select_edge_properteis_for_label(const Graph, const PropertyList,
                                       const EdgeLabel);
#endif

#if defined(WITH_VERTEX_LABEL) && defined(WITH_VERTEX_PRIMARTY_KEYS)
PropertyList get_primary_keys_from_label(const Graph, const VertexLabel);

Vertex get_vertex_from_primary_keys_and_label(const Graph, const Row,
                                              const VertexLabel);
#endif

#if defined(WITH_VERTEX_LABEL) && defined(ENABLE_VERTEX_LIST) && \
    defined(PARTITION_STRATEGY)
VertexList get_local_vertices_from_label(const PartitionedGraph,
                                         const Partition, const VertexLabel);

VertexList get_non_local_vertices_from_label(const PartitionedGraph,
                                             const Partition,
                                             const VertexLabel);
#endif

#endif  // GRIN_PROPERTY_GRAPH_PROPERTY_GRAPH_H_