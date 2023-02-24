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
 @file property.h
 @brief Define the property related APIs
*/

#ifndef GRIN_INCLUDE_PROPERTY_PROPERTY_H_
#define GRIN_INCLUDE_PROPERTY_PROPERTY_H_

#include "../predefine.h"

#ifdef WITH_PROPERTY_NAME
char* get_vertex_property_name(const Graph, const VertexProperty);

/**
 * @brief get the vertex property with a given name under a specific vertex type
 * @param Graph the graph
 * @param VertexType the specific vertex type
 * @param msg the name
 */
VertexProperty get_vertex_property_by_name(const Graph, const VertexType, const char* msg);

/**
 * @brief get all the vertex properties with a given name
 * @param Graph the graph
 * @param msg the name
 */
VertexPropertyList get_vertex_properties_by_name(const Graph, const char*);

char* get_edge_property_name(const Graph, const EdgeProperty);

/**
 * @brief get the edge property with a given name under a specific edge type
 * @param Graph the graph
 * @param EdgeType the specific edge type
 * @param msg the name
 */
EdgeProperty get_edge_property_by_name(const Graph, const EdgeType, const char* msg);

/**
 * @brief get all the edge properties with a given name
 * @param Graph the graph
 * @param msg the name
 */
EdgePropertyList get_edge_properties_by_name(const Graph, const char*);
#endif


#ifdef WITH_VERTEX_PROPERTY
/**
 * @brief destroy vertex property
 * @param VertexProperty vertex property
 */
void destroy_vertex_property(VertexProperty);

/**
 * @brief get property data type
 * @param VertexProperty vertex property
 */
DataType get_vertex_property_data_type(VertexProperty);

/**
 * @brief get the vertex type that the property is bound to
 * @param VertexProperty vertex property
 */
VertexType get_vertex_property_vertex_type(VertexProperty);
#endif

#ifdef WITH_EDGE_PROPERTY
/**
 * @brief destroy edge property
 * @param EdgeProperty edge property
 */
void destroy_edge_property(EdgeProperty);

/**
 * @brief get property data type
 * @param EdgeProperty edge property
 */
DataType get_edge_property_data_type(EdgeProperty);

/**
 * @brief get the edge type that the property is bound to
 * @param EdgeProperty edge property
 */
EdgeType get_edge_property_edge_type(EdgeProperty);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_H_