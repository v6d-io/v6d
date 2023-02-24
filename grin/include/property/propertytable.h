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
 @file propertytable.h
 @brief Define the property table related APIs
*/

#ifndef GRIN_INCLUDE_PROPERTY_PROPERTY_TABLE_H_
#define GRIN_INCLUDE_PROPERTY_PROPERTY_TABLE_H_

#include "../predefine.h"

/** @name Row
 * Row works as the pure value array for the properties of a vertex or an edge.
 * In general, you can think of Row as an array of void*, where each void* points to
 * the value of a property. GRIN assumes the user already knows the corresponding
 * property list beforehead, so that she/he knows how to cast the void* into the
 * property's data type.
 */
///@{
#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
void destroy_row(Row);

/** @brief the value of a property from row by its position in row */
void* get_value_from_row(Row, const size_t);

/** @brief create a row, usually to get vertex/edge by primary keys */
Row create_row();

/** @brief insert a value to the end of the row */
bool insert_value_to_row(Row, const void*);
#endif
///@}

#ifdef WITH_VERTEX_PROPERTY
/**
 * @brief destroy vertex property table
 * @param VertexPropertyTable vertex property table
 */
void destroy_vertex_property_table(VertexPropertyTable);

/**
 * @brief get the vertex property table of a certain vertex type
 * No matter column or row store strategy is used in the storage,
 * GRIN recommends to first get the property table of the vertex type,
 * and then fetch values(rows) by vertex and property(list). However,
 * GRIN does provide direct row fetching API when COLUMN_STORE_TRAIT
 * is NOT set.
 * @param Graph the graph
 * @param VertexType the vertex type
 */
VertexPropertyTable get_vertex_property_table_by_type(const Graph, const VertexType);

/**
 * @brief get vertex property value from table
 * @param VertexPropertyTable vertex property table
 * @param Vertex the vertex which is the row index
 * @param VertexProperty the vertex property which is the column index
 * @return can be casted to the property data type by the caller
 */
void* get_value_from_vertex_property_table(const VertexPropertyTable, const Vertex, const VertexProperty);

/**
 * @brief get vertex row from table
 * @param VertexPropertyTable vertex property table
 * @param Vertex the vertex which is the row index
 * @param VertexPropertyList the vertex property list as columns
 */
Row get_row_from_vertex_property_table(const VertexPropertyTable, const Vertex, const VertexPropertyList);

#ifndef COLUMN_STORE_TRAIT
/**
 * @brief get vertex row directly from the graph, this API only works for row store system
 * @param Graph the graph
 * @param Vertex the vertex which is the row index
 * @param VertexPropertyList the vertex property list as columns
 */
Row get_vertex_row(const Graph, const Vertex, const VertexPropertyList);
#endif
#endif

#ifdef WITH_EDGE_PROPERTY
/**
 * @brief destroy edge property table
 * @param EdgePropertyTable edge property table
 */
void destroy_edge_property_table(EdgePropertyTable);

/**
 * @brief get the edge property table of a certain edge type
 * No matter column or row store strategy is used in the storage,
 * GRIN recommends to first get the property table of the edge type,
 * and then fetch values(rows) by edge and property(list). However,
 * GRIN does provide direct row fetching API when COLUMN_STORE_TRAIT
 * is NOT set.
 * @param Graph the graph
 * @param EdgeType the edge type
 */
EdgePropertyTable get_edge_property_table_by_type(const Graph, const EdgeType);

/**
 * @brief get edge property value from table
 * @param EdgePropertyTable edge property table
 * @param Edge the edge which is the row index
 * @param EdgeProperty the edge property which is the column index
 * @return can be casted to the property data type by the caller
 */
void* get_value_from_edge_property_table(const EdgePropertyTable, const Edge, const EdgeProperty);

/**
 * @brief get edge row from table
 * @param EdgePropertyTable edge property table
 * @param Edge the edge which is the row index
 * @param EdgePropertyList the edge property list as columns
 */
Row get_row_from_edge_property_table(const EdgePropertyTable, const Edge, const EdgePropertyList);

#ifndef COLUMN_STORE_TRAIT
/**
 * @brief get edge row directly from the graph, this API only works for row store system
 * @param Graph the graph
 * @param Edge the edge which is the row index
 * @param EdgePropertyList the edge property list as columns
 */
Row get_edge_row(const Graph, const Edge, const EdgePropertyList);
#endif
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_TABLE_H_