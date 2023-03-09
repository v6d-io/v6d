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

/** @name GRIN_ROW
 * GRIN_ROW works as the pure value array for the properties of a vertex or an edge.
 * In general, you can think of GRIN_ROW as an array of void*, where each void* points to
 * the value of a property. GRIN assumes the user already knows the corresponding
 * property list beforehead, so that she/he knows how to cast the void* into the
 * property's data type.
 */
///@{
#ifdef GRIN_ENABLE_ROW
void grin_destroy_row(GRIN_GRAPH, GRIN_ROW);

/** @brief the value of a property from row by its position in row */
const void* grin_get_value_from_row(GRIN_GRAPH, GRIN_ROW, size_t);

/** @brief create a row, usually to get vertex/edge by primary keys */
GRIN_ROW grin_create_row(GRIN_GRAPH);

/** @brief insert a value to the end of the row */
bool grin_insert_value_to_row(GRIN_GRAPH, GRIN_ROW, void*);
#endif
///@}

#ifdef GRIN_ENABLE_VERTEX_PROPERTY_TABLE
/**
 * @brief destroy vertex property table
 * @param GRIN_VERTEX_PROPERTY_TABLE vertex property table
 */
void grin_destroy_vertex_property_table(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_TABLE);

/**
 * @brief get the vertex property table of a certain vertex type
 * No matter column or row store strategy is used in the storage,
 * GRIN recommends to first get the property table of the vertex type,
 * and then fetch values(rows) by vertex and property(list). However,
 * GRIN does provide direct row fetching API when GRIN_ASSUME_COLUMN_STORE
 * is NOT set.
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_TYPE the vertex type
 */
GRIN_VERTEX_PROPERTY_TABLE grin_get_vertex_property_table_by_type(GRIN_GRAPH, GRIN_VERTEX_TYPE);

/**
 * @brief get vertex property value from table
 * @param GRIN_VERTEX_PROPERTY_TABLE vertex property table
 * @param GRIN_VERTEX the vertex which is the row index
 * @param GRIN_VERTEX_PROPERTY the vertex property which is the column index
 * @return can be casted to the property data type by the caller
 */
const void* grin_get_value_from_vertex_property_table(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_TABLE, GRIN_VERTEX, GRIN_VERTEX_PROPERTY);
#endif

#if defined(GRIN_ENABLE_VERTEX_PROPERTY_TABLE) && defined(GRIN_ENABLE_ROW)
/**
 * @brief get vertex row from table
 * @param GRIN_VERTEX_PROPERTY_TABLE vertex property table
 * @param GRIN_VERTEX the vertex which is the row index
 * @param GRIN_VERTEX_PROPERTY_LIST the vertex property list as columns
 */
GRIN_ROW grin_get_row_from_vertex_property_table(GRIN_GRAPH, GRIN_VERTEX_PROPERTY_TABLE, GRIN_VERTEX, GRIN_VERTEX_PROPERTY_LIST);
#endif

#if !defined(GRIN_ASSUME_COLUMN_STORE) && defined(GRIN_ENABLE_ROW)
/**
 * @brief get vertex row directly from the graph, this API only works for row store system
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX the vertex which is the row index
 * @param GRIN_VERTEX_PROPERTY_LIST the vertex property list as columns
 */
GRIN_ROW grin_get_vertex_row(GRIN_GRAPH, GRIN_VERTEX, GRIN_VERTEX_PROPERTY_LIST);
#endif

#ifdef GRIN_ENABLE_EDGE_PROPERTY_TABLE
/**
 * @brief destroy edge property table
 * @param GRIN_EDGE_PROPERTY_TABLE edge property table
 */
void grin_destroy_edge_property_table(GRIN_GRAPH, GRIN_EDGE_PROPERTY_TABLE);

/**
 * @brief get the edge property table of a certain edge type
 * No matter column or row store strategy is used in the storage,
 * GRIN recommends to first get the property table of the edge type,
 * and then fetch values(rows) by edge and property(list). However,
 * GRIN does provide direct row fetching API when GRIN_ASSUME_COLUMN_STORE
 * is NOT set.
 * @param GRIN_GRAPH the graph
 * @param GRIN_EDGE_TYPE the edge type
 */
GRIN_EDGE_PROPERTY_TABLE grin_get_edge_property_table_by_type(GRIN_GRAPH, GRIN_EDGE_TYPE);

/**
 * @brief get edge property value from table
 * @param GRIN_EDGE_PROPERTY_TABLE edge property table
 * @param GRIN_EDGE the edge which is the row index
 * @param GRIN_EDGE_PROPERTY the edge property which is the column index
 * @return can be casted to the property data type by the caller
 */
const void* grin_get_value_from_edge_property_table(GRIN_GRAPH, GRIN_EDGE_PROPERTY_TABLE, GRIN_EDGE, GRIN_EDGE_PROPERTY);
#endif

#if defined(GRIN_ENABLE_EDGE_PROPERTY_TABLE) && defined(GRIN_ENABLE_ROW)
/**
 * @brief get edge row from table
 * @param GRIN_EDGE_PROPERTY_TABLE edge property table
 * @param GRIN_EDGE the edge which is the row index
 * @param GRIN_EDGE_PROPERTY_LIST the edge property list as columns
 */
GRIN_ROW grin_get_row_from_edge_property_table(GRIN_GRAPH, GRIN_EDGE_PROPERTY_TABLE, GRIN_EDGE, GRIN_EDGE_PROPERTY_LIST);
#endif

#if !defined(GRIN_ASSUME_COLUMN_STORE) && defined(GRIN_ENABLE_ROW)
/**
 * @brief get edge row directly from the graph, this API only works for row store system
 * @param GRIN_GRAPH the graph
 * @param GRIN_EDGE the edge which is the row index
 * @param GRIN_EDGE_PROPERTY_LIST the edge property list as columns
 */
GRIN_ROW grin_get_edge_row(GRIN_GRAPH, GRIN_EDGE, GRIN_EDGE_PROPERTY_LIST);
#endif

#endif  // GRIN_INCLUDE_PROPERTY_PROPERTY_TABLE_H_