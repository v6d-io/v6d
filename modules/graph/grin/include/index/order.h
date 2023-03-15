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
 @file order.h
 @brief Define the vertex ordering predicate APIs
*/

#ifndef GRIN_INCLUDE_INDEX_ORDER_H_
#define GRIN_INCLUDE_INDEX_ORDER_H_

#include "../predefine.h"

#ifndef GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
/** 
 * @brief check whether a vertex list is sorted
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_LIST the vertex list
 * @return whether sorted
*/
bool grin_is_vertex_list_sorted(GRIN_GRAPH, GRIN_VERTEX_LIST);
#endif

#ifdef GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
bool grin_smaller_vertex(GRIN_GRAPH, GRIN_VERTEX, GRIN_VERTEX);
#endif

#if defined(GRIN_ASSUME_ALL_VERTEX_LIST_SORTED) && defined(GRIN_ENABLE_VERTEX_LIST_ARRAY)
/** 
 * @brief get the position of a vertex in a sorted list
 * caller must guarantee the input vertex list is sorted to get the correct result
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_LIST the sorted vertex list
 * @param VERTEX the vertex to find
 * @param pos the returned position of the vertex
 * @return false if the vertex is not found
*/
bool grin_get_position_of_vertex_from_sorted_list(GRIN_GRAPH, GRIN_VERTEX_LIST, GRIN_VERTEX, size_t& pos);
#endif

#endif // GRIN_INCLUDE_INDEX_ORDER_H_