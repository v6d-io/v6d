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

#ifndef GRIN_INCLUDE_PREDICATE_ORDER_H_
#define GRIN_INCLUDE_PREDICATE_ORDER_H_

#include "../predefine.h"

#ifdef GRIN_PREDICATE_VERTEX_ORDERING
/** 
 * @brief sort a vertex list
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_LIST the vertex list to sort
 * @return whether succeed
*/
bool grin_sort_vertex_list(GRIN_GRAPH, GRIN_VERTEX_LIST);

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

#endif // GRIN_INCLUDE_PREDICATE_ORDER_H_