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

#ifndef GRIN_INCLUDE_TOPOLOGY_ADJACENTLIST_H_
#define GRIN_INCLUDE_TOPOLOGY_ADJACENTLIST_H_

#include "../predefine.h"

#ifdef GRIN_ENABLE_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_get_adjacent_list(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);

void grin_destroy_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST);
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
size_t grin_get_adjacent_list_size(GRIN_GRAPH, GRIN_ADJACENT_LIST);

GRIN_VERTEX grin_get_neighbor_from_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST, size_t);

GRIN_EDGE grin_get_edge_from_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST, size_t);
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
GRIN_ADJACENT_LIST_ITERATOR grin_get_adjacent_list_begin(GRIN_GRAPH, GRIN_ADJACENT_LIST);

void grin_destroy_adjacent_list_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

GRIN_ADJACENT_LIST_ITERATOR grin_get_next_adjacent_list_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

bool grin_is_adjacent_list_end(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

GRIN_VERTEX grin_get_neighbor_from_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

GRIN_EDGE grin_get_edge_from_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);
#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_ADJACENTLIST_H_
