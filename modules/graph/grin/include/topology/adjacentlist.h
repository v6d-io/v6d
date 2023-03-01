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

#ifdef ENABLE_ADJACENT_LIST
AdjacentList get_adjacent_list(Graph, Direction, Vertex);

#ifdef WITH_EDGE_PROPERTY
AdjacentList get_adjacent_list_by_edge_type(Graph, Direction, Vertex, EdgeType);
#endif

void destroy_adjacent_list(AdjacentList);

size_t get_adjacent_list_size(AdjacentList);

Vertex get_neighbor_from_adjacent_list(AdjacentList, size_t);

Edge get_edge_from_adjacent_list(AdjacentList, size_t);

#ifdef ENABLE_ADJACENT_LIST_ITERATOR
AdjacentListIterator get_adjacent_list_begin(Graph);

AdjacentListIterator get_next_adjacent_list_iter(AdjacentListIterator);

bool is_adjacent_list_end(AdjacentListIterator);

Vertex get_neighbor_from_iter(AdjacentListIterator);

Edge get_edge_from_iter(AdjacentListIterator);
#endif

#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_ADJACENTLIST_H_
