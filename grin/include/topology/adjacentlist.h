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

AdjacentList get_adjacent_list(const Graph, const Direction, Vertex);

#ifdef WITH_EDGE_LABEL
AdjacentList get_adjacent_list_by_edge_label(const Graph, const Direction, Vertex, EdgeLabel);
#endif

void destroy_adjacent_list(AdjacentList);

AdjacentListIterator get_adjacent_list_begin(const AdjacentList);

AdjacentListIterator get_next_adjacent_iter(const AdjacentList,
                                            AdjacentListIterator);

size_t get_adjacent_list_size(const AdjacentList);

bool has_next_adjacent_iter(const AdjacentList, const AdjacentListIterator);

Vertex get_neighbor_from_iter(const AdjacentList, const AdjacentListIterator);

#ifdef WITH_EDGE_DATA
DataType get_adjacent_edge_data_type(const Graph);

EdgeData get_adjacent_edge_data_value(const AdjacentList,
                                      const AdjacentListIterator);
#endif

#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_ADJACENTLIST_H_
