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

#ifndef GRIN_INCLUDE_TOPOLOGY_EDGELIST_H_
#define GRIN_INCLUDE_TOPOLOGY_EDGELIST_H_

#include "../predefine.h"

#ifdef ENABLE_EDGE_LIST

EdgeList get_edge_list(Graph, Direction);

#ifdef WITH_EDGE_PROPERTY
EdgeList get_edge_list_by_type(Graph, EdgeType);
#endif

void destroy_edge_list(EdgeList);

EdgeList create_edge_list();

bool insert_edge_to_list(EdgeList, Edge);

size_t get_edge_list_size(EdgeList);

Edge get_edge_from_list(EdgeList, size_t);

#ifdef ENABLE_EDGE_LIST_ITERATOR
EdgeListIterator get_edge_list_begin(Graph);

#ifdef WITH_EDGE_PROPERTY
EdgeListIterator get_edge_list_begin_by_type(Graph, EdgeType);
#endif

EdgeListIterator get_next_edge_list_iter(EdgeListIterator);

bool is_edge_list_end(EdgeListIterator);

Edge get_edge_from_iter(EdgeListIterator);
#endif

#endif

#endif  // GRIN_INCLUDE_TOPOLOGY_EDGELIST_H_
