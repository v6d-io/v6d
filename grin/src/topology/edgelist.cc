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

#include "../predefine.h"

#ifdef ENABLE_EDGE_LIST

EdgeList get_edge_list(const Graph, const Direction);

#ifdef WITH_EDGE_LABEL
EdgeList get_edge_list_by_label(const Graph, const EdgeLabel);
#endif

void destroy_edge_list(EdgeList);

EdgeList create_edge_list();

bool insert_edge_to_list(EdgeList, const Edge);

size_t get_edge_list_size(const EdgeList);

Edge get_edge_from_list(const EdgeList, size_t);

#ifdef ENABLE_EDGE_LIST_ITERATOR
EdgeListIterator get_edge_list_begin(const Graph);

#ifdef WITH_EDGE_LABEL
EdgeListIterator get_edge_list_begin_by_label(const Graph, const EdgeLabel);
#endif

bool get_next_edge_list_iter(EdgeListIterator);

Edge get_edge_from_iter(EdgeListIterator);
#endif

#endif
