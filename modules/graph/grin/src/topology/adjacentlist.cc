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

#include "graph/grin/src/predefine.h"
#include "graph/grin/include/topology/adjacentlist.h"

#ifdef GRIN_ENABLE_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_get_adjacent_list(GRIN_GRAPH g, GRIN_DIRECTION d, GRIN_VERTEX v) {
    if (d == GRIN_DIRECTION::BOTH) return GRIN_NULL_LIST;
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto al = new GRIN_ADJACENT_LIST_T();
    al->v = _v;
    al->dir = d;
    al->etype_begin = 0;
    al->etype_end = _g->edge_label_num();
    __grin_init_adjacent_list(_g, al);
    return al;
}

void grin_destroy_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    delete _al;
}

size_t grin_get_adjacent_list_size(GRIN_GRAPH g, GRIN_ADJACENT_LIST al) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    return _al->offsets[_al->etype_end - _al->etype_begin];
}

GRIN_VERTEX grin_get_neighbor_from_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t idx) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    for (auto i = 0; i < _al->etype_end - _al->etype_begin; ++i) {        
        if (idx < _al->offsets[i+1]) {
            auto _idx = idx - _al->offsets[i];
            auto _nbr = _al->data[i].begin() + _idx;
            auto v = new GRIN_VERTEX_T(_nbr->vid);
            return v;
        }
    }
    return GRIN_NULL_VERTEX;
}

GRIN_EDGE grin_get_edge_from_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t idx) {
   auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    for (auto i = 0; i < _al->etype_end - _al->etype_begin; ++i) {        
        if (idx < _al->offsets[i+1]) {
            auto _idx = idx - _al->offsets[i];
            auto _nbr = _al->data[i].begin() + _idx;
            auto v = new GRIN_VERTEX_T(_nbr->vid);
            auto e = new GRIN_EDGE_T();
            e->src = _al->v;
            e->dst = v;
            e->dir = _al->dir;
            e->etype = _al->etype_begin + i;
            e->eid = _nbr->eid;
            return e;        
        }
    }
    return GRIN_NULL_EDGE;
}
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
GRIN_ADJACENT_LIST_ITERATOR grin_get_adjacent_list_begin(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);

void grin_destroy_adjacent_list_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

GRIN_ADJACENT_LIST_ITERATOR grin_get_next_adjacent_list_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

bool grin_is_adjacent_list_end(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

GRIN_VERTEX grin_get_neighbor_from_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);

GRIN_EDGE grin_get_edge_from_iter(GRIN_GRAPH, GRIN_ADJACENT_LIST_ITERATOR);
#endif