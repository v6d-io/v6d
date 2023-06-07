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
#include "topology/adjacentlist.h"


#if defined(GRIN_ENABLE_ADJACENT_LIST) && !defined(GRIN_ENABLE_EDGE_PROPERTY)
GRIN_ADJACENT_LIST grin_get_adjacent_list(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST
void grin_destroy_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    delete _al;
}
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
size_t grin_get_adjacent_list_size(GRIN_GRAPH g, GRIN_ADJACENT_LIST al) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    return _al->end - _al->begin;
}

GRIN_VERTEX grin_get_neighbor_from_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t idx) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    return (_al->begin + idx)->vid;
}

GRIN_EDGE grin_get_edge_from_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t idx) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    auto nbr = _al->begin + idx;
    auto e = new GRIN_EDGE_T();        
    e->dir = _al->dir;
    e->etype = _al->etype;
    e->eid = nbr->eid;
    if (_al->dir == GRIN_DIRECTION::OUT) {
        e->src = _al->vid;
        e->dst = nbr->vid;
    } else {
        e->src = nbr->vid;
        e->dst = _al->vid;
    }
    return e;
}
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
GRIN_ADJACENT_LIST_ITERATOR grin_get_adjacent_list_begin(GRIN_GRAPH g, GRIN_ADJACENT_LIST al) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    auto ali = new GRIN_ADJACENT_LIST_ITERATOR_T();
    ali->vid = _al->vid;
    ali->dir = _al->dir;
    ali->etype = _al->etype;
    ali->current = _al->begin;
    ali->end = _al->end;
    return ali;
}

void grin_destroy_adjacent_list_iter(GRIN_GRAPH g, GRIN_ADJACENT_LIST_ITERATOR ali) {
    auto _ali = static_cast<GRIN_ADJACENT_LIST_ITERATOR_T*>(ali);
    delete _ali;
}

void grin_get_next_adjacent_list_iter(GRIN_GRAPH g, GRIN_ADJACENT_LIST_ITERATOR ali) {
    auto _ali = static_cast<GRIN_ADJACENT_LIST_ITERATOR_T*>(ali);
    _ali->current++;
}

bool grin_is_adjacent_list_end(GRIN_GRAPH g, GRIN_ADJACENT_LIST_ITERATOR ali) {
    auto _ali = static_cast<GRIN_ADJACENT_LIST_ITERATOR_T*>(ali);
    return _ali->current >= _ali->end;
}

GRIN_VERTEX grin_get_neighbor_from_adjacent_list_iter(GRIN_GRAPH g, GRIN_ADJACENT_LIST_ITERATOR ali) {
    auto _ali = static_cast<GRIN_ADJACENT_LIST_ITERATOR_T*>(ali);
    return _ali->current->vid;
}

GRIN_EDGE grin_get_edge_from_adjacent_list_iter(GRIN_GRAPH g, GRIN_ADJACENT_LIST_ITERATOR ali) {
    auto _ali = static_cast<GRIN_ADJACENT_LIST_ITERATOR_T*>(ali);
    auto _nbr = _ali->current;
    auto e = new GRIN_EDGE_T();
    e->dir = _ali->dir;
    e->etype = _ali->etype;
    e->eid = _nbr->eid;
    if (_ali->dir == GRIN_DIRECTION::OUT) {
        e->src = _ali->vid;
        e->dst = _nbr->vid;
    } else {
        e->src = _nbr->vid;
        e->dst = _ali->vid;
    }
    return e;     
}
#endif