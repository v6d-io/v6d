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
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto al = new GRIN_ADJACENT_LIST_T();
    al->v = v;
    al->dir = d;
    al->etype = _g->edge_label_num();
    for (GRIN_EDGE_TYPE_T etype = 0; etype < al->etype; ++etype) {
        if (d == GRIN_DIRECTION::IN) {
            al->data.push_back(_g->GetIncomingRawAdjList(*_v, etype));
        } else {
            al->data.push_back(_g->GetOutgoingRawAdjList(*_v, etype));
        }
    }
    return al;
}

#ifdef GRIN_WITH_EDGE_PROPERTY
GRIN_ADJACENT_LIST grin_get_adjacent_list_by_edge_type(GRIN_GRAPH g, GRIN_DIRECTION d, 
                                            GRIN_VERTEX v, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto al = new GRIN_ADJACENT_LIST_T();
    al->v = v;
    al->dir = d;
    al->etype = *_etype;
    if (d == GRIN_DIRECTION::IN) {
        al->data.push_back(_g->GetIncomingRawAdjList(*_v, *_etype));
    } else {
        al->data.push_back(_g->GetOutgoingRawAdjList(*_v, *_etype));
    }
    return al;
}
#endif

void grin_destroy_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    delete _al;
}

size_t grin_get_adjacent_list_size(GRIN_GRAPH g, GRIN_ADJACENT_LIST al) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    size_t result = 0;
    for (auto &ral : _al->data) {
        result += ral.Size();
    }
    return result;
}

GRIN_VERTEX grin_get_neighbor_from_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t idx) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    size_t result = 0;
    for (auto &ral : _al->data) {
        result += ral.Size();
        if (idx < result) {
            auto _idx = idx - (result - ral.size());
            auto _nbr = ral.begin() + _idx;
            auto v = new GRIN_VERTEX_T(_nbr->vid);
            return v;
        }
    }
    return GRIN_NULL_VERTEX;
}

GRIN_EDGE grin_get_edge_from_adjacent_list(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t idx) {
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    size_t result = 0;
    for (auto i = 0; i < _al->data.size(); ++i) {
        result += _al->data[i].Size();
        if (idx < result) {
            auto _idx = idx - (result - _al->data[i].Size());
            auto _nbr = _al->data[i].begin() + _idx;
            auto v = new GRIN_VERTEX_T(_nbr->vid);
            auto e = new GRIN_EDGE_T();
            e->src = _al->v;
            e->dst = v;
            e->dir = _al->dir;
            e->etype = _al->data.size() > 1 ? i : _al->etype;
            e->eid = _nbr->eid;
            return e;
        }
    }
    return GRIN_NULL_EDGE;
}
#endif