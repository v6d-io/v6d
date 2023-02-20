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

#include "grin/src/predefine.h"
#include "grin/include/topology/adjacentlist.h"

#ifdef ENABLE_ADJACENT_LIST
AdjacentList get_adjacent_list(const Graph g, const Direction d, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto al = new AdjacentList_T();
    al->v = v;
    al->dir = d;
    al->elabel = _g->edge_label_num();
    for (EdgeLabel_T elabel = 0; elabel < al->elabel; ++elabel) {
        if (d == Direction::IN) {
            al->data.push_back(_g->GetIncomingRawAdjList(*_v, elabel));
        } else {
            al->data.push_back(_g->GetOutgoingRawAdjList(*_v, elabel));
        }
    }
    return al;
}

#ifdef WITH_EDGE_LABEL
AdjacentList get_adjacent_list_by_edge_label(const Graph g, const Direction d, 
                                             const Vertex v, const EdgeLabel elabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto al = new AdjacentList_T();
    al->v = v;
    al->dir = d;
    al->elabel = *_elabel;
    if (d == Direction::IN) {
        al->data.push_back(_g->GetIncomingRawAdjList(*_v, *_elabel));
    } else {
        al->data.push_back(_g->GetOutgoingRawAdjList(*_v, *_elabel));
    }
    return al;
}
#endif

void destroy_adjacent_list(AdjacentList al) {
    auto _al = static_cast<AdjacentList_T*>(al);
    delete _al;
}

size_t get_adjacent_list_size(const AdjacentList al) {
    auto _al = static_cast<AdjacentList_T*>(al);
    size_t result = 0;
    for (auto &ral : _al->data) {
        result += ral.Size();
    }
    return result;
}

Vertex get_neighbor_from_adjacent_list(const AdjacentList al, size_t idx) {
    auto _al = static_cast<AdjacentList_T*>(al);
    size_t result = 0;
    for (auto &ral : _al->data) {
        result += ral.Size();
        if (idx < result) {
            auto _idx = idx - (result - ral.size());
            auto _nbr = ral.begin() + _idx;
            auto v = new Vertex_T(_nbr->vid);
            return v;
        }
    }
    return NULL_VERTEX;
}

Edge get_edge_from_adjacent_list(const AdjacentList al, size_t idx) {
    auto _al = static_cast<AdjacentList_T*>(al);
    size_t result = 0;
    for (unsigned i = 0; i < _al->data.size(); ++i) {
        result += _al->data[i].Size();
        if (idx < result) {
            auto _idx = idx - (result - _al->data[i].Size());
            auto _nbr = _al->data[i].begin() + _idx;
            auto v = new Vertex_T(_nbr->vid);
            auto e = new Edge_T();
            e->src = _al->v;
            e->dst = v;
            e->dir = _al->dir;
            e->elabel = _al->data.size() > 1 ? i : _al->elabel;
            e->eid = _nbr->eid;
            return e;
        }
    }
    return NULL_EDGE;
}

#endif