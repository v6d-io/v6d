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
#include "graph/grin/include/topology/vertexlist.h"

#ifdef GRIN_ENABLE_VERTEX_LIST
GRIN_VERTEX_LIST grin_get_vertex_list(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto vl = new GRIN_VERTEX_LIST_T();
    vl->type_begin = 0;
    vl->type_end = _g->vertex_label_num();
    __grin_init_vertex_list(_g, vl);
    return vl;
}

void grin_destroy_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    delete _vl;
}

size_t grin_get_vertex_list_size(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    return _vl->offsets[_vl->type_end - _vl->type_begin];
}

GRIN_VERTEX grin_get_vertex_from_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t idx) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    for (auto i = 0; i < _vl->type_end - _vl->type_begin; ++i) {        
        if (idx < _vl->offsets[i+1]) {
            auto _idx = idx - _vl->offsets[i];
            auto v = new GRIN_VERTEX_T(_vl->vrs[i].begin_value() + _idx);
            return v;
        }
    }
    return GRIN_NULL_VERTEX;
}
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST_ITERATOR
GRIN_VERTEX_LIST_ITERATOR grin_get_vertex_list_begin(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    auto vli = new GRIN_VERTEX_LIST_ITERATOR_T();
    vli->type_begin = _vl->type_begin;
    vli->type_end = _vl->type_end;
    vli->type_current = _vl->type_begin;
    vli->current = 0;
    vli->all_master_mirror = _vl->all_master_mirror;
    if (vli->all_master_mirror == 0) {
        vli->vr = _g->Vertices(vli->type_current);
    } else if (vli->all_master_mirror == 1) {
        vli->vr = _g->InnerVertices(vli->type_current);
    } else {
        vli->vr = _g->OuterVertices(vli->type_current);
    }
    return vli;
}

void grin_destroy_vertex_list_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    delete _vli;
}

void grin_get_next_vertex_list_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    _vli->current++;
    while (_vli->type_current < _vli->type_end) {
        if (_vli->current < _vli->vr.size()) break;
        _vli->type_current++;
        _vli->current = 0;
        if (_vli->type_current < _vli->type_end) {
            if (_vli->all_master_mirror == 0) {
                _vli->vr = _g->Vertices(_vli->type_current);
            } else if (_vli->all_master_mirror == 1) {
                _vli->vr = _g->InnerVertices(_vli->type_current);
            } else {
                _vli->vr = _g->OuterVertices(_vli->type_current);
            }
        }
    }
}

bool grin_is_vertex_list_end(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    return _vli->type_current >= _vli->type_end;
}

GRIN_VERTEX grin_get_vertex_from_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    auto v = new GRIN_VERTEX_T(_vli->vr.begin_value() + _vli->current);
    return v;
}
#endif
