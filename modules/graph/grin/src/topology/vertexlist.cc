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
extern "C" {
#include "graph/grin/include/topology/vertexlist.h"
}

#ifdef GRIN_ENABLE_VERTEX_LIST
GRIN_VERTEX_LIST grin_get_vertex_list(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vl = new GRIN_VERTEX_LIST_T();
    vl->all_master_mirror = 0;
    vl->vtype = _g->vertex_label_num();
    if (vl->vtype == 1) {
        vl->is_simple = true;
        __grin_init_simple_vertex_list(_g, vl);
    } else {
        vl->is_simple = false;
    }
    return vl;
}

void grin_destroy_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    delete _vl;
}

size_t grin_get_vertex_list_size(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    if (_vl->is_simple) return _vl->end_ - _vl->begin_;
    if (_vl->offsets.empty()) __grin_init_complex_vertex_list(static_cast<GRIN_GRAPH_T*>(g)->g, _vl);
    return _vl->offsets[_vl->vtype].first;
}

GRIN_VERTEX grin_get_vertex_from_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t idx) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    auto v = _vl->begin_ + idx;
    if (v < _vl->end_) return v;
    if (_vl->is_simple) return GRIN_NULL_VERTEX;
    if (_vl->offsets.empty()) __grin_init_complex_vertex_list(static_cast<GRIN_GRAPH_T*>(g)->g, _vl);
    for (unsigned i = 0; i < _vl->vtype; ++i) {
        if (idx < _vl->offsets[i+1].first) {
            v = _vl->offsets[i].second + idx - _vl->offsets[i].first;
            return v;
        }
    }
    return GRIN_NULL_VERTEX;
}
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST_ITERATOR
GRIN_VERTEX_LIST_ITERATOR grin_get_vertex_list_begin(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    auto vli = new GRIN_VERTEX_LIST_ITERATOR_T();
    vli->all_master_mirror = _vl->all_master_mirror;
    if (_vl->is_simple) {
        vli->is_simple = true;
        vli->vtype_current = _vl->vtype;
        vli->vtype_end = _vl->vtype + 1;
    } else {
        vli->is_simple = false;
        vli->vtype_current = 0;
        vli->vtype_end = _vl->vtype;
    }
    __grin_next_valid_vertex_list_iterator(_g, vli);
    return vli;
}

void grin_destroy_vertex_list_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    delete _vli;
}

void grin_get_next_vertex_list_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    _vli->current_++;
    if (_vli->current_ < _vli->end_) return;
    if (_vli->is_simple) {
        _vli->vtype_current++;
        return;
    }

    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    _GRIN_GRAPH_T::vertex_range_t vr;
    _vli->vtype_current++;
    __grin_next_valid_vertex_list_iterator(_g, _vli);
}

bool grin_is_vertex_list_end(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    return _vli->vtype_current >= _vli->vtype_end;
}

GRIN_VERTEX grin_get_vertex_from_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    return _vli->current_;
}
#endif
