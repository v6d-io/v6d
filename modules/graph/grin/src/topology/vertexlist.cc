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
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        vl->push_back(_g->Vertices(vtype));
    }
    return vl;
}

#ifdef GRIN_WITH_VERTEX_PROPERTY
GRIN_VERTEX_LIST grin_get_vertex_list_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto vl = new GRIN_VERTEX_LIST_T();
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    vl->push_back(_g->Vertices(*_vtype));
    return vl;
}
#endif

void grin_destroy_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    delete _vl;
}

GRIN_VERTEX_LIST grin_create_vertex_list(GRIN_GRAPH g) {
    auto vl = new GRIN_VERTEX_LIST_T();
    return vl;
}

bool grin_insert_vertex_to_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, GRIN_VERTEX v) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    _vl->push_back(GRIN_GRAPH_T::vertex_range_t(_v->GetValue(), _v->GetValue()));
    return true;
}

size_t grin_get_vertex_list_size(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    size_t result = 0;
    for (auto &vr : *_vl) {
        result += vr.size();
    }
    return result;
}

GRIN_VERTEX grin_get_vertex_from_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t idx) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    size_t result = 0;
    for (auto &vr : *_vl) {
        result += vr.size();
        if (idx < result) {
            auto _idx = idx - (result - vr.size());
            auto v = new GRIN_VERTEX_T(vr.begin_value() + _idx);
            return v;
        }
    }
    return GRIN_NULL_VERTEX;
}
#endif
