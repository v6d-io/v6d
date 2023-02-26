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

#include "modules/graph/grin/src/predefine.h"
#include "modules/graph/grin/include/topology/vertexlist.h"

#ifdef ENABLE_VERTEX_LIST
VertexList get_vertex_list(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto vl = new VertexList_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        vl->push_back(_g->Vertices(vtype));
    }
    return vl;
}

#ifdef WITH_VERTEX_PROPERTY
VertexList get_vertex_list_by_type(const Graph g, const VertexType vtype) {
    auto _g = static_cast<Graph_T*>(g);
    auto vl = new VertexList_T();
    auto _vtype = static_cast<VertexType_T*>(vtype);
    vl->push_back(_g->Vertices(*_vtype));
    return vl;
}
#endif

void destroy_vertex_list(VertexList vl) {
    auto _vl = static_cast<VertexList_T*>(vl);
    delete _vl;
}

VertexList create_vertex_list() {
    auto vl = new VertexList_T();
    return vl;
}

bool insert_vertex_to_list(VertexList vl, const Vertex v) {
    auto _vl = static_cast<VertexList_T*>(vl);
    auto _v = static_cast<Vertex_T*>(v);
    _vl->push_back(Graph_T::vertex_range_t(_v->GetValue(), _v->GetValue()));
    return true;
}

size_t get_vertex_list_size(const VertexList vl) {
    auto _vl = static_cast<VertexList_T*>(vl);
    size_t result = 0;
    for (auto &vr : *_vl) {
        result += vr.size();
    }
    return result;
}

Vertex get_vertex_from_list(const VertexList vl, const size_t idx) {
    auto _vl = static_cast<VertexList_T*>(vl);
    size_t result = 0;
    for (auto &vr : *_vl) {
        result += vr.size();
        if (idx < result) {
            auto _idx = idx - (result - vr.size());
            auto v = new Vertex_T(vr.begin_value() + _idx);
            return v;
        }
    }
    return NULL_VERTEX;
}
#endif
