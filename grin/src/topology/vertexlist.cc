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
#include "grin/include/topology/vertexlist.h"

#ifdef ENABLE_VERTEX_LIST

VertexList get_vertex_list(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto vl = new VertexList_T();
    for (VertexLabel_T vlabel = 0; vlabel < _g->vertex_label_num(); ++vlabel) {
        vl->push_back(_g->Vertices(vlabel));
    }
    return vl;
}

#ifdef WITH_VERTEX_LABEL
VertexList get_vertex_list_by_label(const Graph g, const VertexLabel vlabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto vl = new VertexList_T();
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    vl->push_back(_g->Vertices(*_vlabel));
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

#ifdef CONTINUOUS_VERTEX_ID_TRAIT
bool is_vertex_list_continuous(const VertexList vl) {
    auto _vl = static_cast<VertexList_T*>(vl);
    return _vl->size() == 1;
}

VertexID get_begin_vertex_id_from_list(const VertexList vl) {
    auto _vl = static_cast<VertexList_T*>(vl);
    auto _vid = (*_vl)[0].begin_value();
    auto vid = new VertexID_T(_vid);
    return vid;
} 

VertexID get_end_vertex_id_from_list(const VertexList vl) {
    auto _vl = static_cast<VertexList_T*>(vl);
    auto _vid = (*_vl)[0].end_value();
    auto vid = new VertexID_T(_vid);
    return vid;
}

DataType get_vertex_id_data_type(const Graph g) {
    return DataTypeEnum<VertexID_T>::value;   
}

VertexID get_vertex_id(const Vertex v) {
    auto _v = static_cast<Vertex_T*>(v);
    auto vid = new VertexID_T(_v->GetValue());
    return vid;
}

Vertex get_vertex_from_id(const VertexID vid) {
    auto _vid = static_cast<VertexID_T*>(vid);
    auto v = new Vertex_T(*_vid);
    return v;
}

void destroy_vertex_id(VertexID vid) {
    auto _vid = static_cast<VertexID_T*>(vid);
    delete _vid;
}
#endif

#endif
