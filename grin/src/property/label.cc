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

// This header file is not available for libgrape-lite.

#include "grin/src/predefine.h"
#include "grin/include/property/label.h"

#ifdef WITH_VERTEX_LABEL
// Vertex label
VertexLabel get_vertex_label(const Graph g, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto vl = new VertexLabel_T(_g->vertex_label(*_v));
    return vl;
}

char* get_vertex_label_name(const Graph g, const VertexLabel vl) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vl = static_cast<VertexLabel_T*>(vl);
    auto s = std::move(_g->schema().GetVertexLabelName(*_vl));
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

VertexLabel get_vertex_label_by_name(const Graph g, char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto s = std::string(name);
    auto vl = new VertexLabel_T(_g->schema().GetVertexLabelId(s));
    return vl;
}

#ifdef NATURAL_VERTEX_LABEL_ID_TRAIT
VertexLabelID get_vertex_label_id(const VertexLabel vl) {
    auto _vl = static_cast<VertexLabel_T*>(vl);
    return *_vl;
}

VertexLabel get_vertex_label_from_id(const VertexLabelID vli) {
    auto vl = new VertexLabel_T(vli);
    return vl;
}
#endif

// Vertex label list
VertexLabelList get_vertex_label_list(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto vll = new VertexLabelList_T();
    for (auto i = 0; i < _g->vertex_label_num(); ++i) {
        vll->push_back(i);
    }
    return vll;
}

void destroy_vertex_label_list(VertexLabelList vll) {
    auto _vll = static_cast<VertexLabelList*>(vll);
    delete _vll;
}

VertexLabelList create_vertex_label_list() {
    auto vll = new VertexLabelList_T();
    return vll;
}

bool insert_vertex_label_to_list(VertexLabelList vll, const VertexLabel vl) {
    auto _vll = static_cast<VertexLabelList_T*>(vll);
    auto _vl = static_cast<VertexLabel_T*>(vl);
    _vll->push_back(*_vl);
    return true;
}

size_t get_vertex_label_list_size(const VertexLabelList vll) {
    auto _vll = static_cast<VertexLabelList_T*>(vll);
    return _vll->size();
}

VertexLabel get_vertex_label_from_list(const VertexLabelList vll, const size_t idx) {
    auto _vll = static_cast<VertexLabelList_T*>(vll);
    auto vl = new VertexLabel_T((*_vll)[idx]);
    return vl;
}
#endif

#ifdef WITH_EDGE_LABEL
// Edge label
EdgeLabel get_edge_label(const Graph g, const Edge e) {
    auto _e = static_cast<Edge_T*>(e);
    auto el = new EdgeLabel_T(_e->elabel);
    return el;
}

char* get_edge_label_name(const Graph g, const EdgeLabel el) {
    auto _g = static_cast<Graph_T*>(g);
    auto _el = static_cast<EdgeLabel_T*>(el);
    auto s = std::move(_g->schema().GetEdgeLabelName(*_el));
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;    
}

EdgeLabel get_edge_label_by_name(const Graph g, char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto s = std::string(name);
    auto el = new EdgeLabel_T(_g->schema().GetEdgeLabelId(s));
    return el;
}

#ifdef NATURAL_EDGE_LABEL_ID_TRAIT
EdgeLabelID get_edge_label_id(const EdgeLabel el) {
    auto _el = static_cast<EdgeLabel_T*>(el);
    return *_el;
}

EdgeLabel get_edge_label_from_id(const EdgeLabelID eli) {
    auto el = new EdgeLabel_T(eli);
    return el;
}
#endif

// Edge label list
EdgeLabelList get_edge_label_list(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto ell = new EdgeLabelList_T();
    for (auto i = 0; i < _g->vertex_label_num(); ++i) {
        ell->push_back(i);
    }
    return ell;
}

void destroy_edge_label_list(EdgeLabelList ell) {
    auto _ell = static_cast<EdgeLabelList_T*>(ell);
    delete _ell;
}

EdgeLabelList create_edge_label_list() {
    auto ell = new EdgeLabelList_T();
    return ell;
}

bool insert_edge_label_to_list(EdgeLabelList ell, const EdgeLabel el) {
    auto _ell = static_cast<EdgeLabelList_T*>(ell);
    auto _el = static_cast<EdgeLabel_T*>(el);
    _ell->push_back(*_el);
    return true;
}

size_t get_edge_label_list_size(const EdgeLabelList ell) {
    auto _ell = static_cast<EdgeLabelList_T*>(ell);
    return _ell->size();
}

EdgeLabel get_edge_label_from_list(const EdgeLabelList ell, const size_t idx) {
    auto _ell = static_cast<VertexLabelList_T*>(ell);
    auto el = new VertexLabel_T((*_ell)[idx]);
    return el;
}
#endif


#if defined(WITH_VERTEX_LABEL) && defined(WITH_EDGE_LABEL)
VertexLabel get_src_label_from_edge_label(const Graph g, const EdgeLabel elabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto entry = _g->schema().GetEntry(*_elabel, "EDGE");
    auto pair = entry.relations[0];
    auto vlabel = new VertexLabel_T(_g->schema().GetVertexLabelId(pair.first));
    return vlabel;
}

VertexLabel get_dst_label_from_edge_label(const Graph g, const EdgeLabel elabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto entry = _g->schema().GetEntry(*_elabel, "EDGE");
    auto pair = entry.relations[0];
    auto vlabel = new VertexLabel_T(_g->schema().GetVertexLabelId(pair.second));
    return vlabel;
}
#endif
