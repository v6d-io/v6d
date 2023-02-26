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
#include "modules/graph/grin/include/property/type.h"

#ifdef WITH_VERTEX_PROPERTY
VertexType get_vertex_type(const Graph g, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto vt = new VertexType_T(_g->vertex_label(*_v));
    return vt;
}

char* get_vertex_type_name(const Graph g, const VertexType vt) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vt = static_cast<VertexType_T*>(vt);
    auto s = std::move(_g->schema().GetVertexLabelName(*_vt));
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

VertexType get_vertex_type_by_name(const Graph g, char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto s = std::string(name);
    auto vt = new VertexType_T(_g->schema().GetVertexLabelId(s));
    return vt;
}

VertexTypeList get_vertex_type_list(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto vtl = new VertexTypeList_T();
    for (auto i = 0; i < _g->vertex_label_num(); ++i) {
        vtl->push_back(i);
    }
    return vtl;
}

void destroy_vertex_type_list(VertexTypeList vtl) {
    auto _vtl = static_cast<VertexTypeList*>(vtl);
    delete _vtl;
}

VertexTypeList create_vertex_type_list() {
    auto vtl = new VertexTypeList_T();
    return vtl;
}

bool insert_vertex_type_to_list(VertexTypeList vtl, const VertexType vt) {
    auto _vtl = static_cast<VertexTypeList_T*>(vtl);
    auto _vt = static_cast<VertexType_T*>(vt);
    _vtl->push_back(*_vt);
    return true;
}

size_t get_vertex_type_list_size(const VertexTypeList vtl) {
    auto _vtl = static_cast<VertexTypeList_T*>(vtl);
    return _vtl->size();
}

VertexType get_vertex_type_from_list(const VertexTypeList vtl, const size_t idx) {
    auto _vtl = static_cast<VertexTypeList_T*>(vtl);
    auto vt = new VertexType_T((*_vtl)[idx]);
    return vt;
}
#endif


#ifdef NATURAL_VERTEX_TYPE_ID_TRAIT
VertexTypeID get_vertex_type_id(const VertexType vt) {
    auto _vt = static_cast<VertexType_T*>(vt);
    return *_vt;
}

VertexType get_vertex_type_from_id(const VertexTypeID vti) {
    auto vt = new VertexType_T(vti);
    return vt;
}
#endif


#ifdef WITH_EDGE_PROPERTY
EdgeType get_edge_type(const Graph g, const Edge e) {
    auto _e = static_cast<Edge_T*>(e);
    auto et = new EdgeType_T(_e->etype);
    return et;
}

char* get_edge_type_name(const Graph g, const EdgeType et) {
    auto _g = static_cast<Graph_T*>(g);
    auto _et = static_cast<EdgeType_T*>(et);
    auto s = std::move(_g->schema().GetEdgeLabelName(*_et));
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;    
}

EdgeType get_edge_type_by_name(const Graph g, char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto s = std::string(name);
    auto et = new EdgeType_T(_g->schema().GetEdgeLabelId(s));
    return et;
}

EdgeTypeList get_edge_type_list(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto etl = new EdgeTypeList_T();
    for (auto i = 0; i < _g->edge_label_num(); ++i) {
        etl->push_back(i);
    }
    return etl;
}

void destroy_edge_type_list(EdgeTypeList etl) {
    auto _etl = static_cast<EdgeTypeList_T*>(etl);
    delete _etl;
}

EdgeTypeList create_edge_type_list() {
    auto etl = new EdgeTypeList_T();
    return etl;
}

bool insert_edge_type_to_list(EdgeTypeList etl, const EdgeType et) {
    auto _etl = static_cast<EdgeTypeList_T*>(etl);
    auto _et = static_cast<EdgeType_T*>(et);
    _etl->push_back(*_et);
    return true;
}

size_t get_edge_type_list_size(const EdgeTypeList etl) {
    auto _etl = static_cast<EdgeTypeList_T*>(etl);
    return _etl->size();
}

EdgeType get_edge_type_from_list(const EdgeTypeList etl, const size_t idx) {
    auto _etl = static_cast<VertexTypeList_T*>(etl);
    auto et = new VertexType_T((*_etl)[idx]);
    return et;
}
#endif


#ifdef NATURAL_EDGE_TYPE_ID_TRAIT
EdgeTypeID get_edge_type_id(const EdgeType et) {
    auto _et = static_cast<EdgeType_T*>(et);
    return *_et;
}

EdgeType get_edge_type_from_id(const EdgeTypeID eti) {
    auto et = new EdgeType_T(eti);
    return et;
}
#endif


#if defined(WITH_VERTEX_PROPERTY) && defined(WITH_EDGE_PROPERTY)
VertexTypeList get_src_types_from_edge_type(const Graph g, const EdgeType etype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _etype = static_cast<EdgeType_T*>(etype);
    auto entry = _g->schema().GetEntry(*_etype, "EDGE");
    auto vtl = new VertexTypeList_T();
    for (auto& pair : entry.relations) {
        vtl->push_back(VertexType_T(_g->schema().GetVertexLabelId(pair.first)));
    }
    return vtl;
}

VertexTypeList get_dst_types_from_edge_type(const Graph g, const EdgeType etype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _etype = static_cast<EdgeType_T*>(etype);
    auto entry = _g->schema().GetEntry(*_etype, "EDGE");
    auto vtl = new VertexTypeList_T();
    for (auto& pair : entry.relations) {
        vtl->push_back(VertexType_T(_g->schema().GetVertexLabelId(pair.second)));
    }
    return vtl;
}

EdgeTypeList get_edge_types_from_vertex_type_pair(const Graph g, const VertexType src_vt, 
                                                  const VertexType dst_vt) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v1 = static_cast<VertexType_T*>(src_vt);
    auto _v2 = static_cast<VertexType_T*>(dst_vt);
    auto str_v1 = _g->schema().GetVertexLabelName(*_v1);
    auto str_v2 = _g->schema().GetVertexLabelName(*_v2);

    auto etl = new EdgeTypeList_T();
    for (auto etype = 0; etype < _g->edge_label_num(); ++etype) {
        auto entry = _g->schema().GetEntry(etype, "EDGE");
        for (auto& pair : entry.relations) {
            if (pair.first == str_v1 && pair.second == str_v2) {
                etl->push_back(EdgeType_T(etype));
            }
        }
    }
    return etl;
}
#endif
