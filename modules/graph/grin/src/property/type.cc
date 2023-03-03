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
#include "graph/grin/include/property/type.h"

#ifdef GRIN_WITH_VERTEX_PROPERTY
bool grin_equal_vertex_type(GRIN_VERTEX_TYPE vt1, GRIN_VERTEX_TYPE vt2) {
    auto _vt1 = static_cast<GRIN_VERTEX_TYPE_T*>(vt1);
    auto _vt2 = static_cast<GRIN_VERTEX_TYPE_T*>(vt2);
    return (*_vt1 == *_vt2);
}

GRIN_VERTEX_TYPE grin_get_vertex_type(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto vt = new GRIN_VERTEX_TYPE_T(_g->vertex_label(*_v));
    return vt;
}

const char* grin_get_vertex_type_name(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vt = static_cast<GRIN_VERTEX_TYPE_T*>(vt);
    auto s = _g->schema().GetVertexLabelName(*_vt);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

GRIN_VERTEX_TYPE grin_get_vertex_type_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto s = std::string(name);
    auto vt = new GRIN_VERTEX_TYPE_T(_g->schema().GetVertexLabelId(s));
    return vt;
}

GRIN_VERTEX_TYPE_LIST grin_get_vertex_type_list(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    for (auto i = 0; i < _g->vertex_label_num(); ++i) {
        vtl->push_back(i);
    }
    return vtl;
}

void grin_destroy_vertex_type_list(GRIN_VERTEX_TYPE_LIST vtl) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST*>(vtl);
    delete _vtl;
}

GRIN_VERTEX_TYPE_LIST grin_create_vertex_type_list() {
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    return vtl;
}

bool grin_insert_vertex_type_to_list(GRIN_VERTEX_TYPE_LIST vtl, GRIN_VERTEX_TYPE vt) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(vtl);
    auto _vt = static_cast<GRIN_VERTEX_TYPE_T*>(vt);
    _vtl->push_back(*_vt);
    return true;
}

size_t grin_get_vertex_type_list_size(GRIN_VERTEX_TYPE_LIST vtl) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(vtl);
    return _vtl->size();
}

GRIN_VERTEX_TYPE grin_get_vertex_type_from_list(GRIN_VERTEX_TYPE_LIST vtl, size_t idx) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(vtl);
    auto vt = new GRIN_VERTEX_TYPE_T((*_vtl)[idx]);
    return vt;
}
#endif


#ifdef GRIN_NATURAL_VERTEX_TYPE_ID_TRAIT
GRIN_VERTEX_TYPE_ID grin_get_vertex_type_id(GRIN_VERTEX_TYPE vt) {
    auto _vt = static_cast<GRIN_VERTEX_TYPE_T*>(vt);
    return *_vt;
}

GRIN_VERTEX_TYPE grin_get_vertex_type_from_id(GRIN_VERTEX_TYPE_ID vti) {
    auto vt = new GRIN_VERTEX_TYPE_T(vti);
    return vt;
}
#endif


#ifdef GRIN_WITH_EDGE_PROPERTY
bool grin_equal_edge_type(GRIN_EDGE_TYPE et1, GRIN_EDGE_TYPE et2) {
    auto _et1 = static_cast<GRIN_EDGE_TYPE_T*>(et1);
    auto _et2 = static_cast<GRIN_EDGE_TYPE_T*>(et2);
    return (*_et1 == *_et2);
}

GRIN_EDGE_TYPE grin_get_edge_type(GRIN_GRAPH g, GRIN_EDGE e) {
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    auto et = new GRIN_EDGE_TYPE_T(_e->etype);
    return et;
}

const char* grin_get_edge_type_name(GRIN_GRAPH g, GRIN_EDGE_TYPE et) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _et = static_cast<GRIN_EDGE_TYPE_T*>(et);
    auto s = _g->schema().GetEdgeLabelName(*_et);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;    
}

GRIN_EDGE_TYPE grin_get_edge_type_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto s = std::string(name);
    auto et = new GRIN_EDGE_TYPE_T(_g->schema().GetEdgeLabelId(s));
    return et;
}

GRIN_EDGE_TYPE_LIST grin_get_edge_type_list(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto etl = new GRIN_EDGE_TYPE_LIST_T();
    for (auto i = 0; i < _g->edge_label_num(); ++i) {
        etl->push_back(i);
    }
    return etl;
}

void grin_destroy_edge_type_list(GRIN_EDGE_TYPE_LIST etl) {
    auto _etl = static_cast<GRIN_EDGE_TYPE_LIST_T*>(etl);
    delete _etl;
}

GRIN_EDGE_TYPE_LIST grin_create_edge_type_list() {
    auto etl = new GRIN_EDGE_TYPE_LIST_T();
    return etl;
}

bool grin_insert_edge_type_to_list(GRIN_EDGE_TYPE_LIST etl, GRIN_EDGE_TYPE et) {
    auto _etl = static_cast<GRIN_EDGE_TYPE_LIST_T*>(etl);
    auto _et = static_cast<GRIN_EDGE_TYPE_T*>(et);
    _etl->push_back(*_et);
    return true;
}

size_t grin_get_edge_type_list_size(GRIN_EDGE_TYPE_LIST etl) {
    auto _etl = static_cast<GRIN_EDGE_TYPE_LIST_T*>(etl);
    return _etl->size();
}

GRIN_EDGE_TYPE grin_get_edge_type_from_list(GRIN_EDGE_TYPE_LIST etl, size_t idx) {
    auto _etl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(etl);
    auto et = new GRIN_VERTEX_TYPE_T((*_etl)[idx]);
    return et;
}
#endif


#ifdef GRIN_NATURAL_EDGE_TYPE_ID_TRAIT
GRIN_EDGE_TYPE_ID grin_get_edge_type_id(GRIN_EDGE_TYPE et) {
    auto _et = static_cast<GRIN_EDGE_TYPE_T*>(et);
    return *_et;
}

GRIN_EDGE_TYPE grin_get_edge_type_from_id(GRIN_EDGE_TYPE_ID eti) {
    auto et = new GRIN_EDGE_TYPE_T(eti);
    return et;
}
#endif


#if defined(GRIN_WITH_VERTEX_PROPERTY) && defined(GRIN_WITH_EDGE_PROPERTY)
GRIN_VERTEX_TYPE_LIST grin_get_src_types_from_edge_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto entry = _g->schema().GetEntry(*_etype, "EDGE");
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    for (auto& pair : entry.relations) {
        vtl->push_back(GRIN_VERTEX_TYPE_T(_g->schema().GetVertexLabelId(pair.first)));
    }
    return vtl;
}

GRIN_VERTEX_TYPE_LIST grin_get_dst_types_from_edge_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto entry = _g->schema().GetEntry(*_etype, "EDGE");
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    for (auto& pair : entry.relations) {
        vtl->push_back(GRIN_VERTEX_TYPE_T(_g->schema().GetVertexLabelId(pair.second)));
    }
    return vtl;
}

GRIN_EDGE_TYPE_LIST grin_get_edge_types_from_vertex_type_pair(GRIN_GRAPH g, GRIN_VERTEX_TYPE src_vt, 
                                                  GRIN_VERTEX_TYPE dst_vt) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v1 = static_cast<GRIN_VERTEX_TYPE_T*>(src_vt);
    auto _v2 = static_cast<GRIN_VERTEX_TYPE_T*>(dst_vt);
    auto str_v1 = _g->schema().GetVertexLabelName(*_v1);
    auto str_v2 = _g->schema().GetVertexLabelName(*_v2);

    auto etl = new GRIN_EDGE_TYPE_LIST_T();
    for (auto etype = 0; etype < _g->edge_label_num(); ++etype) {
        auto entry = _g->schema().GetEntry(etype, "EDGE");
        for (auto& pair : entry.relations) {
            if (pair.first == str_v1 && pair.second == str_v2) {
                etl->push_back(GRIN_EDGE_TYPE_T(etype));
            }
        }
    }
    return etl;
}
#endif
