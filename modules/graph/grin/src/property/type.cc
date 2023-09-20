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
#include "property/type.h"

/**************** vertex type ****************/
bool grin_equal_vertex_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt1, GRIN_VERTEX_TYPE vt2) {
    return (vt1 == vt2);
}

GRIN_VERTEX_TYPE grin_get_vertex_type(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->id_parser.GetLabelId(v);
}

void grin_destroy_vertex_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt) {}

const char* grin_get_vertex_type_name(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->vtype_names[vtype].c_str();
}

GRIN_VERTEX_TYPE grin_get_vertex_type_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto s = std::string(name);
    auto _id = _g->schema().GetVertexLabelId(s);
    if (_id == -1 ) return GRIN_NULL_VERTEX_TYPE;
    return _id;
}

GRIN_VERTEX_TYPE_ID grin_get_vertex_type_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    return vtype;
}

GRIN_VERTEX_TYPE grin_get_vertex_type_by_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE_ID vti) {
    return vti;
}


/**************** vertex type list ****************/
GRIN_VERTEX_TYPE_LIST grin_get_vertex_type_list(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    vtl->resize(_g->vertex_label_num());
    for (auto i = 0; i < _g->vertex_label_num(); ++i) {
        (*vtl)[i] = i;
    }
    return vtl;
}

void grin_destroy_vertex_type_list(GRIN_GRAPH g, GRIN_VERTEX_TYPE_LIST vtl) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST*>(vtl);
    delete _vtl;
}

GRIN_VERTEX_TYPE_LIST grin_create_vertex_type_list(GRIN_GRAPH g) {
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    return vtl;
}

bool grin_insert_vertex_type_to_list(GRIN_GRAPH g, GRIN_VERTEX_TYPE_LIST vtl, GRIN_VERTEX_TYPE vtype) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(vtl);
    _vtl->push_back(vtype);
    return true;
}

size_t grin_get_vertex_type_list_size(GRIN_GRAPH g, GRIN_VERTEX_TYPE_LIST vtl) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(vtl);
    return _vtl->size();
}

GRIN_VERTEX_TYPE grin_get_vertex_type_from_list(GRIN_GRAPH g, GRIN_VERTEX_TYPE_LIST vtl, size_t idx) {
    auto _vtl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(vtl);
    return (*_vtl)[idx];
}


/**************** edge type ****************/
bool grin_equal_edge_type(GRIN_GRAPH g, GRIN_EDGE_TYPE et1, GRIN_EDGE_TYPE et2) {
    return (et1 == et2);
}

GRIN_EDGE_TYPE grin_get_edge_type(GRIN_GRAPH g, GRIN_EDGE e) {
    return e.etype;
}

void grin_destroy_edge_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {}

const char* grin_get_edge_type_name(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->etype_names[etype].c_str();
}

GRIN_EDGE_TYPE grin_get_edge_type_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto s = std::string(name);
    auto _id = _g->schema().GetEdgeLabelId(s);
    if (_id == -1) return GRIN_NULL_EDGE_TYPE;
    return _id;
}

GRIN_EDGE_TYPE_ID grin_get_edge_type_id(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    return etype;
}

GRIN_EDGE_TYPE grin_get_edge_type_by_id(GRIN_GRAPH g, GRIN_EDGE_TYPE_ID eti) {
    return eti;
}


/**************** edge type list ****************/
GRIN_EDGE_TYPE_LIST grin_get_edge_type_list(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto etl = new GRIN_EDGE_TYPE_LIST_T();
    etl->resize(_g->edge_label_num());
    for (auto i = 0; i < _g->edge_label_num(); ++i) {
        (*etl)[i] = i;
    }
    return etl;
}

void grin_destroy_edge_type_list(GRIN_GRAPH g, GRIN_EDGE_TYPE_LIST etl) {
    auto _etl = static_cast<GRIN_EDGE_TYPE_LIST_T*>(etl);
    delete _etl;
}

GRIN_EDGE_TYPE_LIST grin_create_edge_type_list(GRIN_GRAPH g) {
    auto etl = new GRIN_EDGE_TYPE_LIST_T();
    return etl;
}

bool grin_insert_edge_type_to_list(GRIN_GRAPH g, GRIN_EDGE_TYPE_LIST etl, GRIN_EDGE_TYPE etype) {
    auto _etl = static_cast<GRIN_EDGE_TYPE_LIST_T*>(etl);
    _etl->push_back(etype);
    return true;
}

size_t grin_get_edge_type_list_size(GRIN_GRAPH g, GRIN_EDGE_TYPE_LIST etl) {
    auto _etl = static_cast<GRIN_EDGE_TYPE_LIST_T*>(etl);
    return _etl->size();
}

GRIN_EDGE_TYPE grin_get_edge_type_from_list(GRIN_GRAPH g, GRIN_EDGE_TYPE_LIST etl, size_t idx) {
    auto _etl = static_cast<GRIN_VERTEX_TYPE_LIST_T*>(etl);
    return (*_etl)[idx];
}


/**************** relations between vertex type pair and edge type ****************/
GRIN_VERTEX_TYPE_LIST grin_get_src_types_by_edge_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto entry = _g->schema().GetEntry(etype, "EDGE");
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    for (auto& pair : entry.relations) {
        vtl->push_back(GRIN_VERTEX_TYPE_T(_g->schema().GetVertexLabelId(pair.first)));
    }
    return vtl; // TODO
}

GRIN_VERTEX_TYPE_LIST grin_get_dst_types_by_edge_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto entry = _g->schema().GetEntry(etype, "EDGE");
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    for (auto& pair : entry.relations) {
        vtl->push_back(GRIN_VERTEX_TYPE_T(_g->schema().GetVertexLabelId(pair.second)));
    }
    return vtl; // TODO
}

GRIN_EDGE_TYPE_LIST grin_get_edge_types_by_vertex_type_pair(GRIN_GRAPH g, GRIN_VERTEX_TYPE src_vt, 
                                                  GRIN_VERTEX_TYPE dst_vt) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto str_v1 = _g->schema().GetVertexLabelName(src_vt);
    auto str_v2 = _g->schema().GetVertexLabelName(dst_vt);

    auto etl = new GRIN_EDGE_TYPE_LIST_T();
    for (auto etype = 0; etype < _g->edge_label_num(); ++etype) {
        auto entry = _g->schema().GetEntry(etype, "EDGE");
        for (auto& pair : entry.relations) {
            if (pair.first == str_v1 && pair.second == str_v2) {
                etl->push_back((unsigned)etype);
            }
        }
    }
    return etl; // TODO
}
