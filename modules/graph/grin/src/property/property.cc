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
#include "property/property.h"

GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_property_list_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    vpl->resize(_g->vertex_property_num(vtype));
    for (auto p = 0; p < _g->vertex_property_num(vtype); ++p) {
        (*vpl)[p] = _grin_create_property(vtype, p);
    }
    return vpl;
}

GRIN_VERTEX_TYPE grin_get_vertex_type_from_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    return _grin_get_type_from_property(vp);
}

const char* grin_get_vertex_property_name(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY vp) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->vprop_names[_grin_get_type_from_property(vp)][_grin_get_prop_from_property(vp)].c_str();
}

GRIN_VERTEX_PROPERTY grin_get_vertex_property_by_name(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype,
                                           const char* name) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    auto s = std::string(name);
    for (auto i = 0; i < _cache->vprop_names[vtype].size(); ++i) {
        if (_cache->vprop_names[vtype][i] == s) return _grin_create_property(vtype, i);
    }
    return GRIN_NULL_VERTEX_PROPERTY;
}

GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_properties_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    auto s = std::string(name);
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        for (auto i = 0; i < _cache->vprop_names[vtype].size(); ++i) {
            if (_cache->vprop_names[vtype][i] == s) {
                vpl->push_back(_grin_create_property(vtype, i));
                break;
            }
        }   
    }
    if (vpl->empty()) {
        delete vpl;
        return GRIN_NULL_VERTEX_PROPERTY_LIST;
    }
    return vpl;
}

GRIN_VERTEX_PROPERTY grin_get_vertex_property_by_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY_ID vpi) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    if (static_cast<int>(vpi) >= _g->vertex_property_num(vtype)) return GRIN_NULL_VERTEX_PROPERTY;
    return _grin_create_property(vtype, vpi);
}

GRIN_VERTEX_PROPERTY_ID grin_get_vertex_property_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY vp) {
    if (vtype != _grin_get_type_from_property(vp)) return GRIN_NULL_VERTEX_PROPERTY_ID;
    return _grin_get_prop_from_property(vp);
}

GRIN_EDGE_PROPERTY_LIST grin_get_edge_property_list_by_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    epl->resize(_g->edge_property_num(etype) - 1);
    for (auto p = 1; p < _g->edge_property_num(etype); ++p) {
        (*epl)[p - 1] = _grin_create_property(etype, p);
    }
    return epl;
}

const char* grin_get_edge_property_name(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY ep) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->eprop_names[_grin_get_type_from_property(ep)][_grin_get_prop_from_property(ep)].c_str();
}

GRIN_EDGE_PROPERTY grin_get_edge_property_by_name(GRIN_GRAPH g, GRIN_EDGE_TYPE etype,
                                           const char* name) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    auto s = std::string(name);
    for (auto i = 0; i < _cache->eprop_names[etype].size(); ++i) {
        if (_cache->eprop_names[etype][i] == s) return _grin_create_property(etype, i);
    }
    return GRIN_NULL_EDGE_PROPERTY;
}

GRIN_EDGE_PROPERTY_LIST grin_get_edge_properties_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    auto s = std::string(name);
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    for (auto etype = 0; etype < _g->edge_label_num(); ++etype) {
        for (auto i = 0; i < _cache->eprop_names[etype].size(); ++i) {
            if (_cache->eprop_names[etype][i] == s) {
                epl->push_back(_grin_create_property(etype, i));
                break;
            }
        }   
    }
    if (epl->empty()) {
        delete epl;
        return GRIN_NULL_EDGE_PROPERTY_LIST;
    }
    return epl;
}

GRIN_EDGE_TYPE grin_get_edge_type_from_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    return _grin_get_type_from_property(ep);
}

GRIN_EDGE_PROPERTY grin_get_edge_property_by_id(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY_ID epi) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    if (static_cast<int>(epi) >= _g->edge_property_num(etype) - 1) return GRIN_NULL_EDGE_PROPERTY;
    return _grin_create_property(etype, epi + 1);
}

GRIN_EDGE_PROPERTY_ID grin_get_edge_property_id(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY ep) {
    if (etype != _grin_get_type_from_property(ep)) return GRIN_NULL_EDGE_PROPERTY_ID;
    return _grin_get_prop_from_property(ep) - 1;
}

bool grin_equal_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp1, GRIN_VERTEX_PROPERTY vp2) {
    return (vp1 == vp2);
}

void grin_destroy_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {}

GRIN_DATATYPE grin_get_vertex_property_datatype(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto properties = _g->schema().GetEntry(_grin_get_type_from_property(vp), "VERTEX").properties();
    auto dt = _g->schema().GetVertexPropertyType(_grin_get_type_from_property(vp), properties[_grin_get_prop_from_property(vp)].id);
    return ArrowToDataType(dt);
}

bool grin_equal_edge_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep1, GRIN_EDGE_PROPERTY ep2) {
    return (ep1 == ep2);
}

void grin_destroy_edge_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {}

GRIN_DATATYPE grin_get_edge_property_datatype(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto dt = _g->schema().GetEdgePropertyType(_grin_get_type_from_property(ep), _grin_get_prop_from_property(ep));
    return ArrowToDataType(dt);
}
