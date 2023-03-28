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
#include "graph/grin/include/property/property.h"

#ifdef GRIN_WITH_VERTEX_PROPERTY_NAME
const char* grin_get_vertex_property_name(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vp = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp);
    auto s = _g->schema().GetVertexPropertyName(_vp->first, _vp->second);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

void destroy_vertex_property_name(GRIN_GRAPH g, const char* name) {
    delete[] name;
}

GRIN_VERTEX_PROPERTY grin_get_vertex_property_by_name(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype,
                                           const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto s = std::string(name);
    auto vp = new GRIN_VERTEX_PROPERTY_T(*_vtype, _g->schema().GetVertexPropertyId(*_vtype, s));
    return vp;
}

GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_properties_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto s = std::string(name);
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        vpl->push_back(GRIN_VERTEX_PROPERTY_T(vtype, _g->schema().GetVertexPropertyId(vtype, s)));
    }
    return vpl;
}
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY_NAME
const char* grin_get_edge_property_name(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _ep = static_cast<GRIN_EDGE_PROPERTY_T*>(ep);
    auto s = _g->schema().GetEdgePropertyName(_ep->first, _ep->second);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

void destroy_edge_property_name(GRIN_GRAPH g, const char* name) {
    delete[] name;
}

GRIN_EDGE_PROPERTY grin_get_edge_property_by_name(GRIN_GRAPH g, GRIN_EDGE_TYPE etype,
                                           const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto s = std::string(name);
    auto ep = new GRIN_EDGE_PROPERTY_T(*_etype, _g->schema().GetEdgePropertyId(*_etype, s));
    return ep;
}

GRIN_EDGE_PROPERTY_LIST grin_get_edge_properties_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto s = std::string(name);
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    for (auto etype = 0; etype < _g->edge_label_num(); ++etype) {
        epl->push_back(GRIN_EDGE_PROPERTY_T(etype, _g->schema().GetVertexPropertyId(etype, s)));
    }
    return epl;
}
#endif


#ifdef GRIN_WITH_VERTEX_PROPERTY
bool grin_equal_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp1, GRIN_VERTEX_PROPERTY vp2) {
    auto _vp1 = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp1);
    auto _vp2 = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp2);
    return (*_vp1 == *_vp2);
}

void grin_destroy_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    auto _vp = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp);
    delete _vp;
}

GRIN_DATATYPE grin_get_vertex_property_data_type(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vp = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp);
    auto dt = _g->schema().GetVertexPropertyType(_vp->first, _vp->second);
    return ArrowToDataType(dt);
}

GRIN_VERTEX_TYPE grin_get_vertex_property_vertex_type(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    auto _vp = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp);
    auto vt = new GRIN_VERTEX_TYPE_T(_vp->first);
    return vt;
}
#endif


#ifdef GRIN_WITH_EDGE_PROPERTY
bool grin_equal_edge_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep1, GRIN_EDGE_PROPERTY ep2) {
    auto _ep1 = static_cast<GRIN_EDGE_PROPERTY_T*>(ep1);
    auto _ep2 = static_cast<GRIN_EDGE_PROPERTY_T*>(ep2);
    return (*_ep1 == *_ep2);
}

void grin_destroy_edge_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    auto _ep = static_cast<GRIN_EDGE_PROPERTY_T*>(ep);
    delete _ep;
}

GRIN_DATATYPE grin_get_edge_property_data_type(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _ep = static_cast<GRIN_EDGE_PROPERTY_T*>(ep);
    auto dt = _g->schema().GetEdgePropertyType(_ep->first, _ep->second);
    return ArrowToDataType(dt);
}

GRIN_EDGE_TYPE grin_get_edge_property_edge_type(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    auto _ep = static_cast<GRIN_EDGE_PROPERTY_T*>(ep);
    auto et = new GRIN_EDGE_TYPE_T(_ep->first);
    return et;
}
#endif
