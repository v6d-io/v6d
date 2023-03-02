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

#if defined(WITH_PROPERTY_NAME) && defined(WITH_VERTEX_PROPERTY)
const char* get_vertex_property_name(Graph g, VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    auto s = _g->schema().GetVertexPropertyName(_vp->first, _vp->second);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

VertexProperty get_vertex_property_by_name(Graph g, VertexType vtype,
                                           const char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto s = std::string(name);
    auto vp = new VertexProperty_T(*_vtype, _g->schema().GetVertexPropertyId(*_vtype, s));
    return vp;
}

VertexPropertyList get_vertex_properties_by_name(Graph g, const char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto s = std::string(name);
    auto vpl = new VertexPropertyList_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        vpl->push_back(VertexProperty_T(vtype, _g->schema().GetVertexPropertyId(vtype, s)));
    }
    return vpl;
}
#endif

#if defined(WITH_PROPERTY_NAME) && defined(WITH_EDGE_PROPERTY)
const char* get_edge_property_name(Graph g, EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    auto s = _g->schema().GetEdgePropertyName(_ep->first, _ep->second);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

EdgeProperty get_edge_property_by_name(Graph g, EdgeType etype,
                                           const char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto _etype = static_cast<EdgeType_T*>(etype);
    auto s = std::string(name);
    auto ep = new EdgeProperty_T(*_etype, _g->schema().GetEdgePropertyId(*_etype, s));
    return ep;
}

EdgePropertyList get_edge_properties_by_name(Graph g, const char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto s = std::string(name);
    auto epl = new EdgePropertyList_T();
    for (auto etype = 0; etype < _g->edge_label_num(); ++etype) {
        epl->push_back(EdgeProperty_T(etype, _g->schema().GetVertexPropertyId(etype, s)));
    }
    return epl;
}
#endif


#ifdef WITH_VERTEX_PROPERTY
bool equal_vertex_property(VertexProperty vp1, VertexProperty vp2) {
    auto _vp1 = static_cast<VertexProperty_T*>(vp1);
    auto _vp2 = static_cast<VertexProperty_T*>(vp2);
    return (*_vp1 == *_vp2);
}

void destroy_vertex_property(VertexProperty vp) {
    auto _vp = static_cast<VertexProperty_T*>(vp);
    delete _vp;
}

DataType get_vertex_property_data_type(Graph g, VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    auto dt = _g->schema().GetVertexPropertyType(_vp->first, _vp->second);
    return ArrowToDataType(dt);
}

VertexType get_vertex_property_vertex_type(VertexProperty vp) {
    auto _vp = static_cast<VertexProperty_T*>(vp);
    auto vt = new VertexType_T(_vp->first);
    return vt;
}
#endif


#ifdef WITH_EDGE_PROPERTY
bool equal_edge_property(EdgeProperty ep1, EdgeProperty ep2) {
    auto _ep1 = static_cast<EdgeProperty_T*>(ep1);
    auto _ep2 = static_cast<EdgeProperty_T*>(ep2);
    return (*_ep1 == *_ep2);
}

void destroy_edge_property(EdgeProperty ep) {
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    delete _ep;
}

DataType get_edge_property_data_type(Graph g, EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    auto dt = _g->schema().GetEdgePropertyType(_ep->first, _ep->second);
    return ArrowToDataType(dt);
}

EdgeType get_edge_property_edge_type(EdgeProperty ep) {
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    auto et = new EdgeType_T(_ep->first);
    return et;
}
#endif
