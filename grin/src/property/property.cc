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

#ifdef WITH_VERTEX_PROPERTY
void destroy_vertex_property(VertexProperty vp) {
    auto _vp = static_cast<VertexProperty_T*>(vp);
    delete _vp;
}

DataType get_vertex_property_type(const Graph g, const VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    auto dt = _g->schema().GetVertexPropertyType(_vp->first, _vp->second);
    return ArrowToDataType(dt);
}

#ifdef WITH_VERTEX_PROPERTY_NAME
char* get_vertex_property_name(const Graph g, const VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    auto s = _g->schema().GetVertexPropertyName(_vp->first, _vp->second);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

VertexProperty get_vertex_property_by_name(const Graph g, const VertexLabel vlabel,
                                           const char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto s = std::string(name);
    auto vp = new VertexProperty_T(*_vlabel, _g->schema().GetVertexPropertyId(*_vlabel, s));
    return vp;
}

#endif

// Vertex Property Table
void destroy_vertex_property_table(VertexPropertyTable vpt) {
    auto _vpt = static_cast<VertexPropertyTable_T*>(vpt);
    delete _vpt;
}

const void* get_value_from_vertex_property_table(const Graph g, const VertexPropertyTable vpt,
                                           const Vertex v, const VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vpt = static_cast<VertexPropertyTable_T*>(vpt);
    auto _v = static_cast<Vertex_T*>(v);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    if (_vp->first != _vpt->first || !_vpt->second.Contain(*_v)) return NULL;
    auto offset = _v->GetValue() - _vpt->second.begin_value();
    auto array = _g->vertex_data_table(_vp->first)->column(_vp->second)->chunk(0);
    auto result = vineyard::get_arrow_array_data_element(array, offset);
    return result;
}

VertexPropertyTable get_vertex_property_table_by_label(const Graph g, const VertexLabel vlabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto vpt = new VertexPropertyTable_T(*_vlabel, _g->InnerVertices(*_vlabel));
    return vpt;
}

#ifdef COLUMN_STORE
VertexPropertyTable get_vertex_property_table_for_property(const Graph g, const VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    auto vlabel = _vp->first;
    auto vpt = new VertexPropertyTable_T(vlabel, _g->InnerVertices(vlabel));
    return vpt;
}
#else
VertexPropertyTable get_vertex_property_table_for_vertex(const Graph, const Vertex);
#endif

#endif


#ifdef WITH_EDGE_PROPERTY
void destroy_edge_property(EdgeProperty ep) {
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    delete _ep;
}

DataType get_edge_property_type(const Graph g, const EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    auto dt = _g->schema().GetEdgePropertyType(_ep->first, _ep->second);
    return ArrowToDataType(dt);
}

#ifdef WITH_EDGE_PROPERTY_NAME
char* get_edge_property_name(const Graph g, const EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    auto s = _g->schema().GetEdgePropertyName(_ep->first, _ep->second);
    int len = s.length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", s.c_str());
    return out;
}

EdgeProperty get_edge_property_by_name(const Graph g, const EdgeLabel elabel,
                                           const char* name) {
    auto _g = static_cast<Graph_T*>(g);
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto s = std::string(name);
    auto ep = new EdgeProperty_T(*_elabel, _g->schema().GetEdgePropertyId(*_elabel, s));
    return ep;
}

#endif

// Edge Property Table
void destroy_edge_property_table(EdgePropertyTable ept) {
    auto _ept = static_cast<EdgePropertyTable_T*>(ept);
    delete _ept;
}

const void* get_value_from_edge_property_table(const Graph g, const EdgePropertyTable ept,
                                           const Edge e, const EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ept = static_cast<EdgePropertyTable_T*>(ept);
    auto _e = static_cast<Edge_T*>(e);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    if (_ep->first != _ept->first || _e->eid >= _ept->second) return NULL;
    auto offset = _e->eid;
    auto array = _g->edge_data_table(_ep->first)->column(_ep->second)->chunk(0);
    auto result = vineyard::get_arrow_array_data_element(array, offset);
    return result;
}

EdgePropertyTable get_edge_property_table_by_label(const Graph g, const EdgeLabel elabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto ept = new EdgePropertyTable_T(*_elabel, _g->edge_data_table(*_elabel)->num_rows());
    return ept;
}

#ifdef COLUMN_STORE
EdgePropertyTable get_edge_property_table_for_property(const Graph g, const EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    auto elabel = _ep->first;
    auto ept = new EdgePropertyTable_T(elabel, _g->edge_data_table(elabel)->num_rows());
    return ept;
}
#else
EdgePropertyTable get_edge_property_table_for_edge(const Graph, const Edge);
#endif

#endif