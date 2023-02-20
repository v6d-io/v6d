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

#ifdef WITH_VERTEX_PROPERTY
void destroy_vertex_property(VertexProperty) {
    
}
#ifdef WITH_VERTEX_PROPERTY_NAME
char* get_vertex_property_name(const Graph, const VertexProperty);

#ifdef COLUMN_STORE
VertexColumn get_vertex_column_by_name(const Graph, char* name);
#endif
#endif

DataType get_vertex_property_type(const Graph g, const VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    auto dt = _g->schema().GetVertexPropertyType(_vp->first, _vp->second);
    return dt.get();
}

#ifdef COLUMN_STORE
void destroy_vertex_column(VertexColumn vc) {
    auto _vc = static_cast<VertexColumn_T*>(vc);
    delete _vc;
}
#ifdef ENABLE_VERTEX_LIST
VertexColumn get_vertex_column_by_list(const Graph g, const VertexList vl,
                                       const VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vl = static_cast<VertexList_T*>(vl);
    auto _vp = static_cast<VertexProperty_T*>(vp);

    if (_vl->size() > 1) return NULL_LIST;
    auto v = (*_vl)[0].begin();
    auto vlabel = _g->vertex_label(*v);
    
    if (_vp->first != vlabel) return NULL_LIST;
    auto vc = new VertexColumn_T();
    vc->push_back(*_vp);
    return vc;
}
#endif
#ifdef WITH_VERTEX_LABEL
VertexColumn get_vertex_column_by_label(const Graph g, const VertexLabel vl,
                                        const VertexProperty vp) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vl = static_cast<VertexLabel_T*>(vl);
    auto _vp = static_cast<VertexProperty_T*>(vp);

    if (*_vl != _vp->first) return NULL_LIST;
    auto vc = new VertexColumn_T();
    vc->push_back(*_vp);
    return vc;    
}
#endif
void* get_value_from_vertex_column(const Graph g, const VertexColumn vc, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vc = static_cast<VertexColumn_T*>(vc);
    auto _v = static_cast<Vertex_T*>(v);
    auto vp = (*_vc)[0];
    auto dt = _g->schema().GetVertexPropertyType(vp.first, vp.second);
    auto c = _g->vertex_data_column<dt.get()>(vp.first, vp.second);
    return &c[_v];
}
#else
void destroy_vertex_row(VertexRow);
VertexRow get_vertex_row_by_list(const Graph, const Vertex, const VertexPropertyList);
#ifdef WITH_VERTEX_LABEL
VertexRow get_vertex_row_by_label(const Graph, const Vertex, const VertexLabel);
#endif
#endif

#endif

#ifdef WITH_EDGE_PROPERTY
#ifdef WITH_EDGE_PROPERTY_NAME
char* get_edge_property_name(const Graph, const EdgeProperty);

#ifdef COLUMN_STORE
EdgeColumn get_edge_column_by_name(const Graph, char* name);
#endif
#endif

DataType get_edge_property_type(const Graph g, const EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    auto dt = _g->schema().GetEdgePropertyType(_ep->first, _ep->second);
    return dt.get();
}

#ifdef COLUMN_STORE
#ifdef WITH_EDGE_LABEL
EdgeColumn get_edge_column_by_label(const Graph g, const EdgeLabel el,
                                        const EdgeProperty ep) {
    auto _g = static_cast<Graph_T*>(g);
    auto _el = static_cast<EdgeLabel_T*>(el);
    auto _ep = static_cast<EdgeProperty_T*>(ep);

    if (*_el != _ep->first) return NULL_LIST;
    auto ec = new EdgeColumn_T();
    ec->push_back(*_ep);
    return ec;    
}
#endif
void* get_value_from_edge_column(const Graph g, const EdgeColumn ec, const Edge e) {
    auto _g = static_cast<Graph_T*>(g);
    auto _ec = static_cast<EdgeColumn_T*>(ec);
    auto _e = static_cast<Edge_T*>(e);
    auto ep = (*_ec)[0];
    auto dt = _g->schema().GetEdgePropertyType(ep.first, ep.second);
    auto c = _g->edge_data_column<dt.get()>(ep.first, ep.second);
    return &c[_e->eid];
}
#else
EdgeRow get_edge_row_by_list(const Graph, const Edge, const EdgePropertyList);
#ifdef WITH_EDGE_LABEL
EdgeRow get_edge_row_by_label(const Graph, const Edge, const EdgeLabel);
#endif
#endif

#endif

