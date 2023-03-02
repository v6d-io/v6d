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
#include "graph/grin/include/property/propertylist.h"

#ifdef WITH_VERTEX_PROPERTY
VertexPropertyList get_vertex_property_list_by_type(Graph g, VertexType vtype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto vpl = new VertexPropertyList_T();
    for (auto p = 0; p < _g->vertex_property_num(*_vtype); ++p) {
        vpl->push_back(VertexProperty_T(*_vtype, p));
    }
    return vpl;
}

size_t get_vertex_property_list_size(VertexPropertyList vpl) {
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    return _vpl->size();
}

VertexProperty get_vertex_property_from_list(VertexPropertyList vpl, size_t idx) {
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    auto vp = new VertexProperty_T((*_vpl)[idx]);
    return vp;
}

VertexPropertyList create_vertex_property_list() {
    auto vpl = new VertexPropertyList_T();
    return vpl;
}

void destroy_vertex_property_list(VertexPropertyList vpl) {
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    delete _vpl;
}

bool insert_vertex_property_to_list(VertexPropertyList vpl, VertexProperty vp) {
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    _vpl->push_back(*_vp);
    return true;
}
#endif


#ifdef NATURAL_VERTEX_PROPERTY_ID_TRAIT
VertexProperty get_vertex_property_from_id(VertexType vtype, VertexPropertyID vpi) {
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto vp = new VertexProperty_T(*_vtype, vpi);
    return vp;
}

VertexPropertyID get_vertex_property_id(VertexType vtype, VertexProperty vp) {
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    if (*_vtype != _vp->first) return NULL_NATURAL_ID;
    return _vp->second;
}
#endif


#ifdef WITH_EDGE_PROPERTY
EdgePropertyList get_edge_property_list_by_type(Graph g, EdgeType etype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _etype = static_cast<EdgeType_T*>(etype);
    auto epl = new EdgePropertyList_T();
    for (auto p = 0; p < _g->edge_property_num(*_etype); ++p) {
        epl->push_back(EdgeProperty_T(*_etype, p));
    }
    return epl;
}

size_t get_edge_property_list_size(EdgePropertyList epl) {
    auto _epl = static_cast<EdgePropertyList_T*>(epl);
    return _epl->size();
}

EdgeProperty get_edge_property_from_list(EdgePropertyList epl, size_t idx) {
    auto _epl = static_cast<EdgePropertyList_T*>(epl);
    auto ep = new EdgeProperty_T((*_epl)[idx]);
    return ep;
}

EdgePropertyList create_edge_property_list() {
    auto epl = new EdgePropertyList_T();
    return epl;
}

void destroy_edge_property_list(EdgePropertyList epl) {
    auto _epl = static_cast<EdgePropertyList_T*>(epl);
    delete _epl;
}

bool insert_edge_property_to_list(EdgePropertyList epl, EdgeProperty ep) {
    auto _epl = static_cast<EdgePropertyList_T*>(epl);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    _epl->push_back(*_ep);
    return true;
}
#endif


#ifdef NATURAL_EDGE_PROPERTY_ID_TRAIT
EdgeProperty get_edge_property_from_id(EdgeType etype, EdgePropertyID epi) {
    auto _etype = static_cast<EdgeType_T*>(etype);
    auto ep = new EdgeProperty_T(*_etype, epi);
    return ep;
}

EdgePropertyID get_edge_property_id(EdgeType etype, EdgeProperty ep) {
    auto _etype = static_cast<EdgeType_T*>(etype);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    if (*_etype != _ep->first) return NULL_NATURAL_ID;
    return _ep->second;
}
#endif


#if defined(WITH_VERTEX_PROPERTY) && defined(COLUMN_STORE_TRAIT)
Graph select_vertex_properties(Graph g, VertexPropertyList vpl) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    std::map<int, std::vector<int>> vertices, edges;
    for (auto& p: *_vpl) {
        int vtype = static_cast<int>(p.first);
        int vp = static_cast<int>(p.second);
        if (vertices.find(vtype) == vertices.end()) {
            vertices[vtype].clear();
        }
        vertices[vtype].push_back(vp);
    }
    vineyard::Client client;
    client.Connect();
    auto object_id = _g->Project(client, vertices, edges);
    return get_graph_by_object_id(client, object_id.value());
}
#endif

#if defined(WITH_EDGE_PROPERTY) && defined(COLUMN_STORE_TRAIT)
Graph select_edge_properteis(Graph g, EdgePropertyList epl) {
    auto _g = static_cast<Graph_T*>(g);
    auto _epl = static_cast<VertexPropertyList_T*>(epl);
    std::map<int, std::vector<int>> vertices, edges;
    for (auto& p: *_epl) {
        int etype = static_cast<int>(p.first);
        int ep = static_cast<int>(p.second);
        if (edges.find(etype) == edges.end()) {
            edges[etype].clear();
        }
        edges[etype].push_back(ep);
    }
    vineyard::Client client;
    client.Connect();
    auto object_id = _g->Project(client, vertices, edges);
    return get_graph_by_object_id(client, object_id.value());
}
#endif