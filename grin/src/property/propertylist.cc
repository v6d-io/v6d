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
#include "grin/src/utils.h"
#include "grin/include/property/propertylist.h"

// Vertex property list
#ifdef WITH_VERTEX_PROPERTY
VertexPropertyList get_all_vertex_properties(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto vpl = new VertexPropertyList_T();
    for (auto vlabel = 0; vlabel < _g->vertex_label_num(); ++vlabel) {
        for (auto p = 0; p < _g->vertex_property_num(vlabel); ++p) {
            vpl->push_back(VertexProperty_T(vlabel, p));
        }
    }
    return vpl;
}

size_t get_vertex_property_list_size(const VertexPropertyList vpl) {
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    return _vpl->size();
}

VertexProperty get_vertex_property_from_list(const VertexPropertyList vpl, const size_t idx) {
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

bool insert_vertex_property_to_list(VertexPropertyList vpl, const VertexProperty vp) {
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    _vpl->push_back(*_vp);
    return true;
}

#ifdef WITH_VERTEX_LABEL
VertexPropertyList get_all_vertex_properties_by_label(const Graph g, const VertexLabel vlabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto vpl = new VertexPropertyList_T();
    for (auto p = 0; p < _g->vertex_property_num(*_vlabel); ++p) {
        vpl->push_back(VertexProperty_T(*_vlabel, p));
    }
    return vpl;
}

#ifdef NATURAL_VERTEX_PROPERTY_ID_TRAIT
VertexProperty get_vertex_property_from_id(const VertexLabel vlabel, const VertexPropertyID vpi) {
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto vp = new VertexProperty_T(*_vlabel, vpi);
    return vp;
}

VertexPropertyID get_vertex_property_id(const VertexLabel vlabel, const VertexProperty vp) {
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    if (*_vlabel != _vp->first) return NULL_NATURAL_ID;
    return _vp->second;
}
#endif
#endif
#endif

// Edge property list
#ifdef WITH_EDGE_PROPERTY
EdgePropertyList get_all_edge_properties(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto epl = new EdgePropertyList_T();
    for (auto elabel = 0; elabel < _g->edge_label_num(); ++elabel) {
        for (auto p = 0; p < _g->edge_property_num(elabel); ++p) {
            epl->push_back(EdgeProperty_T(elabel, p));
        }
    }
    return epl;
}

size_t get_edge_property_list_size(const EdgePropertyList epl) {
    auto _epl = static_cast<EdgePropertyList_T*>(epl);
    return _epl->size();
}

EdgeProperty get_edge_property_from_list(const EdgePropertyList epl, const size_t idx) {
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

bool insert_edge_property_to_list(EdgePropertyList epl, const EdgeProperty ep) {
    auto _epl = static_cast<EdgePropertyList_T*>(epl);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    _epl->push_back(*_ep);
    return true;
}

#ifdef WITH_EDGE_LABEL
EdgePropertyList get_all_edge_properties_by_label(const Graph g, const EdgeLabel elabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto epl = new EdgePropertyList_T();
    for (auto p = 0; p < _g->edge_property_num(*_elabel); ++p) {
        epl->push_back(EdgeProperty_T(*_elabel, p));
    }
    return epl;
}

#ifdef NATURAL_EDGE_PROPERTY_ID_TRAIT
EdgeProperty get_edge_property_from_id(const EdgeLabel elabel, const EdgePropertyID epi) {
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto ep = new EdgeProperty_T(*_elabel, epi);
    return ep;
}

EdgePropertyID get_edge_property_id(const EdgeLabel elabel, const EdgeProperty ep) {
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    if (*_elabel != _ep->first) return NULL_NATURAL_ID;
    return _ep->second;
}
#endif
#endif
#endif


// graph projection
#if defined(WITH_VERTEX_PROPERTY) && defined(COLUMN_STORE)
Graph select_vertex_properties(const Graph g, const VertexPropertyList vpl) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    std::map<unsigned, std::vector<unsigned>> vertices, edges;
    for (auto& p: *_vpl) {
        if (vertices.find(p.first) == vertices.end()) {
            vertices.insert(p.first, {p.second});
        } else {
            vertices[p.first].push_back(p.second);
        }
    }
    vineyard::Client client;
    client.Connect();
    auto object_id = _g->Project(client, vertices, edges);
    return get_graph_by_object_id(client, object_id.value());
}
#endif

#if defined(WITH_EDGE_PROPERTY) && defined(COLUMN_STORE)
Graph select_edge_properteis(const Graph g, const EdgePropertyList epl) {
    auto _g = static_cast<Graph_T*>(g);
    auto _epl = static_cast<VertexPropertyList_T*>(epl);
    std::map<unsigned, std::vector<unsigned>> vertices, edges;
    for (auto& p: *_epl) {
        if (edges.find(p.first) == edges.end()) {
            edges.insert(p.first, {p.second});
        } else {
            edges[p.first].push_back(p.second);
        }
    }
    vineyard::Client client;
    client.Connect();
    auto object_id = _g->Project(client, vertices, edges);
    return get_graph_by_object_id(client, object_id.value());
}
#endif