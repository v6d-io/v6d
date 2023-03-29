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

#ifdef GRIN_WITH_VERTEX_PROPERTY
GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_property_list_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    for (auto p = 0; p < _g->vertex_property_num(*_vtype); ++p) {
        vpl->push_back(GRIN_VERTEX_PROPERTY_T(*_vtype, p));
    }
    return vpl;
}

size_t grin_get_vertex_property_list_size(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    return _vpl->size();
}

GRIN_VERTEX_PROPERTY grin_get_vertex_property_from_list(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl, size_t idx) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    auto vp = new GRIN_VERTEX_PROPERTY_T((*_vpl)[idx]);
    return vp;
}

GRIN_VERTEX_PROPERTY_LIST grin_create_vertex_property_list(GRIN_GRAPH g) {
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    return vpl;
}

void grin_destroy_vertex_property_list(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    delete _vpl;
}

bool grin_insert_vertex_property_to_list(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl, GRIN_VERTEX_PROPERTY vp) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    auto _vp = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp);
    _vpl->push_back(*_vp);
    return true;
}
#endif


#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY
GRIN_VERTEX_PROPERTY grin_get_vertex_property_from_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY_ID vpi) {
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto vp = new GRIN_VERTEX_PROPERTY_T(*_vtype, vpi);
    return vp;
}

GRIN_VERTEX_PROPERTY_ID grin_get_vertex_property_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY vp) {
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto _vp = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp);
    if (*_vtype != _vp->first) return GRIN_NULL_NATURAL_ID;
    return _vp->second;
}
#endif


#ifdef GRIN_WITH_EDGE_PROPERTY
GRIN_EDGE_PROPERTY_LIST grin_get_edge_property_list_by_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    for (auto p = 0; p < _g->edge_property_num(*_etype); ++p) {
        epl->push_back(GRIN_EDGE_PROPERTY_T(*_etype, p));
    }
    return epl;
}

size_t grin_get_edge_property_list_size(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    return _epl->size();
}

GRIN_EDGE_PROPERTY grin_get_edge_property_from_list(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl, size_t idx) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    auto ep = new GRIN_EDGE_PROPERTY_T((*_epl)[idx]);
    return ep;
}

GRIN_EDGE_PROPERTY_LIST grin_create_edge_property_list(GRIN_GRAPH g) {
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    return epl;
}

void grin_destroy_edge_property_list(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    delete _epl;
}

bool grin_insert_edge_property_to_list(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl, GRIN_EDGE_PROPERTY ep) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    auto _ep = static_cast<GRIN_EDGE_PROPERTY_T*>(ep);
    _epl->push_back(*_ep);
    return true;
}
#endif


#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
GRIN_EDGE_PROPERTY grin_get_edge_property_from_id(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY_ID epi) {
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto ep = new GRIN_EDGE_PROPERTY_T(*_etype, epi);
    return ep;
}

GRIN_EDGE_PROPERTY_ID grin_get_edge_property_id(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY ep) {
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto _ep = static_cast<GRIN_EDGE_PROPERTY_T*>(ep);
    if (*_etype != _ep->first) return GRIN_NULL_NATURAL_ID;
    return _ep->second;
}
#endif


// #if defined(GRIN_WITH_VERTEX_PROPERTY) && defined(GRIN_ASSUME_COLUMN_STORE_FOR_VERTEX_PROPERTY)
// GRIN_GRAPH grin_select_vertex_properties(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl) {
//     auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
//     auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
//     std::map<int, std::vector<int>> vertices, edges;
//     for (auto& p: *_vpl) {
//         int vtype = static_cast<int>(p.first);
//         int vp = static_cast<int>(p.second);
//         if (vertices.find(vtype) == vertices.end()) {
//             vertices[vtype].clear();
//         }
//         vertices[vtype].push_back(vp);
//     }
//     vineyard::Client client;
//     client.Connect();
//     auto object_id = _g->Project(client, vertices, edges);
//     return get_graph_by_object_id(client, object_id.value());
// }
// #endif

// #if defined(GRIN_WITH_EDGE_PROPERTY) && defined(GRIN_ASSUME_COLUMN_STORE_FOR_EDGE_PROPERTY)
// GRIN_GRAPH grin_select_edge_properteis(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl) {
//     auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
//     auto _epl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(epl);
//     std::map<int, std::vector<int>> vertices, edges;
//     for (auto& p: *_epl) {
//         int etype = static_cast<int>(p.first);
//         int ep = static_cast<int>(p.second);
//         if (edges.find(etype) == edges.end()) {
//             edges[etype].clear();
//         }
//         edges[etype].push_back(ep);
//     }
//     vineyard::Client client;
//     client.Connect();
//     auto object_id = _g->Project(client, vertices, edges);
//     return get_graph_by_object_id(client, object_id.value());
// }
// #endif