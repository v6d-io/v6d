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

#include "modules/graph/grin/src/predefine.h"
#include "modules/graph/grin/include/topology/structure.h"

bool is_directed(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    return _g->directed();
}

bool is_multigraph(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    return _g->is_multigraph();
}

size_t get_vertex_num(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    size_t result = 0;
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        result += _g->GetVerticesNum(vtype);
    }
    return result;
}

#ifdef WITH_VERTEX_PROPERTY
size_t get_vertex_num_by_type(const Graph g, const VertexType vtype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vtype = static_cast<VertexType_T*>(vtype);
    return _g->GetVerticesNum(*_vtype);
}
#endif

size_t get_edge_num(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    return _g->GetEdgeNum();
}

#ifdef WITH_EDGE_PROPERTY
size_t get_edge_num_by_type(const Graph g, const EdgeType etype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _etype = static_cast<EdgeType_T*>(etype);
    return _g->edge_data_table(*_etype)->num_rows();
}
#endif

// Vertex
void destroy_vertex(Vertex v) {
    auto _v = static_cast<Vertex_T*>(v);
    delete _v;
}

#ifdef WITH_VERTEX_ORIGINAL_ID
Vertex get_vertex_from_original_id(const Graph g, const OriginalID oid) {
    auto _g = static_cast<Graph_T*>(g);
    Vertex result;
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        result = get_vertex_from_original_id_by_type(g, &vtype, oid);
        if (result != NULL_VERTEX) {
            return result;
        }
    }
    return NULL_VERTEX;
}

OriginalID get_vertex_original_id(const Graph g, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto gid = _g->Vertex2Gid(*_v);
    auto oid = new OriginalID_T(_g->Gid2Oid(gid));
    return oid;
}

void destroy_vertex_original_id(OriginalID oid) {
    auto _oid = static_cast<OriginalID_T*>(oid);
    delete _oid;
} 
#endif

#if defined(WITH_VERTEX_ORIGINAL_ID) && defined(WITH_VERTEX_PROPERTY)
Vertex get_vertex_from_original_id_by_type(const Graph g, const VertexType vtype, const OriginalID oid) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto _oid = static_cast<OriginalID_T*>(oid);
    Graph_T::vid_t gid;
    auto v = new Vertex_T();
    if (_g->Oid2Gid(*_vtype, *_oid, gid)) {
        if (_g->Gid2Vertex(gid, *v)) {
            return v;
        }
    }
    return NULL_VERTEX;
}
#endif

// Edge
void destroy_edge(Edge e) {
    auto _e = static_cast<Edge_T*>(e);
    delete _e;
}

Vertex get_edge_src(const Graph g, const Edge e) {
    auto _e = static_cast<Edge_T*>(e);
    return _e->src;
}

Vertex get_edge_dst(const Graph g, const Edge e) {
    auto _e = static_cast<Edge_T*>(e);
    return _e->dst;
}
