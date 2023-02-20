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
#include "grin/include/topology/structure.h"


// Graph 
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
    for (VertexLabel_T vlabel = 0; vlabel < _g->vertex_label_num(); ++vlabel) {
        result += _g->GetVerticesNum(vlabel);
    }
    return result;
}

#ifdef WITH_VERTEX_LABEL
size_t get_vertex_num_by_label(const Graph g, const VertexLabel vlabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    return _g->GetVerticesNum(*_vlabel);
}
#endif

size_t get_edge_num(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    return _g->GetEdgeNum();
}

#ifdef WITH_EDGE_LABEL
size_t get_edge_num_by_label(const Graph g, const EdgeLabel elabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _elabel = static_cast<EdgeLabel_T*>(elabel);
    return _g->edge_data_table(*_elabel)->num_rows();
}
#endif

// Vertex
void destroy_vertex(Vertex v) {
    auto _v = static_cast<Vertex_T*>(v);
    delete _v;
}

#ifdef WITH_VERTEX_ORIGIN_ID
Vertex get_vertex_from_origin_id(const Graph g, const OriginID oid) {
    auto _g = static_cast<Graph_T*>(g);
    Vertex result;
    for (VertexLabel_T vlabel = 0; vlabel < _g->vertex_label_num(); ++vlabel) {
        result = get_vertex_from_label_origin_id(g, &vlabel, oid);
        if (result != NULL_VERTEX) {
            return result;
        }
    }
    return NULL_VERTEX;
}

#ifdef WITH_VERTEX_LABEL
Vertex get_vertex_from_label_origin_id(const Graph g, const VertexLabel vlabel, const OriginID oid) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto _oid = static_cast<OriginID_T*>(oid);
    Graph_T::vid_t gid;
    auto v = new Vertex_T();
    if (_g->Oid2Gid(*_vlabel, *_oid, gid)) {
        if (_g->Gid2Vertex(gid, *v)) {
            return v;
        }
    }
    return NULL_VERTEX;
}
#endif

OriginID get_vertex_origin_id(const Graph g, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto gid = _g->Vertex2Gid(*_v);
    auto oid = new OriginID_T(_g->Gid2Oid(gid));
    return oid;
}

void destroy_vertex_origin_id(OriginID oid) {
    auto _oid = static_cast<OriginID_T*>(oid);
    delete _oid;
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
