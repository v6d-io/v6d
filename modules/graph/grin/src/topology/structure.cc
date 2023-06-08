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
#include "topology/structure.h"
#include "client/client.h"

GRIN_GRAPH grin_get_graph_from_storage(const char* id, const char* version) {
    auto g = new GRIN_GRAPH_T();
    auto ipc_socket = getenv("VINEYARD_IPC_SOCKET");
    g->client.Connect(ipc_socket);

    vineyard::ObjectID obj_id;
    std::stringstream ss(id);
    ss >> obj_id;
 
    g->_g = std::dynamic_pointer_cast<_GRIN_GRAPH_T>(g->client.GetObject(obj_id));
    g->g = g->_g.get();
    _prepare_cache(g);
    return g;
}

void grin_destroy_graph(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    delete _g->cache;
    delete _g;
}

#if defined(GRIN_ASSUME_HAS_DIRECTED_GRAPH) && defined(GRIN_ASSUME_HAS_UNDIRECTED_GRAPH)
bool grin_is_directed(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    return _g->directed();
}
#endif

#ifdef GRIN_ASSUME_HAS_MULTI_EDGE_GRAPH
bool grin_is_multigraph(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    return _g->is_multigraph();
}
#endif

#if !defined(GRIN_WITH_VERTEX_PROPERTY)
size_t grin_get_vertex_num(GRIN_GRAPH);
#endif

#if !defined(GRIN_WITH_EDGE_PROPERTY)
size_t grin_get_edge_num(GRIN_GRAPH);
#endif

// Vertex
void grin_destroy_vertex(GRIN_GRAPH g, GRIN_VERTEX v) {}

bool grin_equal_vertex(GRIN_GRAPH g, GRIN_VERTEX v1, GRIN_VERTEX v2) {
    return v1 == v2;
}

// Data
#ifdef GRIN_WITH_VERTEX_DATA
GRIN_DATATYPE grin_get_vertex_data_datatype(GRIN_GRAPH, GRIN_VERTEX);

const void* grin_get_vertex_data_value(GRIN_GRAPH, GRIN_VERTEX);
#endif

// Edge
void grin_destroy_edge(GRIN_GRAPH g, GRIN_EDGE e) {
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    delete _e;
}

GRIN_VERTEX grin_get_src_vertex_from_edge(GRIN_GRAPH g, GRIN_EDGE e) {
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    return _e->src;
}

GRIN_VERTEX grin_get_dst_vertex_from_edge(GRIN_GRAPH g, GRIN_EDGE e) {
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    return _e->dst;
}

#ifdef GRIN_WITH_EDGE_DATA
GRIN_DATATYPE grin_get_edge_data_datatype(GRIN_GRAPH, GRIN_EDGE);

const void* grin_get_edge_data_value(GRIN_GRAPH, GRIN_EDGE);
#endif