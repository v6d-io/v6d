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
#include "graph/grin/include/topology/structure.h"

bool grin_is_directed(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    return _g->directed();
}

bool grin_is_multigraph(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    return _g->is_multigraph();
}

size_t grin_get_vertex_num(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    size_t result = 0;
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        result += _g->GetVerticesNum(vtype);
    }
    return result;
}

size_t grin_get_edge_num(GRIN_GRAPH g, GRIN_DIRECTION d) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    if (d == IN) {
        return _g->GetInEdgeNum();
    } else if (d == OUT) {
        return _g->GetOutEdgeNum();
    }
    return _g->GetEdgeNum();
}


// Vertex
void grin_destroy_vertex(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    delete _v;
}

#ifdef GRIN_WITH_VERTEX_ORIGINAL_ID
void grin_destroy_vertex_original_id(GRIN_GRAPH g, GRIN_VERTEX_ORIGINAL_ID oid) {
    auto _oid = static_cast<VERTEX_ORIGINAL_ID_T*>(oid);
    delete _oid;
} 

GRIN_DATATYPE grin_get_vertex_original_id_type(GRIN_GRAPH g) {
    return GRIN_DATATYPE_ENUM<VERTEX_ORIGINAL_ID_T>::value;
}


GRIN_VERTEX_ORIGINAL_ID grin_get_vertex_original_id(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto gid = _g->Vertex2Gid(*_v);
    auto oid = new VERTEX_ORIGINAL_ID_T(_g->Gid2Oid(gid));
    return oid;
}
#endif

#if defined(GRIN_WITH_VERTEX_ORIGINAL_ID) && !defined(GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID)
GRIN_VERTEX grin_get_vertex_from_original_id(GRIN_GRAPH, GRIN_VERTEX_ORIGINAL_ID);
#endif

#ifdef GRIN_WITH_VERTEX_DATA
GRIN_DATATYPE grin_get_vertex_data_type(GRIN_GRAPH, GRIN_VERTEX);

GRIN_VERTEX_DATA grin_get_vertex_data_value(GRIN_GRAPH, GRIN_VERTEX);

void grin_destroy_vertex_data(GRIN_GRAPH, GRIN_VERTEX_DATA);
#endif

// Edge
void grin_destroy_edge(GRIN_GRAPH g, GRIN_EDGE e) {
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    delete _e;
}

GRIN_VERTEX grin_get_edge_src(GRIN_GRAPH g, GRIN_EDGE e) {
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    return _e->src;
}

GRIN_VERTEX grin_get_edge_dst(GRIN_GRAPH g, GRIN_EDGE e) {
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    return _e->dst;
}

#ifdef GRIN_WITH_EDGE_DATA
GRIN_DATATYPE grin_get_edge_data_type(GRIN_GRAPH, GRIN_EDGE);

GRIN_EDGE_DATA grin_get_edge_data_value(GRIN_GRAPH, GRIN_EDGE);

void grin_destroy_edge_data(GRIN_GRAPH, GRIN_EDGE_DATA);
#endif