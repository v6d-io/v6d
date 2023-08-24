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
#include "property/topology.h"

#ifdef GRIN_ENABLE_SCHEMA
size_t grin_get_vertex_num_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    return _g->GetVerticesNum(vtype);
}

size_t grin_get_edge_num_by_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    return _g->edge_data_table(etype)->num_rows();
}
#endif

#if defined(GRIN_ENABLE_VERTEX_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_VERTEX_LIST grin_get_vertex_list_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vl = new GRIN_VERTEX_LIST_T(_g->Vertices(vtype));
    return vl;
}
#endif

#if defined(GRIN_ENABLE_EDGE_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_EDGE_LIST grin_get_edge_list_by_type(GRIN_GRAPH, GRIN_EDGE_TYPE);
#endif

#if defined(GRIN_ENABLE_ADJACENT_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_ADJACENT_LIST grin_get_adjacent_list_by_edge_type(GRIN_GRAPH g, GRIN_DIRECTION d, GRIN_VERTEX v, GRIN_EDGE_TYPE etype) {
    assert(d != GRIN_DIRECTION::BOTH);
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    GRIN_ADJACENT_LIST al;
    al.etype = etype;
    al.vid = v;
    _GRIN_GRAPH_T::raw_adj_list_t ral;
    if (d == GRIN_DIRECTION::OUT) {
        al.dir = GRIN_DIRECTION::OUT;
        ral = _g->GetOutgoingRawAdjList(_GRIN_GRAPH_T::vertex_t(v), etype);
    } else if (d == GRIN_DIRECTION::IN) {
        al.dir = GRIN_DIRECTION::IN;
        ral = _g->GetIncomingRawAdjList(_GRIN_GRAPH_T::vertex_t(v), etype);
    } else {
        return GRIN_NULL_ADJACENT_LIST;
    }
    al.begin = ral.begin();
    al.end = ral.end();
    return al;
}
#endif
