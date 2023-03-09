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
#include "graph/grin/include/property/topology.h"


#ifdef GRIN_WITH_VERTEX_PROPERTY
size_t grin_get_vertex_num_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    return _g->GetVerticesNum(*_vtype);
}
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
size_t grin_get_edge_num_by_type(GRIN_GRAPH g, GRIN_DIRECTION d, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    return _g->edge_data_table(*_etype)->num_rows();
}
#endif

#ifdef GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID
GRIN_VERTEX grin_get_vertex_from_original_id_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_ORIGINAL_ID oid) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto _oid = static_cast<VERTEX_ORIGINAL_ID_T*>(oid);
    GRIN_GRAPH_T::vid_t gid;
    auto v = new GRIN_VERTEX_T();
    if (_g->Oid2Gid(*_vtype, *_oid, gid)) {
        if (_g->Gid2Vertex(gid, *v)) {
            return v;
        }
    }
    return GRIN_NULL_VERTEX;
}
#endif

#ifdef GRIN_TRAIT_FILTER_TYPE_FOR_VERTEX_LIST
GRIN_VERTEX_LIST grin_filter_type_for_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_LIST vl) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    _vl->type_begin = *_vtype;
    _vl->type_end = *_vtype + 1;
    __grin_init_vertex_list(_g, _vl);
    return _vl;
}
#endif

#ifdef GRIN_TRAIT_FILTER_TYPE_FOR_EDGE_LIST
GRIN_EDGE_LIST grin_filter_type_for_edge_list(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_EDGE_LIST);
#endif

#ifdef GRIN_TRAIT_FILTER_NEIGHBOR_TYPE_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_filter_neighbor_type_for_adjacent_list(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_ADJACENT_LIST);
#endif

#ifdef GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_filter_edge_type_for_adjacent_list(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_ADJACENT_LIST al) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    _al->etype_begin = *_etype;
    _al->etype_end = *_etype + 1;
    __grin_init_adjacent_list(_g, _al);
    return _al;
}
#endif