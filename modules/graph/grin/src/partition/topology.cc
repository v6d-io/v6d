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
#include "graph/grin/include/partition/topology.h"

#ifdef GRIN_ENABLE_GRAPH_PARTITION
size_t grin_get_total_vertex_num(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    if (_pg->lgs.size() == 0) return 0;
    return _pg->lgs[0]->GetTotalVerticesNum();
}

size_t grin_get_total_edge_num(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    return _pg->pg->total_edge_num();
}
#endif

#ifdef GRIN_TRAIT_SELECT_MASTER_FOR_VERTEX_LIST
GRIN_VERTEX_LIST grin_select_master_for_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    if (_vl->all_master_mirror > 0) return GRIN_NULL_LIST;

    auto fvl = new GRIN_VERTEX_LIST_T();
    fvl->type_begin = _vl->type_begin;
    fvl->type_end = _vl->type_end;
    fvl->all_master_mirror = 1;
    __grin_init_vertex_list(_g, fvl);
    return fvl;
}

GRIN_VERTEX_LIST grin_select_mirror_for_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    if (_vl->all_master_mirror > 0) return GRIN_NULL_LIST;

    auto fvl = new GRIN_VERTEX_LIST_T();
    fvl->type_begin = _vl->type_begin;
    fvl->type_end = _vl->type_end;
    fvl->all_master_mirror = 2;
    __grin_init_vertex_list(_g, fvl);
    return fvl;
}
#endif


#ifdef GRIN_TRAIT_SELECT_PARTITION_FOR_VERTEX_LIST
GRIN_VERTEX_LIST grin_select_partition_for_vertex_list(GRIN_GRAPH, GRIN_PARTITION, GRIN_VERTEX_LIST);
#endif



#ifdef GRIN_TRAIT_SELECT_MASTER_FOR_EDGE_LIST
GRIN_EDGE_LIST grin_select_master_for_edge_list(GRIN_GRAPH, GRIN_EDGE_LIST);

GRIN_EDGE_LIST grin_select_mirror_for_edge_list(GRIN_GRAPH, GRIN_EDGE_LIST);
#endif


#ifdef GRIN_TRAIT_SELECT_PARTITION_FOR_EDGE_LIST
GRIN_EDGE_LIST grin_select_partition_for_edge_list(GRIN_GRAPH, GRIN_PARTITION, GRIN_EDGE_LIST);
#endif

#ifdef GRIN_TRAIT_SELECT_MASTER_NEIGHBOR_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_select_master_neighbor_for_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST);

GRIN_ADJACENT_LIST grin_select_mirror_neighbor_for_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST);
#endif

#ifdef GRIN_TRAIT_SELECT_NEIGHBOR_PARTITION_FOR_ADJACENT_LIST
GRIN_ADJACENT_LIST grin_select_neighbor_partition_for_adjacent_list(GRIN_GRAPH, GRIN_ADJACENT_LIST);
#endif
