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
#include "partition/topology.h"


#if defined(GRIN_TRAIT_SELECT_MASTER_FOR_VERTEX_LIST) && !defined(GRIN_ENABLE_SCHEMA)
GRIN_VERTEX_LIST grin_get_vertex_list_select_master(GRIN_GRAPH);

GRIN_VERTEX_LIST grin_get_vertex_list_select_mirror(GRIN_GRAPH);
#endif

#if defined(GRIN_TRAIT_SELECT_MASTER_FOR_VERTEX_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_VERTEX_LIST grin_get_vertex_list_by_type_select_master(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vl = new GRIN_VERTEX_LIST_T(_g->InnerVertices(vtype));
    return vl;
}

GRIN_VERTEX_LIST grin_get_vertex_list_by_type_select_mirror(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vl = new GRIN_VERTEX_LIST_T(_g->OuterVertices(vtype));
    return vl;
}
#endif


#if defined(GRIN_TRAIT_SELECT_PARTITION_FOR_VERTEX_LIST) && !defined(GRIN_ENABLE_SCHEMA)
GRIN_VERTEX_LIST grin_get_vertex_list_select_partition(GRIN_GRAPH, GRIN_PARTITION);
#endif

#if defined(GRIN_TRAIT_SELECT_PARTITION_FOR_VERTEX_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_VERTEX_LIST grin_get_vertex_list_by_type_select_partition(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_PARTITION);
#endif


#if defined(GRIN_TRAIT_SELECT_MASTER_FOR_EDGE_LIST) && !defined(GRIN_ENABLE_SCHEMA)
GRIN_EDGE_LIST grin_get_edge_list_select_master(GRIN_GRAPH);

GRIN_EDGE_LIST grin_get_edge_list_select_mirror(GRIN_GRAPH);
#endif

#if defined(GRIN_TRAIT_SELECT_MASTER_FOR_EDGE_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_EDGE_LIST grin_get_edge_list_by_type_select_master(GRIN_GRAPH, GRIN_EDGE_TYPE);

GRIN_EDGE_LIST grin_get_edge_list_by_type_select_mirror(GRIN_GRAPH, GRIN_EDGE_TYPE);
#endif


#if defined(GRIN_TRAIT_SELECT_PARTITION_FOR_EDGE_LIST) && !defined(GRIN_ENABLE_SCHEMA)
GRIN_EDGE_LIST grin_get_edge_list_select_partition(GRIN_GRAPH, GRIN_PARTITION);
#endif

#if defined(GRIN_TRAIT_SELECT_PARTITION_FOR_EDGE_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_EDGE_LIST grin_get_edge_list_by_type_select_partition(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_PARTITION);
#endif


#if defined(GRIN_TRAIT_SELECT_MASTER_NEIGHBOR_FOR_ADJACENT_LIST) && !defined(GRIN_ENABLE_SCHEMA)
GRIN_ADJACENT_LIST grin_get_adjacent_list_select_master_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);

GRIN_ADJACENT_LIST grin_get_adjacent_list_select_mirror_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);
#endif

#if defined(GRIN_TRAIT_SELECT_MASTER_NEIGHBOR_FOR_ADJACENT_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_ADJACENT_LIST grin_get_adjacent_list_by_edge_type_select_master_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX, GRIN_EDGE_TYPE);

GRIN_ADJACENT_LIST grin_get_adjacent_list_by_edge_type_select_mirror_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX, GRIN_EDGE_TYPE);
#endif


#if defined(GRIN_TRAIT_SELECT_NEIGHBOR_PARTITION_FOR_ADJACENT_LIST) && !defined(GRIN_ENABLE_SCHEMA)
GRIN_ADJACENT_LIST grin_get_adjacent_list_select_partition_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX, GRIN_PARTITION);
#endif

#if defined(GRIN_TRAIT_SELECT_NEIGHBOR_PARTITION_FOR_ADJACENT_LIST) && defined(GRIN_ENABLE_SCHEMA)
GRIN_ADJACENT_LIST grin_get_adjacent_list_by_edge_type_select_partition_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX, GRIN_EDGE_TYPE, GRIN_PARTITION);
#endif

// Universal Vertices
#if defined(GRIN_ASSUME_WITH_UNIVERSAL_VERTICES) && defined(GRIN_ENABLE_SCHEMA)
GRIN_VERTEX_TYPE_LIST grin_get_vertex_type_list_select_universal(GRIN_GRAPH);

GRIN_VERTEX_TYPE_LIST grin_get_vertex_type_list_select_non_universal(GRIN_GRAPH);

bool grin_is_vertex_type_unisversal(GRIN_GRAPH, GRIN_VERTEX_TYPE);
#endif

#if defined(GRIN_ASSUME_WITH_UNIVERSAL_VERTICES) && !defined(GRIN_ENABLE_SCHEMA) && defined(GRIN_ENABLE_VERTEX_LIST)
GRIN_VERTEX_LIST grin_get_vertex_list_select_universal(GRIN_GRAPH);

GRIN_VERTEX_LIST grin_get_vertex_list_select_non_universal(GRIN_GRAPH);
#endif

#if defined(GRIN_ASSUME_WITH_UNIVERSAL_VERTICES) && !defined(GRIN_ENABLE_SCHEMA) && defined(GRIN_ENABLE_EDGE_LIST) 
GRIN_EDGE_LIST grin_get_edge_list_select_universal(GRIN_GRAPH);

GRIN_EDGE_LIST grin_get_edge_list_select_non_universal(GRIN_GRAPH);
#endif


#if defined(GRIN_ASSUME_WITH_UNIVERSAL_VERTICES) && !defined(GRIN_ENABLE_SCHEMA) && defined(GRIN_ENABLE_ADJACENT_LIST)
GRIN_ADJACENT_LIST grin_get_adjacent_list_select_universal_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);

GRIN_ADJACENT_LIST grin_get_adjacent_list_select_non_universal_neighbor(GRIN_GRAPH, GRIN_DIRECTION, GRIN_VERTEX);
#endif