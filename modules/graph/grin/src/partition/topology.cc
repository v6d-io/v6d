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
