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
#include "index/internal_id.h"

long long int grin_get_vertex_internal_id_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt, GRIN_VERTEX v) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->id_parser.GetOffset(v);
}

GRIN_VERTEX grin_get_vertex_by_internal_id_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt, long long int id) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->id_parser.GenerateId(vt, id);
}

long long int grin_get_vertex_internal_id_upper_bound_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    return _g->GetVerticesNum(vt);
}

long long int grin_get_vertex_internal_id_lower_bound_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt) {
    return 0;
}
