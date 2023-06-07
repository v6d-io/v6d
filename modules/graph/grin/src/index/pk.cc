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

#include "index/pk.h"


#if defined(GRIN_ENABLE_VERTEX_PK_INDEX) && defined(GRIN_ENABLE_VERTEX_PRIMARY_KEYS)
GRIN_VERTEX grin_get_vertex_by_primary_keys_row(GRIN_GRAPH, GRIN_VERTEX_TYPE, GRIN_ROW);
#endif

#if defined(GRIN_ENABLE_VERTEX_PK_INDEX) && defined(GRIN_ENABLE_VERTEX_PK_OF_INT64)
GRIN_VERTEX grin_get_vertex_by_pk_of_int64(GRIN_GRAPH g, GRIN_VERTEX_TYPE vt, long long int vid) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    _GRIN_VERTEX_T v;
    if (_g->GetVertex(vt, vid, v)) {
        return v.GetValue();
    } else {
        return GRIN_NULL_VERTEX;
    }
}
#endif

#if defined(GRIN_ENABLE_EDGE_PK_INDEX) && defined(GRIN_ENABLE_EDGE_PRIMARY_KEYS)
GRIN_EDGE grin_get_edge_by_primary_keys_row(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_ROW);
#endif

#if defined(GRIN_ENABLE_EDGE_PK_INDEX) && defined(GRIN_ENABLE_EDGE_PK_OF_INT64)
GRIN_EDGE grin_get_edge_by_pk_of_int64(GRIN_GRAPH, GRIN_EDGE_TYPE, long long int);
#endif
