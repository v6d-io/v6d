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
#include "graph/grin/include/index/order.h"

#ifdef GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
bool grin_smaller_vertex(GRIN_GRAPH g, GRIN_VERTEX v1, GRIN_VERTEX v2) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _v1 = static_cast<GRIN_VERTEX_T*>(v1);
    auto _v2 = static_cast<GRIN_VERTEX_T*>(v2);
    return _g->Vertex2Gid(*_v1) < _g->Vertex2Gid(*_v2);
}
#endif

#if defined(GRIN_ASSUME_ALL_VERTEX_LIST_SORTED) && defined(GRIN_ENABLE_VERTEX_LIST_ARRAY)
bool grin_get_position_of_vertex_from_sorted_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, GRIN_VERTEX v, size_t& pos) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    auto vtype = _g->vertex_label(*_v);
    if (vtype < _vl->type_begin || vtype >= _vl->type_end) return false;
    pos = 0;
    auto offset = _v->GetValue() - _vl->vrs[vtype - _vl->type_begin].begin_value();
    if (offset < _vl->vrs[vtype - _vl->type_begin].size()) {
        pos = _vl->offsets[vtype - _vl->type_begin] + offset;
        return true;
    }
    return false;
};
#endif
