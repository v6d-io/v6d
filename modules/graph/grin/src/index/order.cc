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

extern "C" {
#include "graph/grin/include/index/order.h"
}

#ifdef GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
bool grin_smaller_vertex(GRIN_GRAPH g, GRIN_VERTEX v1, GRIN_VERTEX v2) {
    return v1 < v2;    
}
#endif

#if defined(GRIN_ASSUME_ALL_VERTEX_LIST_SORTED) && defined(GRIN_ENABLE_VERTEX_LIST_ARRAY)
size_t grin_get_position_of_vertex_from_sorted_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, GRIN_VERTEX v) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    if (v < _vl->end_ && v >= _vl->begin_) return v - _vl->begin_;
    if (_vl->is_simple) return GRIN_NULL_SIZE;
    if (_vl->offsets.empty()) __grin_init_complex_vertex_list(static_cast<GRIN_GRAPH_T*>(g)->g, _vl);
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    auto vtype = _cache->id_parser.GetLabelId(v);
    return v - _vl->offsets[vtype].second + _vl->offsets[vtype].first;
}
#endif
