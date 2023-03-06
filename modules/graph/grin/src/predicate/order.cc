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
#include "graph/grin/include/predicate/order.h"

#ifdef GRIN_PREDICATE_ENABLE_VERTEX_ORDERING
struct less_than_key {
    inline bool operator() (const _GRIN_TYPED_VERTICES_T& tv1, const _GRIN_TYPED_VERTICES_T& tv2) {
        if (tv1.first == tv2.first) {
            return tv1.second.begin_value() < tv2.second.begin_value();
        }
        return tv1.first < tv2.first;
    }
};

bool grin_sort_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    std::sort(_vl->begin(), _vl->end(), less_than_key());
    return true;
};

bool grin_get_position_of_vertex_from_sorted_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, GRIN_VERTEX v, size_t& pos) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    auto vtype = _g->vertex_label(*_v);
    pos = 0;
    for (auto &tv : *_vl) {
        if (tv.first < vtype) pos += tv.second.size();
        else if (tv.first > vtype) return false;
        else {
            if (tv.second.Contain(*_v)) {
                pos += _v->GetValue() - tv.second.begin_value();
                return true;
            } else {
                return false;
            }
        }
    }
    return false;
};
#endif
