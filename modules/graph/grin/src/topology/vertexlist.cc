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
#include "topology/vertexlist.h"

void grin_destroy_vertex_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    delete _vl;
}

size_t grin_get_vertex_list_size(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    return _vl->size();
}

GRIN_VERTEX grin_get_vertex_from_list(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t idx) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    return _vl->begin_value() + idx;
}

GRIN_VERTEX_LIST_ITERATOR grin_get_vertex_list_begin(GRIN_GRAPH g, GRIN_VERTEX_LIST vl) {
    auto _vl = static_cast<GRIN_VERTEX_LIST_T*>(vl);
    auto _vli = new GRIN_VERTEX_LIST_ITERATOR_T();
    _vli->current = _vl->begin_value();
    _vli->end = _vl->end_value();
    return _vli;
}

void grin_destroy_vertex_list_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    delete _vli;
}

void grin_get_next_vertex_list_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    _vli->current++;
}

bool grin_is_vertex_list_end(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    return _vli->current >= _vli->end;
}

GRIN_VERTEX grin_get_vertex_from_iter(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR vli) {
    auto _vli = static_cast<GRIN_VERTEX_LIST_ITERATOR_T*>(vli);
    return _vli->current;
}
