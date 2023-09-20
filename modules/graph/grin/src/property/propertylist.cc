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
#include "property/propertylist.h"

size_t grin_get_vertex_property_list_size(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    return _vpl->size();
}

GRIN_VERTEX_PROPERTY grin_get_vertex_property_from_list(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl, size_t idx) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    return (*_vpl)[idx];
}

GRIN_VERTEX_PROPERTY_LIST grin_create_vertex_property_list(GRIN_GRAPH g) {
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    return vpl;
}

void grin_destroy_vertex_property_list(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    delete _vpl;
}

bool grin_insert_vertex_property_to_list(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_LIST vpl, GRIN_VERTEX_PROPERTY vp) {
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    _vpl->push_back(vp);
    return true;
}

size_t grin_get_edge_property_list_size(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    return _epl->size();
}

GRIN_EDGE_PROPERTY grin_get_edge_property_from_list(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl, size_t idx) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    return (*_epl)[idx];
}

GRIN_EDGE_PROPERTY_LIST grin_create_edge_property_list(GRIN_GRAPH g) {
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    return epl;
}

void grin_destroy_edge_property_list(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    delete _epl;
}

bool grin_insert_edge_property_to_list(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_LIST epl, GRIN_EDGE_PROPERTY ep) {
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    _epl->push_back(ep);
    return true;
}
