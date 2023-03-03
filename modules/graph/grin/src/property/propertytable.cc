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
#include "graph/grin/include/property/propertytable.h"

#if defined(GRIN_WITH_VERTEX_PROPERTY) || defined(GRIN_WITH_EDGE_PROPERTY)
void grin_destroy_row(GRIN_ROW r) {
    auto _r = static_cast<GRIN_ROW_T*>(r);
    delete _r;
}

const void* grin_get_value_from_row(GRIN_ROW r, size_t idx) {
    auto _r = static_cast<GRIN_ROW_T*>(r);
    return (*_r)[idx];
}

GRIN_ROW grin_create_row() {
    auto r = new GRIN_ROW_T();
    return r;
}

bool grin_insert_value_to_row(GRIN_ROW r, void* value) {
    auto _r = static_cast<GRIN_ROW_T*>(r);
    _r->push_back(value);
    return true;
}
#endif


#ifdef GRIN_WITH_VERTEX_PROPERTY
void grin_destroy_vertex_property_table(GRIN_VERTEX_PROPERTY_TABLE vpt) {
    auto _vpt = static_cast<GRIN_VERTEX_PROPERTY_TABLE_T*>(vpt);
    delete _vpt;
}

GRIN_VERTEX_PROPERTY_TABLE grin_get_vertex_property_table_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto vpt = new GRIN_VERTEX_PROPERTY_TABLE_T();
    vpt->g = _g;
    vpt->vtype = *_vtype;
    vpt->vertices = _g->InnerVertices(*_vtype);
    return vpt;
}

const void* grin_get_value_from_vertex_property_table(GRIN_VERTEX_PROPERTY_TABLE vpt,
                                                 GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    auto _vpt = static_cast<GRIN_VERTEX_PROPERTY_TABLE_T*>(vpt);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto _vp = static_cast<GRIN_VERTEX_PROPERTY_T*>(vp);
    if (_vp->first != _vpt->vtype || !_vpt->vertices.Contain(*_v)) return NULL;
    auto offset = _v->GetValue() - _vpt->vertices.begin_value();
    auto array = _vpt->g->vertex_data_table(_vp->first)->column(_vp->second)->chunk(0);
    auto result = vineyard::get_arrow_array_data_element(array, offset);
    return result;
}

GRIN_ROW grin_get_row_from_vertex_property_table(GRIN_VERTEX_PROPERTY_TABLE vpt, GRIN_VERTEX v, 
                                       GRIN_VERTEX_PROPERTY_LIST vpl) {
    auto _vpt = static_cast<GRIN_VERTEX_PROPERTY_TABLE_T*>(vpt);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto _vpl = static_cast<GRIN_VERTEX_PROPERTY_LIST_T*>(vpl);
    if (!_vpt->vertices.Contain(*_v)) return NULL;
    auto offset = _v->GetValue() - _vpt->vertices.begin_value();

    auto r = new GRIN_ROW_T();
    for (auto vp: *_vpl) {
        if (vp.first != _vpt->vtype) return NULL;
        auto array = _vpt->g->vertex_data_table(vp.first)->column(vp.second)->chunk(0);
        auto result = vineyard::get_arrow_array_data_element(array, offset);
        r->push_back(result);
    }
    return r;
}
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
void grin_destroy_edge_property_table(GRIN_EDGE_PROPERTY_TABLE ept) {
    auto _ept = static_cast<GRIN_EDGE_PROPERTY_TABLE_T*>(ept);
    delete _ept;
}

GRIN_EDGE_PROPERTY_TABLE grin_get_edge_property_table_by_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _etype = static_cast<GRIN_EDGE_TYPE_T*>(etype);
    auto ept = new GRIN_EDGE_PROPERTY_TABLE_T();
    ept->g = _g;
    ept->etype = *_etype;
    ept->num = _g->edge_data_table(*_etype)->num_rows();
    return ept;
}

const void* grin_get_value_from_edge_property_table(GRIN_EDGE_PROPERTY_TABLE ept,
                                               GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    auto _ept = static_cast<GRIN_EDGE_PROPERTY_TABLE_T*>(ept);
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    auto _ep = static_cast<GRIN_EDGE_PROPERTY_T*>(ep);
    if (_ep->first != _ept->etype || _e->eid >= _ept->num) return NULL;
    auto offset = _e->eid;
    auto array = _ept->g->edge_data_table(_ep->first)->column(_ep->second)->chunk(0);
    auto result = vineyard::get_arrow_array_data_element(array, offset);
    return result;
}

GRIN_ROW grin_get_row_from_edge_property_table(GRIN_EDGE_PROPERTY_TABLE ept, GRIN_EDGE v, 
                                     GRIN_EDGE_PROPERTY_LIST epl) {
    auto _ept = static_cast<GRIN_EDGE_PROPERTY_TABLE_T*>(ept);
    auto _e = static_cast<GRIN_EDGE_T*>(v);
    auto _epl = static_cast<GRIN_EDGE_PROPERTY_LIST_T*>(epl);
    if (_e->eid >= _ept->num) return NULL;
    auto offset = _e->eid;

    auto r = new GRIN_ROW_T();
    for (auto ep: *_epl) {
        if (ep.first != _ept->etype) return NULL;
        auto array = _ept->g->edge_data_table(ep.first)->column(ep.second)->chunk(0);
        auto result = vineyard::get_arrow_array_data_element(array, offset);
        r->push_back(result);
    }
    return r;
}
#endif
