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

#include "modules/graph/grin/src/predefine.h"
#include "modules/graph/grin/include/property/propertytable.h"

#if defined(WITH_VERTEX_PROPERTY) || defined(WITH_EDGE_PROPERTY)
void destroy_row(Row r) {
    auto _r = static_cast<Row_T*>(r);
    delete _r;
}

const void* get_value_from_row(Row r, const size_t idx) {
    auto _r = static_cast<Row_T*>(r);
    return (*_r)[idx];
}

Row create_row() {
    auto r = new Row_T();
    return r;
}

bool insert_value_to_row(Row r, const void* value) {
    auto _r = static_cast<Row_T*>(r);
    _r->push_back(value);
    return true;
}
#endif


#ifdef WITH_VERTEX_PROPERTY
void destroy_vertex_property_table(VertexPropertyTable vpt) {
    auto _vpt = static_cast<VertexPropertyTable_T*>(vpt);
    delete _vpt;
}

VertexPropertyTable get_vertex_property_table_by_type(const Graph g, const VertexType vtype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto vpt = new VertexPropertyTable_T();
    vpt->g = _g;
    vpt->vtype = *_vtype;
    vpt->vertices = _g->InnerVertices(*_vtype);
    return vpt;
}

const void* get_value_from_vertex_property_table(const VertexPropertyTable vpt,
                                                 const Vertex v, const VertexProperty vp) {
    auto _vpt = static_cast<VertexPropertyTable_T*>(vpt);
    auto _v = static_cast<Vertex_T*>(v);
    auto _vp = static_cast<VertexProperty_T*>(vp);
    if (_vp->first != _vpt->vtype || !_vpt->vertices.Contain(*_v)) return NULL;
    auto offset = _v->GetValue() - _vpt->vertices.begin_value();
    auto array = _vpt->g->vertex_data_table(_vp->first)->column(_vp->second)->chunk(0);
    auto result = vineyard::get_arrow_array_data_element(array, offset);
    return result;
}

Row get_row_from_vertex_property_table(const VertexPropertyTable vpt, const Vertex v, 
                                       const VertexPropertyList vpl) {
    auto _vpt = static_cast<VertexPropertyTable_T*>(vpt);
    auto _v = static_cast<Vertex_T*>(v);
    auto _vpl = static_cast<VertexPropertyList_T*>(vpl);
    if (!_vpt->vertices.Contain(*_v)) return NULL;
    auto offset = _v->GetValue() - _vpt->vertices.begin_value();

    auto r = new Row_T();
    for (auto vp: *_vpl) {
        if (vp.first != _vpt->vtype) return NULL;
        auto array = _vpt->g->vertex_data_table(vp.first)->column(vp.second)->chunk(0);
        auto result = vineyard::get_arrow_array_data_element(array, offset);
        r->push_back(result);
    }
    return r;
}
#endif

#ifdef WITH_EDGE_PROPERTY
void destroy_edge_property_table(EdgePropertyTable ept) {
    auto _ept = static_cast<EdgePropertyTable_T*>(ept);
    delete _ept;
}

EdgePropertyTable get_edge_property_table_by_type(const Graph g, const EdgeType etype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _etype = static_cast<EdgeType_T*>(etype);
    auto ept = new EdgePropertyTable_T();
    ept->g = _g;
    ept->etype = *_etype;
    ept->num = _g->edge_data_table(*_etype)->num_rows();
    return ept;
}

const void* get_value_from_edge_property_table(const EdgePropertyTable ept,
                                               const Edge e, const EdgeProperty ep) {
    auto _ept = static_cast<EdgePropertyTable_T*>(ept);
    auto _e = static_cast<Edge_T*>(e);
    auto _ep = static_cast<EdgeProperty_T*>(ep);
    if (_ep->first != _ept->etype || _e->eid >= _ept->num) return NULL;
    auto offset = _e->eid;
    auto array = _ept->g->edge_data_table(_ep->first)->column(_ep->second)->chunk(0);
    auto result = vineyard::get_arrow_array_data_element(array, offset);
    return result;
}

Row get_row_from_edge_property_table(const EdgePropertyTable ept, const Edge v, 
                                       const EdgePropertyList epl) {
    auto _ept = static_cast<EdgePropertyTable_T*>(ept);
    auto _e = static_cast<Edge_T*>(v);
    auto _epl = static_cast<EdgePropertyList_T*>(epl);
    if (_e->eid >= _ept->num) return NULL;
    auto offset = _e->eid;

    auto r = new Row_T();
    for (auto ep: *_epl) {
        if (ep.first != _ept->etype) return NULL;
        auto array = _ept->g->edge_data_table(ep.first)->column(ep.second)->chunk(0);
        auto result = vineyard::get_arrow_array_data_element(array, offset);
        r->push_back(result);
    }
    return r;
}
#endif
