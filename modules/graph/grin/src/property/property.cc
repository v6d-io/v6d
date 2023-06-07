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
#include "property/property.h"


void grin_destroy_string_value(GRIN_GRAPH g, const char* value) {}

#ifdef GRIN_WITH_VERTEX_PROPERTY_NAME
const char* grin_get_vertex_property_name(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY vp) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->vprop_names[_grin_get_type_from_property(vp)][_grin_get_prop_from_property(vp)].c_str();
}

GRIN_VERTEX_PROPERTY grin_get_vertex_property_by_name(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype,
                                           const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto s = std::string(name);
    auto _id = _g->schema().GetVertexPropertyId(vtype, s);
    if (_id < 0) return GRIN_NULL_VERTEX_PROPERTY;
    return _grin_create_property(vtype, _id);
}

GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_properties_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto s = std::string(name);
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        auto pid = _g->schema().GetVertexPropertyId(vtype, s);
        if (pid >= 0) {
            vpl->push_back(_grin_create_property(vtype, pid));
        }
    }
    if (vpl->empty()) {
        delete vpl;
        return GRIN_NULL_LIST;
    }
    return vpl;
}
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY_NAME
const char* grin_get_edge_property_name(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY ep) {
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    return _cache->eprop_names[_grin_get_type_from_property(ep)][_grin_get_prop_from_property(ep)].c_str();
}

GRIN_EDGE_PROPERTY grin_get_edge_property_by_name(GRIN_GRAPH g, GRIN_EDGE_TYPE etype,
                                           const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto s = std::string(name);
    auto _id = _g->schema().GetEdgePropertyId(etype, s);
    if (_id < 0) return GRIN_NULL_EDGE_PROPERTY;
    return _grin_create_property(etype, _id);
}

GRIN_EDGE_PROPERTY_LIST grin_get_edge_properties_by_name(GRIN_GRAPH g, const char* name) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto s = std::string(name);
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    for (auto etype = 0; etype < _g->edge_label_num(); ++etype) {
        auto pid = _g->schema().GetEdgePropertyId(etype, s);
        if (pid >= 0) {
            epl->push_back(_grin_create_property(etype, pid));
        }
    }
    if (epl->empty()) {
        delete epl;
        return GRIN_NULL_LIST;
    }
    return epl;
}
#endif


#ifdef GRIN_WITH_VERTEX_PROPERTY
bool grin_equal_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp1, GRIN_VERTEX_PROPERTY vp2) {
    return (vp1 == vp2);
}

void grin_destroy_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {}

GRIN_DATATYPE grin_get_vertex_property_datatype(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto dt = _g->schema().GetVertexPropertyType(_grin_get_type_from_property(vp), _grin_get_prop_from_property(vp));
    return ArrowToDataType(dt);
}

int grin_get_vertex_property_value_of_int32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const int*>(result);
}

unsigned int grin_get_vertex_property_value_of_uint32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const unsigned int*>(result);
}

long long int grin_get_vertex_property_value_of_int64(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const long long int*>(result);
}

unsigned long long int grin_get_vertex_property_value_of_uint64(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const unsigned long long int*>(result);
}

float grin_get_vertex_property_value_of_float(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const float*>(result);
}

double grin_get_vertex_property_value_of_double(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const double*>(result);
}

const char* grin_get_vertex_property_value_of_string(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return NULL;
    return static_cast<const std::string*>(result)->c_str();
}

int grin_get_vertex_property_value_of_date32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const int*>(result);
}

int grin_get_vertex_property_value_of_time32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const int*>(result);
}

long long int grin_get_vertex_property_value_of_timestamp64(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    const void* result = _get_value_from_vertex_property_table(g, v, vp);
    if (result == NULL) return 0;
    return *static_cast<const long long int*>(result);
}


GRIN_VERTEX_TYPE grin_get_vertex_type_from_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    return _grin_get_type_from_property(vp);
}
#endif


#ifdef GRIN_WITH_EDGE_PROPERTY
bool grin_equal_edge_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep1, GRIN_EDGE_PROPERTY ep2) {
    return (ep1 == ep2);
}

void grin_destroy_edge_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {}

GRIN_DATATYPE grin_get_edge_property_datatype(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto dt = _g->schema().GetEdgePropertyType(_grin_get_type_from_property(ep), _grin_get_prop_from_property(ep));
    return ArrowToDataType(dt);
}

int grin_get_edge_property_value_of_int32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const int*>(result);
}

unsigned int grin_get_edge_property_value_of_uint32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const unsigned int*>(result);
}

long long int grin_get_edge_property_value_of_int64(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const long long int*>(result);
}

unsigned long long int grin_get_edge_property_value_of_uint64(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const unsigned long long int*>(result);
}

float grin_get_edge_property_value_of_float(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const float*>(result);
}

double grin_get_edge_property_value_of_double(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const double*>(result);
}

const char* grin_get_edge_property_value_of_string(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return NULL;
    return static_cast<const std::string*>(result)->c_str();
}

int grin_get_edge_property_value_of_date32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const int*>(result);
}

int grin_get_edge_property_value_of_time32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const int*>(result);
}

long long int grin_get_edge_property_value_of_timestamp64(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    const void* result = _get_value_from_edge_property_table(g, e, ep);
    if (result == NULL) return 0;
    return *static_cast<const long long int*>(result);
}

GRIN_EDGE_TYPE grin_get_edge_type_from_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    return _grin_get_type_from_property(ep);
}
#endif
