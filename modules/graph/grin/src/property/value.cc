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
#include "property/value.h"

#define GET_VERTEX_VALUE \
    grin_error_code = GRIN_ERROR_CODE::NO_ERROR; \
    unsigned vtype = _grin_get_type_from_property(vp); \
    unsigned vprop = _grin_get_prop_from_property(vp); \
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache; \
    assert((unsigned)_cache->id_parser.GetLabelId(v) == vtype);

#define GET_EDGE_VALUE \
    grin_error_code = GRIN_ERROR_CODE::NO_ERROR; \
    unsigned etype = _grin_get_type_from_property(ep); \
    assert(etype == e.etype); \
    unsigned eprop = _grin_get_prop_from_property(ep); \
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;


void grin_destroy_vertex_property_value_of_string(GRIN_GRAPH g, const char* value) {}

void grin_destroy_vertex_property_value_of_float_array(GRIN_GRAPH g, const float* value, size_t length) {}

int grin_get_vertex_property_value_of_int32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const int*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

unsigned int grin_get_vertex_property_value_of_uint32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const unsigned int*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

long long int grin_get_vertex_property_value_of_int64(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const long long int*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

unsigned long long int grin_get_vertex_property_value_of_uint64(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const unsigned long long int*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

float grin_get_vertex_property_value_of_float(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const float*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

double grin_get_vertex_property_value_of_double(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const double*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

const char* grin_get_vertex_property_value_of_string(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    auto result = _get_arrow_array_data_element(_cache->varrays[vtype][vprop], _cache->id_parser.GetOffset(v));
    return static_cast<const std::string*>(result)->c_str();
}

int grin_get_vertex_property_value_of_date32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const int*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

int grin_get_vertex_property_value_of_time32(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const int*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

long long int grin_get_vertex_property_value_of_timestamp64(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    GET_VERTEX_VALUE
    return static_cast<const long long int*>(_cache->varrs[vtype][vprop])[_cache->id_parser.GetOffset(v)];
}

const float* grin_get_vertex_property_value_of_float_array(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp, size_t* length) {
    GET_VERTEX_VALUE
    *length = _cache->feature_size;
    return static_cast<const float*>(_cache->varrs[vtype][vprop]) + _cache->id_parser.GetOffset(v) * _cache->feature_size;
}

void grin_destroy_edge_property_value_of_string(GRIN_GRAPH g, const char* value) {}

void grin_destroy_edge_property_value_of_float_array(GRIN_GRAPH g, const float* value, size_t length) {}

int grin_get_edge_property_value_of_int32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const int*>(_cache->earrs[etype][eprop])[e.eid];
}

unsigned int grin_get_edge_property_value_of_uint32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const unsigned int*>(_cache->earrs[etype][eprop])[e.eid];
}

long long int grin_get_edge_property_value_of_int64(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const long long int*>(_cache->earrs[etype][eprop])[e.eid];
}

unsigned long long int grin_get_edge_property_value_of_uint64(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const unsigned long long int*>(_cache->earrs[etype][eprop])[e.eid];
}

float grin_get_edge_property_value_of_float(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const float*>(_cache->earrs[etype][eprop])[e.eid];
}

double grin_get_edge_property_value_of_double(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const double*>(_cache->earrs[etype][eprop])[e.eid];
}

const char* grin_get_edge_property_value_of_string(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    auto result = _get_arrow_array_data_element(_cache->earrays[etype][eprop], e.eid);
    return static_cast<const std::string*>(result)->c_str();
}

int grin_get_edge_property_value_of_date32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const int*>(_cache->earrs[etype][eprop])[e.eid];
}

int grin_get_edge_property_value_of_time32(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const int*>(_cache->earrs[etype][eprop])[e.eid];
}

long long int grin_get_edge_property_value_of_timestamp64(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    GET_EDGE_VALUE
    return static_cast<const long long int*>(_cache->earrs[etype][eprop])[e.eid];
}

const float* grin_get_edge_property_value_of_float_array(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep, size_t* length) {
    *length = 0;
    return NULL;
}

