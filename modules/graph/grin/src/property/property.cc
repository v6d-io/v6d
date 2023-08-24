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

#if defined(GRIN_ENABLE_SCHEMA) && defined(GRIN_WITH_VERTEX_PROPERTY)
/**
 * @brief Get the vertex property list of the graph.
 * This API is only available for property graph.
 * @param GRIN_GRAPH The graph.
 * @return The vertex property list.
*/
GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_property_list_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    vpl->resize(_g->vertex_property_num(vtype));
    for (auto p = 0; p < _g->vertex_property_num(vtype); ++p) {
        (*vpl)[p] = _grin_create_property(vtype, p);
    }
    return vpl;
}

/**
 * @brief Get the vertex type that a given vertex property belongs to.
 * @param GRIN_GRAPH The graph
 * @param GRIN_VERTEX_PROPERTY The vertex property
 * @return The vertex type
*/
GRIN_VERTEX_TYPE grin_get_vertex_type_from_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    return _grin_get_type_from_property(vp);
}

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
        return GRIN_NULL_VERTEX_PROPERTY_LIST;
    }
    return vpl;
}


/**
 * @brief Get the vertex property handle by id.
 * In strict schema, storage has naturally increasing ids for vertex properties
 * under a certain vertex type.
 * @param GRIN_GRAPH The graph.
 * @param GRIN_VERTEX_TYPE The vertex type.
 * @param GRIN_VERTEX_PROPERTY_ID The vertex property id.
 * @return The vertex property handle.
*/
GRIN_VERTEX_PROPERTY grin_get_vertex_property_by_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY_ID vpi) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    if (static_cast<int>(vpi) >= _g->vertex_property_num(vtype)) return GRIN_NULL_VERTEX_PROPERTY;
    return _grin_create_property(vtype, vpi);
}

/**
 * @brief Get the vertex property's natural id.
 * In strict schema, the storage has naturally increasing ids for vertex properties
 * under a certain vertex type.
 * @param GRIN_GRAPH The graph.
 * @param GRIN_VERTEX_TYPE The vertex type.
 * @param GRIN_VERTEX_PROPERTY The vertex property handle.
 * @return The vertex property id.
*/
GRIN_VERTEX_PROPERTY_ID grin_get_vertex_property_id(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_VERTEX_PROPERTY vp) {
    if (vtype != _grin_get_type_from_property(vp)) return GRIN_NULL_VERTEX_PROPERTY_ID;
    return _grin_get_prop_from_property(vp);
}
#endif

#if !defined(GRIN_ENABLE_SCHEMA) && defined(GRIN_WITH_VERTEX_PROPERTY)
/**
 * @brief Get the vertex property list of the vertex.
 * When schema is not enabled, each vertex has its own property list.
 * @param GRIN_GRAPH The graph.
 * @param GRIN_VERTEX The vertex.
 * @return The vertex property list.
*/
GRIN_VERTEX_PROPERTY_LIST grin_get_vertex_property_list(GRIN_GRAPH, GRIN_VERTEX);

/**
 * @brief Get the vertex property name
 * @param GRIN_GRAPH The graph
 * @param GRIN_VERTEX The vertex
 * @param GRIN_VERTEX_PROPERTY The vertex property
 * @return The property's name as string
 */
const char* grin_get_vertex_property_name(GRIN_GRAPH, GRIN_VERTEX, GRIN_VERTEX_PROPERTY);

/**
 * @brief Get the vertex property with a given name under a specific vertex
 * @param GRIN_GRAPH The graph
 * @param GRIN_VERTEX The specific vertex
 * @param name The name
 * @return The vertex property
 */
GRIN_VERTEX_PROPERTY grin_get_vertex_property_by_name(GRIN_GRAPH, GRIN_VERTEX, const char* name);
#endif

#if defined(GRIN_ENABLE_SCHEMA) && defined(GRIN_WITH_EDGE_PROPERTY)
GRIN_EDGE_PROPERTY_LIST grin_get_edge_property_list_by_type(GRIN_GRAPH g, GRIN_EDGE_TYPE etype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto epl = new GRIN_EDGE_PROPERTY_LIST_T();
    epl->resize(_g->edge_property_num(etype) - 1);
    for (auto p = 1; p < _g->edge_property_num(etype); ++p) {
        (*epl)[p - 1] = _grin_create_property(etype, p);
    }
    return epl;
}

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
        return GRIN_NULL_EDGE_PROPERTY_LIST;
    }
    return epl;
}

GRIN_EDGE_TYPE grin_get_edge_type_from_property(GRIN_GRAPH g, GRIN_EDGE_PROPERTY ep) {
    return _grin_get_type_from_property(ep);
}

GRIN_EDGE_PROPERTY grin_get_edge_property_by_id(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY_ID epi) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    if (static_cast<int>(epi) >= _g->edge_property_num(etype) - 1) return GRIN_NULL_EDGE_PROPERTY;
    return _grin_create_property(etype, epi + 1);
}

GRIN_EDGE_PROPERTY_ID grin_get_edge_property_id(GRIN_GRAPH g, GRIN_EDGE_TYPE etype, GRIN_EDGE_PROPERTY ep) {
    if (etype != _grin_get_type_from_property(ep)) return GRIN_NULL_EDGE_PROPERTY_ID;
    return _grin_get_prop_from_property(ep) - 1;
}
#endif


#if !defined(GRIN_ENABLE_SCHEMA) && defined(GRIN_WITH_EDGE_PROPERTY)
GRIN_EDGE_PROPERTY_LIST grin_get_edge_property_list(GRIN_GRAPH, GRIN_EDGE);

const char* grin_get_edge_property_name(GRIN_GRAPH, GRIN_EDGE, GRIN_EDGE_PROPERTY);

GRIN_EDGE_PROPERTY grin_get_edge_property_by_name(GRIN_GRAPH, GRIN_EDGE, const char* name);
#endif


#ifdef GRIN_WITH_VERTEX_PROPERTY
bool grin_equal_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp1, GRIN_VERTEX_PROPERTY vp2) {
    return (vp1 == vp2);
}

void grin_destroy_vertex_property(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {}

/**
 * @brief Get the datatype of the vertex property
 * @param GRIN_GRAPH The graph
 * @param GRIN_VERTEX_PROPERTY The vertex property
 * @return The datatype of the vertex property
*/
GRIN_DATATYPE grin_get_vertex_property_datatype(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY vp) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto properties = _g->schema().GetEntry(_grin_get_type_from_property(vp), "VERTEX").properties();
    auto dt = _g->schema().GetVertexPropertyType(_grin_get_type_from_property(vp), properties[_grin_get_prop_from_property(vp)].id);
    return ArrowToDataType(dt);
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
#endif
