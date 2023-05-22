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
#include "graph/grin/include/property/primarykey.h"
}

#ifdef GRIN_ENABLE_VERTEX_PRIMARY_KEYS
/** 
 * @brief get the vertex types with primary keys
 * @param GRIN_GRAPH the graph
*/
GRIN_VERTEX_TYPE_LIST grin_get_vertex_types_with_primary_keys(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vtl = new GRIN_VERTEX_TYPE_LIST_T();
    vtl->resize(_g->vertex_label_num());
    for (auto i = 0; i < _g->vertex_label_num(); ++i) {
        (*vtl)[i] = i;
    }
    return vtl;
}

/** 
 * @brief get the primary keys (property list) of a specific vertex type
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_TYPE the vertex type
*/
GRIN_VERTEX_PROPERTY_LIST grin_get_primary_keys_by_vertex_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto vpl = new GRIN_VERTEX_PROPERTY_LIST_T();
    for (auto p = 0; p < _g->vertex_property_num(vtype); ++p) {
        if (_g->schema().GetVertexPropertyName(vtype, p) == "id") {
            vpl->push_back(_grin_create_property(vtype, p));
            break;
        }
    }
    return vpl;
}

/** 
 * @brief get the vertex with the given primary keys
 * @param GRIN_GRAPH the graph
 * @param GRIN_VERTEX_TYPE the vertex type which determines the property list for primary keys
 * @param GRIN_ROW the values of primary keys
*/
GRIN_VERTEX grin_get_vertex_by_primary_keys(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype, GRIN_ROW r) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _r = static_cast<GRIN_ROW_T*>(r);
    auto value = (*_r)[0];
#if 0
    for (auto p = 0; p < _g->vertex_property_num(vtype); ++p) {
        if (_g->schema().GetVertexPropertyName(vtype, p) == "id") {
            auto arrow_dt = _g->schema().GetVertexPropertyType(vtype, p);
            auto dt = ArrowToDataType(arrow_dt);
            
            if (dt == GRIN_DATATYPE::Int64) {
#endif
                _GRIN_VERTEX_T v;
                auto vid = static_cast<const int64_t*>(value);
                if (!_g->GetVertex(vtype, *vid, v)) return GRIN_NULL_VERTEX;
                return v.GetValue();
#if 0
            } else {
                return GRIN_NULL_VERTEX;
            }
        }
    }
    return GRIN_NULL_VERTEX;
#endif
}
#endif

#ifdef GRIN_WITH_EDGE_PRIMARY_KEYS
/** 
 * @brief get the edge types with primary keys
 * @param GRIN_GRAPH the graph
*/
GRIN_EDGE_TYPE_LIST grin_get_edge_types_with_primary_keys(GRIN_GRAPH);

/** 
 * @brief get the primary keys (property list) of a specific edge type
 * @param GRIN_GRAPH the graph
 * @param GRIN_EDGE_TYPE the edge type
*/
GRIN_EDGE_PROPERTY_LIST grin_get_primary_keys_by_edge_type(GRIN_GRAPH, GRIN_EDGE_TYPE);

/** 
 * @brief get the edge with the given primary keys
 * @param GRIN_GRAPH the graph
 * @param GRIN_EDGE_PROPERTY_LIST the primary keys
 * @param GRIN_ROW the values of primary keys
*/
GRIN_EDGE grin_get_edge_by_primary_keys(GRIN_GRAPH, GRIN_EDGE_TYPE, GRIN_ROW);
#endif
