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
#include "graph/grin/include/common/error.h"
}

std::string GetDataTypeName(GRIN_DATATYPE type) {
  switch (type) {
  case GRIN_DATATYPE::Int32:
    return "int32";
  case GRIN_DATATYPE::UInt32:
    return "uint32";
  case GRIN_DATATYPE::Int64:
    return "int64";
  case GRIN_DATATYPE::UInt64:
    return "uint64";
  case GRIN_DATATYPE::Float:
    return "float";
  case GRIN_DATATYPE::Double:
    return "double";
  case GRIN_DATATYPE::String:
    return "string";
  case GRIN_DATATYPE::Date32:
    return "date32";
  case GRIN_DATATYPE::Time32:
    return "time32";
  case GRIN_DATATYPE::Timestamp64:
    return "timestamp64";
  default:
    return "undefined";
  }
}

GRIN_DATATYPE ArrowToDataType(std::shared_ptr<arrow::DataType> type) {
  if (type == nullptr) {
    return GRIN_DATATYPE::Undefined;
  } else if (arrow::int32()->Equals(type)) {
    return GRIN_DATATYPE::Int32;
  } else if (arrow::int64()->Equals(type)) {
    return GRIN_DATATYPE::Int64;
  } else if (arrow::float32()->Equals(type)) {
    return GRIN_DATATYPE::Float;
  } else if (arrow::uint32()->Equals(type)) {
    return GRIN_DATATYPE::UInt32;
  } else if (arrow::uint64()->Equals(type)) {
    return GRIN_DATATYPE::UInt64;
  } else if (arrow::float64()->Equals(type)) {
    return GRIN_DATATYPE::Double;
  } else if (arrow::utf8()->Equals(type)) {
    return GRIN_DATATYPE::String;
  } else if (arrow::large_utf8()->Equals(type)) {
    return GRIN_DATATYPE::String;
  } 
  return GRIN_DATATYPE::Undefined;
}

const void* _get_value_from_vertex_property_table(GRIN_GRAPH g, GRIN_VERTEX_PROPERTY_TABLE vpt, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    grin_error_code = GRIN_ERROR_CODE::NO_ERROR;
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _vpt = static_cast<GRIN_VERTEX_PROPERTY_TABLE_T*>(vpt);
    unsigned vtype = _grin_get_type_from_property(vp);
    if (v < _vpt->vbegin || v >= _vpt->vend || vtype != _vpt->vtype) {
        grin_error_code = GRIN_ERROR_CODE::INVALID_VALUE;
        return NULL;
    }    
    unsigned vprop = _grin_get_prop_from_property(vp);
    auto offset = v - _vpt->vbegin;
    auto array = _g->vertex_data_table(vtype)->column(vprop)->chunk(0);
    return vineyard::get_arrow_array_data_element(array, offset);
}

const void* _get_value_from_edge_property_table(GRIN_GRAPH g, GRIN_EDGE_PROPERTY_TABLE ept, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    grin_error_code = GRIN_ERROR_CODE::NO_ERROR;
    auto _g = static_cast<GRIN_GRAPH_T*>(g)->g;
    auto _ept = static_cast<GRIN_EDGE_PROPERTY_TABLE_T*>(ept);
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    unsigned etype = _grin_get_type_from_property(ep);  
    unsigned eprop = _grin_get_prop_from_property(ep);
    if (etype != _ept->etype || _e->eid >= _ept->num) {
        grin_error_code = GRIN_ERROR_CODE::INVALID_VALUE;
        return NULL;
    }
    auto offset = _e->eid;
    auto array = _g->edge_data_table(etype)->column(eprop)->chunk(0);
    return vineyard::get_arrow_array_data_element(array, offset);
}


#ifdef GRIN_ENABLE_VERTEX_LIST
void __grin_init_vertex_list(_GRIN_GRAPH_T* g, GRIN_VERTEX_LIST_T* vl) {
    vl->offsets.clear();
    vl->vrs.clear();
    _GRIN_GRAPH_T::vertices_t vr;
    vl->offsets.push_back(0);
    size_t sum = 0;
    for (auto vtype = vl->type_begin; vtype < vl->type_end; ++vtype) {
        if (vl->all_master_mirror == 0) {
            vr = g->Vertices(vtype);
        } else if (vl->all_master_mirror == 1) {
            vr = g->InnerVertices(vtype);
        } else {
            vr = g->OuterVertices(vtype);
        }
        sum += vr.size();
        vl->offsets.push_back(sum);
        vl->vrs.push_back(vr);
    }
}
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST
void __grin_init_adjacent_list(_GRIN_GRAPH_T* g, GRIN_ADJACENT_LIST_T* al) {
    al->offsets.clear();
    al->data.clear();
    _GRIN_GRAPH_T::raw_adj_list_t ral;
    al->offsets.push_back(0);
    size_t sum = 0;
    for (auto etype = al->etype_begin; etype < al->etype_end; ++etype) {
        if (al->dir == GRIN_DIRECTION::IN) {
            ral = g->GetIncomingRawAdjList(_GRIN_GRAPH_T::vertex_t(al->vid), etype);
        } else {
            ral = g->GetOutgoingRawAdjList(_GRIN_GRAPH_T::vertex_t(al->vid), etype);
        }
        sum += ral.size();    
        al->offsets.push_back(sum);
        al->data.push_back(ral);
    }
}
#endif

unsigned long long int _grin_create_property(unsigned type, unsigned prop) {
    return ((unsigned long long int)type << 32) | prop;
}

unsigned _grin_get_type_from_property(unsigned long long int prop) {
    return (unsigned)(prop >> 32);
}

unsigned _grin_get_prop_from_property(unsigned long long int prop) {
    return (unsigned)(prop & 0xffffffff);
}

void _prepare_cache(GRIN_GRAPH_T* g) {
    g->cache = new _GRAPH_CACHE();
    g->cache->id_parser = vineyard::IdParser<_GRIN_GRAPH_T::vid_t>();
    g->cache->id_parser.Init(g->g->fnum(), g->g->vertex_label_num());

    for (int i = 0; i < g->g->vertex_label_num(); ++i) {
        g->cache->vtype_names.push_back(g->g->schema().GetVertexLabelName(i));
        g->cache->vprop_names.push_back(std::vector<std::string>());
        for (int j = 0; j < g->g->vertex_property_num(i); ++j) {
            g->cache->vprop_names[i].push_back(g->g->schema().GetVertexPropertyName(i, j));
        } 
    }

    for (int i = 0; i < g->g->edge_label_num(); ++i) {
        g->cache->etype_names.push_back(g->g->schema().GetEdgeLabelName(i));
        g->cache->eprop_names.push_back(std::vector<std::string>());
        for (int j = 0; j < g->g->edge_property_num(i); ++j) {
            g->cache->eprop_names[i].push_back(g->g->schema().GetEdgePropertyName(i, j));
        } 
    }
}
