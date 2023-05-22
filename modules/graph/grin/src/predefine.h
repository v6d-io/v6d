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

#ifndef GRIN_SRC_PREDEFINE_H_
#define GRIN_SRC_PREDEFINE_H_

extern "C" {
#include "graph/grin/predefine.h"
#include "graph/grin/include/common/error.h"
}

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/arrow_fragment_impl.h"

#include "client/client.h"
#include "arrow/api.h" 

template <typename T>
struct GRIN_DATATYPE_ENUM {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Undefined;
};

template <>
struct GRIN_DATATYPE_ENUM<int32_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Int32;
};

template <>
struct GRIN_DATATYPE_ENUM<uint32_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::UInt32;
};

template <>
struct GRIN_DATATYPE_ENUM<int64_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Int64;
};

template <>
struct GRIN_DATATYPE_ENUM<uint64_t> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::UInt64;
};

template <>
struct GRIN_DATATYPE_ENUM<float> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Float;
};

template <>
struct GRIN_DATATYPE_ENUM<double> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Double;
};

template <>
struct GRIN_DATATYPE_ENUM<std::string> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::String;
};

template <>
struct GRIN_DATATYPE_ENUM<arrow::Date32Type> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Date32;
};

template <>
struct GRIN_DATATYPE_ENUM<arrow::Date64Type> {
  static constexpr GRIN_DATATYPE value = GRIN_DATATYPE::Timestamp64;
};

inline std::string GetDataTypeName(GRIN_DATATYPE type) {
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

inline GRIN_DATATYPE ArrowToDataType(std::shared_ptr<arrow::DataType> type) {
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

#define GRIN_OID_T int64_t
#define GRIN_VID_T uint64_t

/* The following data types shall be defined through typedef. */
typedef vineyard::ArrowFragment<GRIN_OID_T, GRIN_VID_T> _GRIN_GRAPH_T;   
struct _GRAPH_CACHE {
    vineyard::IdParser<_GRIN_GRAPH_T::vid_t> id_parser;
    std::vector<std::string> vtype_names;
    std::vector<std::string> etype_names;
    std::vector<std::vector<std::string>> vprop_names;
    std::vector<std::vector<std::string>> eprop_names;
    std::vector<std::shared_ptr<arrow::Table>> vtables;
    std::vector<std::shared_ptr<arrow::Table>> etables;
};

struct GRIN_GRAPH_T {
    vineyard::Client client;
    std::shared_ptr<_GRIN_GRAPH_T> _g;
    _GRIN_GRAPH_T* g;
    _GRAPH_CACHE* cache;
};

inline void _prepare_cache(GRIN_GRAPH_T* g) {
    g->cache = new _GRAPH_CACHE();
    g->cache->id_parser = vineyard::IdParser<_GRIN_GRAPH_T::vid_t>();
    g->cache->id_parser.Init(g->g->fnum(), g->g->vertex_label_num());

    g->cache->vtype_names.resize(g->g->vertex_label_num());
    g->cache->vtables.resize(g->g->vertex_label_num());
    g->cache->vprop_names.resize(g->g->vertex_label_num());

    for (int i = 0; i < g->g->vertex_label_num(); ++i) {
        g->cache->vtype_names[i] = g->g->schema().GetVertexLabelName(i);
        g->cache->vtables[i] = g->g->vertex_data_table(i);
        g->cache->vprop_names[i].resize(g->g->vertex_property_num(i));
        for (int j = 0; j < g->g->vertex_property_num(i); ++j) {
            g->cache->vprop_names[i][j] = g->g->schema().GetVertexPropertyName(i, j);
        } 
    }

    g->cache->etype_names.resize(g->g->edge_label_num());
    g->cache->etables.resize(g->g->edge_label_num());
    g->cache->eprop_names.resize(g->g->edge_label_num());

    for (int i = 0; i < g->g->edge_label_num(); ++i) {
        g->cache->etype_names[i] = g->g->schema().GetEdgeLabelName(i);
        g->cache->etables[i] = g->g->edge_data_table(i);
        g->cache->eprop_names[i].resize(g->g->edge_property_num(i));
        for (int j = 0; j < g->g->edge_property_num(i); ++j) {
            g->cache->eprop_names[i][j] = g->g->schema().GetEdgePropertyName(i, j);
        } 
    }
}


typedef _GRIN_GRAPH_T::vertex_t _GRIN_VERTEX_T;    

struct GRIN_EDGE_T {
    _GRIN_GRAPH_T::vid_t src;
    _GRIN_GRAPH_T::vid_t dst;
    GRIN_DIRECTION dir;
    unsigned etype;
    _GRIN_GRAPH_T::eid_t eid;
};                     

#ifdef GRIN_ENABLE_VERTEX_LIST
struct GRIN_VERTEX_LIST_T {
    _GRIN_GRAPH_T::vid_t begin_ = 0;
    _GRIN_GRAPH_T::vid_t end_ = 0;
    unsigned all_master_mirror;
    unsigned vtype;
    bool is_simple;
    std::vector<std::pair<size_t, _GRIN_GRAPH_T::vid_t>> offsets;
};
inline void __grin_init_simple_vertex_list(_GRIN_GRAPH_T* g, GRIN_VERTEX_LIST_T* vl) {
    _GRIN_GRAPH_T::vertices_t vr;
    if (vl->all_master_mirror == 0) {
        vr = g->Vertices(vl->vtype);
    } else if (vl->all_master_mirror == 1) {
        vr = g->InnerVertices(vl->vtype);
    } else {
        vr = g->OuterVertices(vl->vtype);
    }
    vl->begin_ = vr.begin_value();
    vl->end_ = vr.end_value();
}
inline void __grin_init_complex_vertex_list(_GRIN_GRAPH_T* g, GRIN_VERTEX_LIST_T* vl) {
    _GRIN_GRAPH_T::vertices_t vr;
    size_t sum = 0;
    vl->offsets.resize(vl->vtype + 1);
    for (unsigned i = 0; i < vl->vtype; ++i) {
        if (vl->all_master_mirror == 0) {
            vr = g->Vertices(i);
        } else if (vl->all_master_mirror == 1) {
            vr = g->InnerVertices(i);
        } else {
            vr = g->OuterVertices(i);
        }
        vl->offsets[i] = std::make_pair(sum, vr.begin_value());
        sum += vr.size();
    }
    vl->offsets[vl->vtype] = std::make_pair(sum, vr.end_value());
}
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST_ITERATOR
struct GRIN_VERTEX_LIST_ITERATOR_T {
    _GRIN_GRAPH_T::vid_t end_;
    _GRIN_GRAPH_T::vid_t current_;
    unsigned all_master_mirror;
    bool is_simple;
    unsigned vtype_current;
    unsigned vtype_end;
};
inline void __grin_next_valid_vertex_list_iterator(_GRIN_GRAPH_T* g, GRIN_VERTEX_LIST_ITERATOR_T* vli) {
    _GRIN_GRAPH_T::vertices_t vr;
    while (vli->vtype_current < vli->vtype_end) {
        if (vli->all_master_mirror == 0) {
            vr = g->Vertices(vli->vtype_current);
        } else if (vli->all_master_mirror == 1) {
            vr = g->InnerVertices(vli->vtype_current);
        } else {
            vr = g->OuterVertices(vli->vtype_current);
        }
        if (vr.size() > 0) {
            vli->current_ = vr.begin_value();
            vli->end_ = vr.end_value();
            break;
        }
        vli->vtype_current++;
    }
}
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST
struct GRIN_ADJACENT_LIST_T {
    const _GRIN_GRAPH_T::nbr_unit_t* begin_ = nullptr;
    const _GRIN_GRAPH_T::nbr_unit_t* end_ = nullptr;
    _GRIN_GRAPH_T::vid_t vid;
    GRIN_DIRECTION dir;
    unsigned etype;
    bool is_simple;
    std::vector<std::pair<size_t, const _GRIN_GRAPH_T::nbr_unit_t*>> offsets;
};
inline void __grin_init_simple_adjacent_list(_GRIN_GRAPH_T* g, GRIN_ADJACENT_LIST_T* al) {
    _GRIN_GRAPH_T::raw_adj_list_t ral;
    if (al->dir == GRIN_DIRECTION::IN) {
          ral = g->GetIncomingRawAdjList(_GRIN_GRAPH_T::vertex_t(al->vid), al->etype);
      } else {
          ral = g->GetOutgoingRawAdjList(_GRIN_GRAPH_T::vertex_t(al->vid), al->etype);
      }
    al->begin_ = ral.begin();
    al->end_ = ral.end();
}
inline void __grin_init_complex_adjacent_list(_GRIN_GRAPH_T* g, GRIN_ADJACENT_LIST_T* al) {
    _GRIN_GRAPH_T::raw_adj_list_t ral;
    size_t sum = 0;
    al->offsets.resize(al->etype + 1);
    for (unsigned i = 0; i < al->etype; ++i) {
        if (al->dir == GRIN_DIRECTION::IN) {
            ral = g->GetIncomingRawAdjList(_GRIN_GRAPH_T::vertex_t(al->vid), i);
        } else {
            ral = g->GetOutgoingRawAdjList(_GRIN_GRAPH_T::vertex_t(al->vid), i);
        }
        al->offsets[i] = std::make_pair(sum, ral.begin());
        sum += ral.size();
    }
    al->offsets[al->etype] = std::make_pair(sum, ral.end());
}
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
struct GRIN_ADJACENT_LIST_ITERATOR_T {
    const _GRIN_GRAPH_T::nbr_unit_t* end_;
    const _GRIN_GRAPH_T::nbr_unit_t* current_;
    _GRIN_GRAPH_T::vid_t vid;
    GRIN_DIRECTION dir;
    bool is_simple;
    unsigned etype_current;
    unsigned etype_end;
};
inline void __grin_next_valid_adjacent_list_iterator(_GRIN_GRAPH_T* g, GRIN_ADJACENT_LIST_ITERATOR_T* ali) {
    _GRIN_GRAPH_T::raw_adj_list_t raj;
    while (ali->etype_current < ali->etype_end) {
        if (ali->dir == GRIN_DIRECTION::IN) {
            raj = g->GetIncomingRawAdjList(_GRIN_GRAPH_T::vertex_t(ali->vid), ali->etype_current);
        } else {
            raj = g->GetOutgoingRawAdjList(_GRIN_GRAPH_T::vertex_t(ali->vid), ali->etype_current);
        }
        if (raj.size() > 0) {
            ali->current_ = raj.begin();
            ali->end_ = raj.end();
            break;
        }
        ali->etype_current++;
    }
}
#endif

#ifdef GRIN_ENABLE_GRAPH_PARTITION
struct GRIN_PARTITIONED_GRAPH_T {
  std::string socket;
  vineyard::Client client;
  std::shared_ptr<vineyard::ArrowFragmentGroup> pg;
  std::vector<vineyard::ObjectID> lgs;
};
typedef unsigned GRIN_PARTITION_T;
typedef std::vector<unsigned> GRIN_PARTITION_LIST_T;
#endif

#ifdef GRIN_ENABLE_VERTEX_REF
typedef _GRIN_GRAPH_T::vid_t GRIN_VERTEX_REF_T;
#endif

inline unsigned long long int _grin_create_property(unsigned type, unsigned prop) {
    return ((unsigned long long int)type << 32) | prop;
}

inline unsigned _grin_get_type_from_property(unsigned long long int prop) {
    return (unsigned)(prop >> 32);
}

inline unsigned _grin_get_prop_from_property(unsigned long long int prop) {
    return (unsigned)(prop & 0xffffffff);
}


#ifdef GRIN_WITH_VERTEX_PROPERTY
typedef unsigned GRIN_VERTEX_TYPE_T;
typedef std::vector<unsigned> GRIN_VERTEX_TYPE_LIST_T;
typedef unsigned long long int GRIN_VERTEX_PROPERTY_T;
typedef std::vector<GRIN_VERTEX_PROPERTY_T> GRIN_VERTEX_PROPERTY_LIST_T;

inline const void* _get_value_from_vertex_property_table(GRIN_GRAPH g, GRIN_VERTEX v, GRIN_VERTEX_PROPERTY vp) {
    grin_error_code = GRIN_ERROR_CODE::NO_ERROR;
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    unsigned vtype0 =  _cache->id_parser.GetLabelId(v);
    unsigned vtype1 = _grin_get_type_from_property(vp);
    if (vtype0 != vtype1) {
        grin_error_code = GRIN_ERROR_CODE::INVALID_VALUE;
        return NULL;
    }
    unsigned vprop = _grin_get_prop_from_property(vp);
    auto array = _cache->vtables[vtype0]->column(vprop)->chunk(0);
    return vineyard::get_arrow_array_data_element(array, _cache->id_parser.GetOffset(v));
}
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
typedef unsigned GRIN_EDGE_TYPE_T;
typedef std::vector<unsigned> GRIN_EDGE_TYPE_LIST_T;
typedef unsigned long long int GRIN_EDGE_PROPERTY_T;
typedef std::vector<GRIN_EDGE_PROPERTY_T> GRIN_EDGE_PROPERTY_LIST_T;

inline const void* _get_value_from_edge_property_table(GRIN_GRAPH g, GRIN_EDGE e, GRIN_EDGE_PROPERTY ep) {
    grin_error_code = GRIN_ERROR_CODE::NO_ERROR;
    auto _e = static_cast<GRIN_EDGE_T*>(e);
    unsigned etype = _grin_get_type_from_property(ep);
    if (_e->etype != etype) {
        grin_error_code = GRIN_ERROR_CODE::INVALID_VALUE;
        return NULL;
    }
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    unsigned eprop = _grin_get_prop_from_property(ep);
    auto array = _cache->etables[etype]->column(eprop)->chunk(0);
    return vineyard::get_arrow_array_data_element(array, _e->eid);
}
#endif

#ifdef GRIN_ENABLE_ROW
typedef std::vector<const void*> GRIN_ROW_T;
#endif

#endif  // GRIN_SRC_PREDEFINE_H_
