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

#include "graph/grin/predefine.h"
#include "common/error.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/arrow_fragment_impl.h"

#include "client/client.h"
#include "arrow/api.h" 

#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__IBMC__) || \
    defined(__INTEL_COMPILER) || defined(__clang__)
#ifndef unlikely
#define unlikely(x_) __builtin_expect(!!(x_), 0)
#endif
#ifndef likely
#define likely(x_) __builtin_expect(!!(x_), 1)
#endif
#else
#ifndef unlikely
#define unlikely(x_) (x_)
#endif
#ifndef likely
#define likely(x_) (x_)
#endif
#endif

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
typedef _GRIN_GRAPH_T::vertices_t GRIN_VERTEX_LIST_T;
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST_ITERATOR
struct GRIN_VERTEX_LIST_ITERATOR_T {
  _GRIN_GRAPH_T::vid_t current;
  _GRIN_GRAPH_T::vid_t end;
};
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST
struct GRIN_ADJACENT_LIST_T {
    const _GRIN_GRAPH_T::nbr_unit_t* begin;
    const _GRIN_GRAPH_T::nbr_unit_t* end;
    _GRIN_GRAPH_T::vid_t vid;
    GRIN_DIRECTION dir;
    unsigned etype;
};
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
struct GRIN_ADJACENT_LIST_ITERATOR_T {
    const _GRIN_GRAPH_T::nbr_unit_t* current;
    const _GRIN_GRAPH_T::nbr_unit_t* end;
    _GRIN_GRAPH_T::vid_t vid;
    GRIN_DIRECTION dir;
    unsigned etype;
};
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

inline const void* _get_arrow_array_data_element(std::shared_ptr<arrow::Array> const& array, unsigned offset) {
  if (array->type()->Equals(arrow::int8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int8Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::uint8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt8Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::int16())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int16Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::uint16())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt16Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::int32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int32Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::uint32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt32Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::int64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int64Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::uint64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt64Array>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::float32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::FloatArray>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::float64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::DoubleArray>(array)->raw_values() + offset);
  } else if (array->type()->Equals(arrow::utf8())) {
    return reinterpret_cast<const void*>(new std::string(
        std::dynamic_pointer_cast<arrow::StringArray>(array).get()->GetView(offset)));
  } else if (array->type()->Equals(arrow::large_utf8())) {
    return reinterpret_cast<const void*>(new std::string(
        std::dynamic_pointer_cast<arrow::LargeStringArray>(array).get()->GetView(offset)));
  } else if (array->type()->id() == arrow::Type::LIST) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::ListArray>(array).get() + offset);
  } else if (array->type()->id() == arrow::Type::LARGE_LIST) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::LargeListArray>(array).get() + offset);
  } else if (array->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::FixedSizeListArray>(array).get() + offset);
  } else if (array->type()->Equals(arrow::null())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::NullArray>(array).get() + offset);
  } else {
    LOG(ERROR) << "Unsupported arrow array type '" << array->type()->ToString()
               << "', type id: " << array->type()->id();
    return NULL;
  }
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
    return _get_arrow_array_data_element(array, _cache->id_parser.GetOffset(v));
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
        printf("INVALID: %u %u\n", _e->etype, etype);
        grin_error_code = GRIN_ERROR_CODE::INVALID_VALUE;
        return NULL;
    }
    auto _cache = static_cast<GRIN_GRAPH_T*>(g)->cache;
    unsigned eprop = _grin_get_prop_from_property(ep);
    auto array = _cache->etables[etype]->column(eprop)->chunk(0);
    return _get_arrow_array_data_element(array, _e->eid);
}
#endif

#ifdef GRIN_ENABLE_ROW
typedef std::vector<const void*> GRIN_ROW_T;
#endif

#endif  // GRIN_SRC_PREDEFINE_H_
