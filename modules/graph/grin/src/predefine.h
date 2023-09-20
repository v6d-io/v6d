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
  case GRIN_DATATYPE::FloatArray:
    return "float_array";
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
  } else if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
    return GRIN_DATATYPE::FloatArray;
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
    std::vector<std::vector<std::shared_ptr<arrow::Array>>> varrays;
    std::vector<std::vector<std::shared_ptr<arrow::Array>>> earrays;
    std::vector<std::vector<const void*>> varrs;
    std::vector<std::vector<const void*>> earrs;
    unsigned feature_size;
};

struct GRIN_GRAPH_T {
    vineyard::Client client;
    std::shared_ptr<_GRIN_GRAPH_T> _g;
    _GRIN_GRAPH_T* g;
    _GRAPH_CACHE* cache;
};

inline const void* _GetArrowArrayData(
    std::shared_ptr<arrow::Array> const& array) {
  if (array->type()->Equals(arrow::int8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int8Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt8Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::int16())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int16Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint16())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt16Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::int32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int32Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt32Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::int64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::Int64Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::uint64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::UInt64Array>(array)->raw_values());
  } else if (array->type()->Equals(arrow::float32())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::FloatArray>(array)->raw_values());
  } else if (array->type()->Equals(arrow::float64())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::DoubleArray>(array)->raw_values());
  } else if (array->type()->Equals(arrow::utf8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::StringArray>(array).get());
  } else if (array->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    auto list_array = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(array);
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::FloatArray>(list_array->values())->raw_values());
  } else if (array->type()->Equals(arrow::large_utf8())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::LargeStringArray>(array).get());
  } else if (array->type()->Equals(arrow::list(arrow::int32())) ||
             array->type()->Equals(arrow::large_list(arrow::uint32())) ||
             array->type()->Equals(arrow::large_list(arrow::int64())) ||
             array->type()->Equals(arrow::large_list(arrow::uint64())) ||
             array->type()->Equals(arrow::large_list(arrow::float32())) ||
             array->type()->Equals(arrow::large_list(arrow::float64()))) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::LargeListArray>(array).get());
  } else if (array->type()->Equals(arrow::null())) {
    return reinterpret_cast<const void*>(
        std::dynamic_pointer_cast<arrow::NullArray>(array).get());
  }
  return nullptr;
}

inline void _prepare_cache(GRIN_GRAPH_T* g) {
    g->cache = new _GRAPH_CACHE();
    g->cache->feature_size = 0;
    g->cache->id_parser = vineyard::IdParser<_GRIN_GRAPH_T::vid_t>();
    g->cache->id_parser.Init(g->g->fnum(), g->g->vertex_label_num());
    g->cache->vtype_names.resize(g->g->vertex_label_num());
    g->cache->varrays.resize(g->g->vertex_label_num());
    g->cache->varrs.resize(g->g->vertex_label_num());
    g->cache->vprop_names.resize(g->g->vertex_label_num());

    for (int i = 0; i < g->g->vertex_label_num(); ++i) {
//    auto properties = _g->schema().GetEntry(vtype, "VERTEX").properties();
        g->cache->vtype_names[i] = g->g->schema().GetVertexLabelName(i);
        g->cache->varrays[i].resize(g->g->vertex_property_num(i));
        g->cache->varrs[i].resize(g->g->vertex_property_num(i));
        g->cache->vprop_names[i].resize(g->g->vertex_property_num(i));
        auto properties = g->g->schema().GetEntry(i, "VERTEX").properties();
        for (int j = 0; j < g->g->vertex_property_num(i); ++j) {
            g->cache->vprop_names[i][j] = properties[j].name; //g->g->schema().GetVertexPropertyName(i, j);
            g->cache->varrays[i][j] = g->g->vertex_data_table(i)->column(j)->chunk(0);
            g->cache->varrs[i][j] = _GetArrowArrayData(g->g->vertex_data_table(i)->column(j)->chunk(0));
            if (g->g->vertex_data_table(i)->column(j)->chunk(0)->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
                assert(g->cache->feature_size == 0);
                auto arr = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(g->g->vertex_data_table(i)->column(j)->chunk(0));
                g->cache->feature_size = arr->list_type()->list_size();
            } 
        }
    }

    g->cache->etype_names.resize(g->g->edge_label_num());
    g->cache->earrays.resize(g->g->edge_label_num());
    g->cache->earrs.resize(g->g->edge_label_num());
    g->cache->eprop_names.resize(g->g->edge_label_num());

    for (int i = 0; i < g->g->edge_label_num(); ++i) {
        g->cache->etype_names[i] = g->g->schema().GetEdgeLabelName(i);
        g->cache->earrays[i].resize(g->g->edge_property_num(i));
        g->cache->earrs[i].resize(g->g->edge_property_num(i));
        g->cache->eprop_names[i].resize(g->g->edge_property_num(i));
        for (int j = 0; j < g->g->edge_property_num(i); ++j) {
            g->cache->eprop_names[i][j] = g->g->schema().GetEdgePropertyName(i, j);
            g->cache->earrays[i][j] = g->g->edge_data_table(i)->column(j)->chunk(0);
            g->cache->earrs[i][j] = _GetArrowArrayData(g->g->edge_data_table(i)->column(j)->chunk(0));
        } 
    }
}


typedef _GRIN_GRAPH_T::vertex_t _GRIN_VERTEX_T;                       

#ifdef GRIN_ENABLE_VERTEX_LIST
typedef _GRIN_GRAPH_T::vertices_t GRIN_VERTEX_LIST_T;
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST_ITERATOR
struct GRIN_VERTEX_LIST_ITERATOR_T {
  _GRIN_GRAPH_T::vid_t current;
  _GRIN_GRAPH_T::vid_t end;
};
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ITERATOR
struct GRIN_ADJACENT_LIST_ITERATOR_T {
    const _GRIN_GRAPH_T::nbr_unit_t* current;
    const _GRIN_GRAPH_T::nbr_unit_t* end;
    _GRIN_GRAPH_T::vid_t vid;
    int dir;
    unsigned etype;
};
#endif

#ifdef GRIN_ENABLE_GRAPH_PARTITION
struct GRIN_PARTITIONED_GRAPH_T {
  vineyard::Client client;
  std::string ipc_socket;
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
    auto list_array = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(array);
    return reinterpret_cast<const void*>(
        static_cast<const float*>(std::dynamic_pointer_cast<arrow::FloatArray>(list_array->values())->raw_values()) + offset * list_array->list_type()->list_size());
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
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
typedef unsigned GRIN_EDGE_TYPE_T;
typedef std::vector<unsigned> GRIN_EDGE_TYPE_LIST_T;
typedef unsigned long long int GRIN_EDGE_PROPERTY_T;
typedef std::vector<GRIN_EDGE_PROPERTY_T> GRIN_EDGE_PROPERTY_LIST_T;
#endif

#ifdef GRIN_ENABLE_ROW
typedef std::vector<const void*> GRIN_ROW_T;
#endif

#endif  // GRIN_SRC_PREDEFINE_H_
