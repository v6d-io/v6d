/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

#ifndef MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_TYPES_H_
#define MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_TYPES_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "libcuckoo/cuckoohash_map.hh"
#include "powturbo/include/ic.h"

#include "grape/config.h"
#include "grape/utils/vertex_array.h"

#include "basic/ds/arrow.h"
#include "basic/ds/hashmap.h"
#include "common/util/arrow.h"

// batching varint decoding for edge list
#ifndef VARINT_ENCODING_BATCH_SIZE
#define VARINT_ENCODING_BATCH_SIZE 16
#endif

namespace vineyard {

template <typename Key, typename Value>
using concurrent_map_t =
    libcuckoo::cuckoohash_map<Key, Value, prime_number_hash_wy<Key>>;

template <typename Key>
using concurrent_set_t = concurrent_map_t<Key, bool>;

using fid_t = grape::fid_t;

template <typename T>
struct InternalType {
  using type = T;
  using vineyard_array_type = vineyard::NumericArray<T>;
  using vineyard_builder_type = vineyard::NumericArrayBuilder<T>;
};

template <>
struct InternalType<std::string> {
  using type = arrow_string_view;
  using vineyard_array_type = vineyard::LargeStringArray;
  using vineyard_builder_type = vineyard::LargeStringArrayBuilder;
};

template <>
struct InternalType<arrow_string_view> {
  using type = arrow_string_view;
  using vineyard_array_type = vineyard::LargeStringArray;
  using vineyard_builder_type = vineyard::LargeStringArrayBuilder;
};

namespace property_graph_types {

using OID_TYPE = int64_t;
using EID_TYPE = uint64_t;
using VID_TYPE = uint64_t;
// using VID_TYPE = uint32_t;

using PROP_ID_TYPE = int;
using LABEL_ID_TYPE = int;

}  // namespace property_graph_types

// Hardcoded the max vertex label num to 128
constexpr int MAX_VERTEX_LABEL_NUM = 128;

static inline int num_to_bitwidth(int num) {
  if (num <= 2) {
    return 1;
  }
  int max = num - 1;
  int width = 0;
  while (max) {
    ++width;
    max >>= 1;
  }
  return width;
}

/**
 * @brief IdParser is designed for parsing the IDs that associated with property
 * graphs
 *
 * @tparam ID_TYPE
 */
template <typename ID_TYPE>
class IdParser {
  using LabelIDT = int;  // LABEL_ID_TYPE

 public:
  IdParser() {}
  ~IdParser() {}

  void Init(fid_t fnum, LabelIDT label_num) {
    CHECK_LE(label_num, MAX_VERTEX_LABEL_NUM);
    int fid_width = num_to_bitwidth(fnum);
    fid_offset_ = (sizeof(ID_TYPE) * 8) - fid_width;
    int label_width = num_to_bitwidth(MAX_VERTEX_LABEL_NUM);
    label_id_offset_ = fid_offset_ - label_width;
    fid_mask_ = ((((ID_TYPE) 1) << fid_width) - (ID_TYPE) 1) << fid_offset_;
    lid_mask_ = (((ID_TYPE) 1) << fid_offset_) - ((ID_TYPE) 1);
    label_id_mask_ = ((((ID_TYPE) 1) << label_width) - (ID_TYPE) 1)
                     << label_id_offset_;
    offset_mask_ = (((ID_TYPE) 1) << label_id_offset_) - (ID_TYPE) 1;
  }

  fid_t GetFid(ID_TYPE v) const { return (v >> fid_offset_); }

  LabelIDT GetLabelId(ID_TYPE v) const {
    return (v & label_id_mask_) >> label_id_offset_;
  }

  int64_t GetOffset(ID_TYPE v) const { return (v & offset_mask_); }

  ID_TYPE GetLid(ID_TYPE v) const { return v & lid_mask_; }

  /**
   * @brief Generate the LID
   */
  ID_TYPE GenerateId(LabelIDT label, int64_t offset) const {
    return (((ID_TYPE) offset) & offset_mask_) |
           ((((ID_TYPE) label) << label_id_offset_) & label_id_mask_);
  }

  /**
   * @brief Generate the GID
   */
  ID_TYPE GenerateId(fid_t fid, LabelIDT label, int64_t offset) const {
    return (((ID_TYPE) offset) & offset_mask_) |
           ((((ID_TYPE) label) << label_id_offset_) & label_id_mask_) |
           ((((ID_TYPE) fid) << fid_offset_) & fid_mask_);
  }

  ID_TYPE offset_mask() const { return offset_mask_; }

 private:
  int fid_offset_;
  int label_id_offset_;
  ID_TYPE fid_mask_;
  ID_TYPE lid_mask_;
  ID_TYPE label_id_mask_;
  ID_TYPE offset_mask_;
};

namespace property_graph_utils {

template <typename VID_T, typename EID_T>
struct NbrUnit {
  using vid_t = VID_T;
  using eid_t = EID_T;

  VID_T vid;
  EID_T eid;
  NbrUnit() = default;
  NbrUnit(VID_T v, EID_T e) : vid(v), eid(e) {}

  grape::Vertex<VID_T> get_neighbor() const {
    return grape::Vertex<VID_T>(vid);
  }
};

template <typename VID_T>
using NbrUnitDefault = NbrUnit<VID_T, property_graph_types::EID_TYPE>;

template <typename DATA_T, typename NBR_T>
class EdgeDataColumn {
 public:
  EdgeDataColumn() = default;

  explicit EdgeDataColumn(std::shared_ptr<arrow::Array> array) {
    if (array->type()->Equals(
            vineyard::ConvertToArrowType<DATA_T>::TypeValue())) {
      data_ = std::dynamic_pointer_cast<ArrowArrayType<DATA_T>>(array)
                  ->raw_values();
    } else {
      data_ = NULL;
    }
  }

  const DATA_T& operator[](const NBR_T& nbr) const { return data_[nbr.eid]; }

  const DATA_T& operator[](const typename NBR_T::eid_t& eid) const {
    return data_[eid];
  }

 private:
  const DATA_T* data_;
};

template <typename NBR_T>
class EdgeDataColumn<std::string, NBR_T> {
 public:
  EdgeDataColumn() = default;

  explicit EdgeDataColumn(std::shared_ptr<arrow::Array> array) {
    if (array->type()->Equals(arrow::large_utf8())) {
      array_ = std::dynamic_pointer_cast<arrow::LargeStringArray>(array);
    } else {
      array_ = nullptr;
    }
  }

  std::string operator[](const NBR_T& nbr) const {
    return array_->GetView(nbr.eid);
  }

  std::string operator[](const typename NBR_T::eid_t& eid) const {
    return array_->GetView(eid);
  }

 private:
  std::shared_ptr<arrow::LargeStringArray> array_;
};

template <typename DATA_T, typename VID_T>
using EdgeDataColumnDefault = EdgeDataColumn<DATA_T, NbrUnitDefault<VID_T>>;

template <typename DATA_T, typename VID_T>
class VertexDataColumn {
 public:
  VertexDataColumn(grape::VertexRange<VID_T> range,
                   std::shared_ptr<arrow::Array> array)
      : range_(range) {
    if (array->type()->Equals(
            vineyard::ConvertToArrowType<DATA_T>::TypeValue())) {
      data_ = std::dynamic_pointer_cast<ArrowArrayType<DATA_T>>(array)
                  ->raw_values() -
              static_cast<ptrdiff_t>(range.begin().GetValue());
    } else {
      data_ = NULL;
    }
  }

  explicit VertexDataColumn(grape::VertexRange<VID_T> range) : range_(range) {
    CHECK_EQ(range.size(), 0);
    data_ = nullptr;
  }

  const DATA_T& operator[](const grape::Vertex<VID_T>& v) const {
    return data_[v.GetValue()];
  }

 private:
  const DATA_T* data_;
  grape::VertexRange<VID_T> range_;
};

template <typename VID_T>
class VertexDataColumn<std::string, VID_T> {
 public:
  VertexDataColumn(grape::VertexRange<VID_T> range,
                   std::shared_ptr<arrow::Array> array)
      : range_(range) {
    if (array->type()->Equals(arrow::large_utf8())) {
      array_ = std::dynamic_pointer_cast<arrow::LargeStringArray>(array);
    } else {
      array_ = nullptr;
    }
  }

  explicit VertexDataColumn(grape::VertexRange<VID_T> range) : range_(range) {
    CHECK_EQ(range.size(), 0);
    array_ = nullptr;
  }

  std::string operator[](const grape::Vertex<VID_T>& v) const {
    return array_->GetView(v.GetValue() - range_.begin().GetValue());
  }

 private:
  grape::VertexRange<VID_T> range_;
  std::shared_ptr<arrow::LargeStringArray> array_;
};

template <typename T>
struct ValueGetter {
  inline static T Value(const void* data, int64_t offset) {
    return reinterpret_cast<const T*>(data)[offset];
  }
};

template <>
struct ValueGetter<std::string> {
  inline static std::string Value(const void* data, int64_t offset) {
    return std::string(
        reinterpret_cast<const arrow::LargeStringArray*>(data)->GetView(
            offset));
  }
};

template <typename VID_T, typename EID_T>
struct Nbr {
 private:
  using vid_t = VID_T;
  using eid_t = EID_T;
  using prop_id_t = property_graph_types::PROP_ID_TYPE;

 public:
  Nbr() : nbr_(NULL), edata_arrays_(nullptr) {}
  Nbr(const NbrUnit<VID_T, EID_T>* nbr, const void** edata_arrays)
      : nbr_(nbr), edata_arrays_(edata_arrays) {}
  Nbr(const Nbr& rhs) : nbr_(rhs.nbr_), edata_arrays_(rhs.edata_arrays_) {}
  Nbr(Nbr&& rhs)
      : nbr_(std::move(rhs.nbr_)), edata_arrays_(rhs.edata_arrays_) {}

  Nbr& operator=(const Nbr& rhs) {
    nbr_ = rhs.nbr_;
    edata_arrays_ = rhs.edata_arrays_;
    return *this;
  }

  Nbr& operator=(Nbr&& rhs) {
    nbr_ = std::move(rhs.nbr_);
    edata_arrays_ = std::move(rhs.edata_arrays_);
    return *this;
  }

  grape::Vertex<VID_T> neighbor() const {
    return grape::Vertex<VID_T>(nbr_->vid);
  }

  grape::Vertex<VID_T> get_neighbor() const {
    return grape::Vertex<VID_T>(nbr_->vid);
  }

  EID_T edge_id() const { return nbr_->eid; }

  template <typename T>
  T get_data(prop_id_t prop_id) const {
    return ValueGetter<T>::Value(edata_arrays_[prop_id], nbr_->eid);
  }

  std::string get_str(prop_id_t prop_id) const {
    return ValueGetter<std::string>::Value(edata_arrays_[prop_id], nbr_->eid);
  }

  double get_double(prop_id_t prop_id) const {
    return ValueGetter<double>::Value(edata_arrays_[prop_id], nbr_->eid);
  }

  int64_t get_int(prop_id_t prop_id) const {
    return ValueGetter<int64_t>::Value(edata_arrays_[prop_id], nbr_->eid);
  }

  inline const Nbr& operator++() const {
    ++nbr_;
    return *this;
  }

  inline Nbr operator++(int) const {
    Nbr ret(*this);
    ++(*this);
    return ret;
  }

  inline const Nbr& operator--() const {
    --nbr_;
    return *this;
  }

  inline Nbr operator--(int) const {
    Nbr ret(*this);
    --(*this);
    return ret;
  }

  inline bool operator==(const Nbr& rhs) const { return nbr_ == rhs.nbr_; }
  inline bool operator!=(const Nbr& rhs) const { return nbr_ != rhs.nbr_; }

  inline bool operator<(const Nbr& rhs) const { return nbr_ < rhs.nbr_; }

  inline const Nbr& operator*() const { return *this; }

 private:
  const mutable NbrUnit<VID_T, EID_T>* nbr_;
  const void** edata_arrays_;
};

template <typename VID_T, typename EID_T>
struct CompactNbr {
 private:
  using vid_t = VID_T;
  using eid_t = EID_T;
  using prop_id_t = property_graph_types::PROP_ID_TYPE;

  CompactNbr() {}

 public:
  CompactNbr(const CompactNbr& rhs)
      : ptr_(rhs.ptr_),
        next_(rhs.next_),
        size_(rhs.size_),
        edata_arrays_(rhs.edata_arrays_),
        data_(rhs.data_),
        current_(rhs.current_) {}
  CompactNbr(CompactNbr&& rhs)
      : ptr_(rhs.ptr_),
        next_(rhs.next_),
        size_(rhs.size_),
        edata_arrays_(rhs.edata_arrays_),
        data_(rhs.data_),
        current_(rhs.current_) {}
  CompactNbr(const uint8_t* ptr, const size_t size, const void** edata_arrays)
      : ptr_(ptr), next_(ptr), size_(size), edata_arrays_(edata_arrays) {
    decode();
  }

  CompactNbr& operator=(const CompactNbr& rhs) {
    ptr_ = rhs.ptr_;
    next_ = rhs.next_;
    size_ = rhs.size_;
    edata_arrays_ = rhs.edata_arrays_;
    data_ = rhs.data_;
    current_ = rhs.current_;
    return *this;
  }

  CompactNbr& operator=(CompactNbr&& rhs) {
    ptr_ = rhs.ptr_;
    next_ = rhs.next_;
    size_ = rhs.size_;
    edata_arrays_ = std::move(rhs.edata_arrays_);
    data_ = rhs.data_;
    current_ = rhs.current_;
    return *this;
  }

  grape::Vertex<VID_T> neighbor() const {
    return grape::Vertex<VID_T>(data_[current_ % batch_size].vid);
  }

  grape::Vertex<VID_T> get_neighbor() const {
    return grape::Vertex<VID_T>(data_[current_ % batch_size].vid);
  }

  EID_T edge_id() const { return data_[current_ % batch_size].eid; }

  template <typename T>
  T get_data(prop_id_t prop_id) const {
    return ValueGetter<T>::Value(edata_arrays_[prop_id], edge_id());
  }

  std::string get_str(prop_id_t prop_id) const {
    return ValueGetter<std::string>::Value(edata_arrays_[prop_id], edge_id());
  }

  double get_double(prop_id_t prop_id) const {
    return ValueGetter<double>::Value(edata_arrays_[prop_id], edge_id());
  }

  int64_t get_int(prop_id_t prop_id) const {
    return ValueGetter<int64_t>::Value(edata_arrays_[prop_id], edge_id());
  }

  inline const CompactNbr& operator++() const {
    VID_T prev_vid = data_[current_ % batch_size].vid;
    current_ += 1;
    decode();
    data_[current_ % batch_size].vid += prev_vid;
    return *this;
  }

  inline CompactNbr operator++(int) const {
    CompactNbr ret(*this);
    ++(*this);
    return ret;
  }

  inline bool operator==(const CompactNbr& rhs) const {
    return ptr_ == rhs.ptr_;
  }
  inline bool operator!=(const CompactNbr& rhs) const {
    return ptr_ != rhs.ptr_;
  }

  inline bool operator<(const CompactNbr& rhs) const { return ptr_ < rhs.ptr_; }

  inline const CompactNbr& operator*() const { return *this; }

 private:
  inline void decode() const {
    if (likely((current_ % batch_size != 0) || current_ >= size_)) {
      if (unlikely(current_ == size_)) {
        ptr_ = next_;
      }
      return;
    }
    ptr_ = next_;
    size_t n =
        (current_ + batch_size) < size_ ? batch_size : (size_ - current_);
    next_ = v8dec32(const_cast<unsigned char*>(
                        reinterpret_cast<const unsigned char*>(next_)),
                    n * element_size, reinterpret_cast<uint32_t*>(data_));
  }

  static constexpr size_t element_size =
      sizeof(property_graph_utils::NbrUnit<VID_T, EID_T>) / sizeof(uint32_t);
  static constexpr size_t batch_size = VARINT_ENCODING_BATCH_SIZE;

  mutable const uint8_t *ptr_, *next_ = nullptr;
  mutable size_t size_ = 0;
  const void** edata_arrays_;

  mutable NbrUnit<VID_T, EID_T> data_[batch_size];
  mutable size_t current_ = 0;
};

template <typename VID_T>
using NbrDefault = Nbr<VID_T, property_graph_types::EID_TYPE>;

template <typename VID_T, typename EID_T>
struct OffsetNbr {
 private:
  using prop_id_t = property_graph_types::PROP_ID_TYPE;

 public:
  OffsetNbr() : nbr_(NULL), edata_table_(nullptr) {}

  OffsetNbr(const NbrUnit<VID_T, EID_T>* nbr,
            std::shared_ptr<arrow::Table> edata_table,
            const vineyard::IdParser<VID_T>* vid_parser, const VID_T* ivnums)
      : nbr_(nbr),
        edata_table_(std::move(edata_table)),
        vid_parser_(vid_parser),
        ivnums_(ivnums) {}

  OffsetNbr(const OffsetNbr& rhs)
      : nbr_(rhs.nbr_),
        edata_table_(rhs.edata_table_),
        vid_parser_(rhs.vid_parser_),
        ivnums_(rhs.ivnums_) {}

  OffsetNbr(OffsetNbr&& rhs)
      : nbr_(std::move(rhs.nbr_)),
        edata_table_(std::move(rhs.edata_table_)),
        vid_parser_(rhs.vid_parser_),
        ivnums_(rhs.ivnums_) {}

  OffsetNbr& operator=(const OffsetNbr& rhs) {
    nbr_ = rhs.nbr_;
    edata_table_ = rhs.edata_table_;
    vid_parser_ = rhs.vid_parser_;
    ivnums_ = rhs.ivnums_;
  }

  OffsetNbr& operator=(OffsetNbr&& rhs) {
    nbr_ = std::move(rhs.nbr_);
    edata_table_ = std::move(rhs.edata_table_);
    vid_parser_ = rhs.vid_parser_;
    ivnums_ = rhs.ivnums_;
  }

  grape::Vertex<VID_T> neighbor() const {
    auto offset_mask = vid_parser_->offset_mask();
    auto offset = nbr_->vid & offset_mask;
    auto v_label = vid_parser_->GetLabelId(nbr_->vid);
    auto ivnum = ivnums_[v_label];
    auto vid =
        offset < (VID_T) ivnum
            ? nbr_->vid
            : ((nbr_->vid & ~offset_mask) | (ivnum + offset_mask - offset));

    return grape::Vertex<VID_T>(vid);
  }

  EID_T edge_id() const { return nbr_->eid; }

  template <typename T>
  T get_data(prop_id_t prop_id) const {
    // the finalized vtables are guaranteed to have been concatenate
    return std::dynamic_pointer_cast<ArrowArrayType<T>>(
               edata_table_->column(prop_id)->chunk(0))
        ->Value(nbr_->eid);
  }

  inline const OffsetNbr& operator++() const {
    ++nbr_;
    return *this;
  }

  inline OffsetNbr operator++(int) const {
    OffsetNbr ret(*this);
    ++ret;
    return ret;
  }

  inline const OffsetNbr& operator--() const {
    --nbr_;
    return *this;
  }

  inline OffsetNbr operator--(int) const {
    OffsetNbr ret(*this);
    --ret;
    return ret;
  }

  inline bool operator==(const OffsetNbr& rhs) const {
    return nbr_ == rhs.nbr_;
  }
  inline bool operator!=(const OffsetNbr& rhs) const {
    return nbr_ != rhs.nbr_;
  }

  inline bool operator<(const OffsetNbr& rhs) const { return nbr_ < rhs.nbr_; }

  inline const OffsetNbr& operator*() const { return *this; }

 private:
  const mutable NbrUnit<VID_T, EID_T>* nbr_;
  std::shared_ptr<arrow::Table> edata_table_;
  const vineyard::IdParser<VID_T>* vid_parser_;
  const VID_T* ivnums_;
};

template <typename VID_T, typename EID_T>
class RawAdjList {
 public:
  RawAdjList() : begin_(NULL), end_(NULL) {}
  RawAdjList(const NbrUnit<VID_T, EID_T>* begin,
             const NbrUnit<VID_T, EID_T>* end)
      : begin_(begin), end_(end) {}

  inline const NbrUnit<VID_T, EID_T>* begin() const { return begin_; }

  inline const NbrUnit<VID_T, EID_T>* end() const { return end_; }

  inline size_t Size() const { return end_ - begin_; }

  inline bool Empty() const { return end_ == begin_; }

  inline bool NotEmpty() const { return end_ != begin_; }

  size_t size() const { return end_ - begin_; }

 private:
  const NbrUnit<VID_T, EID_T>* begin_;
  const NbrUnit<VID_T, EID_T>* end_;
};

template <typename VID_T>
using RawAdjListDefault = RawAdjList<VID_T, property_graph_types::EID_TYPE>;

template <typename VID_T, typename EID_T>
class AdjList {
 public:
  AdjList() : begin_(NULL), end_(NULL), edata_arrays_(nullptr) {}
  AdjList(const NbrUnit<VID_T, EID_T>* begin, const NbrUnit<VID_T, EID_T>* end,
          const void** edata_arrays)
      : begin_(begin), end_(end), edata_arrays_(edata_arrays) {}

  inline Nbr<VID_T, EID_T> begin() const {
    return Nbr<VID_T, EID_T>(begin_, edata_arrays_);
  }

  inline Nbr<VID_T, EID_T> end() const {
    return Nbr<VID_T, EID_T>(end_, edata_arrays_);
  }

  inline size_t Size() const { return end_ - begin_; }

  inline bool Empty() const { return end_ == begin_; }

  inline bool NotEmpty() const { return end_ != begin_; }

  size_t size() const { return end_ - begin_; }

  inline const NbrUnit<VID_T, EID_T>* begin_unit() const { return begin_; }

  inline const NbrUnit<VID_T, EID_T>* end_unit() const { return end_; }

 private:
  const NbrUnit<VID_T, EID_T>* begin_;
  const NbrUnit<VID_T, EID_T>* end_;
  const void** edata_arrays_;
};

template <typename VID_T, typename EID_T>
class CompactAdjList {
 public:
  CompactAdjList()
      : begin_ptr_(nullptr),
        end_ptr_(nullptr),
        size_(0),
        edata_arrays_(nullptr) {}

  CompactAdjList(const CompactAdjList& nbrs) {
    begin_ptr_ = nbrs.begin_ptr_;
    end_ptr_ = nbrs.end_ptr_;
    size_ = nbrs.size_;
    edata_arrays_ = nbrs.edata_arrays_;
  }

  CompactAdjList(CompactAdjList&& nbrs) {
    begin_ptr_ = nbrs.begin_ptr_;
    end_ptr_ = nbrs.end_ptr_;
    size_ = nbrs.size_;
    edata_arrays_ = nbrs.edata_arrays_;
  }

  CompactAdjList& operator=(const CompactAdjList& rhs) {
    begin_ptr_ = rhs.begin_ptr_;
    end_ptr_ = rhs.end_ptr_;
    size_ = rhs.size_;
    edata_arrays_ = rhs.edata_arrays_;
    return *this;
  }

  CompactAdjList& operator=(CompactAdjList&& rhs) {
    begin_ptr_ = rhs.begin_ptr_;
    end_ptr_ = rhs.end_ptr_;
    size_ = rhs.size_;
    edata_arrays_ = rhs.edata_arrays_;
    return *this;
  }

  CompactAdjList(const uint8_t* begin_ptr, const uint8_t* end_ptr,
                 const size_t size, const void** edata_arrays)
      : begin_ptr_(begin_ptr),
        end_ptr_(end_ptr),
        size_(size),
        edata_arrays_(edata_arrays) {}

  inline CompactNbr<VID_T, EID_T> begin() const {
    return CompactNbr<VID_T, EID_T>(begin_ptr_, size_, edata_arrays_);
  }

  inline CompactNbr<VID_T, EID_T> end() const {
    return CompactNbr<VID_T, EID_T>(end_ptr_, 0, edata_arrays_);
  }

  inline size_t Size() const { return size_; }

  inline bool Empty() const { return size_ == 0; }

  inline bool NotEmpty() const { return size_ != 0; }

  size_t size() const { return size_; }

 private:
  const uint8_t* begin_ptr_;
  const uint8_t* end_ptr_;
  size_t size_ = 0;

  const void** edata_arrays_;
};

template <typename VID_T>
using AdjListDefault = AdjList<VID_T, property_graph_types::EID_TYPE>;

/**
 * OffsetAdjList will offset the outer vertices' lid, makes it between "ivnum"
 * and "tvnum" instead of "ivnum ~ tvnum - outer vertex index"
 *
 * @tparam VID_T
 * @tparam EID_T
 */
template <typename VID_T, typename EID_T>
class OffsetAdjList {
 public:
  OffsetAdjList()
      : begin_(NULL),
        end_(NULL),
        edata_table_(nullptr),
        vid_parser_(nullptr),
        ivnums_(nullptr) {}

  OffsetAdjList(const NbrUnit<VID_T, EID_T>* begin,
                const NbrUnit<VID_T, EID_T>* end,
                std::shared_ptr<arrow::Table> edata_table,
                const vineyard::IdParser<VID_T>* vid_parser,
                const VID_T* ivnums)
      : begin_(begin),
        end_(end),
        edata_table_(std::move(edata_table)),
        vid_parser_(vid_parser),
        ivnums_(ivnums) {}

  inline OffsetNbr<VID_T, EID_T> begin() const {
    return OffsetNbr<VID_T, EID_T>(begin_, edata_table_, vid_parser_, ivnums_);
  }

  inline OffsetNbr<VID_T, EID_T> end() const {
    return OffsetNbr<VID_T, EID_T>(end_, edata_table_, vid_parser_, ivnums_);
  }

  inline size_t Size() const { return end_ - begin_; }

  inline bool Empty() const { return end_ == begin_; }

  inline bool NotEmpty() const { return end_ != begin_; }

  size_t size() const { return end_ - begin_; }

 private:
  const NbrUnit<VID_T, EID_T>* begin_;
  const NbrUnit<VID_T, EID_T>* end_;
  std::shared_ptr<arrow::Table> edata_table_;
  const vineyard::IdParser<VID_T>* vid_parser_;
  const VID_T* ivnums_;
};

}  // namespace property_graph_utils

inline std::string generate_type_name(
    const std::string& template_name,
    const std::vector<std::string>& template_params) {
  if (template_params.empty()) {
    return template_name + "<>";
  }
  std::string ret = template_name;
  ret += ("<" + template_params[0]);
  for (size_t i = 1; i < template_params.size(); ++i) {
    ret += ("," + template_params[i]);
  }
  ret += ">";
  return ret;
}

class EmptyArray {
  using value_type = grape::EmptyType;

 public:
  explicit EmptyArray(int64_t size) : size_(size) {}

  value_type Value(int64_t offset) { return value_type(); }
  int64_t length() const { return size_; }

  const value_type* raw_values() { return NULL; }

 private:
  int64_t size_;
};

template <typename DATA_T>
typename std::enable_if<std::is_same<DATA_T, grape::EmptyType>::value,
                        std::shared_ptr<ArrowArrayType<DATA_T>>>::type
assign_array(std::shared_ptr<arrow::Array>, int64_t length) {
  return std::make_shared<EmptyArray>(length);
}

template <typename DATA_T>
typename std::enable_if<!std::is_same<DATA_T, grape::EmptyType>::value,
                        std::shared_ptr<ArrowArrayType<DATA_T>>>::type
assign_array(std::shared_ptr<arrow::Array> array, int64_t) {
  return std::dynamic_pointer_cast<ArrowArrayType<DATA_T>>(array);
}

template <>
struct ConvertToArrowType<::grape::EmptyType> {
  using ArrayType = EmptyArray;
  static std::shared_ptr<arrow::DataType> TypeValue() { return arrow::null(); }
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_TYPES_H_
