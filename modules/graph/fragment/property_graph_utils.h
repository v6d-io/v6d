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

#ifndef MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_H_
#define MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/builder.h"
#include "boost/leaf/all.hpp"

#include "grape/utils/vertex_array.h"
#include "grape/worker/comm_spec.h"

#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/mpi_utils.h"

namespace vineyard {

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
    int fid_width = num_to_bitwidth(fnum);
    fid_offset_ = (sizeof(ID_TYPE) * 8) - fid_width;
    int label_width = num_to_bitwidth(label_num);
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
  VID_T vid;
  EID_T eid;
  NbrUnit() = default;
  NbrUnit(VID_T v, EID_T e) : vid(v), eid(e) {}

  grape::Vertex<VID_T> get_neighbor() const {
    return grape::Vertex<VID_T>(vid);
  }
};

template <typename DATA_T, typename NBR_T>
class EdgeDataColumn {
 public:
  EdgeDataColumn() = default;

  explicit EdgeDataColumn(std::shared_ptr<arrow::Array> array) {
    if (array->type()->Equals(
            vineyard::ConvertToArrowType<DATA_T>::TypeValue())) {
      data_ =
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<DATA_T>::ArrayType>(array)
              ->raw_values();
    } else {
      data_ = NULL;
    }
  }

  const DATA_T& operator[](const NBR_T& nbr) const { return data_[nbr.eid]; }

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

 private:
  std::shared_ptr<arrow::LargeStringArray> array_;
};

template <typename DATA_T, typename VID_T>
class VertexDataColumn {
 public:
  VertexDataColumn(grape::VertexRange<VID_T> range,
                   std::shared_ptr<arrow::Array> array)
      : range_(range) {
    if (array->type()->Equals(
            vineyard::ConvertToArrowType<DATA_T>::TypeValue())) {
      data_ =
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<DATA_T>::ArrayType>(array) -
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
    ++ret;
    return ret;
  }

  inline const Nbr& operator--() const {
    --nbr_;
    return *this;
  }

  inline Nbr operator--(int) const {
    Nbr ret(*this);
    --ret;
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
    return std::dynamic_pointer_cast<
               typename vineyard::ConvertToArrowType<T>::ArrayType>(
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

inline std::string arrow_type_to_type_name(
    std::shared_ptr<arrow::DataType> type) {
  if (vineyard::ConvertToArrowType<bool>::TypeValue()->Equals(type)) {
    return type_name<bool>();
  } else if (vineyard::ConvertToArrowType<int8_t>::TypeValue()->Equals(type)) {
    return type_name<int8_t>();
  } else if (vineyard::ConvertToArrowType<uint8_t>::TypeValue()->Equals(type)) {
    return type_name<uint8_t>();
  } else if (vineyard::ConvertToArrowType<int16_t>::TypeValue()->Equals(type)) {
    return type_name<int16_t>();
  } else if (vineyard::ConvertToArrowType<uint16_t>::TypeValue()->Equals(
                 type)) {
    return type_name<uint16_t>();
  } else if (vineyard::ConvertToArrowType<int32_t>::TypeValue()->Equals(type)) {
    return type_name<int32_t>();
  } else if (vineyard::ConvertToArrowType<uint32_t>::TypeValue()->Equals(
                 type)) {
    return type_name<uint32_t>();
  } else if (vineyard::ConvertToArrowType<int64_t>::TypeValue()->Equals(type)) {
    return type_name<int64_t>();
  } else if (vineyard::ConvertToArrowType<uint64_t>::TypeValue()->Equals(
                 type)) {
    return type_name<uint64_t>();
  } else if (vineyard::ConvertToArrowType<float>::TypeValue()->Equals(type)) {
    return type_name<float>();
  } else if (vineyard::ConvertToArrowType<double>::TypeValue()->Equals(type)) {
    return type_name<double>();
  } else if (vineyard::ConvertToArrowType<std::string>::TypeValue()->Equals(
                 type)) {
    return type_name<std::string>();
  } else {
    return "";
  }
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
                        std::shared_ptr<typename vineyard::ConvertToArrowType<
                            DATA_T>::ArrayType>>::type
assign_array(std::shared_ptr<arrow::Array>, int64_t length) {
  return std::make_shared<EmptyArray>(length);
}

template <typename DATA_T>
typename std::enable_if<!std::is_same<DATA_T, grape::EmptyType>::value,
                        std::shared_ptr<typename vineyard::ConvertToArrowType<
                            DATA_T>::ArrayType>>::type
assign_array(std::shared_ptr<arrow::Array> array, int64_t) {
  return std::dynamic_pointer_cast<
      typename vineyard::ConvertToArrowType<DATA_T>::ArrayType>(array);
}

template <>
struct ConvertToArrowType<::grape::EmptyType> {
  using ArrayType = EmptyArray;
  static std::shared_ptr<arrow::DataType> TypeValue() { return arrow::null(); }
};

inline boost::leaf::result<std::shared_ptr<arrow::Schema>> TypeLoosen(
    const std::vector<std::shared_ptr<arrow::Schema>>& schemas) {
  size_t field_num = 0;
  for (const auto& schema : schemas) {
    if (schema != nullptr) {
      field_num = schema->num_fields();
      break;
    }
  }
  if (field_num == 0) {
    RETURN_GS_ERROR(ErrorCode::kInvalidOperationError, "Every schema is empty");
  }
  // Perform type lossen.
  // timestamp -> int64 -> double -> utf8   binary (not supported)

  // Timestamp value are stored as as number of seconds, milliseconds,
  // microseconds or nanoseconds since UNIX epoch.
  // CSV reader can only produce timestamp in seconds.
  std::vector<std::vector<std::shared_ptr<arrow::Field>>> fields(field_num);
  for (size_t i = 0; i < field_num; ++i) {
    for (const auto& schema : schemas) {
      if (schema != nullptr) {
        fields[i].push_back(schema->field(i));
      }
    }
  }
  std::vector<std::shared_ptr<arrow::Field>> lossen_fields(field_num);

  for (size_t i = 0; i < field_num; ++i) {
    lossen_fields[i] = fields[i][0];
    if (fields[i][0]->type() == arrow::null()) {
      continue;
    }
    auto res = fields[i][0]->type();
    if (res->Equals(arrow::timestamp(arrow::TimeUnit::SECOND))) {
      res = arrow::int64();
    }
    if (res->Equals(arrow::int64())) {
      for (size_t j = 1; j < fields[i].size(); ++j) {
        if (fields[i][j]->type()->Equals(arrow::float64())) {
          res = arrow::float64();
        }
      }
    }
    if (res->Equals(arrow::float64())) {
      for (size_t j = 1; j < fields[i].size(); ++j) {
        if (fields[i][j]->type()->Equals(arrow::utf8())) {
          res = arrow::utf8();
        }
      }
    }
    if (res->Equals(arrow::utf8())) {
      res = arrow::large_utf8();
    }
    lossen_fields[i] = lossen_fields[i]->WithType(res);
  }
  return std::make_shared<arrow::Schema>(lossen_fields);
}

inline boost::leaf::result<std::shared_ptr<arrow::Array>> CastStringToBigString(
    const std::shared_ptr<arrow::Array>& in,
    const std::shared_ptr<arrow::DataType>& to_type) {
  auto array_data = in->data()->Copy();
  auto offset = array_data->buffers[1];
  using from_offset_type = typename arrow::StringArray::offset_type;
  using to_string_offset_type = typename arrow::LargeStringArray::offset_type;
  auto raw_value_offsets_ =
      offset == NULLPTR
          ? NULLPTR
          : reinterpret_cast<const from_offset_type*>(offset->data());
  std::vector<to_string_offset_type> to_offset(offset->size() /
                                               sizeof(from_offset_type));
  for (size_t i = 0; i < to_offset.size(); ++i) {
    to_offset[i] = raw_value_offsets_[i];
  }
  std::shared_ptr<arrow::Buffer> buffer;
  arrow::TypedBufferBuilder<to_string_offset_type> buffer_builder;
  buffer_builder.Append(to_offset.data(), to_offset.size());
  buffer_builder.Finish(&buffer);
  array_data->type = to_type;
  array_data->buffers[1] = buffer;
  auto out = arrow::MakeArray(array_data);
  ARROW_OK_OR_RAISE(out->ValidateFull());
  return out;
}

inline boost::leaf::result<std::shared_ptr<arrow::Array>> CastNullToOthers(
    const std::shared_ptr<arrow::Array>& in,
    const std::shared_ptr<arrow::DataType>& to_type) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  ARROW_OK_OR_RAISE(
      arrow::MakeBuilder(arrow::default_memory_pool(), to_type, &builder));
  ARROW_OK_OR_RAISE(builder->AppendNulls(in->length()));
  std::shared_ptr<arrow::Array> out;
  ARROW_OK_OR_RAISE(builder->Finish(&out));
  ARROW_OK_OR_RAISE(out->ValidateFull());
  return out;
}

inline boost::leaf::result<std::shared_ptr<arrow::Array>> GeneralCast(
    const std::shared_ptr<arrow::Array>& in,
    const std::shared_ptr<arrow::DataType>& to_type) {
  std::shared_ptr<arrow::Array> out;
  CHECK_ARROW_ERROR_AND_ASSIGN(out, arrow::compute::Cast(*in, to_type));
  return out;
}

inline boost::leaf::result<std::shared_ptr<arrow::Table>> CastTableToSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::shared_ptr<arrow::Schema>& schema) {
  if (table->schema()->Equals(schema)) {
    return table;
  }
  CHECK_OR_RAISE(table->num_columns() == schema->num_fields());
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
  for (int64_t i = 0; i < table->num_columns(); ++i) {
    auto col = table->column(i);
    if (!table->field(i)->type()->Equals(schema->field(i)->type())) {
      auto from_type = table->field(i)->type();
      auto to_type = schema->field(i)->type();
      std::vector<std::shared_ptr<arrow::Array>> chunks;
      for (int64_t j = 0; j < col->num_chunks(); ++j) {
        auto array = col->chunk(j);
        if (from_type->Equals(arrow::utf8()) &&
            to_type->Equals(arrow::large_utf8())) {
          BOOST_LEAF_AUTO(new_array, CastStringToBigString(array, to_type));
          chunks.push_back(new_array);
        } else if (from_type->Equals(arrow::null())) {
          BOOST_LEAF_AUTO(new_array, CastNullToOthers(array, to_type));
          chunks.push_back(new_array);
        } else if (arrow::compute::CanCast(*from_type, *to_type)) {
          BOOST_LEAF_AUTO(new_array, GeneralCast(array, to_type));
        } else {
          RETURN_GS_ERROR(ErrorCode::kDataTypeError,
                          "Unsupported cast: To type: " + to_type->ToString() +
                              "; Origin type: " + from_type->ToString());
        }
        VLOG(10) << "Cast " << from_type->ToString() << " To "
                 << to_type->ToString();
      }
      auto chunk_array = std::make_shared<arrow::ChunkedArray>(chunks, to_type);
      new_columns.push_back(chunk_array);
    } else {
      new_columns.push_back(col);
    }
  }
  return arrow::Table::Make(schema, new_columns);
}

// This method used when several workers is loading a file in parallel, each
// worker will read a chunk of the origin file into a arrow::Table.
// We may get different table schemas as some chunks may have zero rows
// or some chunks' data doesn't have any floating numbers, but others might
// have. We could use this method to gather their schemas, and find out most
// inclusive fields, construct a new schema and broadcast back. Note: We perform
// type loosen, timestamp -> int64, int64 -> double. double -> string (big
// string), and any type is prior to null.
inline boost::leaf::result<std::shared_ptr<arrow::Table>> SyncSchema(
    const std::shared_ptr<arrow::Table>& table,
    const grape::CommSpec& comm_spec) {
  std::shared_ptr<arrow::Schema> local_schema =
      table != nullptr ? table->schema() : nullptr;
  std::vector<std::shared_ptr<arrow::Schema>> schemas;

  GlobalAllGatherv(local_schema, schemas, comm_spec);
  BOOST_LEAF_AUTO(normalized_schema, TypeLoosen(schemas));

  if (table == nullptr) {
    std::shared_ptr<arrow::Table> table_out;
    VY_OK_OR_RAISE(
        vineyard::EmptyTableBuilder::Build(normalized_schema, table_out));
    return table_out;
  } else {
    return CastTableToSchema(table, normalized_schema);
  }
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_H_
