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

#include "basic/ds/arrow.h"  // NOLINT(build/include)

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"      // IWYU pragma: keep
#include "arrow/io/api.h"   // IWYU pragma: keep
#include "arrow/ipc/api.h"  // IWYU pragma: keep
#if defined(ARROW_VERSION) && ARROW_VERSION >= 7000000
#include "arrow/visit_type_inline.h"  // IWYU pragma: keep
#else
#include "arrow/visitor_inline.h"
#endif

#include "basic/ds/arrow_shim/concatenate.h"
#include "basic/ds/arrow_shim/memory_pool.h"
#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "common/util/macros.h"

namespace vineyard {

namespace detail {

// can be: array::Array, arrow::ChunkedArray
class ArrowArrayBuilderVisitor {
 public:
  ArrowArrayBuilderVisitor(Client& client,
                           const std::shared_ptr<arrow::ChunkedArray> array)
      : client_(client), array_(array) {}

  Status Visit(const arrow::NullType*) {
    builder_ = std::make_shared<NullArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::BooleanType*) {
    builder_ = std::make_shared<BooleanArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Int8Type*) {
    builder_ = std::make_shared<Int8Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::UInt8Type*) {
    builder_ = std::make_shared<UInt8Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Int16Type*) {
    builder_ = std::make_shared<Int16Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::UInt16Type*) {
    builder_ = std::make_shared<UInt16Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Int32Type*) {
    builder_ = std::make_shared<Int32Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::UInt32Type*) {
    builder_ = std::make_shared<UInt32Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Int64Type*) {
    builder_ = std::make_shared<Int64Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::UInt64Type*) {
    builder_ = std::make_shared<UInt64Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::FloatType*) {
    builder_ = std::make_shared<FloatBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::DoubleType*) {
    builder_ = std::make_shared<DoubleBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Date32Type*) {
    builder_ = std::make_shared<Date32Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Date64Type*) {
    builder_ = std::make_shared<Date64Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Time32Type*) {
    builder_ = std::make_shared<Time32Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::Time64Type*) {
    builder_ = std::make_shared<Time64Builder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::TimestampType*) {
    builder_ = std::make_shared<TimestampBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::StringType*) {
    builder_ = std::make_shared<StringArrayBuilder>(client_, array_);
    return Status::OK();
  }

#if defined(ARROW_VERSION) && ARROW_VERSION >= 15000000
  Status Visit(const arrow::StringViewType* type) {
    return Status::NotImplemented(
        "Type not implemented: " + std::to_string(type->id()) + ", " +
        type->ToString());
  }
#endif

  Status Visit(const arrow::LargeStringType*) {
    builder_ = std::make_shared<LargeStringArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::BinaryType*) {
    builder_ = std::make_shared<BinaryArrayBuilder>(client_, array_);
    return Status::OK();
  }

#if defined(ARROW_VERSION) && ARROW_VERSION >= 15000000
  Status Visit(const arrow::BinaryViewType* type) {
    return Status::NotImplemented(
        "Type not implemented: " + std::to_string(type->id()) + ", " +
        type->ToString());
  }
#endif

  Status Visit(const arrow::LargeBinaryType*) {
    builder_ = std::make_shared<LargeBinaryArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::FixedSizeBinaryType*) {
    builder_ = std::make_shared<FixedSizeBinaryArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::ListType*) {
    builder_ = std::make_shared<ListArrayBuilder>(client_, array_);
    return Status::OK();
  }

#if defined(ARROW_VERSION) && ARROW_VERSION >= 15000000
  Status Visit(const arrow::ListViewType* type) {
    return Status::NotImplemented(
        "Type not implemented: " + std::to_string(type->id()) + ", " +
        type->ToString());
  }
#endif

  Status Visit(const arrow::LargeListType*) {
    builder_ = std::make_shared<LargeListArrayBuilder>(client_, array_);
    return Status::OK();
  }

#if defined(ARROW_VERSION) && ARROW_VERSION >= 15000000
  Status Visit(const arrow::LargeListViewType* type) {
    return Status::NotImplemented(
        "Type not implemented: " + std::to_string(type->id()) + ", " +
        type->ToString());
  }
#endif

  Status Visit(const arrow::FixedSizeListType*) {
    builder_ = std::make_shared<FixedSizeListArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::DataType* type) {
    return Status::NotImplemented(
        "Type not implemented: " + std::to_string(type->id()) + ", " +
        type->ToString());
  }

  std::shared_ptr<ObjectBuilder> Builder() const { return this->builder_; }

 private:
  Client& client_;
  std::shared_ptr<arrow::ChunkedArray> array_;
  std::shared_ptr<ObjectBuilder> builder_;
};

#if defined(ARROW_VERSION) && ARROW_VERSION >= 11000000
#define VINEYARD_TYPE_ID_VISIT_INLINE(TYPE_CLASS)                     \
  case arrow::TYPE_CLASS##Type::type_id: {                            \
    const arrow::TYPE_CLASS##Type* concrete_ptr = NULLPTR;            \
    return visitor->Visit(concrete_ptr, std::forward<ARGS>(args)...); \
  }

template <typename VISITOR, typename... ARGS>
inline Status VineyardVisitTypeIdInline(arrow::Type::type id, VISITOR* visitor,
                                        ARGS&&... args) {
  switch (id) {
    ARROW_GENERATE_FOR_ALL_TYPES(VINEYARD_TYPE_ID_VISIT_INLINE);
  default:
    break;
  }
  return Status::NotImplemented("Type not implemented: " + std::to_string(id));
}
#undef VINEYARD_TYPE_ID_VISIT_INLINE
#else
#define VINEYARD_TYPE_ID_VISIT_INLINE(TYPE_CLASS)          \
  case arrow::TYPE_CLASS##Type::type_id: {                 \
    const arrow::TYPE_CLASS##Type* concrete_ptr = NULLPTR; \
    return visitor->Visit(concrete_ptr);                   \
  }

template <typename VISITOR>
inline Status VineyardVisitTypeIdInline(arrow::Type::type id,
                                        VISITOR* visitor) {
  switch (id) {
    ARROW_GENERATE_FOR_ALL_TYPES(VINEYARD_TYPE_ID_VISIT_INLINE);
  default:
    break;
  }
  return Status::NotImplemented("Type not implemented: " + std::to_string(id));
}
#undef VINEYARD_TYPE_ID_VISIT_INLINE
#endif

Status BuildArray(Client& client, const std::shared_ptr<arrow::Array> array,
                  std::shared_ptr<ObjectBuilder>& builder) {
  return BuildArray(client, std::make_shared<arrow::ChunkedArray>(array),
                    builder);
}

Status BuildArray(Client& client,
                  const std::shared_ptr<arrow::ChunkedArray> array,
                  std::shared_ptr<ObjectBuilder>& builder) {
  ArrowArrayBuilderVisitor visitor(client, array);
  RETURN_ON_ERROR(VineyardVisitTypeIdInline(array->type()->id(), &visitor));
  builder = visitor.Builder();
  return Status::OK();
}

std::shared_ptr<ObjectBuilder> BuildArray(Client& client,
                                          std::shared_ptr<arrow::Array> array) {
  std::shared_ptr<ObjectBuilder> builder;
  VINEYARD_CHECK_OK(BuildArray(client, array, builder));
  return builder;
}

std::shared_ptr<ObjectBuilder> BuildArray(
    Client& client, std::shared_ptr<arrow::ChunkedArray> array) {
  std::shared_ptr<ObjectBuilder> builder;
  VINEYARD_CHECK_OK(BuildArray(client, array, builder));
  return builder;
}

std::shared_ptr<arrow::Array> CastToArray(std::shared_ptr<Object> object) {
  if (auto arr = std::dynamic_pointer_cast<FixedSizeBinaryArray>(object)) {
    return arr->GetArray();
  }
  if (auto arr = std::dynamic_pointer_cast<StringArray>(object)) {
    return arr->GetArray();
  }
  if (auto arr = std::dynamic_pointer_cast<LargeStringArray>(object)) {
    return arr->GetArray();
  }
  if (auto arr = std::dynamic_pointer_cast<FixedSizeBinaryArray>(object)) {
    return arr->GetArray();
  }
  if (auto arr = std::dynamic_pointer_cast<NullArray>(object)) {
    return arr->GetArray();
  }
  if (auto arr = std::dynamic_pointer_cast<ArrowArray>(object)) {
    return arr->ToArray();
  }
  // Don't abort the program, the unresolvable array should be reported lazily.
  //
  // VINEYARD_ASSERT(nullptr != nullptr,
  //                 "Unsupported array type: " + object->meta().GetTypeName());
  return nullptr;
}

}  // namespace detail

#ifndef TAKE_BUFFER_AND_APPLY
#define TAKE_BUFFER_AND_APPLY(builder, FN, pool, buffer) \
  do {                                                   \
    std::unique_ptr<BlobWriter> blob;                    \
    RETURN_ON_ERROR(pool.Take(buffer, blob));            \
    builder->FN(std::move(blob));                        \
  } while (0)
#endif  // TAKE_BUFFER_AND_APPLY

#ifndef TAKE_BUFFER_OR_NULL_AND_APPLY
#define TAKE_BUFFER_OR_NULL_AND_APPLY(builder, client, FN, pool, buffer) \
  do {                                                                   \
    std::unique_ptr<BlobWriter> blob;                                    \
    auto status = pool.Take(buffer, blob);                               \
    if (status.ok()) {                                                   \
      builder->FN(std::move(blob));                                      \
    } else {                                                             \
      if (status.IsObjectNotExists()) {                                  \
        builder->FN(Blob::MakeEmpty(client));                            \
      } else {                                                           \
        RETURN_ON_ERROR(status);                                         \
      }                                                                  \
    }                                                                    \
  } while (0)
#endif  // TAKE_BUFFER_OR_NULL_AND_APPLY

#ifndef TAKE_NULL_BITMAP_AND_APPLY
#define TAKE_NULL_BITMAP_AND_APPLY(builder, client, pool, array) \
  do {                                                           \
    if (array->null_bitmap() && array->null_count() > 0) {       \
      TAKE_BUFFER_AND_APPLY(builder, set_null_bitmap_, pool,     \
                            array->null_bitmap());               \
    } else {                                                     \
      builder->set_null_bitmap_(Blob::MakeEmpty(client));        \
    }                                                            \
  } while (0)
#endif  // TAKE_NULL_BITMAP_AND_APPLY

template <typename T>
void NumericArray<T>::PostConstruct(const ObjectMeta& meta) {
  std::shared_ptr<arrow::DataType> data_type;
  if (this->data_type_.empty()) {
    data_type = ConvertToArrowType<T>::TypeValue();
  } else {
    data_type = type_name_to_arrow_type(this->data_type_);
  }
  this->array_ = std::make_shared<ArrayType>(
      data_type, this->length_, this->buffer_->ArrowBufferOrEmpty(),
      this->null_bitmap_->ArrowBuffer(), this->null_count_, this->offset_);
}

template class NumericArray<int8_t>;
template class NumericArray<int16_t>;
template class NumericArray<int32_t>;
template class NumericArray<int64_t>;
template class NumericArray<uint8_t>;
template class NumericArray<uint16_t>;
template class NumericArray<uint32_t>;
template class NumericArray<uint64_t>;
template class NumericArray<float>;
template class NumericArray<double>;
template class NumericArray<arrow::Date32Type>;
template class NumericArray<arrow::Date64Type>;
template class NumericArray<arrow::Time32Type>;
template class NumericArray<arrow::Time64Type>;
template class NumericArray<arrow::TimestampType>;

template <typename T>
NumericArrayBuilder<T>::NumericArrayBuilder(Client& client)
    : NumericArrayBaseBuilder<T>(client) {}

template <typename T>
NumericArrayBuilder<T>::NumericArrayBuilder(
    Client& client, const std::shared_ptr<ArrayType> array)
    : NumericArrayBaseBuilder<T>(client) {
  std::shared_ptr<arrow::Array> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_.emplace_back(ref);
}

template <typename T>
NumericArrayBuilder<T>::NumericArrayBuilder(
    Client& client, const std::vector<std::shared_ptr<ArrayType>>& arrays)
    : NumericArrayBaseBuilder<T>(client) {
  for (auto const& array : arrays) {
    std::shared_ptr<arrow::Array> ref;
    VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
    this->arrays_.emplace_back(ref);
  }
}

template <typename T>
NumericArrayBuilder<T>::NumericArrayBuilder(
    Client& client, const std::shared_ptr<arrow::ChunkedArray> array)
    : NumericArrayBaseBuilder<T>(client) {
  std::shared_ptr<arrow::ChunkedArray> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_ = ref->chunks();
}

template <typename T>
Status NumericArrayBuilder<T>::Build(Client& client) {
  memory::VineyardMemoryPool pool(client);
  std::shared_ptr<arrow::Array> array;
  if (this->arrays_.empty()) {
    CHECK_ARROW_ERROR(ArrowBuilderType<T>(ConvertToArrowType<T>::TypeValue(),
                                          arrow::default_memory_pool())
                          .Finish(&array));
  } else {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        array, arrow_shim::Concatenate(std::move(this->arrays_), &pool));
  }
  std::shared_ptr<ArrayType> array_ =
      std::dynamic_pointer_cast<ArrayType>(array);

  this->set_length_(array_->length());
  this->set_data_type_(type_name_from_arrow_type(array_->type()));
  this->set_null_count_(array_->null_count());
  this->set_offset_(array_->offset());
  TAKE_BUFFER_OR_NULL_AND_APPLY(this, client, set_buffer_, pool,
                                array_->values());
  TAKE_NULL_BITMAP_AND_APPLY(this, client, pool, array_);
  return Status::OK();
}

template class NumericArrayBuilder<int8_t>;
template class NumericArrayBuilder<int16_t>;
template class NumericArrayBuilder<int32_t>;
template class NumericArrayBuilder<int64_t>;
template class NumericArrayBuilder<uint8_t>;
template class NumericArrayBuilder<uint16_t>;
template class NumericArrayBuilder<uint32_t>;
template class NumericArrayBuilder<uint64_t>;
template class NumericArrayBuilder<float>;
template class NumericArrayBuilder<double>;
template class NumericArrayBuilder<arrow::Date32Type>;
template class NumericArrayBuilder<arrow::Date64Type>;
template class NumericArrayBuilder<arrow::Time32Type>;
template class NumericArrayBuilder<arrow::Time64Type>;
template class NumericArrayBuilder<arrow::TimestampType>;

template <typename T>
FixedNumericArrayBuilder<T>::FixedNumericArrayBuilder(Client& client,
                                                      const size_t size)
    : NumericArrayBaseBuilder<T>(client), client_(client), size_(size) {
  if (size_ > 0) {
    VINEYARD_CHECK_OK(client.CreateBlob(size_ * sizeof(T), writer_));
    data_ = reinterpret_cast<ArrowValueType<T>*>(writer_->data());
  }
}

template <typename T>
FixedNumericArrayBuilder<T>::FixedNumericArrayBuilder(Client& client)
    : NumericArrayBaseBuilder<T>(client), client_(client) {}

template <typename T>
FixedNumericArrayBuilder<T>::~FixedNumericArrayBuilder() {
  if (!this->sealed() && writer_) {
    VINEYARD_DISCARD(writer_->Abort(client_));
  }
}

template <typename T>
Status FixedNumericArrayBuilder<T>::Make(
    Client& client, const size_t size,
    std::shared_ptr<FixedNumericArrayBuilder<T>>& out) {
  out = std::shared_ptr<FixedNumericArrayBuilder<T>>(
      new FixedNumericArrayBuilder<T>(client));
  out->size_ = size;
  if (out->size_ > 0) {
    RETURN_ON_ERROR(client.CreateBlob(out->size_ * sizeof(T), out->writer_));
    out->data_ = reinterpret_cast<ArrowValueType<T>*>(out->writer_->data());
  }
  return Status::OK();
}

template <typename T>
Status FixedNumericArrayBuilder<T>::Make(
    Client& client, std::unique_ptr<BlobWriter> writer, const size_t size,
    std::shared_ptr<FixedNumericArrayBuilder<T>>& out) {
  out = std::shared_ptr<FixedNumericArrayBuilder<T>>(
      new FixedNumericArrayBuilder<T>(client));
  out->size_ = size;
  if (out->size_ > 0) {
    if (!writer) {
      return Status::Invalid(
          "cannot make builder of size > 0 with a null buffer");
    }
    out->writer_ = std::move(writer);
    out->data_ = reinterpret_cast<ArrowValueType<T>*>(out->writer_->data());
  }
  return Status::OK();
}

template <typename T>
Status FixedNumericArrayBuilder<T>::Shrink(const size_t size) {
  Status s;
  if (writer_) {
    s = writer_->Shrink(client_, size * sizeof(T));
    if (s.ok()) {
      size_ = size;
    }
  }
  return s;
}

template <typename T>
Status FixedNumericArrayBuilder<T>::Release(
    std::unique_ptr<BlobWriter>& writer) {
  if (this->sealed()) {
    return Status::ObjectSealed(
        "sealed builder cannot release its internal buffer");
  }
  writer = std::move(writer_);
  data_ = nullptr;
  size_ = 0;
  return Status::OK();
}

template <typename T>
size_t FixedNumericArrayBuilder<T>::size() const {
  return size_;
}

template <typename T>
ArrowValueType<T>* FixedNumericArrayBuilder<T>::MutablePointer(
    int64_t i) const {
  if (data_) {
    return data_ + i;
  }
  return nullptr;
}

template <typename T>
ArrowValueType<T>* FixedNumericArrayBuilder<T>::data() const {
  return data_;
}

template <typename T>
Status FixedNumericArrayBuilder<T>::Build(Client& client) {
  this->set_length_(size_);
  this->set_null_count_(0);
  this->set_offset_(0);
  if (size_ > 0) {
    this->set_buffer_(std::move(writer_));
  } else {
    this->set_buffer_(Blob::MakeEmpty(client));
  }
  this->set_null_bitmap_(Blob::MakeEmpty(client));
  return Status::OK();
}

template class FixedNumericArrayBuilder<int8_t>;
template class FixedNumericArrayBuilder<int16_t>;
template class FixedNumericArrayBuilder<int32_t>;
template class FixedNumericArrayBuilder<int64_t>;
template class FixedNumericArrayBuilder<uint8_t>;
template class FixedNumericArrayBuilder<uint16_t>;
template class FixedNumericArrayBuilder<uint32_t>;
template class FixedNumericArrayBuilder<uint64_t>;
template class FixedNumericArrayBuilder<float>;
template class FixedNumericArrayBuilder<double>;
template class FixedNumericArrayBuilder<arrow::Date32Type>;
template class FixedNumericArrayBuilder<arrow::Date64Type>;
template class FixedNumericArrayBuilder<arrow::Time32Type>;
template class FixedNumericArrayBuilder<arrow::Time64Type>;
template class FixedNumericArrayBuilder<arrow::TimestampType>;

void BooleanArray::PostConstruct(const ObjectMeta& meta) {
  this->array_ = std::make_shared<ArrayType>(
      ConvertToArrowType<bool>::TypeValue(), this->length_,
      this->buffer_->ArrowBufferOrEmpty(), this->null_bitmap_->ArrowBuffer(),
      this->null_count_, this->offset_);
}

BooleanArrayBuilder::BooleanArrayBuilder(Client& client)
    : BooleanArrayBaseBuilder(client) {
  std::shared_ptr<ArrayType> array;
  CHECK_ARROW_ERROR(ArrowBuilderType<bool>{}.Finish(&array));
  this->arrays_.emplace_back(array);
}

BooleanArrayBuilder::BooleanArrayBuilder(Client& client,
                                         const std::shared_ptr<ArrayType> array)
    : BooleanArrayBaseBuilder(client) {
  std::shared_ptr<arrow::Array> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_.emplace_back(ref);
}

BooleanArrayBuilder::BooleanArrayBuilder(
    Client& client, const std::vector<std::shared_ptr<ArrayType>>& arrays)
    : BooleanArrayBaseBuilder(client) {
  for (auto const& array : arrays) {
    std::shared_ptr<arrow::Array> ref;
    VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
    this->arrays_.emplace_back(ref);
  }
}

BooleanArrayBuilder::BooleanArrayBuilder(
    Client& client, const std::shared_ptr<arrow::ChunkedArray> array)
    : BooleanArrayBaseBuilder(client) {
  std::shared_ptr<arrow::ChunkedArray> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_ = ref->chunks();
}

Status BooleanArrayBuilder::Build(Client& client) {
  memory::VineyardMemoryPool pool(client);
  std::shared_ptr<arrow::Array> array;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      array, arrow_shim::Concatenate(std::move(this->arrays_), &pool));
  std::shared_ptr<ArrayType> array_ =
      std::dynamic_pointer_cast<ArrayType>(array);

  this->set_length_(array_->length());
  this->set_null_count_(array_->null_count());
  this->set_offset_(array_->offset());
  TAKE_BUFFER_OR_NULL_AND_APPLY(this, client, set_buffer_, pool,
                                array_->values());
  TAKE_NULL_BITMAP_AND_APPLY(this, client, pool, array_);
  return Status::OK();
}

template <typename ArrayType>
void BaseBinaryArray<ArrayType>::PostConstruct(const ObjectMeta& meta) {
  this->array_ = std::make_shared<ArrayType>(
      this->length_, this->buffer_offsets_->ArrowBufferOrEmpty(),
      this->buffer_data_->ArrowBufferOrEmpty(),
      this->null_bitmap_->ArrowBuffer(), this->null_count_, this->offset_);
}

template class BaseBinaryArray<arrow::BinaryArray>;
template class BaseBinaryArray<arrow::LargeBinaryArray>;
template class BaseBinaryArray<arrow::StringArray>;
template class BaseBinaryArray<arrow::LargeStringArray>;

template <typename ArrayType, typename BuilderType>
GenericBinaryArrayBuilder<ArrayType, BuilderType>::GenericBinaryArrayBuilder(
    Client& client)
    : BaseBinaryArrayBaseBuilder<ArrayType>(client) {
  std::shared_ptr<ArrayType> array;
  CHECK_ARROW_ERROR(BuilderType{}.Finish(&array));
  this->arrays_.emplace_back(array);
}

template <typename ArrayType, typename BuilderType>
GenericBinaryArrayBuilder<ArrayType, BuilderType>::GenericBinaryArrayBuilder(
    Client& client, const std::shared_ptr<ArrayType> array)
    : BaseBinaryArrayBaseBuilder<ArrayType>(client) {
  std::shared_ptr<arrow::Array> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_.emplace_back(ref);
}

template <typename ArrayType, typename BuilderType>
GenericBinaryArrayBuilder<ArrayType, BuilderType>::GenericBinaryArrayBuilder(
    Client& client, const std::vector<std::shared_ptr<ArrayType>>& arrays)
    : BaseBinaryArrayBaseBuilder<ArrayType>(client) {
  for (auto const& array : arrays) {
    std::shared_ptr<arrow::Array> ref;
    VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
    this->arrays_.emplace_back(ref);
  }
}

template <typename ArrayType, typename BuilderType>
GenericBinaryArrayBuilder<ArrayType, BuilderType>::GenericBinaryArrayBuilder(
    Client& client, const std::shared_ptr<arrow::ChunkedArray> array)
    : BaseBinaryArrayBaseBuilder<ArrayType>(client) {
  std::shared_ptr<arrow::ChunkedArray> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_ = ref->chunks();
}

template <typename ArrayType, typename BuilderType>
Status GenericBinaryArrayBuilder<ArrayType, BuilderType>::Build(
    Client& client) {
  memory::VineyardMemoryPool pool(client);
  std::shared_ptr<arrow::Array> array;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      array, arrow_shim::Concatenate(std::move(this->arrays_), &pool));
  std::shared_ptr<ArrayType> array_ =
      std::dynamic_pointer_cast<ArrayType>(array);

  this->set_length_(array_->length());
  this->set_null_count_(array_->null_count());
  this->set_offset_(array_->offset());
  TAKE_BUFFER_OR_NULL_AND_APPLY(this, client, set_buffer_offsets_, pool,
                                array_->value_offsets());
  TAKE_BUFFER_OR_NULL_AND_APPLY(this, client, set_buffer_data_, pool,
                                array_->value_data());
  TAKE_NULL_BITMAP_AND_APPLY(this, client, pool, array_);
  return Status::OK();
}

template class GenericBinaryArrayBuilder<arrow::BinaryArray,
                                         arrow::BinaryBuilder>;
template class GenericBinaryArrayBuilder<arrow::LargeBinaryArray,
                                         arrow::LargeBinaryBuilder>;
template class GenericBinaryArrayBuilder<arrow::StringArray,
                                         arrow::StringBuilder>;
template class GenericBinaryArrayBuilder<arrow::LargeStringArray,
                                         arrow::LargeStringBuilder>;

void FixedSizeBinaryArray::PostConstruct(const ObjectMeta& meta) {
  this->array_ = std::make_shared<arrow::FixedSizeBinaryArray>(
      arrow::fixed_size_binary(this->byte_width_), this->length_,
      this->buffer_->ArrowBufferOrEmpty(), this->null_bitmap_->ArrowBuffer(),
      this->null_count_, this->offset_);
}

FixedSizeBinaryArrayBuilder::FixedSizeBinaryArrayBuilder(
    Client& client, const std::shared_ptr<arrow::DataType>& type)
    : FixedSizeBinaryArrayBaseBuilder(client) {
  std::shared_ptr<ArrayType> array;
  CHECK_ARROW_ERROR(arrow::FixedSizeBinaryBuilder{type}.Finish(&array));
  this->arrays_.emplace_back(array);
}

FixedSizeBinaryArrayBuilder::FixedSizeBinaryArrayBuilder(
    Client& client, const std::shared_ptr<ArrayType> array)
    : FixedSizeBinaryArrayBaseBuilder(client) {
  std::shared_ptr<arrow::Array> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_.emplace_back(ref);
}

FixedSizeBinaryArrayBuilder::FixedSizeBinaryArrayBuilder(
    Client& client, const std::vector<std::shared_ptr<ArrayType>>& arrays)
    : FixedSizeBinaryArrayBaseBuilder(client) {
  for (auto const& array : arrays) {
    std::shared_ptr<arrow::Array> ref;
    VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
    this->arrays_.emplace_back(ref);
  }
}

FixedSizeBinaryArrayBuilder::FixedSizeBinaryArrayBuilder(
    Client& client, const std::shared_ptr<arrow::ChunkedArray> array)
    : FixedSizeBinaryArrayBaseBuilder(client) {
  std::shared_ptr<arrow::ChunkedArray> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_ = ref->chunks();
}

Status FixedSizeBinaryArrayBuilder::Build(Client& client) {
  memory::VineyardMemoryPool pool(client);
  std::shared_ptr<arrow::Array> array;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      array, arrow_shim::Concatenate(std::move(this->arrays_), &pool));
  std::shared_ptr<ArrayType> array_ =
      std::dynamic_pointer_cast<ArrayType>(array);

  VINEYARD_ASSERT(array_->length() == 0 || array_->values()->size() != 0,
                  "Invalid array values");

  this->set_byte_width_(array_->byte_width());
  this->set_length_(array_->length());
  this->set_null_count_(array_->null_count());
  this->set_offset_(array_->offset());
  TAKE_BUFFER_OR_NULL_AND_APPLY(this, client, set_buffer_, pool,
                                array_->values());
  TAKE_NULL_BITMAP_AND_APPLY(this, client, pool, array_);
  return Status::OK();
}

void NullArray::PostConstruct(const ObjectMeta& meta) {
  this->array_ = std::make_shared<arrow::NullArray>(this->length_);
}

NullArrayBuilder::NullArrayBuilder(Client& client)
    : NullArrayBaseBuilder(client) {
  std::shared_ptr<ArrayType> array;
  CHECK_ARROW_ERROR(arrow::NullBuilder{}.Finish(&array));
  this->arrays_.emplace_back(array);
}

NullArrayBuilder::NullArrayBuilder(Client& client,
                                   const std::shared_ptr<ArrayType> array)
    : NullArrayBaseBuilder(client) {
  this->arrays_.emplace_back(array);
}

NullArrayBuilder::NullArrayBuilder(
    Client& client, const std::vector<std::shared_ptr<ArrayType>>& arrays)
    : NullArrayBaseBuilder(client) {
  for (auto const& array : arrays) {
    this->arrays_.emplace_back(array);
  }
}

NullArrayBuilder::NullArrayBuilder(
    Client& client, const std::shared_ptr<arrow::ChunkedArray> array)
    : NullArrayBaseBuilder(client) {
  this->arrays_ = array->chunks();
}

Status NullArrayBuilder::Build(Client& client) {
  memory::VineyardMemoryPool pool(client);
  std::shared_ptr<arrow::Array> array;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      array, arrow_shim::Concatenate(std::move(this->arrays_), &pool));
  std::shared_ptr<ArrayType> array_ =
      std::dynamic_pointer_cast<ArrayType>(array);

  this->set_length_(array_->length());
  return Status::OK();
}

template <typename ArrayType>
void BaseListArray<ArrayType>::PostConstruct(const ObjectMeta& meta) {
  auto array = detail::CastToArray(values_);
  auto list_type =
      std::make_shared<typename ArrayType::TypeClass>(array->type());
  this->array_ = std::make_shared<ArrayType>(
      list_type, this->length_, this->buffer_offsets_->ArrowBufferOrEmpty(),
      array, this->null_bitmap_->ArrowBuffer(), this->null_count_,
      this->offset_);
}

template class BaseListArray<arrow::ListArray>;
template class BaseListArray<arrow::LargeListArray>;

template <typename ArrayType>
BaseListArrayBuilder<ArrayType>::BaseListArrayBuilder(
    Client& client, const std::shared_ptr<ArrayType> array)
    : BaseListArrayBaseBuilder<ArrayType>(client) {
  std::shared_ptr<arrow::Array> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_.emplace_back(ref);
}

template <typename ArrayType>
BaseListArrayBuilder<ArrayType>::BaseListArrayBuilder(
    Client& client, const std::vector<std::shared_ptr<ArrayType>>& arrays)
    : BaseListArrayBaseBuilder<ArrayType>(client) {
  for (auto const& array : arrays) {
    std::shared_ptr<arrow::Array> ref;
    VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
    this->arrays_.emplace_back(ref);
  }
}

template <typename ArrayType>
BaseListArrayBuilder<ArrayType>::BaseListArrayBuilder(
    Client& client, const std::shared_ptr<arrow::ChunkedArray> array)
    : BaseListArrayBaseBuilder<ArrayType>(client) {
  std::shared_ptr<arrow::ChunkedArray> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_ = ref->chunks();
}

template <typename ArrayType>
Status BaseListArrayBuilder<ArrayType>::Build(Client& client) {
  std::shared_ptr<arrow::Array> array;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      array, arrow_shim::Concatenate(std::move(this->arrays_)));
  std::shared_ptr<ArrayType> array_ =
      std::dynamic_pointer_cast<ArrayType>(array);

  // FIXME: list array is not optimized yet.

  this->set_length_(array_->length());
  this->set_null_count_(array_->null_count());
  this->set_offset_(array_->offset());

  {
    std::unique_ptr<BlobWriter> buffer_writer;
    RETURN_ON_ERROR(
        client.CreateBlob(array_->value_offsets()->size(), buffer_writer));
    memcpy(buffer_writer->data(), array_->value_offsets()->data(),
           array_->value_offsets()->size());

    this->set_buffer_offsets_(
        std::shared_ptr<BlobWriter>(std::move(buffer_writer)));
  }
  {
    // Assuming the list is not nested.
    // We need to split the definition to .cc if someday we need to consider
    // nested list in list case.
    this->set_values_(detail::BuildArray(client, array_->values()));
  }
  {
    if (array->null_bitmap() && array->null_count() > 0) {
      std::unique_ptr<BlobWriter> bitmap_buffer_writer;
      RETURN_ON_ERROR(client.CreateBlob(array->null_bitmap()->size(),
                                        bitmap_buffer_writer));
      memcpy(bitmap_buffer_writer->data(), array->null_bitmap()->data(),
             array->null_bitmap()->size());
      this->set_null_bitmap_(
          std::shared_ptr<BlobWriter>(std::move(bitmap_buffer_writer)));
    } else {
      this->set_null_bitmap_(Blob::MakeEmpty(client));
    }
  }
  return Status::OK();
}

template class BaseListArrayBuilder<arrow::ListArray>;
template class BaseListArrayBuilder<arrow::LargeListArray>;

void FixedSizeListArray::PostConstruct(const ObjectMeta& meta) {
  auto array = detail::CastToArray(values_);
  this->array_ = std::make_shared<arrow::FixedSizeListArray>(
      arrow::fixed_size_list(array->type(), this->list_size_), this->length_,
      array);
}

FixedSizeListArrayBuilder::FixedSizeListArrayBuilder(
    Client& client, const std::shared_ptr<ArrayType> array)
    : FixedSizeListArrayBaseBuilder(client) {
  std::shared_ptr<arrow::Array> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_.emplace_back(ref);
}

FixedSizeListArrayBuilder::FixedSizeListArrayBuilder(
    Client& client, const std::vector<std::shared_ptr<ArrayType>>& arrays)
    : FixedSizeListArrayBaseBuilder(client) {
  for (auto const& array : arrays) {
    std::shared_ptr<arrow::Array> ref;
    VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
    this->arrays_.emplace_back(ref);
  }
}

FixedSizeListArrayBuilder::FixedSizeListArrayBuilder(
    Client& client, const std::shared_ptr<arrow::ChunkedArray> array)
    : FixedSizeListArrayBaseBuilder(client) {
  std::shared_ptr<arrow::ChunkedArray> ref;
  VINEYARD_CHECK_OK(detail::Copy(array, ref, true));
  this->arrays_ = ref->chunks();
}

Status FixedSizeListArrayBuilder::Build(Client& client) {
  std::shared_ptr<arrow::Array> array;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      array, arrow_shim::Concatenate(std::move(this->arrays_)));
  std::shared_ptr<ArrayType> array_ =
      std::dynamic_pointer_cast<ArrayType>(array);

  // FIXME: list array is not optimized yet.

  this->set_length_(array_->length());
  this->set_list_size_(array_->list_type()->list_size());

  {
    // Assuming the list is not nested.
    // We need to split the definition to .cc if someday we need to consider
    // nested list in list case.
    this->set_values_(detail::BuildArray(client, array_->values()));
  }
  return Status::OK();
}

void SchemaProxy::PostConstruct(const ObjectMeta& meta) {
  std::shared_ptr<arrow::Buffer> wrapper;
  // the binary value is not roundtrip, see also:
  //   https://json.nlohmann.me/features/binary_values/#serialization
  json::binary_t binary;
  std::vector<uint8_t> binary_vec;
  if (this->schema_binary_.is_binary()) {
    binary = this->schema_binary_.get_binary();
    wrapper = arrow::Buffer::Wrap(binary.data(), binary.size());
  } else if (this->schema_binary_.contains("bytes")) {
    this->schema_binary_["bytes"].get_to(binary_vec);
    wrapper = arrow::Buffer::Wrap(binary_vec.data(), binary_vec.size());
  } else if (this->meta_.HasKey("buffer_")) {
    // for backward compatibility
    std::shared_ptr<Blob> buffer;
    VINEYARD_CHECK_OK(this->meta_.GetMember("buffer_", buffer));
    wrapper = buffer->ArrowBufferOrEmpty();
  }
  if (wrapper == nullptr) {
    LOG(ERROR) << "Invalid schema binary: " << this->schema_binary_.dump(4);
  }
  arrow::io::BufferReader reader(wrapper);
  CHECK_ARROW_ERROR_AND_ASSIGN(this->schema_,
                               arrow::ipc::ReadSchema(&reader, nullptr));
}

SchemaProxyBuilder::SchemaProxyBuilder(
    Client& client, const std::shared_ptr<arrow::Schema> schema)
    : SchemaProxyBaseBuilder(client), schema_(schema) {}

Status SchemaProxyBuilder::Build(Client& client) {
  std::shared_ptr<arrow::Buffer> schema_buffer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 2000000
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      schema_buffer, arrow::ipc::SerializeSchema(*schema_, nullptr,
                                                 arrow::default_memory_pool()));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      schema_buffer,
      arrow::ipc::SerializeSchema(*schema_, arrow::default_memory_pool()));
#endif
  {
    json textual;
    RETURN_ON_ERROR(arrow_shim::SchemaToJSON(schema_, textual));
    this->set_schema_textual_(textual);
  }
  {
    std::vector<uint8_t> buffer(schema_buffer->size());
    memcpy(buffer.data(), schema_buffer->data(), schema_buffer->size());
    this->set_schema_binary_(json::binary(buffer));
  }
  return Status::OK();
}

void RecordBatch::PostConstruct(const ObjectMeta& meta) {
  for (size_t idx = 0; idx < columns_.size(); ++idx) {
    arrow_columns_.emplace_back(detail::CastToArray(columns_[idx]));
  }
}

std::shared_ptr<arrow::RecordBatch> RecordBatch::GetRecordBatch() const {
  if (this->batch_ == nullptr) {
    this->batch_ = arrow::RecordBatch::Make(this->schema_.GetSchema(),
                                            this->row_num_, arrow_columns_);
  }
  return this->batch_;
}

RecordBatchBuilder::RecordBatchBuilder(
    Client& client, const std::shared_ptr<arrow::RecordBatch> batch)
    : RecordBatchBaseBuilder(client) {
  this->batches_.emplace_back(batch);
}

RecordBatchBuilder::RecordBatchBuilder(
    Client& client,
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches)
    : RecordBatchBaseBuilder(client) {
  VINEYARD_ASSERT(batches.size() > 0, "at least one batch is required");
  this->batches_ = std::move(batches);
}

Status RecordBatchBuilder::Build(Client& client) {
  int64_t num_rows = 0, num_columns = batches_[0]->num_columns();
  for (auto const& batch : batches_) {
    num_rows += batch->num_rows();
  }

  this->set_schema_(
      std::make_shared<SchemaProxyBuilder>(client, batches_[0]->schema()));
  this->set_row_num_(num_rows);
  this->set_column_num_(num_columns);

  // column_chunks[column_index][chunk_index]
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> column_chunks(
      num_columns);
  for (auto& batch : batches_) {
    for (int64_t cindex = 0; cindex < batch->num_columns(); ++cindex) {
      column_chunks[cindex].emplace_back(batch->column(cindex));
    }
    batch.reset();  // release the reference
  }
  batches_.clear();  // release the reference

  // build the columns into vineyard
  for (int64_t idx = 0; idx < num_columns; ++idx) {
    this->add_columns_(detail::BuildArray(
        client, std::make_shared<arrow::ChunkedArray>(column_chunks[idx])));
    column_chunks[idx].clear();  // release the reference
  }
  return Status::OK();
}

RecordBatchExtender::RecordBatchExtender(
    Client& client, const std::shared_ptr<RecordBatch> batch)
    : RecordBatchBaseBuilder(client) {
  row_num_ = batch->num_rows();
  column_num_ = batch->num_columns();
  schema_ = batch->schema();
  for (auto const& column : batch->columns()) {
    this->add_columns_(column);
  }
}

Status RecordBatchExtender::AddColumn(
    Client& client, const std::string& field_name,
    const std::shared_ptr<arrow::Array> column) {
  // validate input
  if (static_cast<size_t>(column->length()) != row_num_) {
    return Status::Invalid(
        "The newly added columns doesn't have a matched shape");
  }
  // extend schema
  auto field = ::arrow::field(std::move(field_name), column->type());
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      schema_, schema_->AddField(schema_->num_fields(), field));
  // extend columns
  arrow_columns_.push_back(column);
  column_num_ += 1;
  return Status::OK();
}

Status RecordBatchExtender::Build(Client& client) {
  this->set_row_num_(row_num_);
  this->set_column_num_(column_num_);
  this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
  for (size_t idx = 0; idx < arrow_columns_.size(); ++idx) {
    this->add_columns_(detail::BuildArray(client, arrow_columns_[idx]));
  }
  return Status::OK();
}

RecordBatchConsolidator::RecordBatchConsolidator(
    Client& client, const std::shared_ptr<RecordBatch> batch)
    : RecordBatchBaseBuilder(client) {
  row_num_ = batch->num_rows();
  column_num_ = batch->num_columns();
  schema_ = batch->schema();
  for (auto const& column : batch->columns()) {
    this->add_columns_(column);
  }
  for (auto const& column : batch->arrow_columns()) {
    arrow_columns_.push_back(column);
  }
}

Status RecordBatchConsolidator::ConsolidateColumns(
    Client& client, std::vector<std::string> const& columns,
    std::string const& consolidate_name) {
  std::vector<int64_t> column_indexes;
  for (auto const& column : columns) {
    auto index = schema_->GetFieldIndex(column);
    if (index < 0) {
      return Status::Invalid("The column name '" + column +
                             "' is not found in the schema");
    }
    column_indexes.push_back(index);
  }
  return ConsolidateColumns(client, column_indexes, consolidate_name);
}

Status RecordBatchConsolidator::ConsolidateColumns(
    Client& client, std::vector<int64_t> const& columns,
    std::string const& consolidate_name) {
  std::vector<std::shared_ptr<arrow::Array>> columns_to_consolidate;
  for (int64_t const& column : columns) {
    columns_to_consolidate.push_back(this->arrow_columns_[column]);
  }
  std::shared_ptr<arrow::Array> consolidated_column;
  RETURN_ON_ERROR(vineyard::ConsolidateColumns(columns_to_consolidate,
                                               consolidated_column));

  this->column_num_ -= (columns.size() - 1);
  std::vector<int64_t> sorted_column_indexes(columns);
  std::sort(sorted_column_indexes.begin(), sorted_column_indexes.end());
  for (size_t index = 0; index < sorted_column_indexes.size(); ++index) {
    size_t index_to_remove =
        sorted_column_indexes[sorted_column_indexes.size() - 1 - index];
    this->remove_columns_(index_to_remove);
    this->arrow_columns_.erase(this->arrow_columns_.begin() + index_to_remove);
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(schema_,
                                     schema_->RemoveField(index_to_remove));
  }
  this->arrow_columns_.emplace_back(consolidated_column);
  this->add_columns_(detail::BuildArray(client, consolidated_column));
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      schema_, schema_->AddField(schema_->num_fields(),
                                 ::arrow::field(consolidate_name,
                                                consolidated_column->type())));
  return Status::OK();
}

Status RecordBatchConsolidator::Build(Client& client) {
  this->set_row_num_(row_num_);
  this->set_column_num_(column_num_);
  this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
  return Status::OK();
}

void Table::Construct(const ObjectMeta& meta) {
  Collection<RecordBatch>::Construct(meta);
  this->PostConstruct(meta);
}

void Table::PostConstruct(const ObjectMeta& meta) {
  VINEYARD_CHECK_OK(this->meta_.GetMember("schema_", this->schema_));
  this->meta_.GetKeyValue("num_rows_", this->num_rows_);
  this->meta_.GetKeyValue("num_columns_", this->num_columns_);
  this->meta_.GetKeyValue("batch_num_", this->batch_num_);
  for (auto iter = this->LocalBegin(); iter != this->LocalEnd();
       iter.NextLocal()) {
    this->batches_.emplace_back(std::dynamic_pointer_cast<RecordBatch>(*iter));
  }
}

std::shared_ptr<arrow::Table> Table::GetTable() const {
  if (this->table_ == nullptr) {
    if (batch_num_ > 0) {
      arrow_batches_.resize(batch_num_);
      for (size_t i = 0; i < batch_num_; ++i) {
        arrow_batches_[i] = batches_[i]->GetRecordBatch();
      }
      VINEYARD_CHECK_OK(RecordBatchesToTable(arrow_batches_, &this->table_));
    } else {
      CHECK_ARROW_ERROR_AND_ASSIGN(
          this->table_,
          arrow::Table::FromRecordBatches(this->schema_->GetSchema(), {}));
    }
  }
  return this->table_;
}

TableBuilder::TableBuilder(Client& client,
                           const std::shared_ptr<arrow::Table> table,
                           const bool merge_chunks)
    : CollectionBuilder<RecordBatch>(client), merge_chunks_(merge_chunks) {
  this->tables_.emplace_back(table);
}

TableBuilder::TableBuilder(
    Client& client, const std::vector<std::shared_ptr<arrow::Table>>& tables,
    const bool merge_chunks)
    : CollectionBuilder<RecordBatch>(client), merge_chunks_(merge_chunks) {
  VINEYARD_ASSERT(tables.size() > 0, "at least one batch is required");
  this->tables_ = std::move(tables);
}

Status TableBuilder::Build(Client& client) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  int64_t num_rows = 0;
  for (auto const& table : tables_) {
    num_rows += table->num_rows();
    std::vector<std::shared_ptr<arrow::RecordBatch>> chunks;
    RETURN_ON_ERROR(TableToRecordBatches(table, &chunks));
    batches.insert(batches.end(), chunks.begin(), chunks.end());
  }
  tables_.clear();  // release the reference

  this->set_num_rows_(num_rows);
  this->set_num_columns_(batches[0]->num_columns());
  RETURN_ON_ERROR(this->set_schema(batches[0]->schema()));

  if (merge_chunks_) {
    this->set_batch_num_(1);
    RETURN_ON_ERROR(
        this->AddMember(std::make_shared<RecordBatchBuilder>(client, batches)));
    batches.clear();  // release the reference
  } else {
    this->set_batch_num_(batches.size());
    for (auto const& batch : batches) {
      RETURN_ON_ERROR(
          this->AddMember(std::make_shared<RecordBatchBuilder>(client, batch)));
    }
    batches.clear();  // release the reference
  }
  return Status::OK();
}

void TableBuilder::set_num_rows(const size_t num_rows) {
  this->AddKeyValue("num_rows_", num_rows);
}

void TableBuilder::set_num_rows_(const size_t num_rows) {
  this->set_num_rows(num_rows);
}

void TableBuilder::set_num_columns(const size_t num_columns) {
  this->AddKeyValue("num_columns_", num_columns);
}

void TableBuilder::set_num_columns_(const size_t num_columns) {
  this->set_num_columns(num_columns);
}

void TableBuilder::set_batch_num(const size_t batch_num) {
  this->AddKeyValue("batch_num_", batch_num);
}

void TableBuilder::set_batch_num_(const size_t batch_num) {
  this->set_batch_num(batch_num);
}

Status TableBuilder::set_schema(const std::shared_ptr<arrow::Schema>& schema) {
  auto builder = std::make_shared<SchemaProxyBuilder>(client_, schema);
  return this->AddMember("schema_", builder);
}

Status TableBuilder::set_schema(const std::shared_ptr<ObjectBuilder>& schema) {
  return this->AddMember("schema_", schema);
}

Status TableBuilder::set_schema_(const std::shared_ptr<ObjectBuilder>& schema) {
  return this->set_schema(schema);
}

TableExtender::TableExtender(Client& client, const std::shared_ptr<Table> table)
    : TableBuilder(client, nullptr) {
  row_num_ = table->num_rows();
  column_num_ = table->num_columns();
  schema_ = table->schema();
  for (auto const& batch : table->batches()) {
    record_batch_extenders_.push_back(
        std::make_shared<RecordBatchExtender>(client, batch));
  }
}

Status TableExtender::AddColumn(Client& client, const std::string& field_name,
                                const std::shared_ptr<arrow::Array> column) {
  // validate input
  if (static_cast<size_t>(column->length()) != row_num_) {
    return Status::Invalid(
        "The newly added columns doesn't have a matched shape");
  }
  // extend schema
  auto field = ::arrow::field(field_name, column->type());
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      schema_, schema_->AddField(schema_->num_fields(), field));

  // extend columns on every batch
  size_t offset = 0;
  for (auto& extender : record_batch_extenders_) {
    RETURN_ON_ERROR(extender->AddColumn(
        client, field_name, column->Slice(offset, extender->num_rows())));
    offset += extender->num_rows();
  }
  column_num_ += 1;
  return Status::OK();
}

/**
 * NOTE: `column` is aligned with `table`.
 */
Status TableExtender::AddColumn(
    Client& client, const std::string& field_name,
    const std::shared_ptr<arrow::ChunkedArray> column) {
  // validate input
  if (static_cast<size_t>(column->length()) != row_num_) {
    return Status::Invalid(
        "The newly added columns doesn't have a matched shape");
  }
  // extend schema
  auto field = ::arrow::field(field_name, column->type());
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      schema_, schema_->AddField(schema_->num_fields(), field));

  // extend columns on every batch
  size_t chunk_index = 0;
  for (auto& extender : record_batch_extenders_) {
    RETURN_ON_ERROR(
        extender->AddColumn(client, field_name, column->chunk(chunk_index)));
    chunk_index += 1;
  }
  column_num_ += 1;
  return Status::OK();
}

Status TableExtender::Build(Client& client) {
  this->set_batch_num_(record_batch_extenders_.size());
  this->set_num_rows_(row_num_);
  this->set_num_columns_(column_num_);
  for (auto const& extender : record_batch_extenders_) {
    RETURN_ON_ERROR(this->AddMember(extender));
  }
  RETURN_ON_ERROR(
      this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_)));
  return Status::OK();
}

TableConsolidator::TableConsolidator(Client& client,
                                     const std::shared_ptr<Table> table)
    : TableBuilder(client, nullptr) {
  row_num_ = table->num_rows();
  column_num_ = table->num_columns();
  schema_ = table->schema();
  for (auto const& batch : table->batches()) {
    record_batch_consolidators_.push_back(
        std::make_shared<RecordBatchConsolidator>(client, batch));
  }
}

Status TableConsolidator::ConsolidateColumns(
    Client& client, std::vector<std::string> const& columns,
    std::string const& consolidate_name) {
  std::vector<int64_t> column_indexes;
  for (auto const& column : columns) {
    auto index = schema_->GetFieldIndex(column);
    if (index < 0) {
      return Status::Invalid("The column name '" + column +
                             "' is not found in the schema");
    }
    column_indexes.push_back(index);
  }
  return ConsolidateColumns(client, column_indexes, consolidate_name);
}

Status TableConsolidator::ConsolidateColumns(
    Client& client, std::vector<int64_t> const& columns,
    std::string const& consolidate_name) {
  for (auto& consolidator : record_batch_consolidators_) {
    RETURN_ON_ERROR(
        consolidator->ConsolidateColumns(client, columns, consolidate_name));
  }
  column_num_ -= (columns.size() - 1);
  return Status::OK();
}

Status TableConsolidator::Build(Client& client) {
  this->set_batch_num_(record_batch_consolidators_.size());
  this->set_num_rows_(row_num_);
  this->set_num_columns_(column_num_);
  for (auto const& extender : record_batch_consolidators_) {
    RETURN_ON_ERROR(this->AddMember(extender));
  }
  std::shared_ptr<arrow::Schema> schema;
  if (record_batch_consolidators_.empty()) {
    schema = schema_;
  } else {
    schema = record_batch_consolidators_[0]->schema();
  }
  RETURN_ON_ERROR(
      this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema)));
  return Status::OK();
}

}  // namespace vineyard
