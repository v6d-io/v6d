/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include "basic/ds/arrow.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "basic/ds/arrow_shim/concatenate.h"
#include "basic/ds/arrow_shim/memory_pool.h"
#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "common/memory/memcpy.h"

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

  Status Visit(const arrow::StringType*) {
    builder_ = std::make_shared<StringArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::LargeStringType*) {
    builder_ = std::make_shared<LargeStringArrayBuilder>(client_, array_);
    return Status::OK();
  }

  Status Visit(const arrow::BinaryType*) {
    builder_ = std::make_shared<BinaryArrayBuilder>(client_, array_);
    return Status::OK();
  }

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
  Status Visit(const arrow::LargeListType*) {
    builder_ = std::make_shared<LargeListArrayBuilder>(client_, array_);
    return Status::OK();
  }
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
NumericArrayBuilder<T>::NumericArrayBuilder(Client& client)
    : NumericArrayBaseBuilder<T>(client) {
  std::shared_ptr<ArrayType> array;
  CHECK_ARROW_ERROR(ArrowBuilderType<T>{}.Finish(&array));
  this->arrays_.emplace_back(array);
}

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
  std::unique_ptr<BlobWriter> schema_writer;
  RETURN_ON_ERROR(client.CreateBlob(schema_buffer->size(), schema_writer));
  memcpy(schema_writer->data(), schema_buffer->data(), schema_buffer->size());

  this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(schema_writer)));
  return Status::OK();
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

TableBuilder::TableBuilder(Client& client,
                           const std::shared_ptr<arrow::Table> table,
                           const bool merge_chunks)
    : TableBaseBuilder(client), merge_chunks_(merge_chunks) {
  this->tables_.emplace_back(table);
}

TableBuilder::TableBuilder(
    Client& client, const std::vector<std::shared_ptr<arrow::Table>>& tables,
    const bool merge_chunks)
    : TableBaseBuilder(client), merge_chunks_(merge_chunks) {
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
  this->set_schema_(
      std::make_shared<SchemaProxyBuilder>(client, batches[0]->schema()));

  if (merge_chunks_) {
    this->set_batch_num_(1);
    this->add_batches_(std::make_shared<RecordBatchBuilder>(client, batches));
    batches.clear();  // release the reference
  } else {
    this->set_batch_num_(batches.size());
    for (auto const& batch : batches) {
      this->add_batches_(std::make_shared<RecordBatchBuilder>(client, batch));
    }
    batches.clear();  // release the reference
  }
  return Status::OK();
}

TableExtender::TableExtender(Client& client, const std::shared_ptr<Table> table)
    : TableBaseBuilder(client) {
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
    this->add_batches_(extender);
  }
  this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
  return Status::OK();
}

TableConsolidator::TableConsolidator(Client& client,
                                     const std::shared_ptr<Table> table)
    : TableBaseBuilder(client) {
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
    this->add_batches_(extender);
  }
  if (record_batch_consolidators_.empty()) {
    this->set_schema_(std::make_shared<SchemaProxyBuilder>(client, schema_));
  } else {
    this->set_schema_(std::make_shared<SchemaProxyBuilder>(
        client, record_batch_consolidators_[0]->schema()));
  }
  return Status::OK();
}

}  // namespace vineyard
