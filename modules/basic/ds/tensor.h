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

#ifndef MODULES_BASIC_DS_TENSOR_H_
#define MODULES_BASIC_DS_TENSOR_H_

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "basic/ds/tensor.vineyard.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/arrow.h"
#include "common/util/json.h"

namespace vineyard {

class ITensorBuilder {
 public:
  virtual ~ITensorBuilder() {}
};

/**
 * @brief TensorBuilder is used for building tensors that supported by vineyard
 *
 * @tparam T
 */
template <typename T>
class TensorBuilder : public ITensorBuilder, public TensorBaseBuilder<T> {
 public:
  using value_t = T;
  using value_pointer_t = T*;
  using value_const_pointer_t = const T*;

 public:
  /**
   * @brief Initialize the TensorBuilder with the tensor shape.
   *
   * @param client The client connected to the vineyard server.
   * @param shape The shape of the tensor.
   */
  TensorBuilder(Client& client, std::vector<int64_t> const& shape)
      : TensorBaseBuilder<T>(client) {
    this->set_value_type_(AnyType(AnyTypeEnum<T>::value));
    this->set_shape_(shape);
    int64_t size = std::accumulate(this->shape_.begin(), this->shape_.end(), 1,
                                   std::multiplies<int64_t>{});
    VINEYARD_CHECK_OK(client.CreateBlob(size * sizeof(T), buffer_writer_));
    this->data_ = reinterpret_cast<T*>(buffer_writer_->data());
  }

  /**
   * @brief Initialize the TensorBuilder for a partition of a GlobalTensor.
   *
   * @param client The client connected to the vineyard server.
   * @param shape The shape of the partition.
   * @param partition_index The partition index in the global tensor.
   */
  TensorBuilder(Client& client, std::vector<int64_t> const& shape,
                std::vector<int64_t> const& partition_index)
      : TensorBuilder(client, shape) {
    this->set_partition_index_(partition_index);
  }

  /**
   * @brief Get the shape of the tensor.
   *
   * @return The shape vector where the ith element represents
   * the size of the ith axis.
   */
  std::vector<int64_t> const& shape() const { return this->shape_; }

  /**
   * @brief Get the index of this partition in the global tensor.
   *
   * @return The index vector where the ith element represents the index
   * in the ith axis.
   */
  std::vector<int64_t> const& partition_index() const {
    return this->partition_index_;
  }

  /**
   * @brief Set the shape of the tensor.
   *
   * @param shape The vector for the shape, where the ith element
   * represents the size of the shape in the ith axis.
   */
  void set_shape(std::vector<int64_t> const& shape) { this->set_shape_(shape); }

  /**
   * @brief Set the index in the global tensor.
   *
   * @param partition_index The vector of indices, where the ith element
   * represents the index in the ith axis.
   */
  void set_partition_index(std::vector<int64_t> const& partition_index) {
    this->set_partition_index_(partition_index);
  }

  /**
   * @brief Get the strides of the tensor.
   *
   * @return The strides of the tensor. The definition of the tensor's strides
   * can be found in https://pytorch.org/docs/stable/tensor_attributes.html
   */
  std::vector<int64_t> strides() const {
    std::vector<int64_t> vec(this->shape_.size());
    vec[this->shape_.size() - 1] = sizeof(T);
    for (size_t i = this->shape_.size() - 1; i > 0; --i) {
      vec[i - 1] = vec[i] * this->shape_[i];
    }
    return vec;
  }

  /**
   * @brief Get the data pointer of the tensor.
   *
   */
  inline value_pointer_t data() const { return this->data_; }

  /**
   * @brief Build the tensor.
   *
   * @param client The client connceted to the vineyard server.
   */
  Status Build(Client& client) override {
    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(buffer_writer_)));
    return Status::OK();
  }

 private:
  std::unique_ptr<BlobWriter> buffer_writer_;
  T* data_;
};

/**
 * @brief TensorBuilder is used for building tensors that supported by vineyard
 *
 * @tparam T
 */
template <>
class TensorBuilder<std::string> : public ITensorBuilder,
                                   public TensorBaseBuilder<std::string> {
 public:
  using value_t = arrow_string_view;
  using value_pointer_t = uint8_t*;
  using value_const_pointer_t = const uint8_t*;

 public:
  /**
   * @brief Initialize the TensorBuilder with the tensor shape.
   *
   * @param client The client connected to the vineyard server.
   * @param shape The shape of the tensor.
   */
  TensorBuilder(Client& client, std::vector<int64_t> const& shape)
      : TensorBaseBuilder<std::string>(client) {
    this->set_value_type_(AnyType(AnyTypeEnum<std::string>::value));
    this->set_shape_(shape);
    this->buffer_writer_ = std::make_shared<arrow::LargeStringBuilder>();
  }

  /**
   * @brief Initialize the TensorBuilder for a partition of a GlobalTensor.
   *
   * @param client The client connected to the vineyard server.
   * @param shape The shape of the partition.
   * @param partition_index The partition index in the global tensor.
   */
  TensorBuilder(Client& client, std::vector<int64_t> const& shape,
                std::vector<int64_t> const& partition_index)
      : TensorBuilder(client, shape) {
    this->set_partition_index_(partition_index);
  }

  /**
   * @brief Get the shape of the tensor.
   *
   * @return The shape vector where the ith element represents
   * the size of the ith axis.
   */
  std::vector<int64_t> const& shape() const { return this->shape_; }

  /**
   * @brief Get the index of this partition in the global tensor.
   *
   * @return The index vector where the ith element represents the index
   * in the ith axis.
   */
  std::vector<int64_t> const& partition_index() const {
    return this->partition_index_;
  }

  /**
   * @brief Set the shape of the tensor.
   *
   * @param shape The vector for the shape, where the ith element
   * represents the size of the shape in the ith axis.
   */
  void set_shape(std::vector<int64_t> const& shape) { this->set_shape_(shape); }

  /**
   * @brief Set the index in the global tensor.
   *
   * @param partition_index The vector of indices, where the ith element
   * represents the index in the ith axis.
   */
  void set_partition_index(std::vector<int64_t> const& partition_index) {
    this->set_partition_index_(partition_index);
  }

  /**
   * @brief Get the strides of the tensor.
   *
   * @return The strides of the tensor. The definition of the tensor's strides
   * can be found in https://pytorch.org/docs/stable/tensor_attributes.html
   */
  std::vector<int64_t> strides() const {
    std::vector<int64_t> vec(this->shape_.size());
    vec[this->shape_.size() - 1] = 1 /* special case for std::string */;
    for (size_t i = this->shape_.size() - 1; i > 0; --i) {
      vec[i - 1] = vec[i] * this->shape_[i];
    }
    return vec;
  }

  /**
   * @brief Get the data pointer of the tensor.
   *
   */
  inline value_pointer_t data() const {
    return const_cast<value_pointer_t>(this->buffer_writer_->value_data());
  }

  /**
   * @brief Append value to the builder.
   */
  inline Status Append(value_t const& value) {
    RETURN_ON_ARROW_ERROR(
        this->buffer_writer_->Append(value.data(), value.size()));
    return Status::OK();
  }

  /**
   * @brief Append value to the builder.
   */
  inline Status Append(value_const_pointer_t value, const size_t length) {
    RETURN_ON_ARROW_ERROR(this->buffer_writer_->Append(value, length));
    return Status::OK();
  }

  /**
   * @brief Append value to the builder.
   */
  inline Status Append(std::string const& value) {
    RETURN_ON_ARROW_ERROR(this->buffer_writer_->Append(value));
    return Status::OK();
  }

  /**
   * @brief Build the tensor.
   *
   * @param client The client connceted to the vineyard server.
   */
  Status Build(Client& client) override {
    std::shared_ptr<arrow::Array> array;
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(array, buffer_writer_->Finish());
    this->set_buffer_(std::make_shared<LargeStringArrayBuilder>(
        client, std::dynamic_pointer_cast<arrow::LargeStringArray>(array)));
    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::LargeStringBuilder> buffer_writer_;
};

class GlobalTensorBaseBuilder;

/**
 * @brief GlobalTensor is a holder for a set of tensor chunks that are
 * distributed over many vineyard nodes.
 */
class GlobalTensor : public Registered<GlobalTensor>, GlobalObject {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<GlobalTensor>{new GlobalTensor()});
  }

  void Construct(const ObjectMeta& meta) override;

  std::vector<int64_t> const& shape() const;
  std::vector<int64_t> const& partition_shape() const;

  /**
   * @brief Get the local partitions of the vineyard instance that is
   * connected from the client.
   *
   * @param client The client connected to a vineyard instance.
   * @return The vector of pointers to the local partitions.
   */
  const std::vector<std::shared_ptr<ITensor>>& LocalPartitions(
      Client& client) const;

  /**
   * @brief Get the local partitions stored in the given vineyard instance.
   *
   * @param instance_id The given ID of the vineyard instance.
   * @return The vector of pointers to the local partitions.
   */
  const std::vector<std::shared_ptr<ITensor>>& LocalPartitions(
      const InstanceID instance_id) const;

 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> partition_shape_;

  mutable std::map<InstanceID, std::vector<std::shared_ptr<ITensor>>>
      partitions_;

  friend class Client;
  friend class GlobalTensorBaseBuilder;
};

class GlobalTensorBaseBuilder : public ObjectBuilder {
 public:
  explicit GlobalTensorBaseBuilder(Client& client) {}

  explicit GlobalTensorBaseBuilder(GlobalTensor const& __value) {
    this->set_shape_(__value.shape_);
    this->set_partition_shape_(__value.partition_shape_);
    for (auto const& __partitions__items : __value.partitions_) {
      for (auto const& __partition : __partitions__items.second) {
        this->add_partitions_(__partition->id());
      }
    }
  }

  explicit GlobalTensorBaseBuilder(std::shared_ptr<GlobalTensor> const& __value)
      : GlobalTensorBaseBuilder(*__value) {}

  std::shared_ptr<Object> _Seal(Client& client) override;

  Status Build(Client& client) override { return Status::OK(); }

 protected:
  std::vector<int64_t> shape_;
  std::vector<int64_t> partition_shape_;
  std::vector<ObjectID> partitions_;

  void set_shape_(std::vector<int64_t> const& shape__) {
    this->shape_ = shape__;
  }

  void set_partition_shape_(std::vector<int64_t> const& partition_shape__) {
    this->partition_shape_ = partition_shape__;
  }

  void set_partitions_(std::vector<ObjectID> const& partitions__) {
    this->partitions_ = partitions__;
  }
  void set_partitions_(size_t const idx, ObjectID const& partitions__) {
    if (idx >= this->partitions_.size()) {
      this->partitions_.resize(idx + 1);
    }
    this->partitions_[idx] = partitions__;
  }
  void add_partitions_(ObjectID const& partitions__) {
    this->partitions_.emplace_back(partitions__);
  }
};

/**
 * @brief GlobalTensorBuilder is designed for building global tensors
 *
 */
class GlobalTensorBuilder : public GlobalTensorBaseBuilder {
 public:
  explicit GlobalTensorBuilder(Client& client)
      : GlobalTensorBaseBuilder(client) {}

  /**
   * @brief Get the partition shape of the global tensor.
   * Here the ith element represents how many partitions
   * are made on the ith axis
   */
  std::vector<int64_t> const& partition_shape() const;

  /**
   * @brief Set the partition shape of the global tensor.
   * Here the ith element represents how many partitions
   * are made on the ith axis
   */
  void set_partition_shape(std::vector<int64_t> const& partition_shape);

  /**
   * @brief Get the entire shape of the global tensor.
   *
   */
  std::vector<int64_t> const& shape() const;

  /**
   * @brief Set the entire shape of the global tensor.
   *
   */
  void set_shape(std::vector<int64_t> const& shape);

  /**
   * @brief Add a partition in the vineyard instance to the global tensor.
   *
   * @param instance_id The ID of the vineyard instance.
   * @param partition_id The ObjectID of the partition to added.
   *
   */
  void AddPartition(const ObjectID partition_id);

  /**
   * @brief Add a group of partitions in the vineyard instance
   * to the global tensor.
   *
   * @param instance_id The ID of the vineyard instance.
   * @param partition_id The vector of ObjectIDs for the
   * group of partitions to add.
   *
   */
  void AddPartitions(const std::vector<ObjectID>& partition_ids);

  /**
   * @brief Seal the meta data of the global tensor.
   * When creating a global tensor, clients from different
   * machines that are connected
   * to different vineyard instances will sync the partition info
   * to seal the meta data for the global tensor.
   *
   * @param client The client connected to the vineyard server.
   */
  std::shared_ptr<Object> _Seal(Client& client) override;

  /**
   * @brief Build the global tensor.
   *
   * @param client The client connected to the vineyard server.
   */
  Status Build(Client& client) override;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_TENSOR_H_
