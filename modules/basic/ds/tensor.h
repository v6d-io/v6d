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

#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/tensor.h"

#include "basic/ds/array.h"
#include "basic/ds/tensor.vineyard.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/ptree.h"

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
  inline T* data() const { return this->data_; }

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
 * @brief GlobalTensorBuilder is designed for building global tensors
 *
 */
class GlobalTensorBuilder : public GlobalTensorBaseBuilder {
 public:
  explicit GlobalTensorBuilder(Client& client)
      : GlobalTensorBaseBuilder(client), partitions_builder_(client) {}

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
  void AddPartition(const InstanceID instance_id, const ObjectID partition_id);

  /**
   * @brief Add a group of partitions in the vineyard instance
   * to the global tensor.
   *
   * @param instance_id The ID of the vineyard instance.
   * @param partition_id The vector of ObjectIDs for the
   * group of partitions to add.
   *
   */
  void AddPartitions(const InstanceID instance_id,
                     const std::vector<ObjectID>& partition_ids);

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

 protected:
  std::shared_ptr<ObjectSet> partitions_;
  ObjectSetBuilder partitions_builder_;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_TENSOR_H_
