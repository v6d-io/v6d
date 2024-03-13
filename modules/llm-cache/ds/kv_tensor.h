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

#ifndef MODULES_LLM_CACHE_DS_KV_TENSOR_H_
#define MODULES_LLM_CACHE_DS_KV_TENSOR_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"

namespace vineyard {

class KVTensor : public Object, public BareRegistered<KVTensor> {
 public:
  /**
   * @brief Construct a new KVTensor object.
   *
   * @return The pointer to the constructed KVTensor object.
   */
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<KVTensor>{new KVTensor()});
  }

  /**
   * @brief Construct a new KVTensor object.
   *
   * @param meta The metadata of the KVTensor object.
   *
   * @return The pointer to the constructed KVTensor object.
   */
  void Construct(const ObjectMeta& meta) override {
    std::string __type_name = type_name<KVTensor>();
    VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                    "Expect typename '" + __type_name + "', but got '" +
                        meta.GetTypeName() + "'");
    Object::Construct(meta);

    meta.GetKeyValue("value_type_", this->value_type_);
    this->buffer_ = std::dynamic_pointer_cast<Blob>(meta.GetMember("buffer_"));
    meta.GetKeyValue("shape_", this->shape_);
  }

  /**
   * @brief Get the shape of the tensor.
   *
   * @return The shape vector where the ith element represents
   * the size of the ith axis.
   */
  __attribute__((annotate("shared"))) std::vector<int64_t> const& shape()
      const {
    return shape_;
  }

  /**
   * @brief Get the type of tensor's elements.
   *
   * @return The type of the tensor's elements.
   */
  __attribute__((annotate("shared"))) std::string value_type() const {
    return this->value_type_;
  }

  /**
   * @brief Get the data pointer to the tensor's data buffer.
   *
   * @return The data pointer.
   */
  __attribute__((annotate("shared"))) const uint8_t* data() const {
    return reinterpret_cast<const uint8_t*>(buffer_->data());
  }

  /**
   * @brief Get the data in the tensor by index.
   *
   * @return The data reference.
   */
  __attribute__((annotate("shared"))) const uint8_t operator[](
      size_t index) const {
    return this->data()[index];
  }

  /**
   * @brief Get the buffer of the tensor.
   *
   * @return The shared pointer to an buffer which
   * holds the data buffer of the tensor.
   */
  __attribute__((annotate("shared"))) const std::shared_ptr<Buffer> buffer()
      const {
    return this->buffer_->Buffer();
  }

 private:
  __attribute__((annotate("shared"))) std::string value_type_;
  __attribute__((annotate("shared"))) std::shared_ptr<Blob> buffer_;
  __attribute__((annotate("shared"))) Tuple<int64_t> shape_;

  friend class Client;
  friend class KVTensorBuilder;
};

class KVTensorBuilder : public vineyard::ObjectBuilder {
 public:
  explicit KVTensorBuilder(Client& client) {}

  KVTensorBuilder(Client& client, std::vector<int64_t> const& shape) {
    this->set_value_type_("uint8");
    this->set_shape_(shape);
    int64_t size = std::accumulate(this->shape_.begin(), this->shape_.end(), 1,
                                   std::multiplies<int64_t>{});
    VINEYARD_CHECK_OK(
        client.CreateBlob(size * sizeof(uint8_t), buffer_writer_));
    this->data_ = reinterpret_cast<uint8_t*>(buffer_writer_->data());
  }

  /**
   * @brief Get the meta of the value.
   *
   * @param __value The value to get the meta from.
   *
   * @return The meta of the value.
   */
  ObjectMeta& ValueMetaRef(std::shared_ptr<KVTensor>& __value) {
    return __value->meta_;
  }

  /**
   * @brief Get the data pointer to the tensor's data buffer.
   *
   * @return The data pointer.
   */
  inline uint8_t* data() const { return this->data_; }

  /**
   * @brief Build the kv tensor.
   *
   * @param client The client connceted to the vineyard server.
   */
  Status Build(Client& client) override {
    this->set_buffer_(std::shared_ptr<BlobWriter>(std::move(buffer_writer_)));
    return Status::OK();
  }

  /**
   * @brief Seal the kv tensor.
   *
   * @param client The client connceted to the vineyard server.
   * @param object The object to seal.
   *
   * @return Status::OK() if the object is sealed successfully.
   */
  Status _Seal(Client& client, std::shared_ptr<Object>& object) override {
    // ensure the builder hasn't been sealed yet.
    ENSURE_NOT_SEALED(this);

    RETURN_ON_ERROR(this->Build(client));
    auto __value = std::make_shared<KVTensor>();
    object = __value;

    size_t __value_nbytes = 0;

    __value->meta_.SetTypeName(type_name<KVTensor>());

    __value->value_type_ = value_type_;
    __value->meta_.AddKeyValue("value_type_", __value->value_type_);

    using __buffer__value_type =
        typename decltype(__value->buffer_)::element_type;
    auto __value_buffer_ =
        std::dynamic_pointer_cast<__buffer__value_type>(buffer_->_Seal(client));
    __value->buffer_ = __value_buffer_;
    __value->meta_.AddMember("buffer_", __value->buffer_);
    __value_nbytes += __value_buffer_->nbytes();

    __value->shape_ = shape_;
    __value->meta_.AddKeyValue("shape_", __value->shape_);

    __value->meta_.SetNBytes(__value_nbytes);

    RETURN_ON_ERROR(client.CreateMetaData(__value->meta_, __value->id_));

    // mark the builder as sealed
    this->set_sealed(true);

    return Status::OK();
  }

 protected:
  std::string value_type_;
  std::shared_ptr<ObjectBase> buffer_;
  Tuple<int64_t> shape_;

  void set_value_type_(std::string const& value_type__) {
    this->value_type_ = value_type__;
  }

  void set_buffer_(std::shared_ptr<ObjectBase> const& buffer__) {
    this->buffer_ = buffer__;
  }

  void set_shape_(Tuple<int64_t> const& shape__) { this->shape_ = shape__; }

 private:
  std::unique_ptr<BlobWriter> buffer_writer_;
  uint8_t* data_;
  friend class KVTensor;
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_KV_TENSOR_H_
