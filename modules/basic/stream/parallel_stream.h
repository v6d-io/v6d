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

#ifndef MODULES_BASIC_STREAM_PARALLEL_STREAM_H_
#define MODULES_BASIC_STREAM_PARALLEL_STREAM_H_

#include <memory>
#include <string>
#include <vector>

#include "client/client.h"

namespace vineyard {

class ParallelStreamBuilder;

class ParallelStream : public Registered<ParallelStream> {
 public:
  static std::shared_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(std::make_shared<ParallelStream>());
  }

  void Construct(const ObjectMeta& meta) override {
    std::string __type_name = type_name<ParallelStream>();
    CHECK(meta.GetTypeName() == __type_name);
    this->meta_ = meta;
    this->id_ = meta.GetId();

    meta.GetKeyValue("size_", this->size_);
    for (size_t idx = 0; idx < this->size_; ++idx) {
      streams_.emplace_back(meta.GetMember("stream_" + std::to_string(idx)));
    }
  }

  template <typename T>
  std::shared_ptr<T> GetStream(int index) {
    return std::dynamic_pointer_cast<T>(streams_[index]);
  }

  ObjectMeta GetStreamMeta(int index) { return streams_[index]->meta(); }

  size_t GetStreamSize() { return size_; }

 private:
  size_t size_;
  std::vector<std::shared_ptr<Object>> streams_;

  friend class Client;
  friend class ParallelStreamBuilder;
};

/**
 * @brief ParallelStreamBuilder is desinged for building parallel stremas
 *
 */
class ParallelStreamBuilder : public ObjectBuilder {
 public:
  explicit ParallelStreamBuilder(Client& client) {}

  void AddStream(const ObjectID stream_id) { streams_.emplace_back(stream_id); }

  Status Build(Client& client) override { return Status::OK(); }

  std::shared_ptr<Object> _Seal(Client& client) override {
    // ensure the builder hasn't been sealed yet.
    ENSURE_NOT_SEALED(this);

    VINEYARD_CHECK_OK(this->Build(client));
    auto __value = std::make_shared<ParallelStream>();

    __value->meta_.SetTypeName(type_name<ParallelStream>());

    __value->size_ = streams_.size();
    __value->meta_.AddKeyValue("size_", __value->size_);

    for (size_t idx = 0; idx < streams_.size(); ++idx) {
      __value->meta_.AddMember("stream_" + std::to_string(idx), streams_[idx]);
    }
    __value->meta_.SetNBytes(0);

    VINEYARD_CHECK_OK(client.CreateMetaData(__value->meta_, __value->id_));
    VINEYARD_CHECK_OK(client.GetMetaData(__value->id_, __value->meta_));

    // mark the builder as sealed
    VINEYARD_CHECK_OK(client.Persist(__value->id()));
    this->set_sealed(true);

    return std::static_pointer_cast<Object>(__value);
  }

 private:
  std::vector<ObjectID> streams_;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_STREAM_PARALLEL_STREAM_H_
