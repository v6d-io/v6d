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

#ifndef SRC_CLIENT_DS_STREAM_H_
#define SRC_CLIENT_DS_STREAM_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/util/config.h"
#include "arrow/util/key_value_metadata.h"

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/core_types.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

template <typename T>
class StreamBuilder;

template <typename T>
class Stream : public Object {
 public:
  template <typename S>
  static ObjectID Make(Client& client,
                       std::map<std::string, std::string> const& params) {
    static_assert(std::is_base_of<Object, S>::value,
                  "Not a vineyard object type");

    ObjectID id = InvalidObjectID();
    ObjectMeta meta;

    meta.SetTypeName(type_name<S>());
    meta.AddKeyValue("params_", params);
    meta.SetNBytes(0);

    VINEYARD_CHECK_OK(client.CreateMetaData(meta, id));
    VINEYARD_CHECK_OK(client.CreateStream(id));
    return id;
  }

  template <typename S>
  static ObjectID Make(
      Client& client,
      std::unordered_map<std::string, std::string> const& params) {
    return Make<S>(client, std::map<std::string, std::string>(params.begin(),
                                                              params.end()));
  }

  Status Next(std::shared_ptr<T>& chunk) {
    RETURN_ON_ASSERT(client_ != nullptr && readonly_ == true,
                     "Expect a readonly stream");
    std::shared_ptr<Object> result = nullptr;
    auto status = client_->ClientBase::PullNextStreamChunk(this->id_, result);
    if (status.ok()) {
      chunk = std::dynamic_pointer_cast<T>(result);
      if (chunk == nullptr) {
        return Status::Invalid("Failed to cast object with type '" +
                               result->meta().GetTypeName() + "' to type '" +
                               type_name<T>() + "'");
      }
    }
    return status;
  }

  Status Push(std::shared_ptr<T> const& chunk) {
    RETURN_ON_ASSERT(client_ != nullptr && readonly_ == false,
                     "Expect a writeable stream");
    return client_->ClientBase::PushNextStreamChunk(this->id_, chunk->id());
  }

  Status Push(std::shared_ptr<Object> const& chunk) {
    RETURN_ON_ASSERT(client_ != nullptr && readonly_ == false,
                     "Expect a writeable stream");
    return client_->ClientBase::PushNextStreamChunk(this->id_, chunk->id());
  }

  Status Push(ObjectMeta const& chunk) {
    RETURN_ON_ASSERT(client_ != nullptr && readonly_ == false,
                     "Expect a writeable stream");
    return client_->ClientBase::PushNextStreamChunk(this->id_, chunk.GetId());
  }

  Status Push(ObjectID const& chunk) {
    RETURN_ON_ASSERT(client_ != nullptr && readonly_ == false,
                     "Expect a writeable stream");
    return client_->ClientBase::PushNextStreamChunk(this->id_, chunk);
  }

  Status Abort() {
    RETURN_ON_ASSERT(client_ != nullptr && readonly_ == false,
                     "Expect a writeable stream");
    if (stoped_) {
      return Status::OK();
    }
    stoped_ = true;
    return client_->ClientBase::StopStream(this->id_, true);
  }

  Status Finish() {
    RETURN_ON_ASSERT(client_ != nullptr && readonly_ == false,
                     "Expect a writeable stream");
    if (stoped_) {
      return Status::OK();
    }
    stoped_ = true;
    return client_->ClientBase::StopStream(this->id_, false);
  }

  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<Stream<T>>{new Stream<T>()});
  }

  void Construct(const ObjectMeta& meta) override {
    std::string __type_name = this->GetTypeName();
    VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                    "Expect typename '" + __type_name + "', but got '" +
                        meta.GetTypeName() + "'");
    this->meta_ = meta;
    this->id_ = meta.GetId();
    meta.GetKeyValue("params_", this->params_);
  }

  std::map<std::string, std::string> const& GetParams() {
    return this->params_;
  }

  Status OpenReader(Client* client) {
    if (client_ != nullptr) {
      return Status::StreamOpened();
    }
    RETURN_ON_ASSERT(client_ == nullptr && client != nullptr,
                     "Cannot open a stream multiple times or with null client");
    client_ = client;
    RETURN_ON_ERROR(client->OpenStream(this->id_, StreamOpenMode::read));
    readonly_ = true;
    return Status::OK();
  }

  Status OpenWriter(Client* client) {
    if (client_ != nullptr) {
      return Status::StreamOpened();
    }
    RETURN_ON_ASSERT(client_ == nullptr && client != nullptr,
                     "Cannot open a stream multiple times or with null client");
    client_ = client;
    RETURN_ON_ERROR(client->OpenStream(this->id_, StreamOpenMode::write));
    readonly_ = false;
    return Status::OK();
  }

  bool IsOpen() const { return client_ != nullptr; }

 protected:
  Client* client_ = nullptr;
  bool readonly_ = false;
  std::map<std::string, std::string> params_;

  virtual std::string GetTypeName() const { return type_name<Stream<T>>(); }

 private:
  bool stoped_;  // an optimization: avoid repeated idempotent requests.

  friend class StreamBuilder<T>;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_STREAM_H_
