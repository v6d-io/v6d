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

#ifndef MODULES_BASIC_STREAM_FIXED_BLOB_STREAM_H_
#define MODULES_BASIC_STREAM_FIXED_BLOB_STREAM_H_

#include <memory>
#include <string>
#include <vector>

#include "arrow/builder.h"
#include "arrow/status.h"

#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/stream.h"
#include "common/util/uuid.h"

namespace vineyard {

class FixedBlobStream : public BareRegistered<FixedBlobStream>,
                        public Stream<std::vector<Blob>> {
 public:
  explicit FixedBlobStream(ObjectMeta meta) { Construct(meta); }

  FixedBlobStream() {}

  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<FixedBlobStream>{new FixedBlobStream()});
  }

  Status Open(Client* client, StreamOpenMode mode, bool wait = false,
              uint64_t timeout = 0);

  Status ActivateStreamWithBuffer(std::vector<void*>& buffers);

  Status ActivateStreamWithBlob(std::vector<ObjectID>& blob_list);

  Status ActivateStreamWithOffset(std::vector<uint64_t>& offset_list);

  Status Push(uint64_t offset);

  Status CheckBlockReceived(int index, bool& finished);

  Status Abort(bool& success);

  Status Close();

  static Status Delete(Client* client, FixedBlobStream stream);

  Status PrintRecvInfo();

  void Construct(const ObjectMeta& meta) override {
    std::string __type_name = type_name<FixedBlobStream>();
    VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                    "Expect typename '" + __type_name + "', but got '" +
                        meta.GetTypeName() + "'");
    Object::Construct(meta);
    this->meta_.GetKeyValue("nums", this->buffer_nums_);
    this->meta_.GetKeyValue("size", this->buffer_size_);
    this->meta_.GetKeyValue("is_remote", this->is_remote_);
    this->meta_.GetKeyValue("rpc_endpoint", this->rpc_endpoint_);
    this->meta_.GetKeyValue("stream_name", this->stream_name_);
  }

  ObjectID GetId() const { return this->id_; }

 protected:
  int buffer_nums_ = 0;
  size_t buffer_size_ = 0;
  std::string stream_name_ = "";
  bool is_remote_ = false;
  std::string rpc_endpoint_ = "";
  void* recv_flag_mem_ = nullptr;
  int recv_mem_fd_ = -1;
  Client* client_ = nullptr;
};

class FixedStreamBuilder {
 public:
  explicit FixedStreamBuilder(Client& client) : client_(client) {
    meta_.SetTypeName(type_name<FixedBlobStream>());
    meta_.SetNBytes(0);
  }

  template <typename Value>
  void AddKeyValue(const std::string& key, const Value& value) {
    meta_.AddKeyValue(key, value);
  }

  Status Finish(std::string stream_name, int nums = 0, size_t size = 0) {
    ObjectID id = InvalidObjectID();
    RETURN_ON_ERROR(client_.CreateFixedStream(id, stream_name, nums, size));
    meta_.SetId(id);
    return Status::OK();
  }

  static Status Make(Client& client, std::shared_ptr<FixedBlobStream>& stream,
                     ObjectID remote_id, int nums, size_t size, bool is_remote,
                     std::string rpc_endpoint) {
    FixedStreamBuilder builder(client);
    builder.AddKeyValue("nums", nums);
    builder.AddKeyValue("size", size);
    builder.AddKeyValue("stream_name", "");
    builder.AddKeyValue("is_remote", is_remote);
    builder.AddKeyValue("rpc_endpoint", rpc_endpoint);
    builder.AddKeyValue("remote_id", remote_id);
    RETURN_ON_ERROR(builder.Finish("", nums, size));
    stream = std::make_shared<FixedBlobStream>(builder.meta_);
    return Status::OK();
  }

  static Status Make(Client& client, std::shared_ptr<FixedBlobStream>& stream,
                     std::string stream_name, int nums, size_t size,
                     bool is_remote = false, std::string rpc_endpoint = "") {
    FixedStreamBuilder builder(client);
    builder.AddKeyValue("nums", nums);
    builder.AddKeyValue("size", size);
    builder.AddKeyValue("stream_name", stream_name);
    builder.AddKeyValue("is_remote", is_remote);
    builder.AddKeyValue("rpc_endpoint", rpc_endpoint);
    builder.AddKeyValue("remote_id", InvalidObjectID());
    RETURN_ON_ERROR(builder.Finish(stream_name, nums, size));
    stream = std::make_shared<FixedBlobStream>(builder.meta_);
    return Status::OK();
  }

 private:
  Client& client_;
  ObjectMeta meta_;
};

template <>
struct stream_type<std::vector<Blob>> {
  using type = FixedBlobStream;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_STREAM_FIXED_BLOB_STREAM_H_
