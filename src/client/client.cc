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

#include "client/client.h"

#include <mutex>
#include <utility>

#include "boost/range/combine.hpp"

#include "client/ds/blob.h"
#include "client/io.h"
#include "client/utils.h"
#include "common/memory/fling.h"
#include "common/util/boost.h"
#include "common/util/protocols.h"

namespace vineyard {

Status Client::Connect() {
  if (const char* env_p = std::getenv("VINEYARD_IPC_SOCKET")) {
    return Connect(std::string(env_p));
  }
  return Status::ConnectionError(
      "Environment variable VINEYARD_IPC_SOCKET does't exists");
}

Status Client::Connect(const std::string& ipc_socket) {
  std::lock_guard<std::recursive_mutex> guard(client_mutex_);
  RETURN_ON_ASSERT(!connected_ || ipc_socket == ipc_socket_);
  if (connected_) {
    return Status::OK();
  }
  ipc_socket_ = ipc_socket;
  RETURN_ON_ERROR(connect_ipc_socket_retry(ipc_socket, vineyard_conn_));
  std::string message_out;
  WriteRegisterRequest(message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  std::string ipc_socket_value, rpc_endpoint_value;
  RETURN_ON_ERROR(ReadRegisterReply(message_in, ipc_socket_value,
                                    rpc_endpoint_value, instance_id_,
                                    server_version_));
  rpc_endpoint_ = rpc_endpoint_value;
  connected_ = true;

  if (!compatible_server(server_version_)) {
    LOG(ERROR) << "Warning: this version of vineyard client may be "
                  "incompatible with connected server: "
               << "client's version is " << vineyard_version()
               << ", while the server's version is " << server_version_;
  }

  return Status::OK();
}

Status Client::Fork(Client& client) {
  RETURN_ON_ASSERT(client.Connected(),
                   "The client has already been connected to vineyard server");
  return client.Connect(ipc_socket_);
}

Client& Client::Default() {
  static std::once_flag flag;
  static Client* client = new Client();
  std::call_once(flag, [&] { VINEYARD_CHECK_OK(client->Connect()); });
  return *client;
}

Status Client::GetMetaData(const ObjectID id, ObjectMeta& meta,
                           const bool sync_remote) {
  ENSURE_CONNECTED(this);
  json tree;
  RETURN_ON_ERROR(GetData(id, tree, sync_remote));
  meta.SetMetaData(this, tree);

  std::unordered_map<ObjectID, Payload> buffers;
  RETURN_ON_ERROR(GetBuffers(meta.GetBlobSet()->AllBlobIds(), buffers));

  for (auto const& id : meta.GetBlobSet()->AllBlobIds()) {
    auto object = buffers.find(id);
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    if (object != buffers.end()) {
      uint8_t* mmapped_ptr = nullptr;
      if (object->second.data_size > 0) {
        RETURN_ON_ERROR(mmapToClient(object->second.store_fd,
                                     object->second.map_size, true,
                                     &mmapped_ptr));
      }
      buffer = arrow::Buffer::Wrap(mmapped_ptr + object->second.data_offset,
                                   object->second.data_size);
    }
    meta.SetBlob(id, buffer);
  }
  return Status::OK();
}

Status Client::GetMetaData(const std::vector<ObjectID>& ids,
                           std::vector<ObjectMeta>& metas,
                           const bool sync_remote) {
  ENSURE_CONNECTED(this);
  std::vector<json> trees;
  RETURN_ON_ERROR(GetData(ids, trees, sync_remote));
  metas.resize(trees.size());

  std::unordered_set<ObjectID> blob_ids;
  for (size_t idx = 0; idx < trees.size(); ++idx) {
    metas[idx].SetMetaData(this, trees[idx]);
    for (const auto& id : metas[idx].GetBlobSet()->AllBlobIds()) {
      blob_ids.emplace(id);
    }
  }

  std::unordered_map<ObjectID, Payload> buffers;
  RETURN_ON_ERROR(GetBuffers(blob_ids, buffers));

  for (auto& meta : metas) {
    for (auto const id : meta.GetBlobSet()->AllBlobIds()) {
      auto object = buffers.find(id);
      std::shared_ptr<arrow::Buffer> buffer = nullptr;
      if (object != buffers.end()) {
        uint8_t* mmapped_ptr = nullptr;
        if (object->second.data_size > 0) {
          RETURN_ON_ERROR(mmapToClient(object->second.store_fd,
                                       object->second.map_size, true,
                                       &mmapped_ptr));
        }
        buffer = std::make_shared<arrow::Buffer>(
            mmapped_ptr + object->second.data_offset, object->second.data_size);
      }
      meta.SetBlob(id, buffer);
    }
  }

  return Status::OK();
}

Status Client::CreateBlob(size_t size, std::unique_ptr<BlobWriter>& blob) {
  ENSURE_CONNECTED(this);

  ObjectID object_id;
  Payload object;
  RETURN_ON_ERROR(CreateBuffer(size, object_id, object));
  RETURN_ON_ASSERT((size_t) object.data_size == size);
  uint8_t* mmapped_ptr = nullptr;
  if (object.data_size > 0) {
    RETURN_ON_ERROR(
        mmapToClient(object.store_fd, object.map_size, false, &mmapped_ptr));
  }
  std::shared_ptr<arrow::MutableBuffer> buffer =
      std::make_shared<arrow::MutableBuffer>(mmapped_ptr + object.data_offset,
                                             object.data_size);
  blob.reset(new BlobWriter(object_id, buffer));
  return Status::OK();
}

Status Client::CreateStream(const ObjectID& id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteCreateStreamRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadCreateStreamReply(message_in));
  return Status::OK();
}

Status Client::OpenStream(const ObjectID& id, OpenStreamMode mode) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteOpenStreamRequest(id, static_cast<int64_t>(mode), message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadOpenStreamReply(message_in));
  return Status::OK();
}

Status Client::GetNextStreamChunk(ObjectID const id, size_t const size,
                                  std::unique_ptr<arrow::MutableBuffer>& blob) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteGetNextStreamChunkRequest(id, size, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  Payload object;
  RETURN_ON_ERROR(ReadGetNextStreamChunkReply(message_in, object));
  RETURN_ON_ASSERT(size == static_cast<size_t>(object.data_size),
                   "The size of returned chunk doesn't match");
  uint8_t* mmapped_ptr = nullptr;
  if (object.data_size != 0) {
    RETURN_ON_ERROR(
        mmapToClient(object.store_fd, object.map_size, false, &mmapped_ptr));
  }
  blob.reset(new arrow::MutableBuffer(mmapped_ptr + object.data_offset,
                                      object.data_size));
  return Status::OK();
}

Status Client::PullNextStreamChunk(ObjectID const id,
                                   std::unique_ptr<arrow::Buffer>& blob) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePullNextStreamChunkRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  Payload object;
  RETURN_ON_ERROR(ReadPullNextStreamChunkReply(message_in, object));
  uint8_t* mmapped_ptr = nullptr;
  if (object.data_size != 0) {
    RETURN_ON_ERROR(
        mmapToClient(object.store_fd, object.map_size, true, &mmapped_ptr));
  }
  blob.reset(
      new arrow::Buffer(mmapped_ptr + object.data_offset, object.data_size));
  return Status::OK();
}

Status Client::StopStream(ObjectID const id, const bool failed) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteStopStreamRequest(id, failed, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadStopStreamReply(message_in));
  return Status::OK();
}

std::shared_ptr<Object> Client::GetObject(const ObjectID id) {
  ObjectMeta meta;
  VINEYARD_CHECK_OK(this->GetMetaData(id, meta, true));
  VINEYARD_ASSERT(!meta.MetaData().empty());
  auto object = ObjectFactory::Create(meta.GetTypeName());
  if (object == nullptr) {
    object = std::shared_ptr<Object>(new Object());
  }
  object->Construct(meta);
  return object;
}

Status Client::GetObject(const ObjectID id, std::shared_ptr<Object>& object) {
  ObjectMeta meta;
  RETURN_ON_ERROR(this->GetMetaData(id, meta, true));
  RETURN_ON_ASSERT(!meta.MetaData().empty());
  object = ObjectFactory::Create(meta.GetTypeName());
  if (object == nullptr) {
    object = std::shared_ptr<Object>(new Object());
  }
  object->Construct(meta);
  return Status::OK();
}

std::vector<std::shared_ptr<Object>> Client::GetObjects(
    const std::vector<ObjectID>& ids) {
  std::vector<ObjectMeta> metas;
  VINEYARD_CHECK_OK(this->GetMetaData(ids, metas, true));
  for (auto const& meta : metas) {
    if (meta.MetaData().empty()) {
      VINEYARD_ASSERT(!meta.MetaData().empty());
    }
  }
  std::vector<std::shared_ptr<Object>> objects;
  objects.reserve(ids.size());
  for (auto const& meta : metas) {
    auto object = ObjectFactory::Create(meta.GetTypeName());
    if (object == nullptr) {
      object = std::shared_ptr<Object>(new Object());
    }
    object->Construct(meta);
    objects.emplace_back(object);
  }
  return objects;
}

std::vector<std::shared_ptr<Object>> Client::ListObjects(
    std::string const& pattern, const bool regex, size_t const limit) {
  std::unordered_map<ObjectID, json> meta_trees;
  VINEYARD_CHECK_OK(ListData(pattern, regex, limit, meta_trees));

  std::vector<ObjectMeta> metas;
  std::unordered_set<ObjectID> blob_ids;
  metas.resize(meta_trees.size());
  size_t cnt = 0;
  for (auto const& kv : meta_trees) {
    metas[cnt].SetMetaData(this, kv.second);
    for (auto const& id : metas[cnt].GetBlobSet()->AllBlobIds()) {
      blob_ids.emplace(id);
    }
    cnt += 1;
  }

  // retrive blobs
  std::unordered_map<ObjectID, Payload> buffers;
  VINEYARD_CHECK_OK(GetBuffers(blob_ids, buffers));

  // construct objects
  std::vector<std::shared_ptr<Object>> objects;
  objects.reserve(metas.size());
  for (auto& meta : metas) {
    for (auto const id : meta.GetBlobSet()->AllBlobIds()) {
      auto object = buffers.find(id);
      std::shared_ptr<arrow::Buffer> buffer = nullptr;
      if (object != buffers.end()) {
        uint8_t* mmapped_ptr = nullptr;
        if (object->second.data_size) {
          VINEYARD_CHECK_OK(mmapToClient(object->second.store_fd,
                                         object->second.map_size, true,
                                         &mmapped_ptr));
        }
        buffer = std::make_shared<arrow::Buffer>(
            mmapped_ptr + object->second.data_offset, object->second.data_size);
      }
      meta.SetBlob(id, buffer);
    }

    auto object = ObjectFactory::Create(meta.GetTypeName());
    if (object == nullptr) {
      object = std::shared_ptr<Object>(new Object());
    }
    object->Construct(meta);
    objects.emplace_back(object);
  }
  return objects;
}

Status Client::CreateBuffer(const size_t size, ObjectID& id, Payload& object) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteCreateBufferRequest(size, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadCreateBufferReply(message_in, id, object));
  return Status::OK();
}

Status Client::GetBuffer(const ObjectID id, Payload& object) {
  std::unordered_map<ObjectID, Payload> objects;
  RETURN_ON_ERROR(GetBuffers({id}, objects));
  if (objects.empty()) {
    return Status::ObjectNotExists();
  }
  object = objects.at(id);
  return Status::OK();
}

Status Client::GetBuffers(const std::unordered_set<ObjectID>& ids,
                          std::unordered_map<ObjectID, Payload>& objects) {
  if (ids.empty()) {
    return Status::OK();
  }
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteGetBuffersRequest(ids, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, objects));
  return Status::OK();
}

Status Client::mmapToClient(int fd, int64_t map_size, bool readonly,
                            uint8_t** ptr) {
  auto entry = mmap_table_.find(fd);
  if (entry == mmap_table_.end()) {
    int client_fd = recv_fd(vineyard_conn_);
    if (fd <= 0) {
      return Status::IOError(
          "Failed to receieve file descriptor from the socket");
    }
    auto mmap_entry = std::unique_ptr<MmapEntry>(
        new MmapEntry(client_fd, map_size, readonly));
    entry = mmap_table_.emplace(fd, std::move(mmap_entry)).first;
  }
  if (readonly) {
    *ptr = entry->second->map_readonly();
    if (*ptr == nullptr) {
      return Status::IOError("Failed to mmap received fd as a readonly buffer");
    }

  } else {
    *ptr = entry->second->map_readwrite();
    if (*ptr == nullptr) {
      return Status::IOError("Failed to mmap received fd as a writable buffer");
    }
  }
  return Status::OK();
}

Client::~Client() { Disconnect(); }

}  // namespace vineyard
