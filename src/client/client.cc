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

#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
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
  RETURN_ON_ASSERT(!client.Connected(),
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

  std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
  RETURN_ON_ERROR(GetBuffers(meta.GetBufferSet()->AllBufferIds(), buffers));

  for (auto const& id : meta.GetBufferSet()->AllBufferIds()) {
    const auto& buffer = buffers.find(id);
    if (buffer != buffers.end()) {
      meta.SetBuffer(id, buffer->second);
    }
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

  std::set<ObjectID> blob_ids;
  for (size_t idx = 0; idx < trees.size(); ++idx) {
    metas[idx].SetMetaData(this, trees[idx]);
    for (const auto& id : metas[idx].GetBufferSet()->AllBufferIds()) {
      blob_ids.emplace(id);
    }
  }

  std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
  RETURN_ON_ERROR(GetBuffers(blob_ids, buffers));

  for (auto& meta : metas) {
    for (auto const id : meta.GetBufferSet()->AllBufferIds()) {
      const auto& buffer = buffers.find(id);
      if (buffer != buffers.end()) {
        meta.SetBuffer(id, buffer->second);
      }
    }
  }
  return Status::OK();
}

Status Client::CreateBlob(size_t size, std::unique_ptr<BlobWriter>& blob) {
  ENSURE_CONNECTED(this);

  ObjectID object_id = InvalidObjectID();
  Payload object;
  std::shared_ptr<arrow::MutableBuffer> buffer = nullptr;
  RETURN_ON_ERROR(CreateBuffer(size, object_id, object, buffer));
  blob.reset(new BlobWriter(object_id, object, buffer));
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
    RETURN_ON_ERROR(mmapToClient(object.store_fd, object.map_size, false, true,
                                 &mmapped_ptr));
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
    RETURN_ON_ERROR(mmapToClient(object.store_fd, object.map_size, true, true,
                                 &mmapped_ptr));
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
  std::set<ObjectID> blob_ids;
  metas.resize(meta_trees.size());
  size_t cnt = 0;
  for (auto const& kv : meta_trees) {
    metas[cnt].SetMetaData(this, kv.second);
    for (auto const& id : metas[cnt].GetBufferSet()->AllBufferIds()) {
      blob_ids.emplace(id);
    }
    cnt += 1;
  }

  // retrive blobs
  std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
  VINEYARD_CHECK_OK(GetBuffers(blob_ids, buffers));

  // construct objects
  std::vector<std::shared_ptr<Object>> objects;
  objects.reserve(metas.size());
  for (auto& meta : metas) {
    for (auto const id : meta.GetBufferSet()->AllBufferIds()) {
      const auto& buffer = buffers.find(id);
      if (buffer != buffers.end()) {
        meta.SetBuffer(id, buffer->second);
      }
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

Status Client::CreateArena(const size_t size, int& fd, size_t& available_size,
                           uintptr_t& base, uintptr_t& space) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteMakeArenaRequest(size, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMakeArenaReply(message_in, fd, available_size, base));
  VINEYARD_ASSERT(size == std::numeric_limits<size_t>::max() ||
                  size == available_size);
  uint8_t* mmapped_ptr = nullptr;
  VINEYARD_CHECK_OK(
      mmapToClient(fd, available_size, false, false, &mmapped_ptr));
  space = reinterpret_cast<uintptr_t>(mmapped_ptr);
  return Status::OK();
}

Status Client::ReleaseArena(const int fd, std::vector<size_t> const& offsets,
                            std::vector<size_t> const& sizes) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteFinalizeArenaRequest(fd, offsets, sizes, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadFinalizeArenaReply(message_in));
  return Status::OK();
}

Status Client::CreateBuffer(const size_t size, ObjectID& id, Payload& payload,
                            std::shared_ptr<arrow::MutableBuffer>& buffer) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteCreateBufferRequest(size, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadCreateBufferReply(message_in, id, payload));
  RETURN_ON_ASSERT(static_cast<size_t>(payload.data_size) == size);
  if (payload.data_size > 0) {
    uint8_t* shared = nullptr;
    RETURN_ON_ERROR(
        mmapToClient(payload.store_fd, payload.map_size, false, true, &shared));
    buffer = std::make_shared<arrow::MutableBuffer>(
        shared + payload.data_offset, payload.data_size);
  }
  return Status::OK();
}

Status Client::GetBuffer(const ObjectID id,
                         std::shared_ptr<arrow::Buffer>& buffer) {
  std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
  RETURN_ON_ERROR(GetBuffers({id}, buffers));
  if (buffers.empty()) {
    return Status::ObjectNotExists();
  }
  buffer = buffers.at(id);
  return Status::OK();
}

Status Client::GetBuffers(
    const std::set<ObjectID>& ids,
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>>& buffers) {
  if (ids.empty()) {
    return Status::OK();
  }
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteGetBuffersRequest(ids, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  std::map<ObjectID, Payload> payloads;
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, payloads));
  for (auto const& item : payloads) {
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    if (item.second.data_size > 0) {
      uint8_t* shared = nullptr;
      VINEYARD_CHECK_OK(mmapToClient(item.second.store_fd, item.second.map_size,
                                     true, true, &shared));
      buffer = std::make_shared<arrow::Buffer>(shared + item.second.data_offset,
                                               item.second.data_size);
    }
    buffers.emplace(item.first, buffer);
  }
  return Status::OK();
}

Status Client::DropBuffer(const ObjectID id, const int fd) {
  ENSURE_CONNECTED(this);

  // unmap from client
  auto entry = mmap_table_.find(fd);
  if (entry != mmap_table_.end()) {
    mmap_table_.erase(entry);
  }

  // free on server
  std::string message_out;
  WriteDropBufferRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDropBufferReply(message_in));
  return Status::OK();
}

Status Client::mmapToClient(int fd, int64_t map_size, bool readonly,
                            bool realign, uint8_t** ptr) {
  auto entry = mmap_table_.find(fd);
  if (entry == mmap_table_.end()) {
    int client_fd = recv_fd(vineyard_conn_);
    if (fd <= 0) {
      return Status::IOError(
          "Failed to receieve file descriptor from the socket");
    }
    auto mmap_entry = std::unique_ptr<MmapEntry>(
        new MmapEntry(client_fd, map_size, readonly, realign));
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
