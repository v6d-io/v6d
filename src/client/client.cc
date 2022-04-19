/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
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
#include "common/util/logging.h"
#include "common/util/protocols.h"
#include "common/util/uuid.h"

namespace vineyard {

BasicIPCClient::BasicIPCClient() : shm_(new detail::SharedMemoryManager(-1)) {}

Status BasicIPCClient::Connect(const std::string& ipc_socket,
                               std::string const& store_type) {
  std::lock_guard<std::recursive_mutex> guard(client_mutex_);
  RETURN_ON_ASSERT(!connected_ || ipc_socket == ipc_socket_);
  if (connected_) {
    return Status::OK();
  }
  ipc_socket_ = ipc_socket;
  RETURN_ON_ERROR(connect_ipc_socket_retry(ipc_socket, vineyard_conn_));
  std::string message_out;
  WriteRegisterRequest(message_out, store_type);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  std::string ipc_socket_value, rpc_endpoint_value;
  bool store_match;
  RETURN_ON_ERROR(ReadRegisterReply(message_in, ipc_socket_value,
                                    rpc_endpoint_value, instance_id_,
                                    session_id_, server_version_, store_match));
  rpc_endpoint_ = rpc_endpoint_value;
  connected_ = true;

  if (!compatible_server(server_version_)) {
    std::clog << "[warn] Warning: this version of vineyard client may be "
                 "incompatible with connected server: "
              << "client's version is " << vineyard_version()
              << ", while the server's version is " << server_version_
              << std::endl;
  }

  shm_.reset(new detail::SharedMemoryManager(vineyard_conn_));

  if (!store_match) {
    Disconnect();
    return Status::Invalid("Mismatched store type");
  }
  return Status::OK();
}

Status BasicIPCClient::Open(std::string const& ipc_socket,
                            std::string const& bulk_store_type) {
  RETURN_ON_ASSERT(!this->connected_,
                   "The client has already been connected to vineyard server");
  std::string socket_path;
  VINEYARD_CHECK_OK(Connect(ipc_socket, "Any"));

  {
    std::lock_guard<std::recursive_mutex> guard(client_mutex_);
    std::string message_out;
    WriteNewSessionRequest(message_out, bulk_store_type);
    RETURN_ON_ERROR(doWrite(message_out));
    json message_in;
    RETURN_ON_ERROR(doRead(message_in));
    RETURN_ON_ERROR(ReadNewSessionReply(message_in, socket_path));
  }

  Disconnect();
  VINEYARD_CHECK_OK(Connect(socket_path, bulk_store_type));
  return Status::OK();
}

Client::~Client() { Disconnect(); }

Status Client::Connect() {
  auto ep = read_env("VINEYARD_IPC_SOCKET");
  if (!ep.empty()) {
    return Connect(ep);
  }
  return Status::ConnectionError(
      "Environment variable VINEYARD_IPC_SOCKET does't exists");
}

Status Client::Connect(const std::string& ipc_socket) {
  return BasicIPCClient::Connect(ipc_socket, /*bulk_store_type=*/"Normal");
}

Status Client::Open(std::string const& ipc_socket) {
  return BasicIPCClient::Open(ipc_socket, /*bulk_store_type=*/"Normal");
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
  meta.Reset();
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
    metas[idx].Reset();
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
  uint8_t *mmapped_ptr = nullptr, *dist = nullptr;
  if (object.data_size > 0) {
    RETURN_ON_ERROR(shm_->Mmap(object.store_fd, object.map_size,
                               object.pointer - object.data_offset, false, true,
                               &mmapped_ptr));
    dist = mmapped_ptr + object.data_offset;
  }
  blob.reset(new arrow::MutableBuffer(dist, object.data_size));
  return Status::OK();
}

Status Client::PullNextStreamChunk(ObjectID const id,
                                   std::unique_ptr<arrow::Buffer>& chunk) {
  std::shared_ptr<Object> buffer;
  RETURN_ON_ERROR(ClientBase::PullNextStreamChunk(id, buffer));
  if (auto casted = std::dynamic_pointer_cast<vineyard::Blob>(buffer)) {
    chunk.reset(
        new arrow::Buffer(reinterpret_cast<const uint8_t*>(casted->data()),
                          casted->allocated_size()));
    return Status::OK();
  }
  return Status::Invalid("Expect buffer, but got '" +
                         buffer->meta().GetTypeName() + "'");
}

std::shared_ptr<Object> Client::GetObject(const ObjectID id) {
  ObjectMeta meta;
  VINEYARD_CHECK_OK(this->GetMetaData(id, meta, true));
  VINEYARD_ASSERT(!meta.MetaData().empty());
  auto object = ObjectFactory::Create(meta.GetTypeName());
  if (object == nullptr) {
    object = std::unique_ptr<Object>(new Object());
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
    object = std::unique_ptr<Object>(new Object());
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
      object = std::unique_ptr<Object>(new Object());
    }
    object->Construct(meta);
    objects.emplace_back(std::shared_ptr<Object>(object.release()));
  }
  return objects;
}

std::vector<ObjectMeta> Client::ListObjectMeta(std::string const& pattern,
                                               const bool regex,
                                               size_t const limit,
                                               bool nobuffer) {
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

  if (nobuffer) {
    return metas;
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
  }
  return metas;
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
      object = std::unique_ptr<Object>(new Object());
    }
    object->Construct(meta);
    objects.emplace_back(std::shared_ptr<Object>(object.release()));
  }
  return objects;
}

bool Client::IsSharedMemory(const void* target) const {
  return shm_->Exists(target);
}

bool Client::IsSharedMemory(const uintptr_t target) const {
  return shm_->Exists(target);
}

bool Client::IsSharedMemory(const void* target, ObjectID& object_id) const {
  return shm_->Exists(target, object_id);
}

bool Client::IsSharedMemory(const uintptr_t target, ObjectID& object_id) const {
  return shm_->Exists(target, object_id);
}

Status Client::AllocatedSize(const ObjectID id, size_t& size) {
  ENSURE_CONNECTED(this);
  json tree;
  RETURN_ON_ERROR(GetData(id, tree, false));
  ObjectMeta meta;
  meta.SetMetaData(this, tree);

  std::map<ObjectID, size_t> sizes;
  RETURN_ON_ERROR(GetBufferSizes(meta.GetBufferSet()->AllBufferIds(), sizes));
  size = 0;
  for (auto const& sz : sizes) {
    if (sz.second > 0) {
      size += sz.second;
    }
  }
  return Status::OK();
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
      shm_->Mmap(fd, available_size, nullptr, false, false, &mmapped_ptr));
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

  uint8_t *shared = nullptr, *dist = nullptr;
  if (payload.data_size > 0) {
    RETURN_ON_ERROR(shm_->Mmap(payload.store_fd, payload.map_size,
                               payload.pointer - payload.data_offset, false,
                               true, &shared));
    dist = shared + payload.data_offset;
  }
  buffer = std::make_shared<arrow::MutableBuffer>(dist, payload.data_size);

  return Status::OK();
}

Status Client::GetBuffer(const ObjectID id,
                         std::shared_ptr<arrow::Buffer>& buffer) {
  std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
  RETURN_ON_ERROR(GetBuffers({id}, buffers));
  if (buffers.empty()) {
    return Status::ObjectNotExists("buffer not exists: " +
                                   ObjectIDToString(id));
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
  std::vector<Payload> payloads;
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, payloads));

  for (auto const& item : payloads) {
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    uint8_t *shared = nullptr, *dist = nullptr;
    if (item.data_size > 0) {
      VINEYARD_CHECK_OK(shm_->Mmap(item.store_fd, item.map_size,
                                   item.pointer - item.data_offset, true, true,
                                   &shared));
      dist = shared + item.data_offset;
    }
    buffer = std::make_shared<arrow::Buffer>(dist, item.data_size);
    buffers.emplace(item.object_id, buffer);
  }
  return Status::OK();
}

Status Client::GetBufferSizes(const std::set<ObjectID>& ids,
                              std::map<ObjectID, size_t>& sizes) {
  if (ids.empty()) {
    return Status::OK();
  }
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteGetBuffersRequest(ids, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  std::vector<Payload> payloads;
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, payloads));
  for (auto const& item : payloads) {
    uint8_t* shared = nullptr;
    if (item.data_size > 0) {
      VINEYARD_CHECK_OK(shm_->Mmap(item.store_fd, item.map_size,
                                   item.pointer - item.data_offset, true, true,
                                   &shared));
    }
    sizes.emplace(item.object_id, item.data_size);
  }
  return Status::OK();
}

Status Client::DropBuffer(const ObjectID id, const int fd) {
  ENSURE_CONNECTED(this);

  // unmap from client
  //
  // FIXME: the erase may cause re-recv fd problem, needs further inspection.

  // auto entry = mmap_table_.find(fd);
  // if (entry != mmap_table_.end()) {
  //   mmap_table_.erase(entry);
  // }

  // free on server
  std::string message_out;
  WriteDropBufferRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDropBufferReply(message_in));
  return Status::OK();
}

Status Client::Seal(ObjectID const& object_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteSealRequest(object_id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));

  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadSealReply(message_in));
  return Status::OK();
}

Status Client::ShallowCopy(ObjectID const id, ObjectID& target_id,
                           Client& source_client) {
  ENSURE_CONNECTED(this);
  ObjectMeta meta;
  json tree;

  RETURN_ON_ERROR(source_client.GetData(id, tree, /*sync_remote==*/true));
  meta.SetMetaData(this, tree);
  auto bids = meta.GetBufferSet()->AllBufferIds();
  std::map<ObjectID, ObjectID> mapping;
  for (auto const& id : bids) {
    mapping.emplace(id, id);
  }

  // create buffers in normal bulk store.
  std::string message_out;
  WriteMoveBuffersOwnershipRequest(mapping, source_client.session_id(),
                                   message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMoveBuffersOwnershipReply(message_in));

  // reconstruct meta tree
  auto meta_tree = meta.MutMetaData();
  std::function<ObjectID(json&)> reconstruct =
      [&](json& meta_tree) -> ObjectID {
    for (auto& item : meta_tree.items()) {
      if (item.value().is_object() && !item.value().empty()) {
        auto sub_id = ObjectIDFromString(
            item.value()["id"].get_ref<std::string const&>());
        auto new_sub_id = sub_id;
        if (mapping.find(sub_id) == mapping.end()) {
          new_sub_id = reconstruct(item.value());
          mapping.emplace(sub_id, new_sub_id);
        } else {
          new_sub_id = mapping[sub_id];
        }
        if (!IsBlob(new_sub_id)) {
          ObjectMeta sub_meta;
          VINEYARD_CHECK_OK(GetMetaData(new_sub_id, sub_meta));
          meta_tree[item.key()] = sub_meta.MetaData();
        }
      }
    }
    ObjectMeta new_meta;
    ObjectID new_id;
    new_meta.SetMetaData(this, meta_tree);
    VINEYARD_CHECK_OK(CreateMetaData(new_meta, new_id));
    return new_id;
  };

  target_id = reconstruct(meta_tree);

  return Status::OK();
}

Status Client::ShallowCopy(PlasmaID const plasma_id, ObjectID& target_id,
                           PlasmaClient& source_client) {
  ENSURE_CONNECTED(this);
  std::set<PlasmaID> plasma_ids;
  std::map<PlasmaID, PlasmaPayload> plasma_payloads;
  plasma_ids.emplace(plasma_id);
  // get PlasmaPayload to get the object_id and data_size
  VINEYARD_CHECK_OK(source_client.GetPayloads(plasma_ids, plasma_payloads));

  std::map<PlasmaID, ObjectID> mapping;
  for (auto const& item : plasma_payloads) {
    mapping.emplace(item.first, item.second.object_id);
  }

  // create buffers in normal bulk store.
  std::string message_out;
  WriteMoveBuffersOwnershipRequest(mapping, source_client.session_id(),
                                   message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMoveBuffersOwnershipReply(message_in));

  /// no need to reconstruct meta_tree since we do not support composable object
  /// for plasma store.
  target_id = plasma_payloads.at(plasma_id).object_id;
  return Status::OK();
}

PlasmaClient::~PlasmaClient() {}

// dummy implementation
Status PlasmaClient::GetMetaData(const ObjectID id, ObjectMeta& meta_data,
                                 const bool sync_remote) {
  return Status::Invalid("Unsupported.");
}

Status PlasmaClient::Seal(PlasmaID const& plasma_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePlasmaSealRequest(plasma_id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));

  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadSealReply(message_in));
  RETURN_ON_ERROR(SealUsage(plasma_id));
  return Status::OK();
}

Status PlasmaClient::Open(std::string const& ipc_socket) {
  return BasicIPCClient::Open(ipc_socket, /*bulk_store_type=*/"Plasma");
}

Status PlasmaClient::Connect(const std::string& ipc_socket) {
  return BasicIPCClient::Connect(ipc_socket, /*bulk_store_type=*/"Plasma");
}

Status PlasmaClient::CreateBuffer(PlasmaID plasma_id, size_t size,
                                  size_t plasma_size,
                                  std::unique_ptr<BlobWriter>& blob) {
  ENSURE_CONNECTED(this);
  ObjectID object_id = InvalidObjectID();
  PlasmaPayload plasma_payload;
  std::shared_ptr<arrow::MutableBuffer> buffer = nullptr;

  std::string message_out;
  WriteCreateBufferByPlasmaRequest(plasma_id, size, plasma_size, message_out);
  RETURN_ON_ERROR(doWrite(message_out));

  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(
      ReadCreateBufferByPlasmaReply(message_in, object_id, plasma_payload));

  RETURN_ON_ASSERT(static_cast<size_t>(plasma_payload.data_size) == size);
  uint8_t *shared = nullptr, *dist = nullptr;
  if (plasma_payload.data_size > 0) {
    RETURN_ON_ERROR(
        this->shm_->Mmap(plasma_payload.store_fd, plasma_payload.map_size,
                         plasma_payload.pointer - plasma_payload.data_offset,
                         false, true, &shared));
    dist = shared + plasma_payload.data_offset;
  }
  buffer =
      std::make_shared<arrow::MutableBuffer>(dist, plasma_payload.data_size);

  auto payload = plasma_payload.ToNormalPayload();
  object_id = payload.object_id;
  blob.reset(new BlobWriter(object_id, payload, buffer));
  RETURN_ON_ERROR(AddUsage(plasma_id, plasma_payload));
  return Status::OK();
}

Status PlasmaClient::GetPayloads(
    std::set<PlasmaID> const& plasma_ids,
    std::map<PlasmaID, PlasmaPayload>& plasma_payloads) {
  if (plasma_ids.empty()) {
    return Status::OK();
  }
  ENSURE_CONNECTED(this);
  std::set<PlasmaID> remote_ids;
  std::vector<PlasmaPayload> local_payloads;
  std::vector<PlasmaPayload> _payloads;

  /// Lookup in local cache
  for (auto const& id : plasma_ids) {
    PlasmaPayload tmp;
    if (FetchOnLocal(id, tmp).ok()) {
      local_payloads.emplace_back(tmp);
    } else {
      remote_ids.emplace(id);
    }
  }

  /// Lookup in remote server
  std::string message_out;
  WriteGetBuffersByPlasmaRequest(remote_ids, message_out);
  RETURN_ON_ERROR(doWrite(message_out));

  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetBuffersByPlasmaReply(message_in, _payloads));

  _payloads.insert(_payloads.end(), local_payloads.begin(),
                   local_payloads.end());

  for (auto const& item : _payloads) {
    plasma_payloads.emplace(item.plasma_id, item);
  }
  return Status::OK();
}

Status PlasmaClient::GetBuffers(
    std::set<PlasmaID> const& plasma_ids,
    std::map<PlasmaID, std::shared_ptr<arrow::Buffer>>& buffers) {
  std::map<PlasmaID, PlasmaPayload> plasma_payloads;
  RETURN_ON_ERROR(GetPayloads(plasma_ids, plasma_payloads));

  for (auto const& item : plasma_payloads) {
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    uint8_t *shared = nullptr, *dist = nullptr;
    if (item.second.data_size > 0) {
      VINEYARD_CHECK_OK(this->shm_->Mmap(
          item.second.store_fd, item.second.map_size,
          item.second.pointer - item.second.data_offset, true, true, &shared));
      dist = shared + item.second.data_offset;
    }
    buffer = std::make_shared<arrow::Buffer>(dist, item.second.data_size);
    buffers.emplace(item.second.plasma_id, buffer);

    RETURN_ON_ERROR(AddUsage(item.second.plasma_id, item.second));
  }
  return Status::OK();
}

Status PlasmaClient::ShallowCopy(PlasmaID const plasma_id, PlasmaID& target_pid,
                                 PlasmaClient& source_client) {
  ENSURE_CONNECTED(this);
  std::map<PlasmaID, PlasmaID> mapping;
  mapping.emplace(plasma_id, plasma_id);

  // create a new plasma object in plasma bulk store.
  std::string message_out;
  WriteMoveBuffersOwnershipRequest(mapping, source_client.session_id(),
                                   message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMoveBuffersOwnershipReply(message_in));

  /// no need to reconstruct meta_tree since we do not support composable object
  /// for plasma store.
  target_pid = plasma_id;
  return Status::OK();
}

Status PlasmaClient::ShallowCopy(ObjectID const id,
                                 std::set<PlasmaID>& target_pids,
                                 Client& source_client) {
  ENSURE_CONNECTED(this);
  ObjectMeta meta;
  json tree;

  RETURN_ON_ERROR(source_client.GetData(id, tree, /*sync_remote==*/true));
  meta.SetMetaData(this, tree);
  auto bids = meta.GetBufferSet()->AllBufferIds();

  std::map<ObjectID, PlasmaID> mapping;
  for (auto const& bid : bids) {
    PlasmaID new_pid = PlasmaIDFromString(ObjectIDToString(bid));
    mapping.emplace(bid, new_pid);
  }

  // create a new plasma object in plasma bulk store.
  std::string message_out;
  WriteMoveBuffersOwnershipRequest(mapping, source_client.session_id(),
                                   message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMoveBuffersOwnershipReply(message_in));

  /// no need to reconstruct meta_tree since we do not support composable object
  /// for plasma store.
  return Status::OK();
}

/// Release an plasma blob.
Status PlasmaClient::OnRelease(PlasmaID const& plasma_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePlasmaReleaseRequest(plasma_id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));

  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadPlasmaReleaseReply(message_in));
  return Status::OK();
}

/// Delete an plasma blob.
Status PlasmaClient::OnDelete(PlasmaID const& plasma_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePlasmaDelDataRequest(plasma_id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));

  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadPlasmaDelDataReply(message_in));
  return Status::OK();
}

Status PlasmaClient::Release(const PlasmaID& id) { return RemoveUsage(id); }

Status PlasmaClient::Delete(const PlasmaID& id) { return PreDelete(id); }

namespace detail {

MmapEntry::MmapEntry(int fd, int64_t map_size, uint8_t* pointer, bool readonly,
                     bool realign)
    : fd_(fd),
      pointer(pointer),
      ro_pointer_(nullptr),
      rw_pointer_(nullptr),
      length_(0) {
  // fake_mmap in malloc.h leaves a gap between memory segments, to make
  // map_size page-aligned again.
  if (realign) {
    length_ = map_size - sizeof(size_t);
  } else {
    length_ = map_size;
  }
}

MmapEntry::~MmapEntry() {
  if (ro_pointer_) {
    int r = munmap(ro_pointer_, length_);
    if (r != 0) {
      std::clog << "[error] munmap returned " << r << ", errno = " << errno
                << ": " << strerror(errno) << std::endl;
    }
  }
  if (rw_pointer_) {
    int r = munmap(rw_pointer_, length_);
    if (r != 0) {
      std::clog << "[error] munmap returned " << r << ", errno = " << errno
                << ": " << strerror(errno) << std::endl;
    }
  }
  close(fd_);
}

uint8_t* MmapEntry::map_readonly() {
  if (!ro_pointer_) {
    ro_pointer_ = reinterpret_cast<uint8_t*>(
        mmap(NULL, length_, PROT_READ, MAP_SHARED, fd_, 0));
    if (ro_pointer_ == MAP_FAILED) {
      std::clog << "[error] mmap failed: errno = " << errno << ": "
                << strerror(errno) << std::endl;
      ro_pointer_ = nullptr;
    }
  }
  return ro_pointer_;
}

uint8_t* MmapEntry::map_readwrite() {
  if (!rw_pointer_) {
    rw_pointer_ = reinterpret_cast<uint8_t*>(
        mmap(NULL, length_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (rw_pointer_ == MAP_FAILED) {
      std::clog << "[error] mmap failed: errno = " << errno << ": "
                << strerror(errno) << std::endl;
      rw_pointer_ = nullptr;
    }
  }
  return rw_pointer_;
}

SharedMemoryManager::SharedMemoryManager(int vineyard_conn)
    : vineyard_conn_(vineyard_conn) {}

Status SharedMemoryManager::Mmap(int fd, int64_t map_size, uint8_t* pointer,
                                 bool readonly, bool realign, uint8_t** ptr) {
  auto entry = mmap_table_.find(fd);
  if (entry == mmap_table_.end()) {
    int client_fd = recv_fd(vineyard_conn_);
    if (fd <= 0) {
      return Status::IOError(
          "Failed to receieve file descriptor from the socket");
    }
    auto mmap_entry = std::unique_ptr<MmapEntry>(
        new MmapEntry(client_fd, map_size, pointer, readonly, realign));
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
  segments_.emplace(reinterpret_cast<uintptr_t>(*ptr), entry->second.get());
  return Status::OK();
}

bool SharedMemoryManager::Exists(const uintptr_t target) {
  ObjectID id;
  return Exists(target, id);
}

bool SharedMemoryManager::Exists(const void* target) {
  ObjectID id;
  return Exists(target, id);
}

bool SharedMemoryManager::Exists(const uintptr_t target, ObjectID& object_id) {
  if (segments_.empty()) {
    return false;
  }
#if defined(WITH_VERBOSE)
  std::clog
      << "[trace] ---------------- shared memory segments: ----------------"
      << std::endl;
  std::clog << "[trace] pointer that been queried: "
            << reinterpret_cast<void*>(target) << std::endl;
  for (auto const& item : segments_) {
    std::clog << "[trace] [" << reinterpret_cast<void*>(item.first) << ", "
              << reinterpret_cast<void*>(item.first + item.second->length_)
              << ")" << std::endl;
  }
#endif

  auto loc = segments_.upper_bound(target);
  if (segments_.empty()) {
    return false;
  } else if (loc == segments_.begin()) {
    return false;
  } else if (loc == segments_.end()) {
    // check rbegin
    auto const item = segments_.rbegin();
    object_id = resolveObjectID(target, item->first, item->second);
    return object_id != InvalidObjectID();
  } else {
    // check prev
    auto const item = std::prev(loc);
    object_id = resolveObjectID(target, item->first, item->second);
    return object_id != InvalidObjectID();
  }
}

bool SharedMemoryManager::Exists(const void* target, ObjectID& object_id) {
  return Exists(reinterpret_cast<const uintptr_t>(target), object_id);
}

ObjectID SharedMemoryManager::resolveObjectID(const uintptr_t target,
                                              const uintptr_t key,
                                              const MmapEntry* entry) {
  if (key <= target && target < key + entry->length_) {
    ptrdiff_t offset = 0;
    if (key == reinterpret_cast<uintptr_t>(entry->ro_pointer_)) {
      offset = target - reinterpret_cast<uintptr_t>(entry->ro_pointer_);
    } else if (key == reinterpret_cast<uintptr_t>(entry->rw_pointer_)) {
      offset = target - reinterpret_cast<uintptr_t>(entry->rw_pointer_);
    } else {
      return InvalidObjectID();
    }
    ObjectID object_id = GenerateBlobID(entry->pointer + offset);
#if defined(WITH_VERBOSE)
    std::clog << "[trace] resuing blob " << ObjectIDToString(object_id)
              << " (offset is " << offset << ")"
              << " for pointer " << reinterpret_cast<void*>(target)
              << std::endl;
#endif
    return object_id;
  } else {
    return InvalidObjectID();
  }
}

}  // namespace detail

}  // namespace vineyard
