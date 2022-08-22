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

#include "client/rpc_client.h"

#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "client/ds/object_factory.h"
#include "client/ds/remote_blob.h"
#include "client/io.h"
#include "client/utils.h"
#include "common/util/env.h"
#include "common/util/protocols.h"

namespace vineyard {

RPCClient::~RPCClient() { Disconnect(); }

Status RPCClient::Connect() {
  auto ep = read_env("VINEYARD_RPC_ENDPOINT");
  if (!ep.empty()) {
    return Connect(ep);
  }
  return Status::ConnectionError(
      "Environment variable VINEYARD_RPC_ENDPOINT does't exists");
}

Status RPCClient::Connect(const std::string& rpc_endpoint) {
  size_t pos = rpc_endpoint.find(":");
  std::string host, port;
  if (pos == std::string::npos) {
    host = rpc_endpoint;
    port = "9600";
  } else {
    host = rpc_endpoint.substr(0, pos);
    port = rpc_endpoint.substr(pos + 1);
  }
  return this->Connect(host, static_cast<uint32_t>(std::stoul(port)));
}

Status RPCClient::Connect(const std::string& host, uint32_t port) {
  std::lock_guard<std::recursive_mutex> guard(client_mutex_);
  std::string rpc_endpoint = host + ":" + std::to_string(port);
  RETURN_ON_ASSERT(!connected_ || rpc_endpoint == rpc_endpoint_);
  if (connected_) {
    return Status::OK();
  }
  rpc_endpoint_ = rpc_endpoint;
  RETURN_ON_ERROR(connect_rpc_socket_retry(host, port, vineyard_conn_));
  std::string message_out;
  WriteRegisterRequest(message_out, StoreType::kDefault);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  std::string ipc_socket_value, rpc_endpoint_value;
  bool store_match;
  RETURN_ON_ERROR(ReadRegisterReply(message_in, ipc_socket_value,
                                    rpc_endpoint_value, remote_instance_id_,
                                    session_id_, server_version_, store_match));
  ipc_socket_ = ipc_socket_value;
  connected_ = true;

  if (!compatible_server(server_version_)) {
    std::clog << "[warn] Warning: this version of vineyard client may be "
                 "incompatible with connected server: "
              << "client's version is " << vineyard_version()
              << ", while the server's version is " << server_version_
              << std::endl;
  }

  // RPC client doesn't have a concrete instance id, even the unspecified
  // instance id.
  instance_id_ = UnspecifiedInstanceID() - 1;
  return Status::OK();
}

Status RPCClient::Fork(RPCClient& client) {
  RETURN_ON_ASSERT(!client.Connected(),
                   "The client has already been connected to vineyard server");
  return client.Connect(rpc_endpoint_);
}

Status RPCClient::GetMetaData(const ObjectID id, ObjectMeta& meta,
                              const bool sync_remote) {
  ENSURE_CONNECTED(this);
  json tree;
  RETURN_ON_ERROR(GetData(id, tree, sync_remote));
  meta.Reset();
  meta.SetMetaData(this, tree);
  return Status::OK();
}

Status RPCClient::GetMetaData(const std::vector<ObjectID>& ids,
                              std::vector<ObjectMeta>& metas,
                              const bool sync_remote) {
  ENSURE_CONNECTED(this);
  std::vector<json> trees;
  RETURN_ON_ERROR(GetData(ids, trees, sync_remote));
  metas.resize(trees.size());

  for (size_t idx = 0; idx < trees.size(); ++idx) {
    metas[idx].Reset();
    metas[idx].SetMetaData(this, trees[idx]);
  }
  return Status::OK();
}

std::shared_ptr<Object> RPCClient::GetObject(const ObjectID id) {
  ObjectMeta meta;
  RETURN_NULL_ON_ERROR(this->GetMetaData(id, meta, true));
  RETURN_NULL_ON_ASSERT(!meta.MetaData().empty());
  auto object = ObjectFactory::Create(meta.GetTypeName());
  if (object == nullptr) {
    object = std::unique_ptr<Object>(new Object());
  }
  object->Construct(meta);
  return object;
}

Status RPCClient::GetObject(const ObjectID id,
                            std::shared_ptr<Object>& object) {
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

std::vector<std::shared_ptr<Object>> RPCClient::GetObjects(
    const std::vector<ObjectID>& ids) {
  std::vector<std::shared_ptr<Object>> objects(ids.size());
  std::vector<ObjectMeta> metas;
  if (!this->GetMetaData(ids, metas, true).ok()) {
    for (size_t index = 0; index < ids.size(); ++index) {
      objects[index] = nullptr;
    }
    return objects;
  }
  for (size_t index = 0; index < metas.size(); ++index) {
    if (metas[index].MetaData().empty()) {
      objects[index] = nullptr;
    } else {
      auto object = ObjectFactory::Create(metas[index].GetTypeName());
      if (object == nullptr) {
        object = std::unique_ptr<Object>(new Object());
      }
      object->Construct(metas[index]);
      objects[index] = std::shared_ptr<Object>(object.release());
    }
  }
  return objects;
}

std::vector<ObjectMeta> RPCClient::ListObjectMeta(std::string const& pattern,
                                                  const bool regex,
                                                  size_t const limit, bool) {
  std::unordered_map<ObjectID, json> meta_trees;
  VINEYARD_CHECK_OK(ListData(pattern, regex, limit, meta_trees));

  // construct object metadatas
  std::vector<ObjectMeta> objects;
  objects.reserve(meta_trees.size());
  for (auto const& kv : meta_trees) {
    ObjectMeta meta;
    meta.SetMetaData(this, kv.second);
    objects.emplace_back(meta);
  }
  return objects;
}

std::vector<std::shared_ptr<Object>> RPCClient::ListObjects(
    std::string const& pattern, const bool regex, size_t const limit) {
  std::unordered_map<ObjectID, json> meta_trees;
  VINEYARD_CHECK_OK(ListData(pattern, regex, limit, meta_trees));

  // construct objects
  std::vector<std::shared_ptr<Object>> objects;
  objects.reserve(meta_trees.size());
  for (auto const& kv : meta_trees) {
    ObjectMeta meta;
    meta.SetMetaData(this, kv.second);
    auto object = ObjectFactory::Create(meta.GetTypeName());
    if (object == nullptr) {
      object = std::unique_ptr<Object>(new Object());
    }
    object->Construct(meta);
    objects.emplace_back(std::shared_ptr<Object>(object.release()));
  }
  return objects;
}

Status RPCClient::migrateBuffers(RPCClient& remote,
                                 const std::set<ObjectID> blobs,
                                 std::map<ObjectID, ObjectID>& results) {
  ENSURE_CONNECTED(this);

  std::vector<Payload> payloads;
  std::vector<int> fd_sent;

  std::string message_out;
  WriteGetRemoteBuffersRequest(blobs, false, message_out);
  RETURN_ON_ERROR(remote.doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(remote.doRead(message_in));
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, payloads, fd_sent));
  RETURN_ON_ASSERT(payloads.size() == blobs.size(),
                   "The result size doesn't match with the requested sizes: " +
                       std::to_string(payloads.size()) + " vs. " +
                       std::to_string(blobs.size()));

  for (auto const& payload : payloads) {
    if (payload.data_size == 0) {
      results[payload.object_id] = EmptyBlobID();
      continue;
    }
    auto remote_blob_writer = std::shared_ptr<RemoteBlobWriter>(
        new RemoteBlobWriter(payload.data_size));
    RETURN_ON_ERROR(recv_bytes(remote.vineyard_conn_,
                               remote_blob_writer->data(), payload.data_size));
    ObjectID target_blob_id = InvalidObjectID();
    RETURN_ON_ERROR(this->CreateRemoteBlob(remote_blob_writer, target_blob_id));
    results[payload.object_id] = target_blob_id;
  }
  return Status::OK();
}

Status RPCClient::CreateRemoteBlob(
    std::shared_ptr<RemoteBlobWriter> const& buffer, ObjectID& id) {
  ENSURE_CONNECTED(this);
  VINEYARD_ASSERT(buffer != nullptr, "Expects a non-null remote blob rewriter");

  Payload payload;
  int fd_sent = -1;

  std::string message_out;
  WriteCreateRemoteBufferRequest(buffer->size(), message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  // send the actual payload
  RETURN_ON_ERROR(send_bytes(vineyard_conn_, buffer->data(), buffer->size()));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadCreateBufferReply(message_in, id, payload, fd_sent));
  RETURN_ON_ASSERT(
      static_cast<size_t>(payload.data_size) == buffer->size(),
      "The result blob size doesn't match with the requested size");
  return Status::OK();
}

Status RPCClient::GetRemoteBlob(const ObjectID& id,
                                std::shared_ptr<RemoteBlob>& buffer) {
  return this->GetRemoteBlob(id, false, buffer);
}

Status RPCClient::GetRemoteBlob(const ObjectID& id, const bool unsafe,
                                std::shared_ptr<RemoteBlob>& buffer) {
  ENSURE_CONNECTED(this);

  std::vector<Payload> payloads;
  std::vector<int> fd_sent;

  std::string message_out;
  WriteGetRemoteBuffersRequest(std::set<ObjectID>{id}, unsafe, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, payloads, fd_sent));
  RETURN_ON_ASSERT(payloads.size() == 1, "Expects only one payload");

  // read the actual payload
  buffer = std::shared_ptr<RemoteBlob>(new RemoteBlob(
      payloads[0].object_id, remote_instance_id_, payloads[0].data_size));
  RETURN_ON_ERROR(recv_bytes(vineyard_conn_, buffer->mutable_data(),
                             payloads[0].data_size));
  return Status::OK();
}
Status RPCClient::GetRemoteBlobs(
    std::vector<ObjectID> const& ids,
    std::vector<std::shared_ptr<RemoteBlob>>& remote_blobs) {
  return this->GetRemoteBlobs(ids, false, remote_blobs);
}

Status RPCClient::GetRemoteBlobs(
    std::vector<ObjectID> const& ids, const bool unsafe,
    std::vector<std::shared_ptr<RemoteBlob>>& remote_blobs) {
  ENSURE_CONNECTED(this);

  std::unordered_set<ObjectID> id_set(ids.begin(), ids.end());
  std::vector<Payload> payloads;
  std::vector<int> fd_sent;

  std::string message_out;
  WriteGetRemoteBuffersRequest(id_set, unsafe, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, payloads, fd_sent));
  RETURN_ON_ASSERT(payloads.size() == id_set.size(),
                   "The result size doesn't match with the requested sizes: " +
                       std::to_string(payloads.size()) + " vs. " +
                       std::to_string(id_set.size()));

  std::unordered_map<ObjectID, std::shared_ptr<RemoteBlob>> id_payload_map;
  for (auto const& payload : payloads) {
    auto remote_blob = std::shared_ptr<RemoteBlob>(new RemoteBlob(
        payload.object_id, remote_instance_id_, payload.data_size));
    RETURN_ON_ERROR(recv_bytes(vineyard_conn_, remote_blob->mutable_data(),
                               payload.data_size));
    id_payload_map[payload.object_id] = remote_blob;
  }
  // clear the result container
  remote_blobs.clear();
  for (auto const& id : ids) {
    auto it = id_payload_map.find(id);
    if (it == id_payload_map.end()) {
      remote_blobs.emplace_back(nullptr);
    } else {
      remote_blobs.emplace_back(it->second);
    }
  }
  return Status::OK();
}

}  // namespace vineyard
