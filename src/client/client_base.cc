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

#include "client/client_base.h"

#include <future>
#include <iostream>
#include <utility>

#include "boost/range/combine.hpp"

#include "client/client.h"
#include "client/io.h"
#include "client/rpc_client.h"
#include "client/utils.h"
#include "common/util/protocols.h"

namespace vineyard {

ClientBase::ClientBase() : connected_(false), vineyard_conn_(0) {}

Status ClientBase::GetData(const ObjectID id, json& tree,
                           const bool sync_remote, const bool wait) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteGetDataRequest(id, sync_remote, wait, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetDataReply(message_in, tree));
  return Status::OK();
}

Status ClientBase::GetData(const std::vector<ObjectID>& ids,
                           std::vector<json>& trees, const bool sync_remote,
                           const bool wait) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteGetDataRequest(ids, sync_remote, wait, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  std::unordered_map<ObjectID, json> meta_trees;
  RETURN_ON_ERROR(ReadGetDataReply(message_in, meta_trees));
  trees.reserve(ids.size());
  for (auto const& id : ids) {
    trees.emplace_back(meta_trees.at(id));
  }
  return Status::OK();
}

Status ClientBase::CreateData(const json& tree, ObjectID& id,
                              Signature& signature, InstanceID& instance_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteCreateDataRequest(tree, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadCreateDataReply(message_in, id, signature, instance_id));
  return Status::OK();
}

Status ClientBase::CreateMetaData(ObjectMeta& meta_data, ObjectID& id) {
  InstanceID instance_id = this->instance_id_;
  meta_data.SetInstanceId(instance_id);
  meta_data.AddKeyValue("transient", true);
  // nbytes is optional
  if (!meta_data.Haskey("nbytes")) {
    meta_data.SetNBytes(0);
  }
  // if the metadata has incomplete components, trigger an remote meta sync.
  if (meta_data.incomplete()) {
    VINEYARD_SUPPRESS(SyncMetaData());
  }
  Signature signature;
  auto status = CreateData(meta_data.MetaData(), id, signature, instance_id);
  if (status.ok()) {
    meta_data.SetId(id);
    meta_data.SetSignature(signature);
    meta_data.SetClient(this);
    meta_data.SetInstanceId(instance_id);
    if (meta_data.incomplete()) {
      // N.B.: don't use `meta_data` directly to `GetMetaData` otherwise it may
      // violate the invariant of `BufferSet` in `ObjectMeta`.
      ObjectMeta result_meta;
      RETURN_ON_ERROR(this->GetMetaData(id, result_meta));
      meta_data = result_meta;
    }
  }
  return status;
}

Status ClientBase::SyncMetaData() {
  json __dummy;
  return GetData(InvalidObjectID(), __dummy, true, false);
}

Status ClientBase::DelData(const ObjectID id, const bool force,
                           const bool deep) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteDelDataRequest(id, force, deep, false, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDelDataReply(message_in));
  return Status::OK();
}

Status ClientBase::DelData(const std::vector<ObjectID>& ids, const bool force,
                           const bool deep) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteDelDataRequest(ids, force, deep, false, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDelDataReply(message_in));
  return Status::OK();
}

Status ClientBase::ListData(std::string const& pattern, bool const regex,
                            size_t const limit,
                            std::unordered_map<ObjectID, json>& meta_trees) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteListDataRequest(pattern, regex, limit, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetDataReply(message_in, meta_trees));
  return Status::OK();
}

Status ClientBase::CreateStream(const ObjectID& id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteCreateStreamRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadCreateStreamReply(message_in));
  return Status::OK();
}

Status ClientBase::OpenStream(const ObjectID& id, StreamOpenMode mode) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteOpenStreamRequest(id, static_cast<int64_t>(mode), message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadOpenStreamReply(message_in));
  return Status::OK();
}

Status ClientBase::PushNextStreamChunk(ObjectID const id,
                                       ObjectID const chunk) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePushNextStreamChunkRequest(id, chunk, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadPushNextStreamChunkReply(message_in));
  return Status::OK();
}

Status ClientBase::PullNextStreamChunk(ObjectID const id, ObjectID& chunk) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePullNextStreamChunkRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadPullNextStreamChunkReply(message_in, chunk));
  return Status::OK();
}

Status ClientBase::PullNextStreamChunk(ObjectID const id, ObjectMeta& chunk) {
  ObjectID chunk_id = InvalidObjectID();
  RETURN_ON_ERROR(this->PullNextStreamChunk(id, chunk_id));
  return GetMetaData(chunk_id, chunk, false);
}

Status ClientBase::PullNextStreamChunk(ObjectID const id,
                                       std::shared_ptr<Object>& chunk) {
  ObjectMeta meta;
  RETURN_ON_ERROR(this->PullNextStreamChunk(id, meta));
  RETURN_ON_ASSERT(!meta.MetaData().empty());
  chunk = ObjectFactory::Create(meta.GetTypeName());
  if (chunk == nullptr) {
    chunk = std::unique_ptr<Object>(new Object());
  }
  chunk->Construct(meta);
  return Status::OK();
}

Status ClientBase::StopStream(ObjectID const id, const bool failed) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteStopStreamRequest(id, failed, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadStopStreamReply(message_in));
  return Status::OK();
}

Status ClientBase::Persist(const ObjectID id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePersistRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadPersistReply(message_in));
  return Status::OK();
}

Status ClientBase::IfPersist(const ObjectID id, bool& persist) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteIfPersistRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadIfPersistReply(message_in, persist));
  return Status::OK();
}

Status ClientBase::Exists(const ObjectID id, bool& exists) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteExistsRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadExistsReply(message_in, exists));
  return Status::OK();
}

Status ClientBase::ShallowCopy(const ObjectID id, ObjectID& target_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteShallowCopyRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadShallowCopyReply(message_in, target_id));
  return Status::OK();
}

Status ClientBase::ShallowCopy(const ObjectID id, json const& extra_metadata,
                               ObjectID& target_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteShallowCopyRequest(id, extra_metadata, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadShallowCopyReply(message_in, target_id));
  return Status::OK();
}

Status ClientBase::DeepCopy(const ObjectID object_id, ObjectID& target_id) {
  ENSURE_CONNECTED(this);

  ObjectMeta meta;
  RETURN_ON_ERROR(this->GetMetaData(object_id, meta, true));

  std::map<InstanceID, json> cluster;
  RETURN_ON_ERROR(this->ClusterInfo(cluster));
  auto selfhost =
      cluster.at(this->instance_id())["hostname"].get_ref<std::string const&>();
  auto selfEndpoint = cluster.at(meta.GetInstanceId())["rpc_endpoint"]
                          .get_ref<std::string const&>();

  auto receiver = std::async(std::launch::async, [&]() -> Status {
    RETURN_ON_ERROR(
        this->deepCopyImpl(object_id, target_id, selfhost, selfEndpoint));
    return Status::OK();
  });

  return receiver.get();
}

Status ClientBase::deepCopyImpl(const ObjectID object_id, ObjectID& target_id,
                                std::string const& peer,
                                std::string const& peer_rpc_endpoint) {
  std::string message_out;
  WriteDeepCopyRequest(object_id, peer, peer_rpc_endpoint, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDeepCopyReply(message_in, target_id));
  return Status::OK();
}

Status ClientBase::PutName(const ObjectID id, std::string const& name) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WritePutNameRequest(id, name, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadPutNameReply(message_in));
  return Status::OK();
}

Status ClientBase::GetName(const std::string& name, ObjectID& id,
                           const bool wait) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteGetNameRequest(name, wait, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetNameReply(message_in, id));
  return Status::OK();
}

Status ClientBase::DropName(const std::string& name) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteDropNameRequest(name, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDropNameReply(message_in));
  return Status::OK();
}

Status ClientBase::MigrateStream(const ObjectID stream_id,
                                 ObjectID& result_id) {
  return MigrateObject(stream_id, result_id, true);
}

Status ClientBase::MigrateObject(const ObjectID object_id, ObjectID& result_id,
                                 bool is_stream) {
  ENSURE_CONNECTED(this);

  // query the object location info
  ObjectMeta meta;
  RETURN_ON_ERROR(this->GetMetaData(object_id, meta, true));
#ifndef NDEBUG
  std::clog << "migrate local: " << this->instance_id()
            << ", remote: " << meta.GetInstanceId() << std::endl;
#endif
  if (meta.GetInstanceId() == this->instance_id()) {
    result_id = object_id;
    return Status::OK();
  }

  // findout remote server
  std::map<InstanceID, json> cluster;
  RETURN_ON_ERROR(this->ClusterInfo(cluster));
  auto selfhost =
      cluster.at(this->instance_id())["hostname"].get_ref<std::string const&>();
  auto otherHost = cluster.at(meta.GetInstanceId())["hostname"]
                       .get_ref<std::string const&>();
  auto otherEndpoint = cluster.at(meta.GetInstanceId())["rpc_endpoint"]
                           .get_ref<std::string const&>();

  // launch remote migrate sender
  auto sender = std::async(std::launch::async, [&]() -> Status {
    RPCClient other;
    RETURN_ON_ERROR(other.Connect(otherEndpoint));
    ObjectID dummy = InvalidObjectID();
    RETURN_ON_ERROR(other.migrateObjectImpl(object_id, dummy, true, is_stream,
                                            selfhost, otherEndpoint));
    return Status::OK();
  });

  // local migrate receiver
  auto receiver = std::async(std::launch::async, [&]() -> Status {
    RETURN_ON_ERROR(this->migrateObjectImpl(
        object_id, result_id, false, is_stream, otherHost, otherEndpoint));
#ifndef NDEBUG
    std::clog << "receive from migration: " << ObjectIDToString(object_id)
              << " -> " << ObjectIDToString(result_id) << std::endl;
#endif
    return Status::OK();
  });

  return sender.get() & receiver.get();
}

Status ClientBase::migrateObjectImpl(const ObjectID object_id,
                                     ObjectID& result_id, bool const local,
                                     bool const is_stream,
                                     std::string const& peer,
                                     std::string const& peer_rpc_endpoint) {
  std::string message_out;
  WriteMigrateObjectRequest(object_id, local, is_stream, peer,
                            peer_rpc_endpoint, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMigrateObjectReply(message_in, result_id));
  return Status::OK();
}

Status ClientBase::Clear() {
  std::string message_out;
  WriteClearRequest(message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadClearReply(message_in));
  return Status::OK();
}

bool ClientBase::Connected() const {
  if (connected_ &&
      recv(vineyard_conn_, NULL, 1, MSG_PEEK | MSG_DONTWAIT) != -1) {
    connected_ = false;
  }
  return connected_;
}

void ClientBase::Disconnect() {
  std::lock_guard<std::recursive_mutex> __guard(this->client_mutex_);
  if (!this->connected_) {
    return;
  }
  std::string message_out;
  WriteExitRequest(message_out);
  VINEYARD_SUPPRESS(doWrite(message_out));
  close(vineyard_conn_);
  connected_ = false;
}

void ClientBase::CloseSession() {
  std::lock_guard<std::recursive_mutex> guard(client_mutex_);
  if (!Connected()) {
    return;
  }
  std::string message_out;
  WriteDeleteSessionRequest(message_out);
  VINEYARD_SUPPRESS(doWrite(message_out));
  json message_in;
  VINEYARD_SUPPRESS(doRead(message_in));
  close(vineyard_conn_);
  connected_ = false;
}

Status ClientBase::doWrite(const std::string& message_out) {
  auto status = send_message(vineyard_conn_, message_out);
  if (!status.ok()) {
    connected_ = false;
  }
  return status;
}

Status ClientBase::doRead(std::string& message_in) {
  return recv_message(vineyard_conn_, message_in);
}

Status ClientBase::doRead(json& root) {
  std::string message_in;
  auto status = recv_message(vineyard_conn_, message_in);
  if (!status.ok()) {
    connected_ = false;
    return status;
  }
  status = CATCH_JSON_ERROR([&]() -> Status {
    root = json::parse(message_in);
    return Status::OK();
  }());
  if (!status.ok()) {
    connected_ = false;
  }
  return status;
}

Status ClientBase::ClusterInfo(std::map<InstanceID, json>& meta) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteClusterMetaRequest(message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  json cluster_meta;
  RETURN_ON_ERROR(ReadClusterMetaReply(message_in, cluster_meta));
  for (auto& kv : cluster_meta.items()) {
    InstanceID instance_id = UnspecifiedInstanceID();
    std::stringstream(kv.key().substr(1)) >> instance_id;
    meta.emplace(instance_id, kv.value());
  }
  return Status::OK();
}

Status ClientBase::InstanceStatus(
    std::shared_ptr<struct InstanceStatus>& status) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteInstanceStatusRequest(message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  json status_json;
  RETURN_ON_ERROR(ReadInstanceStatusReply(message_in, status_json));
  status.reset(new struct InstanceStatus(status_json));
  return Status::OK();
}

Status ClientBase::Instances(std::vector<InstanceID>& instances) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteClusterMetaRequest(message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  json cluster_meta;
  RETURN_ON_ERROR(ReadClusterMetaReply(message_in, cluster_meta));
  for (auto& kv : cluster_meta.items()) {
    InstanceID instance_id;
    std::stringstream(kv.key().substr(1)) >> instance_id;
    instances.emplace_back(instance_id);
  }
  return Status::OK();
}

Status ClientBase::Debug(const json& debug, json& result) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteDebugRequest(debug, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDebugReply(message_in, result));
  return Status::OK();
}

InstanceStatus::InstanceStatus(const json& tree)
    : instance_id(tree["instance_id"].get<InstanceID>()),
      deployment(tree["deployment"].get_ref<const std::string&>()),
      memory_usage(tree["memory_usage"].get<size_t>()),
      memory_limit(tree["memory_limit"].get<size_t>()),
      deferred_requests(tree["deferred_requests"].get<size_t>()),
      ipc_connections(tree["ipc_connections"].get<size_t>()),
      rpc_connections(tree["rpc_connections"].get<size_t>()) {}

}  // namespace vineyard
