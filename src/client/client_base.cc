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

#include "client/client_base.h"

#include <sys/socket.h>

#include "client/ds/i_object.h"
#include "client/ds/object_factory.h"
#include "client/io.h"
#include "client/utils.h"
#include "common/util/env.h"
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
  auto status = ReadGetDataReply(message_in, tree);
  return Status::Wrap(
      status, "failed to get metadata for '" + ObjectIDToString(id) + "'");
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

Status ClientBase::CreateData(const std::vector<json>& trees,
                              std::vector<ObjectID>& ids,
                              std::vector<Signature>& signatures,
                              std::vector<InstanceID>& instance_ids) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteCreateDatasRequest(trees, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(
      ReadCreateDatasReply(message_in, ids, signatures, instance_ids));
  return Status::OK();
}

Status ClientBase::CreateMetaData(ObjectMeta& meta_data, ObjectID& id) {
  auto instance_id = this->instance_id_;
  if (this->IsRPC()) {
    instance_id = this->remote_instance_id();
  }
  return this->CreateMetaData(meta_data, instance_id, std::ref(id));
}

Status ClientBase::CreateMetaData(std::vector<ObjectMeta>& meta_datas,
                                  std::vector<ObjectID>& ids) {
  auto instance_id = this->instance_id_;
  if (this->IsRPC()) {
    instance_id = this->remote_instance_id();
  }
  return this->CreateMetaData(meta_datas, instance_id, std::ref(ids));
}

Status ClientBase::CreateMetaData(ObjectMeta& meta_data,
                                  InstanceID const& instance_id, ObjectID& id) {
  const char* labels[3] = {"JOB_NAME", "POD_NAME", "POD_NAMESPACE"};
  InstanceID computed_instance_id = instance_id;
  meta_data.SetInstanceId(instance_id);
  meta_data.AddKeyValue("transient", true);
  // add the key from env to the metadata for k8s environment.
  for (auto l : labels) {
    auto value = read_env(l);
    if (!value.empty()) {
      meta_data.AddKeyValue(std::string(l), std::string(value));
    }
  }
  // nbytes is optional
  if (!meta_data.HasKey("nbytes")) {
    meta_data.SetNBytes(0);
  }
  // if the metadata has incomplete components, trigger an remote meta sync.
  if (meta_data.incomplete()) {
    VINEYARD_SUPPRESS(SyncMetaData());
  }
  Signature signature;
  RETURN_ON_ERROR(
      CreateData(meta_data.MetaData(), id, signature, computed_instance_id));

  meta_data.SetId(id);
  meta_data.SetSignature(signature);
  meta_data.SetClient(this);
  meta_data.SetInstanceId(computed_instance_id);
  if (meta_data.incomplete()) {
    // N.B.: don't use `meta_data` directly to `GetMetaData` otherwise it may
    // violate the invariant of `BufferSet` in `ObjectMeta`.
    ObjectMeta result_meta;
    RETURN_ON_ERROR(this->GetMetaData(id, result_meta));
    meta_data = result_meta;
  }
  return Status::OK();
}

Status ClientBase::CreateMetaData(std::vector<ObjectMeta>& meta_datas,
                                  InstanceID const& instance_id,
                                  std::vector<ObjectID>& ids) {
  const char* labels[3] = {"JOB_NAME", "POD_NAME", "POD_NAMESPACE"};
  std::vector<InstanceID> computed_instance_ids(meta_datas.size(), instance_id);
  bool has_incomplete = false;
  for (auto& meta_data : meta_datas) {
    meta_data.SetInstanceId(instance_id);
    meta_data.AddKeyValue("transient", true);
    // add the key from env to the metadata for k8s environment.
    for (auto l : labels) {
      auto value = read_env(l);
      if (!value.empty()) {
        meta_data.AddKeyValue(std::string(l), std::string(value));
      }
    }
    // nbytes is optional
    if (!meta_data.HasKey("nbytes")) {
      meta_data.SetNBytes(0);
    }
    has_incomplete = has_incomplete || meta_data.incomplete();
  }
  // if the metadata has incomplete components, trigger an remote meta sync.
  if (has_incomplete) {
    VINEYARD_SUPPRESS(SyncMetaData());
  }
  std::vector<Signature> signatures;
  std::vector<json> trees;
  for (auto const& meta_data : meta_datas) {
    trees.emplace_back(meta_data.MetaData());
  }
  RETURN_ON_ERROR(CreateData(trees, ids, signatures, computed_instance_ids));

  for (size_t i = 0; i < meta_datas.size(); ++i) {
    meta_datas[i].SetId(ids[i]);
    meta_datas[i].SetSignature(signatures[i]);
    meta_datas[i].SetClient(this);
    meta_datas[i].SetInstanceId(computed_instance_ids[i]);
    if (meta_datas[i].incomplete()) {
      // N.B.: don't use `meta_data` directly to `GetMetaData` otherwise it may
      // violate the invariant of `BufferSet` in `ObjectMeta`.
      ObjectMeta result_meta;
      RETURN_ON_ERROR(this->GetMetaData(ids[i], result_meta));
      meta_datas[i] = result_meta;
    }
  }
  return Status::OK();
}

Status ClientBase::SyncMetaData() {
  json __dummy;
  return GetData(InvalidObjectID(), __dummy, true, false);
}

Status ClientBase::DelData(const ObjectID id, const bool force,
                           const bool deep) {
  return DelData(id, force, deep, false);
}

Status ClientBase::DelData(const ObjectID id, const bool force, const bool deep,
                           const bool memory_trim) {
  return DelData(std::vector<ObjectID>{id}, force, deep, memory_trim);
}

Status ClientBase::DelData(const std::vector<ObjectID>& ids, const bool force,
                           const bool deep) {
  return DelData(ids, force, deep, false);
}

Status ClientBase::DelData(const std::vector<ObjectID>& ids, const bool force,
                           const bool deep, const bool memory_trim) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteDelDataRequest(ids, force, deep, memory_trim, false, message_out);
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

Status ClientBase::ListNames(std::string const& pattern, bool const regex,
                             size_t const limit,
                             std::map<std::string, ObjectID>& names) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteListNameRequest(pattern, regex, limit, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadListNameReply(message_in, names));
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

Status ClientBase::DropStream(ObjectID const id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteDropStreamRequest(id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadDropStreamReply(message_in));
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

Status ClientBase::MigrateObject(const ObjectID object_id,
                                 ObjectID& result_id) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteMigrateObjectRequest(object_id, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMigrateObjectReply(message_in, result_id));
  return Status::OK();
}

Status ClientBase::Clear() {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteClearRequest(message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadClearReply(message_in));
  return Status::OK();
}

Status ClientBase::MemoryTrim(bool& trimmed) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteMemoryTrimRequest(message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadMemoryTrimReply(message_in, trimmed));
  return Status::OK();
}

Status ClientBase::Label(const ObjectID object, std::string const& key,
                         std::string const& value) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteLabelRequest(object, key, value, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadLabelReply(message_in));
  return Status::OK();
}

Status ClientBase::Label(const ObjectID object,
                         std::map<std::string, std::string> const& labels) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteLabelRequest(object, labels, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadLabelReply(message_in));
  return Status::OK();
}

Status ClientBase::Evict(std::vector<ObjectID> const& objects) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteEvictRequest(objects, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadEvictReply(message_in));
  return Status::OK();
}

Status ClientBase::Load(std::vector<ObjectID> const& objects, const bool pin) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteLoadRequest(objects, pin, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadLoadReply(message_in));
  return Status::OK();
}

Status ClientBase::Unpin(std::vector<ObjectID> const& objects) {
  ENSURE_CONNECTED(this);
  std::string message_out;
  WriteUnpinRequest(objects, message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadUnpinReply(message_in));
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
  auto status = recv_message(vineyard_conn_, message_in);
  if (!status.ok()) {
    connected_ = false;
  }
  return status;
}

Status ClientBase::doRead(json& root) {
  std::string message_in;
  RETURN_ON_ERROR(doRead(message_in));
  Status status;
  CATCH_JSON_ERROR(root, status, json::parse(message_in));
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
