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

#include "common/util/protocols.h"

#include <sstream>
#include <unordered_set>

#include "boost/algorithm/string.hpp"

#include "common/util/uuid.h"
#include "common/util/version.h"

namespace vineyard {

#define CHECK_IPC_ERROR(tree, type)                                      \
  do {                                                                   \
    if (tree.contains("code")) {                                         \
      Status st = Status(static_cast<StatusCode>(tree.value("code", 0)), \
                         tree.value("message", ""));                     \
      if (!st.ok()) {                                                    \
        return st;                                                       \
      }                                                                  \
    }                                                                    \
    RETURN_ON_ASSERT(root["type"] == (type));                            \
  } while (0)

CommandType ParseCommandType(const std::string& str_type) {
  if (str_type == "exit_request") {
    return CommandType::ExitRequest;
  } else if (str_type == "exit_reply") {
    return CommandType::ExitReply;
  } else if (str_type == "register_request") {
    return CommandType::RegisterRequest;
  } else if (str_type == "register_reply") {
    return CommandType::RegisterReply;
  } else if (str_type == "get_data_request") {
    return CommandType::GetDataRequest;
  } else if (str_type == "get_data_reply") {
    return CommandType::GetDataReply;
  } else if (str_type == "create_data_request") {
    return CommandType::CreateDataRequest;
  } else if (str_type == "persist_request") {
    return CommandType::PersistRequest;
  } else if (str_type == "exists_request") {
    return CommandType::ExistsRequest;
  } else if (str_type == "del_data_request") {
    return CommandType::DelDataRequest;
  } else if (str_type == "cluster_meta") {
    return CommandType::ClusterMetaRequest;
  } else if (str_type == "list_data_request") {
    return CommandType::ListDataRequest;
  } else if (str_type == "create_buffer_request") {
    return CommandType::CreateBufferRequest;
  } else if (str_type == "get_buffers_request") {
    return CommandType::GetBuffersRequest;
  } else if (str_type == "create_stream_request") {
    return CommandType::CreateStreamRequest;
  } else if (str_type == "get_next_stream_chunk_request") {
    return CommandType::GetNextStreamChunkRequest;
  } else if (str_type == "pull_next_stream_chunk_request") {
    return CommandType::PullNextStreamChunkRequest;
  } else if (str_type == "stop_stream_request") {
    return CommandType::StopStreamRequest;
  } else if (str_type == "put_name_request") {
    return CommandType::PutNameRequest;
  } else if (str_type == "get_name_request") {
    return CommandType::GetNameRequest;
  } else if (str_type == "drop_name_request") {
    return CommandType::DropNameRequest;
  } else if (str_type == "if_persist_request") {
    return CommandType::IfPersistRequest;
  } else if (str_type == "instance_status_request") {
    return CommandType::InstanceStatusRequest;
  } else if (str_type == "shallow_copy_request") {
    return CommandType::ShallowCopyRequest;
  } else {
    return CommandType::NullCommand;
  }
}

static inline void encode_msg(const json& root, std::string& msg) {
  msg = json_to_string(root);
}

void WriteErrorReply(Status const& status, std::string& msg) {
  encode_msg(status.ToJSON(), msg);
}

void WriteRegisterRequest(std::string& msg) {
  json root;
  root["type"] = "register_request";
  root["version"] = vineyard_version();

  encode_msg(root, msg);
}

Status ReadRegisterRequest(const json& root, std::string& version) {
  RETURN_ON_ASSERT(root["type"] == "register_request");

  // When the "version" field is missing from the client, we treat it
  // as default unknown version number: 0.0.0.
  version = root.value<std::string>("version", "0.0.0");
  return Status::OK();
}

void WriteRegisterReply(const std::string& ipc_socket,
                        const std::string& rpc_endpoint,
                        const InstanceID instance_id, std::string& msg) {
  json root;
  root["type"] = "register_reply";
  root["ipc_socket"] = ipc_socket;
  root["rpc_endpoint"] = rpc_endpoint;
  root["instance_id"] = instance_id;
  root["version"] = vineyard_version();
  encode_msg(root, msg);
}

Status ReadRegisterReply(const json& root, std::string& ipc_socket,
                         std::string& rpc_endpoint, InstanceID& instance_id,
                         std::string& version) {
  CHECK_IPC_ERROR(root, "register_reply");
  ipc_socket = root["ipc_socket"].get_ref<std::string const&>();
  rpc_endpoint = root["rpc_endpoint"].get_ref<std::string const&>();
  instance_id = root["instance_id"].get<InstanceID>();

  // When the "version" field is missing from the server, we treat it
  // as default unknown version number: 0.0.0.
  version = root.value<std::string>("version", "0.0.0");
  return Status::OK();
}

void WriteExitRequest(std::string& msg) {
  json root;
  root["type"] = "exit_request";

  encode_msg(root, msg);
}

void WriteGetDataRequest(const ObjectID id, const bool sync_remote,
                         const bool wait, std::string& msg) {
  json root;
  root["type"] = "get_data_request";
  root["id"] = std::vector<ObjectID>{id};
  root["sync_remote"] = sync_remote;
  root["wait"] = wait;

  encode_msg(root, msg);
}

void WriteGetDataRequest(const std::vector<ObjectID>& ids,
                         const bool sync_remote, const bool wait,
                         std::string& msg) {
  json root;
  root["type"] = "get_data_request";
  root["id"] = ids;
  root["sync_remote"] = sync_remote;
  root["wait"] = wait;

  encode_msg(root, msg);
}

Status ReadGetDataRequest(const json& root, std::vector<ObjectID>& ids,
                          bool& sync_remote, bool& wait) {
  RETURN_ON_ASSERT(root["type"] == "get_data_request");
  ids = root["id"].get_to(ids);
  sync_remote = root.value("sync_remote", false);
  wait = root.value("wait", false);
  return Status::OK();
}

void WriteGetDataReply(const json& content, std::string& msg) {
  json root;
  root["type"] = "get_data_reply";
  root["content"] = content;

  encode_msg(root, msg);
}

Status ReadGetDataReply(const json& root, json& content) {
  CHECK_IPC_ERROR(root, "get_data_reply");
  // should be only one item
  auto content_group = root["content"];
  if (content_group.size() != 1) {
    return Status::ObjectNotExists();
  }
  content = *content_group.begin();
  return Status::OK();
}

Status ReadGetDataReply(const json& root,
                        std::unordered_map<ObjectID, json>& content) {
  CHECK_IPC_ERROR(root, "get_data_reply");
  for (auto const& kv : json::iterator_wrapper(root["content"])) {
    content.emplace(VYObjectIDFromString(kv.key()), kv.value());
  }
  return Status::OK();
}

void WriteListDataRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg) {
  json root;
  root["type"] = "list_data_request";
  root["pattern"] = pattern;
  root["regex"] = regex;
  root["limit"] = limit;

  encode_msg(root, msg);
}

Status ReadListDataRequest(const json& root, std::string& pattern, bool& regex,
                           size_t& limit) {
  RETURN_ON_ASSERT(root["type"] == "list_data_request");
  pattern = root["pattern"].get_ref<std::string const&>();
  regex = root.value("regex", false);
  limit = root["limit"].get<size_t>();
  return Status::OK();
}

void WriteCreateBufferRequest(const size_t size, std::string& msg) {
  json root;
  root["type"] = "create_buffer_request";
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadCreateBufferRequest(const json& root, size_t& size) {
  RETURN_ON_ASSERT(root["type"] == "create_buffer_request");
  size = root["size"].get<size_t>();
  return Status::OK();
}

void WriteCreateBufferReply(const ObjectID id,
                            const std::shared_ptr<Payload>& object,
                            std::string& msg) {
  json root;
  root["type"] = "create_buffer_reply";
  root["id"] = id;
  json tree;
  object->ToJSON(tree);
  root["created"] = tree;

  encode_msg(root, msg);
}

Status ReadCreateBufferReply(const json& root, ObjectID& id, Payload& object) {
  CHECK_IPC_ERROR(root, "create_buffer_reply");
  json tree = root["created"];
  id = root["id"].get<ObjectID>();
  object.FromJSON(tree);
  return Status::OK();
}

void WriteGetBuffersRequest(const std::unordered_set<ObjectID>& ids,
                            std::string& msg) {
  json root;
  root["type"] = "get_buffers_request";
  int idx = 0;
  for (auto const& id : ids) {
    root[std::to_string(idx++)] = id;
  }
  root["num"] = ids.size();

  encode_msg(root, msg);
}

Status ReadGetBuffersRequest(const json& root, std::vector<ObjectID>& ids) {
  RETURN_ON_ASSERT(root["type"] == "get_buffers_request");
  size_t num = root["num"].get<size_t>();
  for (size_t i = 0; i < num; ++i) {
    ids.push_back(root[std::to_string(i)].get<ObjectID>());
  }
  return Status::OK();
}

void WriteGetBuffersReply(const std::vector<std::shared_ptr<Payload>>& objects,
                          std::string& msg) {
  json root;
  root["type"] = "get_buffers_reply";
  for (size_t i = 0; i < objects.size(); ++i) {
    json tree;
    objects[i]->ToJSON(tree);
    root[std::to_string(i)] = tree;
  }
  root["num"] = objects.size();

  encode_msg(root, msg);
}

Status ReadGetBuffersReply(const json& root,
                           std::unordered_map<ObjectID, Payload>& objects) {
  CHECK_IPC_ERROR(root, "get_buffers_reply");
  for (size_t i = 0; i < root["num"]; ++i) {
    json tree = root[std::to_string(i)];
    Payload object;
    object.FromJSON(tree);
    objects.emplace(object.object_id, object);
  }
  return Status::OK();
}

void WriteCreateDataRequest(const json& content, std::string& msg) {
  json root;
  root["type"] = "create_data_request";
  root["content"] = content;

  encode_msg(root, msg);
}

Status ReadCreateDataRequest(const json& root, json& content) {
  RETURN_ON_ASSERT(root["type"] == "create_data_request");
  content = root["content"];
  return Status::OK();
}

void WriteCreateDataReply(const ObjectID& id, const InstanceID& instance_id,
                          std::string& msg) {
  json root;
  root["type"] = "create_data_reply";
  root["id"] = id;
  root["instance_id"] = instance_id;

  encode_msg(root, msg);
}

Status ReadCreateDataReply(const json& root, ObjectID& id,
                           InstanceID& instance_id) {
  CHECK_IPC_ERROR(root, "create_data_reply");
  id = root["id"].get<ObjectID>();
  instance_id = root["instance_id"].get<InstanceID>();
  return Status::OK();
}

void WritePersistRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = "persist_request";
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadPersistRequest(const json& root, ObjectID& id) {
  RETURN_ON_ASSERT(root["type"] == "persist_request");
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WritePersistReply(std::string& msg) {
  json root;
  root["type"] = "persist_reply";

  encode_msg(root, msg);
}

Status ReadPersistReply(const json& root) {
  CHECK_IPC_ERROR(root, "persist_reply");
  return Status::OK();
}

void WriteIfPersistRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = "if_persist_request";
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadIfPersistRequest(const json& root, ObjectID& id) {
  RETURN_ON_ASSERT(root["type"] == "if_persist_request");
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteIfPersistReply(bool persist, std::string& msg) {
  json root;
  root["type"] = "if_persist_reply";
  root["persist"] = persist;

  encode_msg(root, msg);
}

Status ReadIfPersistReply(const json& root, bool& persist) {
  CHECK_IPC_ERROR(root, "if_persist_reply");
  persist = root.value("persist", false);
  return Status::OK();
}

void WriteExistsRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = "exists_request";
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadExistsRequest(const json& root, ObjectID& id) {
  RETURN_ON_ASSERT(root["type"] == "exists_request");
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteExistsReply(bool exists, std::string& msg) {
  json root;
  root["type"] = "exists_reply";
  root["exists"] = exists;

  encode_msg(root, msg);
}

Status ReadExistsReply(const json& root, bool& exists) {
  CHECK_IPC_ERROR(root, "exists_reply");
  exists = root.value("exists", false);
  return Status::OK();
}

void WriteDelDataRequest(const ObjectID id, const bool force, const bool deep,
                         std::string& msg) {
  json root;
  root["type"] = "del_data_request";
  root["id"] = std::vector<ObjectID>{id};
  root["force"] = force;
  root["deep"] = deep;

  encode_msg(root, msg);
}

void WriteDelDataRequest(const std::vector<ObjectID>& ids, const bool force,
                         const bool deep, std::string& msg) {
  json root;
  root["type"] = "del_data_request";
  root["id"] = ids;
  root["force"] = force;
  root["deep"] = deep;

  encode_msg(root, msg);
}

Status ReadDelDataRequest(const json& root, std::vector<ObjectID>& ids,
                          bool& force, bool& deep) {
  RETURN_ON_ASSERT(root["type"] == "del_data_request");
  ids = root["id"].get_to(ids);
  force = root.value("force", false);
  deep = root.value("deep", false);
  return Status::OK();
}

void WriteDelDataReply(std::string& msg) {
  json root;
  root["type"] = "del_data_reply";

  encode_msg(root, msg);
}

Status ReadDelDataReply(const json& root) {
  CHECK_IPC_ERROR(root, "del_data_reply");
  return Status::OK();
}

void WriteClusterMetaRequest(std::string& msg) {
  json root;
  root["type"] = "cluster_meta";

  encode_msg(root, msg);
}

Status ReadClusterMetaRequest(const json& root) {
  RETURN_ON_ASSERT(root["type"] == "cluster_meta");
  return Status::OK();
}

void WriteClusterMetaReply(const json& meta, std::string& msg) {
  json root;
  root["type"] = "cluster_meta";
  root["meta"] = meta;

  encode_msg(root, msg);
}

Status ReadClusterMetaReply(const json& root, json& meta) {
  CHECK_IPC_ERROR(root, "cluster_meta");
  meta = root["meta"];
  return Status::OK();
}

void WriteInstanceStatusRequest(std::string& msg) {
  json root;
  root["type"] = "instance_status_request";

  encode_msg(root, msg);
}

Status ReadInstanceStatusRequest(const json& root) {
  RETURN_ON_ASSERT(root["type"] == "instance_status_request");
  return Status::OK();
}

void WriteInstanceStatusReply(const json& meta, std::string& msg) {
  json root;
  root["type"] = "instance_status_reply";
  root["meta"] = meta;

  encode_msg(root, msg);
}

Status ReadInstanceStatusReply(const json& root, json& meta) {
  CHECK_IPC_ERROR(root, "instance_status_reply");
  meta = root["meta"];
  return Status::OK();
}

void WritePutNameRequest(const ObjectID object_id, const std::string& name,
                         std::string& msg) {
  json root;
  root["type"] = "put_name_request";
  root["object_id"] = object_id;
  root["name"] = name;

  encode_msg(root, msg);
}

Status ReadPutNameRequest(const json& root, ObjectID& object_id,
                          std::string& name) {
  RETURN_ON_ASSERT(root["type"] == "put_name_request");
  object_id = root["object_id"].get<ObjectID>();
  name = root["name"].get_ref<std::string const&>();
  return Status::OK();
}

void WritePutNameReply(std::string& msg) {
  json root;
  root["type"] = "put_name_reply";

  encode_msg(root, msg);
}

Status ReadPutNameReply(const json& root) {
  CHECK_IPC_ERROR(root, "put_name_reply");
  return Status::OK();
}

void WriteGetNameRequest(const std::string& name, const bool wait,
                         std::string& msg) {
  json root;
  root["type"] = "get_name_request";
  root["name"] = name;
  root["wait"] = wait;

  encode_msg(root, msg);
}

Status ReadGetNameRequest(const json& root, std::string& name, bool& wait) {
  RETURN_ON_ASSERT(root["type"] == "get_name_request");
  name = root["name"].get_ref<std::string const&>();
  wait = root["wait"].get<bool>();
  return Status::OK();
}

void WriteGetNameReply(const ObjectID& object_id, std::string& msg) {
  json root;
  root["type"] = "get_name_reply";
  root["object_id"] = object_id;

  encode_msg(root, msg);
}

Status ReadGetNameReply(const json& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, "get_name_reply");
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteDropNameRequest(const std::string& name, std::string& msg) {
  json root;
  root["type"] = "drop_name_request";
  root["name"] = name;

  encode_msg(root, msg);
}

Status ReadDropNameRequest(const json& root, std::string& name) {
  RETURN_ON_ASSERT(root["type"] == "drop_name_request");
  name = root["name"].get_ref<std::string const&>();
  return Status::OK();
}

void WriteDropNameReply(std::string& msg) {
  json root;
  root["type"] = "drop_name_reply";

  encode_msg(root, msg);
}

Status ReadDropNameReply(const json& root) {
  CHECK_IPC_ERROR(root, "drop_name_reply");
  return Status::OK();
}

void WriteCreateStreamRequest(const ObjectID& object_id, std::string& msg) {
  json root;
  root["type"] = "create_stream_request";
  root["object_id"] = object_id;

  encode_msg(root, msg);
}

Status ReadCreateStreamRequest(const json& root, ObjectID& object_id) {
  RETURN_ON_ASSERT(root["type"] == "create_stream_request");
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteCreateStreamReply(std::string& msg) {
  json root;
  root["type"] = "create_stream_reply";

  encode_msg(root, msg);
}

Status ReadCreateStreamReply(const json& root) {
  CHECK_IPC_ERROR(root, "create_stream_reply");
  return Status::OK();
}

void WriteGetNextStreamChunkRequest(const ObjectID stream_id, const size_t size,
                                    std::string& msg) {
  json root;
  root["type"] = "get_next_stream_chunk_request";
  root["id"] = stream_id;
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadGetNextStreamChunkRequest(const json& root, ObjectID& stream_id,
                                     size_t& size) {
  RETURN_ON_ASSERT(root["type"] == "get_next_stream_chunk_request");
  stream_id = root["id"].get<ObjectID>();
  size = root["size"].get<size_t>();
  return Status::OK();
}

void WriteGetNextStreamChunkReply(std::shared_ptr<Payload>& object,
                                  std::string& msg) {
  json root;
  root["type"] = "get_next_stream_chunk_reply";
  json buffer_meta;
  object->ToJSON(buffer_meta);
  root["buffer"] = buffer_meta;

  encode_msg(root, msg);
}

Status ReadGetNextStreamChunkReply(const json& root, Payload& object) {
  CHECK_IPC_ERROR(root, "get_next_stream_chunk_reply");
  object.FromJSON(root["buffer"]);
  return Status::OK();
}

void WritePullNextStreamChunkRequest(const ObjectID stream_id,
                                     std::string& msg) {
  json root;
  root["type"] = "pull_next_stream_chunk_request";
  root["id"] = stream_id;

  encode_msg(root, msg);
}

Status ReadPullNextStreamChunkRequest(const json& root, ObjectID& stream_id) {
  RETURN_ON_ASSERT(root["type"] == "pull_next_stream_chunk_request");
  stream_id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WritePullNextStreamChunkReply(std::shared_ptr<Payload>& object,
                                   std::string& msg) {
  json root;
  root["type"] = "pull_next_stream_chunk_reply";
  json buffer_meta;
  object->ToJSON(buffer_meta);
  root["buffer"] = buffer_meta;

  encode_msg(root, msg);
}

Status ReadPullNextStreamChunkReply(const json& root, Payload& object) {
  CHECK_IPC_ERROR(root, "pull_next_stream_chunk_reply");
  object.FromJSON(root["buffer"]);
  return Status::OK();
}

void WriteStopStreamRequest(const ObjectID stream_id, const bool failed,
                            std::string& msg) {
  json root;
  root["type"] = "stop_stream_request";
  root["id"] = stream_id;
  root["failed"] = failed;

  encode_msg(root, msg);
}

Status ReadStopStreamRequest(const json& root, ObjectID& stream_id,
                             bool& failed) {
  RETURN_ON_ASSERT(root["type"] == "stop_stream_request");
  stream_id = root["id"].get<ObjectID>();
  failed = root["failed"].get<bool>();
  return Status::OK();
}

void WriteStopStreamReply(std::string& msg) {
  json root;
  root["type"] = "stop_stream_reply";

  encode_msg(root, msg);
}

Status ReadStopStreamReply(const json& root) {
  CHECK_IPC_ERROR(root, "stop_stream_reply");
  return Status::OK();
}

void WriteShallowCopyRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = "shallow_copy_request";
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadShallowCopyRequest(const json& root, ObjectID& id) {
  RETURN_ON_ASSERT(root["type"] == "shallow_copy_request");
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteShallowCopyReply(const ObjectID target_id, std::string& msg) {
  json root;
  root["type"] = "shallow_copy_reply";
  root["target_id"] = target_id;

  encode_msg(root, msg);
}

Status ReadShallowCopyReply(const json& root, ObjectID& target_id) {
  CHECK_IPC_ERROR(root, "shallow_copy_reply");
  target_id = root["target_id"].get<ObjectID>();
  return Status::OK();
}

}  // namespace vineyard
