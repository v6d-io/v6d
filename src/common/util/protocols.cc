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

#include "common/util/boost.h"
#include "common/util/uuid.h"

namespace vineyard {

#define CHECK_IPC_ERROR(tree, type)                             \
  do {                                                          \
    auto stcode = tree.get_optional<int>("code");               \
    if (stcode) {                                               \
      Status st = Status(static_cast<StatusCode>(stcode.get()), \
                         tree.get<std::string>("message", "")); \
      if (!st.ok()) {                                           \
        return st;                                              \
      }                                                         \
    }                                                           \
    RETURN_ON_ASSERT(root.get<std::string>("type") == (type));  \
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

static inline void encode_msg(const ptree& root, std::string& msg) {
  std::stringstream ss;
  bpt::write_json(ss, root, false);
  msg = ss.str();
}

void WriteErrorReply(Status const& status, std::string& msg) {
  encode_msg(status.ToJSON(), msg);
}

void WriteRegisterRequest(std::string& msg) {
  ptree root;
  root.put("type", "register_request");

  encode_msg(root, msg);
}

Status ReadRegisterRequest(const ptree& root) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "register_request");
  return Status::OK();
}

void WriteRegisterReply(const std::string& ipc_socket,
                        const std::string& rpc_endpoint,
                        const uint64_t instance_id, std::string& msg) {
  ptree root;
  root.put("type", "register_reply");
  root.put("ipc_socket", ipc_socket);
  root.put("rpc_endpoint", rpc_endpoint);
  root.put("instance_id", instance_id);

  encode_msg(root, msg);
}

Status ReadRegisterReply(const ptree& root, std::string& ipc_socket,
                         std::string& rpc_endpoint, uint64_t& instance_id) {
  CHECK_IPC_ERROR(root, "register_reply");
  ipc_socket = root.get<std::string>("ipc_socket");
  rpc_endpoint = root.get<std::string>("rpc_endpoint");
  instance_id = root.get<uint64_t>("instance_id");
  return Status::OK();
}

void WriteExitRequest(std::string& msg) {
  ptree root;
  root.put("type", "exit_request");

  encode_msg(root, msg);
}

void WriteGetDataRequest(const ObjectID id, const bool sync_remote,
                         const bool wait, std::string& msg) {
  ptree root;
  root.put("type", "get_data_request");
  root.put("id", VYObjectIDToString(id));
  root.put("sync_remote", sync_remote);
  root.put("wait", wait);

  encode_msg(root, msg);
}

void WriteGetDataRequest(const std::vector<ObjectID>& ids,
                         const bool sync_remote, const bool wait,
                         std::string& msg) {
  ptree root;
  root.put("type", "get_data_request");

  std::vector<std::string> ids_string;
  ids_string.reserve(ids.size());
  for (ObjectID const& id : ids) {
    ids_string.emplace_back(VYObjectIDToString(id));
  }
  root.put("id", boost::algorithm::join(ids_string, ";"));
  root.put("sync_remote", sync_remote);
  root.put("wait", wait);

  encode_msg(root, msg);
}

Status ReadGetDataRequest(const ptree& root, std::vector<ObjectID>& ids,
                          bool& sync_remote, bool& wait) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "get_data_request");
  std::vector<std::string> id_strings;
  std::string id_string = root.get<std::string>("id");
  boost::algorithm::split(id_strings, id_string, boost::is_any_of(";"));
  for (auto const& s : id_strings) {
    ids.emplace_back(VYObjectIDFromString(s));
  }
  sync_remote = root.get<bool>("sync_remote");
  wait = root.get<bool>("wait");
  return Status::OK();
}

void WriteGetDataReply(const ptree& content, std::string& msg) {
  ptree root;
  root.put("type", "get_data_reply");
  root.add_child("content", content);

  encode_msg(root, msg);
}

Status ReadGetDataReply(const ptree& root, ptree& content) {
  CHECK_IPC_ERROR(root, "get_data_reply");
  // should be only one item
  auto content_group = root.get_child("content");
  if (content_group.size() != 1) {
    return Status::ObjectNotExists();
  }
  content = content_group.begin()->second;
  return Status::OK();
}

Status ReadGetDataReply(const ptree& root,
                        std::unordered_map<ObjectID, ptree>& content) {
  CHECK_IPC_ERROR(root, "get_data_reply");
  for (auto const& kv : root.get_child("content")) {
    content.emplace(VYObjectIDFromString(kv.first), kv.second);
  }
  return Status::OK();
}

void WriteListDataRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg) {
  ptree root;
  root.put("type", "list_data_request");
  root.put("pattern", pattern);
  root.put("regex", regex);
  root.put("limit", limit);

  encode_msg(root, msg);
}

Status ReadListDataRequest(const ptree& root, std::string& pattern, bool& regex,
                           size_t& limit) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "list_data_request");
  pattern = root.get<std::string>("pattern");
  regex = root.get<bool>("regex");
  limit = root.get<size_t>("limit");
  return Status::OK();
}

void WriteCreateBufferRequest(const size_t size, std::string& msg) {
  ptree root;
  root.put("type", "create_buffer_request");
  root.put("size", size);

  encode_msg(root, msg);
}

Status ReadCreateBufferRequest(const ptree& root, size_t& size) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "create_buffer_request");
  size = root.get<size_t>("size");
  return Status::OK();
}

void WriteCreateBufferReply(const ObjectID id,
                            const std::shared_ptr<Payload>& object,
                            std::string& msg) {
  ptree root;
  root.put("type", "create_buffer_reply");
  root.put("id", id);
  ptree tree;
  object->ToJSON(tree);
  root.add_child("created", tree);

  encode_msg(root, msg);
}

Status ReadCreateBufferReply(const ptree& root, ObjectID& id, Payload& object) {
  CHECK_IPC_ERROR(root, "create_buffer_reply");
  ptree tree = root.get_child("created");
  id = root.get<ObjectID>("id");
  object.FromJSON(tree);
  return Status::OK();
}

void WriteGetBuffersRequest(const std::unordered_set<ObjectID>& ids,
                            std::string& msg) {
  ptree root;
  root.put("type", "get_buffers_request");
  int idx = 0;
  for (auto const& id : ids) {
    root.put(std::to_string(idx++), id);
  }
  root.put("num", ids.size());

  encode_msg(root, msg);
}

Status ReadGetBuffersRequest(const ptree& root, std::vector<ObjectID>& ids) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "get_buffers_request");
  size_t num = root.get<size_t>("num");
  for (size_t i = 0; i < num; ++i) {
    ids.push_back(root.get<ObjectID>(std::to_string(i)));
  }
  return Status::OK();
}

void WriteGetBuffersReply(const std::vector<std::shared_ptr<Payload>>& objects,
                          std::string& msg) {
  ptree root;
  root.put("type", "get_buffers_reply");
  for (size_t i = 0; i < objects.size(); ++i) {
    ptree tree;
    objects[i]->ToJSON(tree);
    root.add_child(std::to_string(i), tree);
  }
  root.put("num", objects.size());

  encode_msg(root, msg);
}

Status ReadGetBuffersReply(const ptree& root,
                           std::unordered_map<ObjectID, Payload>& objects) {
  CHECK_IPC_ERROR(root, "get_buffers_reply");
  for (size_t i = 0; i < root.get<size_t>("num"); ++i) {
    ptree tree = root.get_child(std::to_string(i));
    Payload object;
    object.FromJSON(tree);
    objects.emplace(object.object_id, object);
  }
  return Status::OK();
}

void WriteCreateDataRequest(const ptree& content, std::string& msg) {
  ptree root;
  root.put("type", "create_data_request");
  root.add_child("content", content);

  encode_msg(root, msg);
}

Status ReadCreateDataRequest(const ptree& root, ptree& content) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "create_data_request");
  content = root.get_child("content");
  return Status::OK();
}

void WriteCreateDataReply(const ObjectID& id, const InstanceID& instance_id,
                          std::string& msg) {
  ptree root;
  root.put("type", "create_data_reply");
  root.put("id", id);
  root.put("instance_id", instance_id);

  encode_msg(root, msg);
}

Status ReadCreateDataReply(const ptree& root, ObjectID& id,
                           InstanceID& instance_id) {
  CHECK_IPC_ERROR(root, "create_data_reply");
  id = root.get<ObjectID>("id");
  instance_id = root.get<ObjectID>("instance_id");
  return Status::OK();
}

void WritePersistRequest(const ObjectID id, std::string& msg) {
  ptree root;
  root.put("type", "persist_request");
  root.put("id", id);

  encode_msg(root, msg);
}

Status ReadPersistRequest(const ptree& root, ObjectID& id) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "persist_request");
  id = root.get<ObjectID>("id");
  return Status::OK();
}

void WritePersistReply(std::string& msg) {
  ptree root;
  root.put("type", "persist_reply");

  encode_msg(root, msg);
}

Status ReadPersistReply(const ptree& root) {
  CHECK_IPC_ERROR(root, "persist_reply");
  return Status::OK();
}

void WriteIfPersistRequest(const ObjectID id, std::string& msg) {
  ptree root;
  root.put("type", "if_persist_request");
  root.put("id", id);

  encode_msg(root, msg);
}

Status ReadIfPersistRequest(const ptree& root, ObjectID& id) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "if_persist_request");
  id = root.get<ObjectID>("id");
  return Status::OK();
}

void WriteIfPersistReply(bool persist, std::string& msg) {
  ptree root;
  root.put("type", "if_persist_reply");
  root.put("persist", persist);

  encode_msg(root, msg);
}

Status ReadIfPersistReply(const ptree& root, bool& persist) {
  CHECK_IPC_ERROR(root, "if_persist_reply");
  persist = root.get<bool>("persist");
  return Status::OK();
}

void WriteExistsRequest(const ObjectID id, std::string& msg) {
  ptree root;
  root.put("type", "exists_request");
  root.put("id", id);

  encode_msg(root, msg);
}

Status ReadExistsRequest(const ptree& root, ObjectID& id) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "exists_request");
  id = root.get<ObjectID>("id");
  return Status::OK();
}

void WriteExistsReply(bool exists, std::string& msg) {
  ptree root;
  root.put("type", "exists_reply");
  root.put("exists", exists);

  encode_msg(root, msg);
}

Status ReadExistsReply(const ptree& root, bool& exists) {
  CHECK_IPC_ERROR(root, "exists_reply");
  exists = root.get<bool>("exists");
  return Status::OK();
}

void WriteDelDataRequest(const ObjectID id, const bool force, const bool deep,
                         std::string& msg) {
  ptree root;
  root.put("type", "del_data_request");
  root.put("id", VYObjectIDToString(id));
  root.put("force", force);
  root.put("deep", deep);

  encode_msg(root, msg);
}

void WriteDelDataRequest(const std::vector<ObjectID>& ids, const bool force,
                         const bool deep, std::string& msg) {
  ptree root;
  root.put("type", "del_data_request");

  std::vector<std::string> ids_string;
  ids_string.reserve(ids.size());
  for (ObjectID const& id : ids) {
    ids_string.emplace_back(VYObjectIDToString(id));
  }
  root.put("id", boost::algorithm::join(ids_string, ";"));

  root.put("force", force);
  root.put("deep", deep);

  encode_msg(root, msg);
}

Status ReadDelDataRequest(const ptree& root, std::vector<ObjectID>& ids,
                          bool& force, bool& deep) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "del_data_request");
  std::vector<std::string> id_strings;
  std::string id_string = root.get<std::string>("id");
  boost::algorithm::split(id_strings, id_string, boost::is_any_of(";"));
  for (auto const& s : id_strings) {
    ids.emplace_back(VYObjectIDFromString(s));
  }
  force = root.get<bool>("force", false);
  deep = root.get<bool>("deep", false);
  return Status::OK();
}

void WriteDelDataReply(std::string& msg) {
  ptree root;
  root.put("type", "del_data_reply");

  encode_msg(root, msg);
}

Status ReadDelDataReply(const ptree& root) {
  CHECK_IPC_ERROR(root, "del_data_reply");
  return Status::OK();
}

void WriteClusterMetaRequest(std::string& msg) {
  ptree root;
  root.put("type", "cluster_meta");

  encode_msg(root, msg);
}

Status ReadClusterMetaRequest(const ptree& root) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "cluster_meta");
  return Status::OK();
}

void WriteClusterMetaReply(const ptree& meta, std::string& msg) {
  ptree root;
  root.put("type", "cluster_meta");
  root.add_child("meta", meta);

  encode_msg(root, msg);
}

Status ReadClusterMetaReply(const ptree& root, ptree& meta) {
  CHECK_IPC_ERROR(root, "cluster_meta");
  meta = root.get_child("meta");
  return Status::OK();
}

void WriteInstanceStatusRequest(std::string& msg) {
  ptree root;
  root.put("type", "instance_status_request");

  encode_msg(root, msg);
}

Status ReadInstanceStatusRequest(const ptree& root) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "instance_status_request");
  return Status::OK();
}

void WriteInstanceStatusReply(const ptree& meta, std::string& msg) {
  ptree root;
  root.put("type", "instance_status_reply");
  root.add_child("meta", meta);

  encode_msg(root, msg);
}

Status ReadInstanceStatusReply(const ptree& root, ptree& meta) {
  CHECK_IPC_ERROR(root, "instance_status_reply");
  meta = root.get_child("meta");
  return Status::OK();
}

void WritePutNameRequest(const ObjectID object_id, const std::string& name,
                         std::string& msg) {
  ptree root;
  root.put("type", "put_name_request");
  root.put("object_id", object_id);
  root.put("name", name);

  encode_msg(root, msg);
}

Status ReadPutNameRequest(const ptree& root, ObjectID& object_id,
                          std::string& name) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "put_name_request");
  object_id = root.get<ObjectID>("object_id");
  name = root.get<std::string>("name");
  return Status::OK();
}

void WritePutNameReply(std::string& msg) {
  ptree root;
  root.put("type", "put_name_reply");

  encode_msg(root, msg);
}

Status ReadPutNameReply(const ptree& root) {
  CHECK_IPC_ERROR(root, "put_name_reply");
  return Status::OK();
}

void WriteGetNameRequest(const std::string& name, const bool wait,
                         std::string& msg) {
  ptree root;
  root.put("type", "get_name_request");
  root.put("name", name);
  root.put("wait", wait);

  encode_msg(root, msg);
}

Status ReadGetNameRequest(const ptree& root, std::string& name, bool& wait) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "get_name_request");
  name = root.get<std::string>("name");
  wait = root.get<bool>("wait");
  return Status::OK();
}

void WriteGetNameReply(const ObjectID& object_id, std::string& msg) {
  ptree root;
  root.put("type", "get_name_reply");
  root.put("object_id", object_id);

  encode_msg(root, msg);
}

Status ReadGetNameReply(const ptree& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, "get_name_reply");
  object_id = root.get<ObjectID>("object_id");
  return Status::OK();
}

void WriteDropNameRequest(const std::string& name, std::string& msg) {
  ptree root;
  root.put("type", "drop_name_request");
  root.put("name", name);

  encode_msg(root, msg);
}

Status ReadDropNameRequest(const ptree& root, std::string& name) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "drop_name_request");
  name = root.get<std::string>("name");
  return Status::OK();
}

void WriteDropNameReply(std::string& msg) {
  ptree root;
  root.put("type", "drop_name_reply");

  encode_msg(root, msg);
}

Status ReadDropNameReply(const ptree& root) {
  CHECK_IPC_ERROR(root, "drop_name_reply");
  return Status::OK();
}

void WriteCreateStreamRequest(const ObjectID& object_id, std::string& msg) {
  ptree root;
  root.put("type", "create_stream_request");
  root.put("object_id", object_id);

  encode_msg(root, msg);
}

Status ReadCreateStreamRequest(const ptree& root, ObjectID& object_id) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "create_stream_request");
  object_id = root.get<ObjectID>("object_id");
  return Status::OK();
}

void WriteCreateStreamReply(std::string& msg) {
  ptree root;
  root.put("type", "create_stream_reply");

  encode_msg(root, msg);
}

Status ReadCreateStreamReply(const ptree& root) {
  CHECK_IPC_ERROR(root, "create_stream_reply");
  return Status::OK();
}

void WriteGetNextStreamChunkRequest(const ObjectID stream_id, const size_t size,
                                    std::string& msg) {
  ptree root;
  root.put("type", "get_next_stream_chunk_request");
  root.put("id", stream_id);
  root.put("size", size);

  encode_msg(root, msg);
}

Status ReadGetNextStreamChunkRequest(const ptree& root, ObjectID& stream_id,
                                     size_t& size) {
  RETURN_ON_ASSERT(root.get<std::string>("type") ==
                   "get_next_stream_chunk_request");
  stream_id = root.get<ObjectID>("id");
  size = root.get<size_t>("size");
  return Status::OK();
}

void WriteGetNextStreamChunkReply(std::shared_ptr<Payload>& object,
                                  std::string& msg) {
  ptree root;
  root.put("type", "get_next_stream_chunk_reply");
  ptree buffer_meta;
  object->ToJSON(buffer_meta);
  root.add_child("buffer", buffer_meta);

  encode_msg(root, msg);
}

Status ReadGetNextStreamChunkReply(const ptree& root, Payload& object) {
  CHECK_IPC_ERROR(root, "get_next_stream_chunk_reply");
  object.FromJSON(root.get_child("buffer"));
  return Status::OK();
}

void WritePullNextStreamChunkRequest(const ObjectID stream_id,
                                     std::string& msg) {
  ptree root;
  root.put("type", "pull_next_stream_chunk_request");
  root.put("id", stream_id);

  encode_msg(root, msg);
}

Status ReadPullNextStreamChunkRequest(const ptree& root, ObjectID& stream_id) {
  RETURN_ON_ASSERT(root.get<std::string>("type") ==
                   "pull_next_stream_chunk_request");
  stream_id = root.get<ObjectID>("id");
  return Status::OK();
}

void WritePullNextStreamChunkReply(std::shared_ptr<Payload>& object,
                                   std::string& msg) {
  ptree root;
  root.put("type", "pull_next_stream_chunk_reply");
  ptree buffer_meta;
  object->ToJSON(buffer_meta);
  root.add_child("buffer", buffer_meta);

  encode_msg(root, msg);
}

Status ReadPullNextStreamChunkReply(const ptree& root, Payload& object) {
  CHECK_IPC_ERROR(root, "pull_next_stream_chunk_reply");
  object.FromJSON(root.get_child("buffer"));
  return Status::OK();
}

void WriteStopStreamRequest(const ObjectID stream_id, const bool failed,
                            std::string& msg) {
  ptree root;
  root.put("type", "stop_stream_request");
  root.put("id", stream_id);
  root.put("failed", failed);

  encode_msg(root, msg);
}

Status ReadStopStreamRequest(const ptree& root, ObjectID& stream_id,
                             bool& failed) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "stop_stream_request");
  stream_id = root.get<ObjectID>("id");
  failed = root.get<bool>("failed");
  return Status::OK();
}

void WriteStopStreamReply(std::string& msg) {
  ptree root;
  root.put("type", "stop_stream_reply");

  encode_msg(root, msg);
}

Status ReadStopStreamReply(const ptree& root) {
  CHECK_IPC_ERROR(root, "stop_stream_reply");
  return Status::OK();
}

void WriteShallowCopyRequest(const ObjectID id, std::string& msg) {
  ptree root;
  root.put("type", "shallow_copy_request");
  root.put("id", id);

  encode_msg(root, msg);
}

Status ReadShallowCopyRequest(const ptree& root, ObjectID& id) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "shallow_copy_request");
  id = root.get<ObjectID>("id");
  return Status::OK();
}

void WriteShallowCopyReply(const ObjectID target_id, std::string& msg) {
  ptree root;
  root.put("type", "shallow_copy_reply");
  root.put("target_id", target_id);

  encode_msg(root, msg);
}

Status ReadShallowCopyReply(const ptree& root, ObjectID& target_id) {
  CHECK_IPC_ERROR(root, "shallow_copy_reply");
  target_id = root.get<ObjectID>("target_id");
  return Status::OK();
}

}  // namespace vineyard
