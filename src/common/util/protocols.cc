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

#include "common/util/protocols.h"

#include <sstream>
#include <unordered_set>

#include "common/util/uuid.h"
#include "common/util/version.h"

namespace vineyard {

#define CHECK_IPC_ERROR(tree, type)                                      \
  do {                                                                   \
    if (tree.contains("code")) {                                         \
      Status st = Status(static_cast<StatusCode>(tree.value("code", 0)), \
                         tree.value("message", ""));                     \
      if (!st.ok()) {                                                    \
        std::stringstream err_message;                                   \
        err_message << "IPC error at " << __FILE__ << ":" << __LINE__;   \
        return Status::Wrap(st, err_message.str());                      \
      }                                                                  \
    }                                                                    \
    RETURN_ON_ASSERT(root.value("type", "UNKNOWN") == (type));           \
  } while (0)

static inline void encode_msg(const json& root, std::string& msg) {
  msg = json_to_string(root);
}

const std::string command_t::REGISTER_REQUEST = "register_request";
const std::string command_t::REGISTER_REPLY = "register_reply";
const std::string command_t::EXIT_REQUEST = "exit_request";
const std::string command_t::EXIT_REPLY = "exit_reply";

// Blobs APIs
const std::string command_t::CREATE_BUFFER_REQUEST = "create_buffer_request";
const std::string command_t::CREATE_BUFFER_REPLY = "create_buffer_reply";
const std::string command_t::CREATE_BUFFERS_REQUEST = "create_buffers_request";
const std::string command_t::CREATE_BUFFERS_REPLY = "create_buffers_reply";
const std::string command_t::CREATE_DISK_BUFFER_REQUEST =
    "create_disk_buffer_request";
const std::string command_t::CREATE_DISK_BUFFER_REPLY =
    "create_disk_buffer_reply";
const std::string command_t::CREATE_GPU_BUFFER_REQUEST =
    "create_gpu_buffer_request";
const std::string command_t::CREATE_GPU_BUFFER_REPLY =
    "create_gpu_buffer_reply";
const std::string command_t::SEAL_BUFFER_REQUEST = "seal_request";
const std::string command_t::SEAL_BUFFER_REPLY = "seal_reply";
const std::string command_t::GET_BUFFERS_REQUEST = "get_buffers_request";
const std::string command_t::GET_BUFFERS_REPLY = "get_buffers_reply";
const std::string command_t::GET_GPU_BUFFERS_REQUEST =
    "get_gpu_buffers_request";
const std::string command_t::GET_GPU_BUFFERS_REPLY = "get_gpu_buffers_reply";
const std::string command_t::DROP_BUFFER_REQUEST = "drop_buffer_request";
const std::string command_t::DROP_BUFFER_REPLY = "drop_buffer_reply";
const std::string command_t::SHRINK_BUFFER_REQUEST = "shrink_buffer_request";
const std::string command_t::SHRINK_BUFFER_REPLY = "shrink_buffer_reply";

const std::string command_t::REQUEST_FD_REQUEST = "request_fd_request";
const std::string command_t::REQUEST_FD_REPLY = "request_fd_reply";

const std::string command_t::CREATE_REMOTE_BUFFER_REQUEST =
    "create_remote_buffer_request";
const std::string command_t::CREATE_REMOTE_BUFFERS_REQUEST =
    "create_remote_buffers_request";
const std::string command_t::GET_REMOTE_BUFFERS_REQUEST =
    "get_remote_buffers_request";

const std::string command_t::INCREASE_REFERENCE_COUNT_REQUEST =
    "increase_reference_count_request";
const std::string command_t::INCREASE_REFERENCE_COUNT_REPLY =
    "increase_reference_count_reply";
const std::string command_t::RELEASE_REQUEST = "release_request";
const std::string command_t::RELEASE_REPLY = "release_reply";
const std::string command_t::DEL_DATA_WITH_FEEDBACKS_REQUEST =
    "del_data_with_feedbacks_request";
const std::string command_t::DEL_DATA_WITH_FEEDBACKS_REPLY =
    "del_data_with_feedbacks_reply";

const std::string command_t::CREATE_BUFFER_PLASMA_REQUEST =
    "create_buffer_by_plasma_request";
const std::string command_t::CREATE_BUFFER_PLASMA_REPLY =
    "create_buffer_by_plasma_reply";
const std::string command_t::GET_BUFFERS_PLASMA_REQUEST =
    "get_buffers_by_plasma_request";
const std::string command_t::GET_BUFFERS_PLASMA_REPLY =
    "get_buffers_by_plasma_reply";
const std::string command_t::PLASMA_SEAL_REQUEST = "plasma_seal_request";
const std::string command_t::PLASMA_SEAL_REPLY = "plasma_seal_reply";
const std::string command_t::PLASMA_RELEASE_REQUEST = "plasma_release_request";
const std::string command_t::PLASMA_RELEASE_REPLY = "plasma_release_reply";
const std::string command_t::PLASMA_DEL_DATA_REQUEST =
    "plasma_delete_data_request";
const std::string command_t::PLASMA_DEL_DATA_REPLY = "plasma_delete_data_reply";

// Metadata APIs
const std::string command_t::CREATE_DATA_REQUEST = "create_data_request";
const std::string command_t::CREATE_DATA_REPLY = "create_data_reply";
const std::string command_t::CREATE_DATAS_REQUEST = "create_datas_request";
const std::string command_t::CREATE_DATAS_REPLY = "create_datas_reply";
const std::string command_t::GET_DATA_REQUEST = "get_data_request";
const std::string command_t::GET_DATA_REPLY = "get_data_reply";
const std::string command_t::LIST_DATA_REQUEST = "list_data_request";
const std::string command_t::LIST_DATA_REPLY = "list_data_reply";
const std::string command_t::DELETE_DATA_REQUEST = "del_data_request";
const std::string command_t::DELETE_DATA_REPLY = "del_data_reply";
const std::string command_t::EXISTS_REQUEST = "exists_request";
const std::string command_t::EXISTS_REPLY = "exists_reply";
const std::string command_t::PERSIST_REQUEST = "persist_request";
const std::string command_t::PERSIST_REPLY = "persist_reply";
const std::string command_t::IF_PERSIST_REQUEST = "if_persist_request";
const std::string command_t::IF_PERSIST_REPLY = "if_persist_reply";
const std::string command_t::LABEL_REQUEST = "label_request";
const std::string command_t::LABEL_REPLY = "label_reply";
const std::string command_t::CLEAR_REQUEST = "clear_request";
const std::string command_t::CLEAR_REPLY = "clear_reply";
const std::string command_t::MEMORY_TRIM_REQUEST = "memory_trim_request";
const std::string command_t::MEMORY_TRIM_REPLY = "memory_trim_reply";

// Stream APIs
const std::string command_t::CREATE_STREAM_REQUEST = "create_stream_request";
const std::string command_t::CREATE_STREAM_REPLY = "create_stream_reply";
const std::string command_t::OPEN_STREAM_REQUEST = "open_stream_request";
const std::string command_t::OPEN_STREAM_REPLY = "open_stream_reply";
const std::string command_t::GET_NEXT_STREAM_CHUNK_REQUEST =
    "get_next_stream_chunk_request";
const std::string command_t::GET_NEXT_STREAM_CHUNK_REPLY =
    "get_next_stream_chunk_reply";
const std::string command_t::PUSH_NEXT_STREAM_CHUNK_REQUEST =
    "push_next_stream_chunk_request";
const std::string command_t::PUSH_NEXT_STREAM_CHUNK_REPLY =
    "push_next_stream_chunk_reply";
const std::string command_t::PULL_NEXT_STREAM_CHUNK_REQUEST =
    "pull_next_stream_chunk_request";
const std::string command_t::PULL_NEXT_STREAM_CHUNK_REPLY =
    "pull_next_stream_chunk_reply";
const std::string command_t::STOP_STREAM_REQUEST = "stop_stream_request";
const std::string command_t::STOP_STREAM_REPLY = "stop_stream_reply";
const std::string command_t::DROP_STREAM_REQUEST = "drop_stream_request";
const std::string command_t::DROP_STREAM_REPLY = "drop_stream_reply";

// Names APIs
const std::string command_t::PUT_NAME_REQUEST = "put_name_request";
const std::string command_t::PUT_NAME_REPLY = "put_name_reply";
const std::string command_t::GET_NAME_REQUEST = "get_name_request";
const std::string command_t::GET_NAME_REPLY = "get_name_reply";
const std::string command_t::LIST_NAME_REQUEST = "list_name_request";
const std::string command_t::LIST_NAME_REPLY = "list_name_reply";
const std::string command_t::DROP_NAME_REQUEST = "drop_name_request";
const std::string command_t::DROP_NAME_REPLY = "drop_name_reply";

// Arena APIs
const std::string command_t::MAKE_ARENA_REQUEST = "make_arena_request";
const std::string command_t::MAKE_ARENA_REPLY = "make_arena_reply";
const std::string command_t::FINALIZE_ARENA_REQUEST = "finalize_arena_request";
const std::string command_t::FINALIZE_ARENA_REPLY = "finalize_arena_reply";

// Session APIs
const std::string command_t::NEW_SESSION_REQUEST = "new_session_request";
const std::string command_t::NEW_SESSION_REPLY = "new_session_reply";
const std::string command_t::DELETE_SESSION_REQUEST = "delete_session_request";
const std::string command_t::DELETE_SESSION_REPLY = "delete_session_reply";

const std::string command_t::MOVE_BUFFERS_OWNERSHIP_REQUEST =
    "move_buffers_ownership_request";
const std::string command_t::MOVE_BUFFERS_OWNERSHIP_REPLY =
    "move_buffers_ownership_reply";

// Spill APIs
const std::string command_t::EVICT_REQUEST = "evict_request";
const std::string command_t::EVICT_REPLY = "evict_reply";
const std::string command_t::LOAD_REQUEST = "load_request";
const std::string command_t::LOAD_REPLY = "load_reply";
const std::string command_t::UNPIN_REQUEST = "unpin_request";
const std::string command_t::UNPIN_REPLY = "unpin_reply";
const std::string command_t::IS_SPILLED_REQUEST = "is_spilled_request";
const std::string command_t::IS_SPILLED_REPLY = "is_spilled_reply";
const std::string command_t::IS_IN_USE_REQUEST = "is_in_use_request";
const std::string command_t::IS_IN_USE_REPLY = "is_in_use_reply";

// Meta APIs
const std::string command_t::CLUSTER_META_REQUEST = "cluster_meta";
const std::string command_t::CLUSTER_META_REPLY = "cluster_meta";
const std::string command_t::INSTANCE_STATUS_REQUEST =
    "instance_status_request";
const std::string command_t::INSTANCE_STATUS_REPLY = "instance_status_reply";
const std::string command_t::MIGRATE_OBJECT_REQUEST = "migrate_object_request";
const std::string command_t::MIGRATE_OBJECT_REPLY = "migrate_object_reply";
const std::string command_t::SHALLOW_COPY_REQUEST = "shallow_copy_request";
const std::string command_t::SHALLOW_COPY_REPLY = "shallow_copy_reply";
const std::string command_t::DEBUG_REQUEST = "debug_command";
const std::string command_t::DEBUG_REPLY = "debug_reply";

// distributed lock
const std::string command_t::ACQUIRE_LOCK_REQUEST = "acquire_lock_request";
const std::string command_t::ACQUIRE_LOCK_REPLY = "acquire_lock_reply";
const std::string command_t::RELEASE_LOCK_REQUEST = "release_lock_request";
const std::string command_t::RELEASE_LOCK_REPLY = "release_lock_reply";

void WriteErrorReply(Status const& status, std::string& msg) {
  encode_msg(status.ToJSON(), msg);
}

void WriteRegisterRequest(std::string& msg, StoreType const& bulk_store_type,
                          const std::string& username,
                          const std::string& password) {
  WriteRegisterRequest(msg, bulk_store_type, RootSessionID(), username,
                       password);
}

void WriteRegisterRequest(std::string& msg, StoreType const& bulk_store_type,
                          const ObjectID& session_id,
                          const std::string& username,
                          const std::string& password) {
  json root;
  root["type"] = command_t::REGISTER_REQUEST;
  root["version"] = vineyard_version();
  root["store_type"] = bulk_store_type;
  root["session_id"] = session_id;
  root["username"] = username;
  root["password"] = password;

  encode_msg(root, msg);
}

Status ReadRegisterRequest(const json& root, std::string& version,
                           StoreType& store_type, SessionID& session_id,
                           std::string& username, std::string& password) {
  CHECK_IPC_ERROR(root, command_t::REGISTER_REQUEST);

  // When the "version" field is missing from the client, we treat it
  // as default unknown version number: 0.0.0.
  version =
      root.value<std::string>("version", /* default */ std::string("0.0.0"));
  session_id = root.value("session_id", /* default */ RootSessionID());

  // Keep backwards compatibility.
  if (root.contains("store_type")) {
    if (root["store_type"].is_number()) {
      store_type = root.value("store_type", /* default */ StoreType::kDefault);
    } else {
      std::string store_type_name =
          root.value("store_type", /* default */ std::string("Normal"));
      if (store_type_name == "Plasma") {
        store_type = StoreType::kPlasma;
      } else {
        store_type = StoreType::kDefault;
      }
    }
  }

  // userpass
  username = root.value("username", /* default */ "");
  password = root.value("password", /* default */ "");

  return Status::OK();
}

void WriteRegisterReply(const std::string& ipc_socket,
                        const std::string& rpc_endpoint,
                        const InstanceID instance_id,
                        const SessionID session_id, const bool store_match,
                        const bool support_rpc_compression, std::string& msg) {
  json root;
  root["type"] = command_t::REGISTER_REPLY;
  root["ipc_socket"] = ipc_socket;
  root["rpc_endpoint"] = rpc_endpoint;
  root["instance_id"] = instance_id;
  root["session_id"] = session_id;
  root["version"] = vineyard_version();
  root["store_match"] = store_match;
  root["support_rpc_compression"] = support_rpc_compression;
  encode_msg(root, msg);
}

Status ReadRegisterReply(const json& root, std::string& ipc_socket,
                         std::string& rpc_endpoint, InstanceID& instance_id,
                         SessionID& session_id, std::string& version,
                         bool& store_match, bool& support_rpc_compression) {
  CHECK_IPC_ERROR(root, command_t::REGISTER_REPLY);
  ipc_socket = root["ipc_socket"].get_ref<std::string const&>();
  rpc_endpoint = root["rpc_endpoint"].get_ref<std::string const&>();
  instance_id = root["instance_id"].get<InstanceID>();
  session_id = root["session_id"].get<SessionID>();

  // When the "version" field is missing from the server, we treat it
  // as default unknown version number: 0.0.0.
  version = root.value<std::string>("version", std::string("0.0.0"));

  store_match = root.value("store_match", true);
  support_rpc_compression = root.value("support_rpc_compression", false);
  return Status::OK();
}

void WriteExitRequest(std::string& msg) {
  json root;
  root["type"] = command_t::EXIT_REQUEST;

  encode_msg(root, msg);
}

void WriteCreateBufferRequest(const size_t size, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_BUFFER_REQUEST;
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadCreateBufferRequest(const json& root, size_t& size) {
  CHECK_IPC_ERROR(root, command_t::CREATE_BUFFER_REQUEST);
  size = root["size"].get<size_t>();
  return Status::OK();
}

void WriteCreateBufferReply(const ObjectID id,
                            const std::shared_ptr<Payload>& object,
                            const int fd_to_send, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_BUFFER_REPLY;
  root["id"] = id;
  root["fd"] = fd_to_send;
  json tree;
  object->ToJSON(tree);
  root["created"] = tree;

  encode_msg(root, msg);
}

Status ReadCreateBufferReply(const json& root, ObjectID& id, Payload& object,
                             int& fd_sent) {
  CHECK_IPC_ERROR(root, command_t::CREATE_BUFFER_REPLY);
  json tree = root["created"];
  id = root["id"].get<ObjectID>();
  object.FromJSON(tree);
  fd_sent = root.value("fd", -1);
  return Status::OK();
}

void WriteCreateBuffersRequest(const std::vector<size_t>& sizes,
                               std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_BUFFERS_REQUEST;
  root["num"] = sizes.size();
  root["sizes"] = sizes;

  encode_msg(root, msg);
}

Status ReadCreateBuffersRequest(const json& root, std::vector<size_t>& sizes) {
  CHECK_IPC_ERROR(root, command_t::CREATE_BUFFERS_REQUEST);
  sizes = root["sizes"].get<std::vector<size_t>>();
  return Status::OK();
}

void WriteCreateBuffersReply(
    const std::vector<ObjectID>& ids,
    const std::vector<std::shared_ptr<Payload>>& objects,
    const std::vector<int>& fds_to_send, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_BUFFERS_REPLY;
  root["num"] = ids.size();
  root["ids"] = ids;
  root["fds"] = fds_to_send;
  json payloads = json::array();
  for (size_t i = 0; i < objects.size(); ++i) {
    json tree;
    objects[i]->ToJSON(tree);
    root[std::to_string(i)] = tree;
    payloads.push_back(tree);
  }
  root["payloads"] = payloads;

  encode_msg(root, msg);
}

Status ReadCreateBuffersReply(const json& root, std::vector<ObjectID>& ids,
                              std::vector<Payload>& objects,
                              std::vector<int>& fds_sent) {
  CHECK_IPC_ERROR(root, command_t::CREATE_BUFFERS_REPLY);
  ids = root["ids"].get<std::vector<ObjectID>>();
  fds_sent = root["fds"].get<std::vector<int>>();
  for (size_t i = 0; i < root["num"]; ++i) {
    json tree = root[std::to_string(i)];
    Payload object;
    object.FromJSON(tree);
    objects.emplace_back(object);
  }
  return Status::OK();
}

void WriteCreateDiskBufferRequest(const size_t size, const std::string& path,
                                  std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_DISK_BUFFER_REQUEST;
  root["size"] = size;
  root["path"] = path;

  encode_msg(root, msg);
}

Status ReadCreateDiskBufferRequest(const json& root, size_t& size,
                                   std::string& path) {
  CHECK_IPC_ERROR(root, command_t::CREATE_DISK_BUFFER_REQUEST);
  size = root["size"].get<size_t>();
  path = root["path"].get<std::string>();
  return Status::OK();
}

void WriteCreateDiskBufferReply(const ObjectID id,
                                const std::shared_ptr<Payload>& object,
                                const int fd_to_send, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_DISK_BUFFER_REPLY;
  root["id"] = id;
  root["fd"] = fd_to_send;
  json tree;
  object->ToJSON(tree);
  root["created"] = tree;

  encode_msg(root, msg);
}

Status ReadCreateDiskBufferReply(const json& root, ObjectID& id,
                                 Payload& object, int& fd_sent) {
  CHECK_IPC_ERROR(root, command_t::CREATE_DISK_BUFFER_REPLY);
  json tree = root["created"];
  id = root["id"].get<ObjectID>();
  object.FromJSON(tree);
  fd_sent = root.value("fd", -1);
  return Status::OK();
}

// GPU related implementations
void WriteCreateGPUBufferRequest(const size_t size, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_GPU_BUFFER_REQUEST;
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadCreateGPUBufferRequest(const json& root, size_t& size) {
  CHECK_IPC_ERROR(root, command_t::CREATE_GPU_BUFFER_REQUEST);
  size = root["size"].get<size_t>();
  return Status::OK();
}

void WriteGPUCreateBufferReply(const ObjectID id,
                               const std::shared_ptr<Payload>& object,
                               const std::vector<int64_t>& handle,
                               std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_GPU_BUFFER_REPLY;
  root["id"] = id;
  std::cout << std::endl;
  root["handle"] = handle;
  json tree;
  object->ToJSON(tree);
  root["created"] = tree;
  encode_msg(root, msg);
}

Status ReadGPUCreateBufferReply(const json& root, ObjectID& id, Payload& object,
                                std::vector<int64_t>& handle) {
  CHECK_IPC_ERROR(root, command_t::CREATE_GPU_BUFFER_REPLY);
  json tree = root["created"];
  id = root["id"].get<ObjectID>();
  object.FromJSON(tree);
  handle = root["handle"].get<std::vector<int64_t>>();
  return Status::OK();
}

void WriteSealRequest(ObjectID const& object_id, std::string& msg) {
  json root;
  root["type"] = command_t::SEAL_BUFFER_REQUEST;
  root["object_id"] = object_id;
  encode_msg(root, msg);
}

Status ReadSealRequest(json const& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, command_t::SEAL_BUFFER_REQUEST);
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteSealReply(std::string& msg) {
  json root;
  root["type"] = command_t::SEAL_BUFFER_REPLY;
  encode_msg(root, msg);
}

Status ReadSealReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::SEAL_BUFFER_REPLY);
  return Status::OK();
}

void WriteGetBuffersRequest(const std::set<ObjectID>& ids, const bool unsafe,
                            std::string& msg) {
  json root;
  root["type"] = command_t::GET_BUFFERS_REQUEST;
  int idx = 0;
  for (auto const& id : ids) {
    root[std::to_string(idx++)] = id;
  }
  root["num"] = ids.size();
  root["unsafe"] = unsafe;

  encode_msg(root, msg);
}

void WriteGetBuffersRequest(const std::unordered_set<ObjectID>& ids,
                            const bool unsafe, std::string& msg) {
  json root;
  root["type"] = command_t::GET_BUFFERS_REQUEST;
  int idx = 0;
  for (auto const& id : ids) {
    root[std::to_string(idx++)] = id;
  }
  root["num"] = ids.size();
  root["unsafe"] = unsafe;

  encode_msg(root, msg);
}

Status ReadGetBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                             bool& unsafe) {
  CHECK_IPC_ERROR(root, command_t::GET_BUFFERS_REQUEST);
  if (root.contains("ids") && root["ids"].is_array()) {
    root["ids"].get_to(ids);
  } else {
    size_t num = root["num"].get<size_t>();
    for (size_t i = 0; i < num; ++i) {
      ids.push_back(root[std::to_string(i)].get<ObjectID>());
    }
  }
  unsafe = root.value("unsafe", false);
  return Status::OK();
}

void WriteGetBuffersReply(const std::vector<std::shared_ptr<Payload>>& objects,
                          const std::vector<int>& fd_to_send,
                          const bool compress, std::string& msg) {
  json root;
  root["type"] = command_t::GET_BUFFERS_REPLY;
  json payloads = json::array();
  for (size_t i = 0; i < objects.size(); ++i) {
    json tree;
    objects[i]->ToJSON(tree);
    root[std::to_string(i)] = tree;
    payloads.push_back(tree);
  }
  // store payloads twice for backwards compatibility
  root["payloads"] = payloads;
  root["fds"] = fd_to_send;
  root["num"] = objects.size();
  root["compress"] = compress;

  encode_msg(root, msg);
}

Status ReadGetBuffersReply(const json& root, std::vector<Payload>& objects,
                           std::vector<int>& fd_sent) {
  CHECK_IPC_ERROR(root, command_t::GET_BUFFERS_REPLY);

  if (root.contains("payloads") && root["payloads"].is_array()) {
    for (auto const& payload : root["payloads"]) {
      Payload object;
      object.FromJSON(payload);
      objects.emplace_back(object);
    }
  } else {
    for (size_t i = 0; i < root.value("num", static_cast<size_t>(0)); ++i) {
      json tree = root[std::to_string(i)];
      Payload object;
      object.FromJSON(tree);
      objects.emplace_back(object);
    }
  }
  if (root.contains("fds")) {
    fd_sent = root["fds"].get<std::vector<int>>();
  }
  return Status::OK();
}

Status ReadGetBuffersReply(const json& root, std::vector<Payload>& objects,
                           std::vector<int>& fd_sent, bool& compress) {
  RETURN_ON_ERROR(ReadGetBuffersReply(root, objects, fd_sent));
  compress = root.value("compress", false);
  return Status::OK();
}

void WriteGetGPUBuffersRequest(const std::set<ObjectID>& ids, const bool unsafe,
                               std::string& msg) {
  json root;
  root["type"] = command_t::GET_GPU_BUFFERS_REQUEST;
  int idx = 0;
  for (auto const& id : ids) {
    root[std::to_string(idx++)] = id;
  }
  root["num"] = ids.size();
  root["unsafe"] = unsafe;

  encode_msg(root, msg);
}

Status ReadGetGPUBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                                bool& unsafe) {
  CHECK_IPC_ERROR(root, command_t::GET_GPU_BUFFERS_REQUEST);
  size_t num = root["num"].get<size_t>();
  for (size_t i = 0; i < num; ++i) {
    ids.push_back(root[std::to_string(i)].get<ObjectID>());
  }
  unsafe = root.value("unsafe", false);
  return Status::OK();
}

void WriteGetGPUBuffersReply(
    const std::vector<std::shared_ptr<Payload>>& objects,
    const std::vector<std::vector<int64_t>>& handles, std::string& msg) {
  json root;
  root["type"] = command_t::GET_GPU_BUFFERS_REPLY;
  for (size_t i = 0; i < objects.size(); ++i) {
    json tree;
    objects[i]->ToJSON(tree);
    root[std::to_string(i)] = tree;
  }
  root["handles"] = handles;
  root["num"] = objects.size();

  encode_msg(root, msg);
}

Status ReadGetGPUBuffersReply(const json& root, std::vector<Payload>& objects,
                              std::vector<std::vector<int64_t>>& handles) {
  CHECK_IPC_ERROR(root, command_t::GET_GPU_BUFFERS_REPLY);
  for (size_t i = 0; i < root["num"]; ++i) {
    json tree = root[std::to_string(i)];
    Payload object;
    object.FromJSON(tree);
    objects.emplace_back(object);
  }
  if (root.contains("handles")) {
    handles = root["handles"].get<std::vector<std::vector<int64_t>>>();
  }
  return Status::OK();
}

void WriteDropBufferRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = command_t::DROP_BUFFER_REQUEST;
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadDropBufferRequest(const json& root, ObjectID& id) {
  CHECK_IPC_ERROR(root, command_t::DROP_BUFFER_REQUEST);
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteDropBufferReply(std::string& msg) {
  json root;
  root["type"] = command_t::DROP_BUFFER_REPLY;

  encode_msg(root, msg);
}

Status ReadDropBufferReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::DROP_BUFFER_REPLY);
  return Status::OK();
}

void WriteShrinkBufferRequest(const ObjectID id, const size_t size,
                              std::string& msg) {
  json root;
  root["type"] = command_t::SHRINK_BUFFER_REQUEST;
  root["id"] = id;
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadShrinkBufferRequest(const json& root, ObjectID& id, size_t& size) {
  CHECK_IPC_ERROR(root, command_t::SHRINK_BUFFER_REQUEST);
  id = root["id"].get<ObjectID>();
  size = root["size"].get<size_t>();
  return Status::OK();
}

void WriteShrinkBufferReply(std::string& msg) {
  json root;
  root["type"] = command_t::SHRINK_BUFFER_REPLY;

  encode_msg(root, msg);
}

Status ReadShrinkBufferReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::SHRINK_BUFFER_REPLY);
  return Status::OK();
}

void WriteCreateRemoteBufferRequest(const size_t size, std::string& msg) {
  WriteCreateRemoteBufferRequest(size, false, msg);
}

void WriteCreateRemoteBufferRequest(const size_t size, const bool compress,
                                    std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_REMOTE_BUFFER_REQUEST;
  root["size"] = size;
  root["compress"] = compress;

  encode_msg(root, msg);
}

Status ReadCreateRemoteBufferRequest(const json& root, size_t& size,
                                     bool& compress) {
  CHECK_IPC_ERROR(root, command_t::CREATE_REMOTE_BUFFER_REQUEST);
  size = root["size"].get<size_t>();
  compress = root.value("compress", false);
  return Status::OK();
}

void WriteCreateRemoteBuffersRequest(const std::vector<size_t>& size,
                                     const bool compress, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_REMOTE_BUFFERS_REQUEST;
  root["num"] = size.size();
  root["sizes"] = size;
  root["compress"] = compress;

  encode_msg(root, msg);
}

Status ReadCreateRemoteBuffersRequest(const json& root,
                                      std::vector<size_t>& size,
                                      bool& compress) {
  CHECK_IPC_ERROR(root, command_t::CREATE_REMOTE_BUFFERS_REQUEST);
  size = root["sizes"].get<std::vector<size_t>>();
  compress = root.value("compress", false);
  return Status::OK();
}

void WriteGetRemoteBuffersRequest(const std::set<ObjectID>& ids,
                                  const bool unsafe, std::string& msg) {
  WriteGetRemoteBuffersRequest(ids, unsafe, false, msg);
}

void WriteGetRemoteBuffersRequest(const std::set<ObjectID>& ids,
                                  const bool unsafe, const bool compress,
                                  std::string& msg) {
  json root;
  root["type"] = command_t::GET_REMOTE_BUFFERS_REQUEST;
  int idx = 0;
  for (auto const& id : ids) {
    root[std::to_string(idx++)] = id;
  }
  root["num"] = ids.size();
  root["unsafe"] = unsafe;
  root["compress"] = compress;

  encode_msg(root, msg);
}

void WriteGetRemoteBuffersRequest(const std::unordered_set<ObjectID>& ids,
                                  const bool unsafe, std::string& msg) {
  WriteGetRemoteBuffersRequest(ids, unsafe, false, msg);
}

void WriteGetRemoteBuffersRequest(const std::unordered_set<ObjectID>& ids,
                                  const bool unsafe, const bool compress,
                                  std::string& msg) {
  json root;
  root["type"] = command_t::GET_REMOTE_BUFFERS_REQUEST;
  int idx = 0;
  for (auto const& id : ids) {
    root[std::to_string(idx++)] = id;
  }
  root["num"] = ids.size();
  root["unsafe"] = unsafe;
  root["compress"] = compress;

  encode_msg(root, msg);
}

Status ReadGetRemoteBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                                   bool& unsafe, bool& compress) {
  CHECK_IPC_ERROR(root, command_t::GET_REMOTE_BUFFERS_REQUEST);
  size_t num = root["num"].get<size_t>();
  for (size_t i = 0; i < num; ++i) {
    ids.push_back(root[std::to_string(i)].get<ObjectID>());
  }
  unsafe = root.value("unsafe", false);
  compress = root.value("compress", false);
  return Status::OK();
}

void WriteIncreaseReferenceCountRequest(const std::vector<ObjectID>& ids,
                                        std::string& msg) {
  json root;
  root["type"] = command_t::INCREASE_REFERENCE_COUNT_REQUEST;
  root["ids"] = ids;
  encode_msg(root, msg);
}

Status ReadIncreaseReferenceCountRequest(json const& root,
                                         std::vector<ObjectID>& ids) {
  CHECK_IPC_ERROR(root, command_t::INCREASE_REFERENCE_COUNT_REQUEST);
  root["ids"].get_to(ids);
  return Status::OK();
}

void WriteIncreaseReferenceCountReply(std::string& msg) {
  json root;
  root["type"] = command_t::INCREASE_REFERENCE_COUNT_REPLY;
  encode_msg(root, msg);
}

Status ReadIncreaseReferenceCountReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::INCREASE_REFERENCE_COUNT_REPLY);
  return Status::OK();
}

void WriteReleaseRequest(ObjectID const& object_id, std::string& msg) {
  json root;
  root["type"] = command_t::RELEASE_REQUEST;
  root["object_id"] = object_id;
  encode_msg(root, msg);
}

Status ReadReleaseRequest(json const& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, command_t::RELEASE_REQUEST);
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteReleaseReply(std::string& msg) {
  json root;
  root["type"] = command_t::RELEASE_REPLY;
  encode_msg(root, msg);
}

Status ReadReleaseReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::RELEASE_REPLY);
  return Status::OK();
}

void WriteDelDataWithFeedbacksRequest(const std::vector<ObjectID>& id,
                                      const bool force, const bool deep,
                                      const bool memory_trim,
                                      const bool fastpath, std::string& msg) {
  json root;
  root["type"] = command_t::DEL_DATA_WITH_FEEDBACKS_REQUEST;
  root["id"] = std::vector<ObjectID>{id};
  root["force"] = force;
  root["deep"] = deep;
  root["memory_trim"] = memory_trim;
  root["fastpath"] = fastpath;

  encode_msg(root, msg);
}

Status ReadDelDataWithFeedbacksRequest(json const& root,
                                       std::vector<ObjectID>& ids, bool& force,
                                       bool& deep, bool& memory_trim,
                                       bool& fastpath) {
  CHECK_IPC_ERROR(root, command_t::DEL_DATA_WITH_FEEDBACKS_REQUEST);
  root["id"].get_to(ids);
  force = root.value("force", false);
  deep = root.value("deep", false);
  memory_trim = root.value("memory_trim", false);
  fastpath = root.value("fastpath", false);
  return Status::OK();
}

void WriteDelDataWithFeedbacksReply(const std::vector<ObjectID>& deleted_bids,
                                    std::string& msg) {
  json root;
  root["type"] = command_t::DEL_DATA_WITH_FEEDBACKS_REPLY;
  root["deleted_bids"] = deleted_bids;

  encode_msg(root, msg);
}

Status ReadDelDataWithFeedbacksReply(json const& root,
                                     std::vector<ObjectID>& deleted_bids) {
  CHECK_IPC_ERROR(root, command_t::DEL_DATA_WITH_FEEDBACKS_REPLY);
  root["deleted_bids"].get_to(deleted_bids);
  return Status::OK();
}

void WriteCreateBufferByPlasmaRequest(PlasmaID const plasma_id,
                                      size_t const size,
                                      size_t const plasma_size,
                                      std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_BUFFER_PLASMA_REQUEST;
  root["plasma_id"] = plasma_id;
  root["plasma_size"] = plasma_size;
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadCreateBufferByPlasmaRequest(json const& root, PlasmaID& plasma_id,
                                       size_t& size, size_t& plasma_size) {
  CHECK_IPC_ERROR(root, command_t::CREATE_BUFFER_PLASMA_REQUEST);
  plasma_id = root["plasma_id"].get<PlasmaID>();
  size = root["size"].get<size_t>();
  plasma_size = root["plasma_size"].get<size_t>();

  return Status::OK();
}

void WriteCreateBufferByPlasmaReply(
    ObjectID const object_id,
    const std::shared_ptr<PlasmaPayload>& plasma_object, int fd_to_send,
    std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_BUFFER_PLASMA_REPLY;
  root["id"] = object_id;
  json tree;
  plasma_object->ToJSON(tree);
  root["created"] = tree;
  root["fd"] = fd_to_send;

  encode_msg(root, msg);
}

Status ReadCreateBufferByPlasmaReply(json const& root, ObjectID& object_id,
                                     PlasmaPayload& plasma_object,
                                     int& fd_sent) {
  CHECK_IPC_ERROR(root, command_t::CREATE_BUFFER_PLASMA_REPLY);
  json tree = root["created"];
  object_id = root["id"].get<ObjectID>();
  plasma_object.FromJSON(tree);
  fd_sent = root.value("fd", -1);
  return Status::OK();
}

void WriteGetBuffersByPlasmaRequest(std::set<PlasmaID> const& plasma_ids,
                                    const bool unsafe, std::string& msg) {
  json root;
  root["type"] = command_t::GET_BUFFERS_PLASMA_REQUEST;
  int idx = 0;
  for (auto const& eid : plasma_ids) {
    root[std::to_string(idx++)] = eid;
  }
  root["num"] = plasma_ids.size();
  root["unsafe"] = unsafe;

  encode_msg(root, msg);
}

Status ReadGetBuffersByPlasmaRequest(const json& root,
                                     std::vector<PlasmaID>& plasma_ids,
                                     bool& unsafe) {
  CHECK_IPC_ERROR(root, command_t::GET_BUFFERS_PLASMA_REQUEST);
  size_t num = root["num"].get<size_t>();
  for (size_t i = 0; i < num; ++i) {
    plasma_ids.push_back(root[std::to_string(i)].get<PlasmaID>());
  }
  unsafe = root.value("unsafe", false);
  return Status::OK();
}

void WriteGetBuffersByPlasmaReply(
    std::vector<std::shared_ptr<PlasmaPayload>> const& plasma_objects,
    std::string& msg) {
  json root;
  root["type"] = command_t::GET_BUFFERS_PLASMA_REPLY;
  for (size_t i = 0; i < plasma_objects.size(); ++i) {
    json tree;
    plasma_objects[i]->ToJSON(tree);
    root[std::to_string(i)] = tree;
  }
  root["num"] = plasma_objects.size();

  encode_msg(root, msg);
}

Status ReadGetBuffersByPlasmaReply(json const& root,
                                   std::vector<PlasmaPayload>& plasma_objects) {
  CHECK_IPC_ERROR(root, command_t::GET_BUFFERS_PLASMA_REPLY);
  for (size_t i = 0; i < root["num"]; ++i) {
    json tree = root[std::to_string(i)];
    PlasmaPayload plasma_object;
    plasma_object.FromJSON(tree);
    plasma_objects.emplace_back(plasma_object);
  }
  return Status::OK();
}
void WritePlasmaSealRequest(PlasmaID const& plasma_id, std::string& msg) {
  json root;
  root["type"] = command_t::PLASMA_SEAL_REQUEST;
  root["plasma_id"] = plasma_id;
  encode_msg(root, msg);
}

Status ReadPlasmaSealRequest(json const& root, PlasmaID& plasma_id) {
  CHECK_IPC_ERROR(root, command_t::PLASMA_SEAL_REQUEST);
  plasma_id = root["plasma_id"].get<PlasmaID>();
  return Status::OK();
}
void WritePlasmaReleaseRequest(PlasmaID const& plasma_id, std::string& msg) {
  json root;
  root["type"] = command_t::PLASMA_RELEASE_REQUEST;
  root["plasma_id"] = plasma_id;
  encode_msg(root, msg);
}

Status ReadPlasmaReleaseRequest(json const& root, PlasmaID& plasma_id) {
  CHECK_IPC_ERROR(root, command_t::PLASMA_RELEASE_REQUEST);
  plasma_id = root["plasma_id"].get<PlasmaID>();
  return Status::OK();
}

void WritePlasmaReleaseReply(std::string& msg) {
  json root;
  root["type"] = command_t::PLASMA_RELEASE_REPLY;
  encode_msg(root, msg);
}

Status ReadPlasmaReleaseReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::PLASMA_RELEASE_REPLY);
  return Status::OK();
}

void WritePlasmaDelDataRequest(PlasmaID const& plasma_id, std::string& msg) {
  json root;
  root["type"] = command_t::PLASMA_DEL_DATA_REQUEST;
  root["plasma_id"] = plasma_id;
  encode_msg(root, msg);
}

Status ReadPlasmaDelDataRequest(json const& root, PlasmaID& plasma_id) {
  CHECK_IPC_ERROR(root, command_t::PLASMA_DEL_DATA_REQUEST);
  plasma_id = root["plasma_id"].get<PlasmaID>();
  return Status::OK();
}

void WritePlasmaDelDataReply(std::string& msg) {
  json root;
  root["type"] = command_t::PLASMA_DEL_DATA_REPLY;
  encode_msg(root, msg);
}

Status ReadPlasmaDelDataReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::PLASMA_DEL_DATA_REPLY);
  return Status::OK();
}

void WriteCreateDataRequest(const json& content, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_DATA_REQUEST;
  root["content"] = content;

  encode_msg(root, msg);
}

Status ReadCreateDataRequest(const json& root, json& content) {
  CHECK_IPC_ERROR(root, command_t::CREATE_DATA_REQUEST);
  content = root["content"];
  return Status::OK();
}

void WriteCreateDataReply(const ObjectID& id, const Signature& signature,
                          const InstanceID& instance_id, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_DATA_REPLY;
  root["id"] = id;
  root["signature"] = signature;
  root["instance_id"] = instance_id;

  encode_msg(root, msg);
}

Status ReadCreateDataReply(const json& root, ObjectID& id, Signature& signature,
                           InstanceID& instance_id) {
  CHECK_IPC_ERROR(root, command_t::CREATE_DATA_REPLY);
  id = root["id"].get<ObjectID>();
  signature = root["signature"].get<Signature>();
  instance_id = root["instance_id"].get<InstanceID>();
  return Status::OK();
}

void WriteCreateDatasRequest(const std::vector<json>& contents,
                             std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_DATAS_REQUEST;
  root["num"] = contents.size();
  root["contents"] = contents;

  encode_msg(root, msg);
}

Status ReadCreateDatasRequest(const json& root, std::vector<json>& contents) {
  CHECK_IPC_ERROR(root, command_t::CREATE_DATAS_REQUEST);
  contents = root["contents"].get<std::vector<json>>();
  return Status::OK();
}

void WriteCreateDatasReply(const std::vector<ObjectID>& ids,
                           const std::vector<Signature>& signatures,
                           const std::vector<InstanceID>& instance_ids,
                           std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_DATAS_REPLY;
  root["num"] = ids.size();
  root["ids"] = ids;
  root["signatures"] = signatures;
  root["instance_ids"] = instance_ids;

  encode_msg(root, msg);
}

Status ReadCreateDatasReply(const json& root, std::vector<ObjectID>& ids,
                            std::vector<Signature>& signatures,
                            std::vector<InstanceID>& instance_ids) {
  CHECK_IPC_ERROR(root, command_t::CREATE_DATAS_REPLY);
  ids = root["ids"].get<std::vector<ObjectID>>();
  signatures = root["signatures"].get<std::vector<Signature>>();
  instance_ids = root["instance_ids"].get<std::vector<InstanceID>>();
  return Status::OK();
}

void WriteGetDataRequest(const ObjectID id, const bool sync_remote,
                         const bool wait, std::string& msg) {
  json root;
  root["type"] = command_t::GET_DATA_REQUEST;
  root["id"] = std::vector<ObjectID>{id};
  root["sync_remote"] = sync_remote;
  root["wait"] = wait;

  encode_msg(root, msg);
}

void WriteGetDataRequest(const std::vector<ObjectID>& ids,
                         const bool sync_remote, const bool wait,
                         std::string& msg) {
  json root;
  root["type"] = command_t::GET_DATA_REQUEST;
  root["id"] = ids;
  root["sync_remote"] = sync_remote;
  root["wait"] = wait;

  encode_msg(root, msg);
}

Status ReadGetDataRequest(const json& root, std::vector<ObjectID>& ids,
                          bool& sync_remote, bool& wait) {
  CHECK_IPC_ERROR(root, command_t::GET_DATA_REQUEST);
  root["id"].get_to(ids);
  sync_remote = root.value("sync_remote", false);
  wait = root.value("wait", false);
  return Status::OK();
}

void WriteGetDataReply(const json& content, std::string& msg) {
  json root;
  root["type"] = command_t::GET_DATA_REPLY;
  root["content"] = content;

  encode_msg(root, msg);
}

Status ReadGetDataReply(const json& root, json& content) {
  CHECK_IPC_ERROR(root, command_t::GET_DATA_REPLY);
  // should be only one item
  auto content_group = root["content"];
  if (content_group.size() != 1) {
    return Status::ObjectNotExists("failed to read get_data reply: " +
                                   root.dump());
  }
  content = *content_group.begin();
  return Status::OK();
}

Status ReadGetDataReply(const json& root,
                        std::unordered_map<ObjectID, json>& content) {
  CHECK_IPC_ERROR(root, command_t::GET_DATA_REPLY);
  for (auto const& kv : root["content"].items()) {
    content.emplace(ObjectIDFromString(kv.key()), kv.value());
  }
  return Status::OK();
}

void WriteListDataRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg) {
  json root;
  root["type"] = command_t::LIST_DATA_REQUEST;
  root["pattern"] = pattern;
  root["regex"] = regex;
  root["limit"] = limit;

  encode_msg(root, msg);
}

Status ReadListDataRequest(const json& root, std::string& pattern, bool& regex,
                           size_t& limit) {
  CHECK_IPC_ERROR(root, command_t::LIST_DATA_REQUEST);
  pattern = root["pattern"].get_ref<std::string const&>();
  regex = root.value("regex", false);
  limit = root["limit"].get<size_t>();
  return Status::OK();
}

void WriteDelDataRequest(const ObjectID id, const bool force, const bool deep,
                         const bool memory_trim, const bool fastpath,
                         std::string& msg) {
  json root;
  root["type"] = command_t::DELETE_DATA_REQUEST;
  root["id"] = std::vector<ObjectID>{id};
  root["force"] = force;
  root["deep"] = deep;
  root["fastpath"] = fastpath;
  root["memory_trim"] = memory_trim;

  encode_msg(root, msg);
}

void WriteDelDataRequest(const std::vector<ObjectID>& ids, const bool force,
                         const bool deep, const bool memory_trim,
                         const bool fastpath, std::string& msg) {
  json root;
  root["type"] = command_t::DELETE_DATA_REQUEST;
  root["id"] = ids;
  root["force"] = force;
  root["deep"] = deep;
  root["fastpath"] = fastpath;
  root["memory_trim"] = memory_trim;

  encode_msg(root, msg);
}

Status ReadDelDataRequest(const json& root, std::vector<ObjectID>& ids,
                          bool& force, bool& deep, bool& memory_trim,
                          bool& fastpath) {
  CHECK_IPC_ERROR(root, command_t::DELETE_DATA_REQUEST);
  root["id"].get_to(ids);
  force = root.value("force", false);
  deep = root.value("deep", false);
  fastpath = root.value("fastpath", false);
  memory_trim = root.value("memory_trim", false);
  return Status::OK();
}

void WriteDelDataReply(std::string& msg) {
  json root;
  root["type"] = command_t::DELETE_DATA_REPLY;

  encode_msg(root, msg);
}

Status ReadDelDataReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::DELETE_DATA_REPLY);
  return Status::OK();
}

void WriteExistsRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = command_t::EXISTS_REQUEST;
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadExistsRequest(const json& root, ObjectID& id) {
  CHECK_IPC_ERROR(root, command_t::EXISTS_REQUEST);
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteExistsReply(bool exists, std::string& msg) {
  json root;
  root["type"] = command_t::EXISTS_REPLY;
  root["exists"] = exists;

  encode_msg(root, msg);
}

Status ReadExistsReply(const json& root, bool& exists) {
  CHECK_IPC_ERROR(root, command_t::EXISTS_REPLY);
  exists = root.value("exists", false);
  return Status::OK();
}

void WritePersistRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = command_t::PERSIST_REQUEST;
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadPersistRequest(const json& root, ObjectID& id) {
  CHECK_IPC_ERROR(root, command_t::PERSIST_REQUEST);
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WritePersistReply(std::string& msg) {
  json root;
  root["type"] = command_t::PERSIST_REPLY;

  encode_msg(root, msg);
}

Status ReadPersistReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::PERSIST_REPLY);
  return Status::OK();
}

void WriteIfPersistRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = command_t::IF_PERSIST_REQUEST;
  root["id"] = id;

  encode_msg(root, msg);
}

Status ReadIfPersistRequest(const json& root, ObjectID& id) {
  CHECK_IPC_ERROR(root, command_t::IF_PERSIST_REQUEST);
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteIfPersistReply(bool persist, std::string& msg) {
  json root;
  root["type"] = command_t::IF_PERSIST_REPLY;
  root["persist"] = persist;

  encode_msg(root, msg);
}

Status ReadIfPersistReply(const json& root, bool& persist) {
  CHECK_IPC_ERROR(root, command_t::IF_PERSIST_REPLY);
  persist = root.value("persist", false);
  return Status::OK();
}

void WriteLabelRequest(const ObjectID id, const std::string& key,
                       const std::string& value, std::string& msg) {
  json root;
  root["type"] = command_t::LABEL_REQUEST;
  root["id"] = id;
  root["keys"] = std::vector<std::string>{key};
  root["values"] = std::vector<std::string>{value};
  encode_msg(root, msg);
}

void WriteLabelRequest(const ObjectID id, const std::vector<std::string>& keys,
                       const std::vector<std::string>& values,
                       std::string& msg) {
  json root;
  root["type"] = command_t::LABEL_REQUEST;
  root["id"] = id;
  root["keys"] = keys;
  root["values"] = values;
  encode_msg(root, msg);
}

void WriteLabelRequest(const ObjectID id,
                       const std::map<std::string, std::string>& kvs,
                       std::string& msg) {
  json root;
  std::vector<std::string> keys, values;
  for (auto const& item : kvs) {
    keys.push_back(item.first);
    values.push_back(item.second);
  }
  root["type"] = command_t::LABEL_REQUEST;
  root["id"] = id;
  root["keys"] = keys;
  root["values"] = values;
  encode_msg(root, msg);
}

Status ReadLabelRequest(json const& root, ObjectID& id,
                        std::vector<std::string>& keys,
                        std::vector<std::string>& values) {
  CHECK_IPC_ERROR(root, command_t::LABEL_REQUEST);
  id = root["id"].get<ObjectID>();
  root["keys"].get_to(keys);
  root["values"].get_to(values);
  return Status::OK();
}

void WriteLabelReply(std::string& msg) {
  json root;
  root["type"] = command_t::LABEL_REPLY;
  encode_msg(root, msg);
}

Status ReadLabelReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::LABEL_REPLY);
  return Status::OK();
}

void WriteClearRequest(std::string& msg) {
  json root;
  root["type"] = command_t::CLEAR_REQUEST;

  encode_msg(root, msg);
}

Status ReadClearRequest(const json& root) {
  CHECK_IPC_ERROR(root, command_t::CLEAR_REQUEST);
  return Status::OK();
}

void WriteClearReply(std::string& msg) {
  json root;
  root["type"] = command_t::CLEAR_REPLY;
  encode_msg(root, msg);
}

Status ReadClearReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::CLEAR_REPLY);
  return Status::OK();
}

void WriteMemoryTrimRequest(std::string& msg) {
  json root;
  root["type"] = command_t::MEMORY_TRIM_REQUEST;

  encode_msg(root, msg);
}

Status ReadMemoryTrimRequest(const json& root) {
  CHECK_IPC_ERROR(root, command_t::MEMORY_TRIM_REQUEST);
  return Status::OK();
}

void WriteMemoryTrimReply(const bool trimmed, std::string& msg) {
  json root;
  root["type"] = command_t::MEMORY_TRIM_REPLY;
  root["trimmed"] = trimmed;

  encode_msg(root, msg);
}

Status ReadMemoryTrimReply(const json& root, bool& trimmed) {
  CHECK_IPC_ERROR(root, command_t::MEMORY_TRIM_REPLY);
  trimmed = root.value("trimmed", false);
  return Status::OK();
}

void WriteCreateStreamRequest(const ObjectID& object_id, std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_STREAM_REQUEST;
  root["object_id"] = object_id;

  encode_msg(root, msg);
}

Status ReadCreateStreamRequest(const json& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, command_t::CREATE_STREAM_REQUEST);
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteCreateStreamReply(std::string& msg) {
  json root;
  root["type"] = command_t::CREATE_STREAM_REPLY;

  encode_msg(root, msg);
}

Status ReadCreateStreamReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::CREATE_STREAM_REPLY);
  return Status::OK();
}

void WriteOpenStreamRequest(const ObjectID& object_id, const int64_t& mode,
                            std::string& msg) {
  json root;
  root["type"] = command_t::OPEN_STREAM_REQUEST;
  root["object_id"] = object_id;
  root["mode"] = mode;

  encode_msg(root, msg);
}

Status ReadOpenStreamRequest(const json& root, ObjectID& object_id,
                             int64_t& mode) {
  CHECK_IPC_ERROR(root, command_t::OPEN_STREAM_REQUEST);
  object_id = root["object_id"].get<ObjectID>();
  mode = root["mode"].get<int64_t>();
  return Status::OK();
}

void WriteOpenStreamReply(std::string& msg) {
  json root;
  root["type"] = command_t::OPEN_STREAM_REPLY;

  encode_msg(root, msg);
}

Status ReadOpenStreamReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::OPEN_STREAM_REPLY);
  return Status::OK();
}

void WriteGetNextStreamChunkRequest(const ObjectID stream_id, const size_t size,
                                    std::string& msg) {
  json root;
  root["type"] = command_t::GET_NEXT_STREAM_CHUNK_REQUEST;
  root["id"] = stream_id;
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadGetNextStreamChunkRequest(const json& root, ObjectID& stream_id,
                                     size_t& size) {
  CHECK_IPC_ERROR(root, command_t::GET_NEXT_STREAM_CHUNK_REQUEST);
  stream_id = root["id"].get<ObjectID>();
  size = root["size"].get<size_t>();
  return Status::OK();
}

void WriteGetNextStreamChunkReply(std::shared_ptr<Payload> const& object,
                                  int fd_sent, std::string& msg) {
  json root;
  root["type"] = command_t::GET_NEXT_STREAM_CHUNK_REPLY;
  json buffer_meta;
  object->ToJSON(buffer_meta);
  root["buffer"] = buffer_meta;
  root["fd"] = fd_sent;

  encode_msg(root, msg);
}

Status ReadGetNextStreamChunkReply(const json& root, Payload& object,
                                   int& fd_sent) {
  CHECK_IPC_ERROR(root, command_t::GET_NEXT_STREAM_CHUNK_REPLY);
  object.FromJSON(root["buffer"]);
  fd_sent = root.value("fd", -1);
  return Status::OK();
}

void WritePushNextStreamChunkRequest(const ObjectID stream_id,
                                     const ObjectID chunk, std::string& msg) {
  json root;
  root["type"] = command_t::PUSH_NEXT_STREAM_CHUNK_REQUEST;
  root["id"] = stream_id;
  root["chunk"] = chunk;

  encode_msg(root, msg);
}

Status ReadPushNextStreamChunkRequest(const json& root, ObjectID& stream_id,
                                      ObjectID& chunk) {
  CHECK_IPC_ERROR(root, command_t::PUSH_NEXT_STREAM_CHUNK_REQUEST);
  stream_id = root["id"].get<ObjectID>();
  chunk = root["chunk"].get<ObjectID>();
  return Status::OK();
}

void WritePushNextStreamChunkReply(std::string& msg) {
  json root;
  root["type"] = command_t::PUSH_NEXT_STREAM_CHUNK_REPLY;
  encode_msg(root, msg);
}

Status ReadPushNextStreamChunkReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::PUSH_NEXT_STREAM_CHUNK_REPLY);
  return Status::OK();
}

void WritePullNextStreamChunkRequest(const ObjectID stream_id,
                                     std::string& msg) {
  json root;
  root["type"] = command_t::PULL_NEXT_STREAM_CHUNK_REQUEST;
  root["id"] = stream_id;

  encode_msg(root, msg);
}

Status ReadPullNextStreamChunkRequest(const json& root, ObjectID& stream_id) {
  CHECK_IPC_ERROR(root, command_t::PULL_NEXT_STREAM_CHUNK_REQUEST);
  stream_id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WritePullNextStreamChunkReply(ObjectID const chunk, std::string& msg) {
  json root;
  root["type"] = command_t::PULL_NEXT_STREAM_CHUNK_REPLY;
  root["chunk"] = chunk;

  encode_msg(root, msg);
}

Status ReadPullNextStreamChunkReply(const json& root, ObjectID& chunk) {
  CHECK_IPC_ERROR(root, command_t::PULL_NEXT_STREAM_CHUNK_REPLY);
  chunk = root["chunk"].get<ObjectID>();
  return Status::OK();
}

void WriteStopStreamRequest(const ObjectID stream_id, const bool failed,
                            std::string& msg) {
  json root;
  root["type"] = command_t::STOP_STREAM_REQUEST;
  root["id"] = stream_id;
  root["failed"] = failed;

  encode_msg(root, msg);
}

Status ReadStopStreamRequest(const json& root, ObjectID& stream_id,
                             bool& failed) {
  CHECK_IPC_ERROR(root, command_t::STOP_STREAM_REQUEST);
  stream_id = root["id"].get<ObjectID>();
  failed = root["failed"].get<bool>();
  return Status::OK();
}

void WriteStopStreamReply(std::string& msg) {
  json root;
  root["type"] = command_t::STOP_STREAM_REPLY;

  encode_msg(root, msg);
}

Status ReadStopStreamReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::STOP_STREAM_REPLY);
  return Status::OK();
}

void WriteDropStreamRequest(const ObjectID stream_id, std::string& msg) {
  json root;
  root["type"] = command_t::DROP_STREAM_REQUEST;
  root["id"] = stream_id;

  encode_msg(root, msg);
}

Status ReadDropStreamRequest(const json& root, ObjectID& stream_id) {
  CHECK_IPC_ERROR(root, command_t::DROP_STREAM_REQUEST);
  stream_id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteDropStreamReply(std::string& msg) {
  json root;
  root["type"] = command_t::DROP_STREAM_REPLY;

  encode_msg(root, msg);
}

Status ReadDropStreamReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::DROP_STREAM_REPLY);
  return Status::OK();
}

void WritePutNameRequest(const ObjectID object_id, const std::string& name,
                         std::string& msg) {
  json root;
  root["type"] = command_t::PUT_NAME_REQUEST;
  root["object_id"] = object_id;
  root["name"] = name;

  encode_msg(root, msg);
}

Status ReadPutNameRequest(const json& root, ObjectID& object_id,
                          std::string& name) {
  CHECK_IPC_ERROR(root, command_t::PUT_NAME_REQUEST);
  object_id = root["object_id"].get<ObjectID>();
  name = root["name"].get_ref<std::string const&>();
  return Status::OK();
}

void WritePutNameReply(std::string& msg) {
  json root;
  root["type"] = command_t::PUT_NAME_REPLY;

  encode_msg(root, msg);
}

Status ReadPutNameReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::PUT_NAME_REPLY);
  return Status::OK();
}

void WriteGetNameRequest(const std::string& name, const bool wait,
                         std::string& msg) {
  json root;
  root["type"] = command_t::GET_NAME_REQUEST;
  root["name"] = name;
  root["wait"] = wait;

  encode_msg(root, msg);
}

Status ReadGetNameRequest(const json& root, std::string& name, bool& wait) {
  CHECK_IPC_ERROR(root, command_t::GET_NAME_REQUEST);
  name = root["name"].get_ref<std::string const&>();
  wait = root["wait"].get<bool>();
  return Status::OK();
}

void WriteGetNameReply(const ObjectID& object_id, std::string& msg) {
  json root;
  root["type"] = command_t::GET_NAME_REPLY;
  root["object_id"] = object_id;

  encode_msg(root, msg);
}

Status ReadGetNameReply(const json& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, command_t::GET_NAME_REPLY);
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteListNameRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg) {
  json root;
  root["type"] = command_t::LIST_NAME_REQUEST;
  root["pattern"] = pattern;
  root["regex"] = regex;
  root["limit"] = limit;

  encode_msg(root, msg);
}

Status ReadListNameRequest(const json& root, std::string& pattern, bool& regex,
                           size_t& limit) {
  CHECK_IPC_ERROR(root, command_t::LIST_NAME_REQUEST);
  pattern = root["pattern"].get_ref<std::string const&>();
  regex = root.value("regex", false);
  limit = root["limit"].get<size_t>();
  return Status::OK();
}

void WriteListNameReply(std::map<std::string, ObjectID> const& names,
                        std::string& msg) {
  json root;
  root["type"] = command_t::LIST_NAME_REPLY;
  root["size"] = names.size();
  root["names"] = names;

  encode_msg(root, msg);
}

Status ReadListNameReply(const json& root,
                         std::map<std::string, ObjectID>& names) {
  CHECK_IPC_ERROR(root, command_t::LIST_NAME_REPLY);
  names = root.value("names", std::map<std::string, ObjectID>{});
  return Status::OK();
}

void WriteDropNameRequest(const std::string& name, std::string& msg) {
  json root;
  root["type"] = command_t::DROP_NAME_REQUEST;
  root["name"] = name;

  encode_msg(root, msg);
}

Status ReadDropNameRequest(const json& root, std::string& name) {
  CHECK_IPC_ERROR(root, command_t::DROP_NAME_REQUEST);
  name = root["name"].get_ref<std::string const&>();
  return Status::OK();
}

void WriteDropNameReply(std::string& msg) {
  json root;
  root["type"] = command_t::DROP_NAME_REPLY;

  encode_msg(root, msg);
}

Status ReadDropNameReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::DROP_NAME_REPLY);
  return Status::OK();
}

void WriteMakeArenaRequest(const size_t size, std::string& msg) {
  json root;
  root["type"] = command_t::MAKE_ARENA_REQUEST;
  root["size"] = size;

  encode_msg(root, msg);
}

Status ReadMakeArenaRequest(const json& root, size_t& size) {
  CHECK_IPC_ERROR(root, command_t::MAKE_ARENA_REQUEST);
  size = root["size"].get<size_t>();
  return Status::OK();
}

void WriteMakeArenaReply(const int fd, const size_t size, const uintptr_t base,
                         std::string& msg) {
  json root;
  root["type"] = command_t::MAKE_ARENA_REPLY;
  root["fd"] = fd;
  root["size"] = size;
  root["base"] = base;

  encode_msg(root, msg);
}

Status ReadMakeArenaReply(const json& root, int& fd, size_t& size,
                          uintptr_t& base) {
  CHECK_IPC_ERROR(root, command_t::MAKE_ARENA_REPLY);
  fd = root["fd"].get<int>();
  size = root["size"].get<size_t>();
  base = root["base"].get<uintptr_t>();
  return Status::OK();
}

void WriteFinalizeArenaRequest(const int fd, std::vector<size_t> const& offsets,
                               std::vector<size_t> const& sizes,
                               std::string& msg) {
  json root;
  root["type"] = command_t::FINALIZE_ARENA_REQUEST;
  root["fd"] = fd;
  root["offsets"] = offsets;
  root["sizes"] = sizes;

  encode_msg(root, msg);
}

Status ReadFinalizeArenaRequest(const json& root, int& fd,
                                std::vector<size_t>& offsets,
                                std::vector<size_t>& sizes) {
  CHECK_IPC_ERROR(root, command_t::FINALIZE_ARENA_REQUEST);
  fd = root["fd"].get<int>();
  offsets = root["offsets"].get<std::vector<size_t>>();
  sizes = root["sizes"].get<std::vector<size_t>>();
  return Status::OK();
}

void WriteFinalizeArenaReply(std::string& msg) {
  json root;
  root["type"] = command_t::FINALIZE_ARENA_REPLY;
  encode_msg(root, msg);
}

Status ReadFinalizeArenaReply(const json& root) {
  CHECK_IPC_ERROR(root, command_t::FINALIZE_ARENA_REPLY);
  return Status::OK();
}

void WriteNewSessionRequest(std::string& msg,
                            StoreType const& bulk_store_type) {
  json root;
  root["type"] = command_t::NEW_SESSION_REQUEST;
  root["bulk_store_type"] = bulk_store_type;
  encode_msg(root, msg);
}

Status ReadNewSessionRequest(json const& root, StoreType& bulk_store_type) {
  CHECK_IPC_ERROR(root, command_t::NEW_SESSION_REQUEST);
  bulk_store_type =
      root.value("bulk_store_type", /* default */ StoreType::kDefault);
  return Status::OK();
}

void WriteNewSessionReply(std::string& msg, std::string const& socket_path) {
  json root;
  root["type"] = command_t::NEW_SESSION_REPLY;
  root["socket_path"] = socket_path;
  encode_msg(root, msg);
}

Status ReadNewSessionReply(const json& root, std::string& socket_path) {
  CHECK_IPC_ERROR(root, command_t::NEW_SESSION_REPLY);
  socket_path = root["socket_path"].get_ref<std::string const&>();
  return Status::OK();
}

void WriteDeleteSessionRequest(std::string& msg) {
  json root;
  root["type"] = command_t::DELETE_SESSION_REQUEST;
  encode_msg(root, msg);
}

void WriteDeleteSessionReply(std::string& msg) {
  json root;
  root["type"] = command_t::DELETE_SESSION_REPLY;
  encode_msg(root, msg);
}

template <>
void WriteMoveBuffersOwnershipRequest<ObjectID, ObjectID>(
    std::map<ObjectID, ObjectID> const& id_to_id, SessionID const session_id,
    std::string& msg) {
  json root;
  root["type"] = command_t::MOVE_BUFFERS_OWNERSHIP_REQUEST;
  root["id_to_id"] = id_to_id;
  root["session_id"] = session_id;
  encode_msg(root, msg);
}

template <>
void WriteMoveBuffersOwnershipRequest<ObjectID, PlasmaID>(
    std::map<ObjectID, PlasmaID> const& id_to_pid, SessionID const session_id,
    std::string& msg) {
  json root;
  root["type"] = command_t::MOVE_BUFFERS_OWNERSHIP_REQUEST;
  root["id_to_pid"] = id_to_pid;
  root["session_id"] = session_id;
  encode_msg(root, msg);
}

template <>
void WriteMoveBuffersOwnershipRequest<PlasmaID, ObjectID>(
    std::map<PlasmaID, ObjectID> const& pid_to_id, SessionID const session_id,
    std::string& msg) {
  json root;
  root["type"] = command_t::MOVE_BUFFERS_OWNERSHIP_REQUEST;
  root["pid_to_id"] = pid_to_id;
  root["session_id"] = session_id;
  encode_msg(root, msg);
}

template <>
void WriteMoveBuffersOwnershipRequest<PlasmaID, PlasmaID>(
    std::map<PlasmaID, PlasmaID> const& pid_to_pid, SessionID const session_id,
    std::string& msg) {
  json root;
  root["type"] = command_t::MOVE_BUFFERS_OWNERSHIP_REQUEST;
  root["pid_to_pid"] = pid_to_pid;
  root["session_id"] = session_id;
  encode_msg(root, msg);
}

Status ReadMoveBuffersOwnershipRequest(json const& root,
                                       std::map<ObjectID, ObjectID>& id_to_id,
                                       std::map<PlasmaID, ObjectID>& pid_to_id,
                                       std::map<ObjectID, PlasmaID>& id_to_pid,
                                       std::map<PlasmaID, PlasmaID>& pid_to_pid,
                                       SessionID& session_id) {
  CHECK_IPC_ERROR(root, command_t::MOVE_BUFFERS_OWNERSHIP_REQUEST);
  id_to_id = root.value<std::map<ObjectID, ObjectID>>("id_to_id", {});
  pid_to_id = root.value<std::map<PlasmaID, ObjectID>>("pid_to_id", {});
  id_to_pid = root.value<std::map<ObjectID, PlasmaID>>("id_to_pid", {});
  pid_to_pid = root.value<std::map<PlasmaID, PlasmaID>>("pid_to_pid", {});
  session_id = root["session_id"].get<SessionID>();
  return Status::OK();
}

void WriteMoveBuffersOwnershipReply(std::string& msg) {
  json root;
  root["type"] = command_t::MOVE_BUFFERS_OWNERSHIP_REPLY;
  encode_msg(root, msg);
}

Status ReadMoveBuffersOwnershipReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::MOVE_BUFFERS_OWNERSHIP_REPLY);
  return Status::OK();
}

void WriteEvictRequest(const std::vector<ObjectID>& ids, std::string& msg) {
  json root;
  root["type"] = command_t::EVICT_REQUEST;
  root["ids"] = ids;
  encode_msg(root, msg);
}

Status ReadEvictRequest(json const& root, std::vector<ObjectID>& ids) {
  CHECK_IPC_ERROR(root, command_t::EVICT_REQUEST);
  root["ids"].get_to(ids);
  return Status::OK();
}

void WriteEvictReply(std::string& msg) {
  json root;
  root["type"] = command_t::EVICT_REPLY;
  encode_msg(root, msg);
}

Status ReadEvictReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::EVICT_REPLY);
  return Status::OK();
}

void WriteLoadRequest(const std::vector<ObjectID>& ids, const bool pin,
                      std::string& msg) {
  json root;
  root["type"] = command_t::LOAD_REQUEST;
  root["ids"] = std::vector<ObjectID>{ids};
  root["pin"] = pin;
  encode_msg(root, msg);
}

Status ReadLoadRequest(json const& root, std::vector<ObjectID>& ids,
                       bool& pin) {
  CHECK_IPC_ERROR(root, command_t::LOAD_REQUEST);
  root["ids"].get_to(ids);
  pin = root.value("pin", false);
  return Status::OK();
}

void WriteLoadReply(std::string& msg) {
  json root;
  root["type"] = command_t::LOAD_REPLY;
  encode_msg(root, msg);
}

Status ReadLoadReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::LOAD_REPLY);
  return Status::OK();
}

void WriteUnpinRequest(const std::vector<ObjectID>& ids, std::string& msg) {
  json root;
  root["type"] = command_t::UNPIN_REQUEST;
  root["ids"] = ids;
  encode_msg(root, msg);
}

Status ReadUnpinRequest(json const& root, std::vector<ObjectID>& ids) {
  CHECK_IPC_ERROR(root, command_t::UNPIN_REQUEST);
  root["ids"].get_to(ids);
  return Status::OK();
}

void WriteUnpinReply(std::string& msg) {
  json root;
  root["type"] = command_t::UNPIN_REPLY;
  encode_msg(root, msg);
}

Status ReadUnpinReply(json const& root) {
  CHECK_IPC_ERROR(root, command_t::UNPIN_REPLY);
  return Status::OK();
}

void WriteIsSpilledRequest(const ObjectID& id, std::string& msg) {
  json root;
  root["type"] = command_t::IS_SPILLED_REQUEST;
  root["id"] = id;
  encode_msg(root, msg);
}

Status ReadIsSpilledRequest(json const& root, ObjectID& id) {
  CHECK_IPC_ERROR(root, command_t::IS_SPILLED_REQUEST);
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteIsSpilledReply(const bool is_spilled, std::string& msg) {
  json root;
  root["type"] = command_t::IS_SPILLED_REPLY;
  root["is_spilled"] = is_spilled;
  encode_msg(root, msg);
}

Status ReadIsSpilledReply(json const& root, bool& is_spilled) {
  CHECK_IPC_ERROR(root, command_t::IS_SPILLED_REPLY);
  is_spilled = root["is_spilled"].get<bool>();
  return Status::OK();
}

void WriteIsInUseRequest(const ObjectID& id, std::string& msg) {
  json root;
  root["type"] = command_t::IS_IN_USE_REQUEST;
  root["id"] = id;
  encode_msg(root, msg);
}

Status ReadIsInUseRequest(json const& root, ObjectID& id) {
  CHECK_IPC_ERROR(root, command_t::IS_IN_USE_REQUEST);
  id = root["id"].get<ObjectID>();
  return Status::OK();
}

void WriteIsInUseReply(const bool is_in_use, std::string& msg) {
  json root;
  root["type"] = command_t::IS_IN_USE_REPLY;
  root["is_in_use"] = is_in_use;
  encode_msg(root, msg);
}

Status ReadIsInUseReply(json const& root, bool& is_in_use) {
  CHECK_IPC_ERROR(root, command_t::IS_IN_USE_REPLY);
  is_in_use = root["is_in_use"].get<bool>();
  return Status::OK();
}

void WriteClusterMetaRequest(std::string& msg) {
  json root;
  root["type"] = command_t::CLUSTER_META_REQUEST;

  encode_msg(root, msg);
}

Status ReadClusterMetaRequest(const json& root) {
  CHECK_IPC_ERROR(root, command_t::CLUSTER_META_REQUEST);
  return Status::OK();
}

void WriteClusterMetaReply(const json& meta, std::string& msg) {
  json root;
  root["type"] = command_t::CLUSTER_META_REPLY;
  root["meta"] = meta;

  encode_msg(root, msg);
}

Status ReadClusterMetaReply(const json& root, json& meta) {
  CHECK_IPC_ERROR(root, command_t::CLUSTER_META_REPLY);
  meta = root["meta"];
  return Status::OK();
}

void WriteInstanceStatusRequest(std::string& msg) {
  json root;
  root["type"] = command_t::INSTANCE_STATUS_REQUEST;

  encode_msg(root, msg);
}

Status ReadInstanceStatusRequest(const json& root) {
  CHECK_IPC_ERROR(root, command_t::INSTANCE_STATUS_REQUEST);
  return Status::OK();
}

void WriteInstanceStatusReply(const json& meta, std::string& msg) {
  json root;
  root["type"] = command_t::INSTANCE_STATUS_REPLY;
  root["meta"] = meta;

  encode_msg(root, msg);
}

Status ReadInstanceStatusReply(const json& root, json& meta) {
  CHECK_IPC_ERROR(root, command_t::INSTANCE_STATUS_REPLY);
  meta = root["meta"];
  return Status::OK();
}

void WriteMigrateObjectRequest(const ObjectID object_id, std::string& msg) {
  json root;
  root["type"] = command_t::MIGRATE_OBJECT_REQUEST;
  root["object_id"] = object_id;

  encode_msg(root, msg);
}

Status ReadMigrateObjectRequest(const json& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, command_t::MIGRATE_OBJECT_REQUEST);
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteMigrateObjectRequest(const ObjectID object_id, const bool local,
                               const bool is_stream, const std::string& peer,
                               std::string const& peer_rpc_endpoint,
                               std::string& msg) {
  json root;
  root["type"] = command_t::MIGRATE_OBJECT_REQUEST;
  root["object_id"] = object_id;
  root["local"] = local;
  root["is_stream"] = is_stream;
  root["peer"] = peer;
  root["peer_rpc_endpoint"] = peer_rpc_endpoint,

  encode_msg(root, msg);
}

Status ReadMigrateObjectRequest(const json& root, ObjectID& object_id,
                                bool& local, bool& is_stream, std::string& peer,
                                std::string& peer_rpc_endpoint) {
  CHECK_IPC_ERROR(root, command_t::MIGRATE_OBJECT_REQUEST);
  object_id = root["object_id"].get<ObjectID>();
  local = root["local"].get<bool>();
  is_stream = root["is_stream"].get<bool>();
  peer = root["peer"].get_ref<std::string const&>();
  peer_rpc_endpoint = root["peer_rpc_endpoint"].get_ref<std::string const&>();
  return Status::OK();
}

void WriteMigrateObjectReply(const ObjectID& object_id, std::string& msg) {
  json root;
  root["type"] = command_t::MIGRATE_OBJECT_REPLY;
  root["object_id"] = object_id;

  encode_msg(root, msg);
}

Status ReadMigrateObjectReply(const json& root, ObjectID& object_id) {
  CHECK_IPC_ERROR(root, command_t::MIGRATE_OBJECT_REPLY);
  object_id = root["object_id"].get<ObjectID>();
  return Status::OK();
}

void WriteShallowCopyRequest(const ObjectID id, std::string& msg) {
  json root;
  root["type"] = command_t::SHALLOW_COPY_REQUEST;
  root["id"] = id;

  encode_msg(root, msg);
}

void WriteShallowCopyRequest(const ObjectID id, json const& extra_metadata,
                             std::string& msg) {
  json root;
  root["type"] = command_t::SHALLOW_COPY_REQUEST;
  root["id"] = id;
  root["extra"] = extra_metadata;

  encode_msg(root, msg);
}

Status ReadShallowCopyRequest(const json& root, ObjectID& id,
                              json& extra_metadata) {
  CHECK_IPC_ERROR(root, command_t::SHALLOW_COPY_REQUEST);
  id = root["id"].get<ObjectID>();
  extra_metadata = root.value("extra", json::object());
  return Status::OK();
}

void WriteShallowCopyReply(const ObjectID target_id, std::string& msg) {
  json root;
  root["type"] = command_t::SHALLOW_COPY_REPLY;
  root["target_id"] = target_id;

  encode_msg(root, msg);
}

Status ReadShallowCopyReply(const json& root, ObjectID& target_id) {
  CHECK_IPC_ERROR(root, command_t::SHALLOW_COPY_REPLY);
  target_id = root["target_id"].get<ObjectID>();
  return Status::OK();
}

void WriteDebugRequest(const json& debug, std::string& msg) {
  json root;
  root["type"] = command_t::DEBUG_REQUEST;
  root["debug"] = debug;
  encode_msg(root, msg);
}

Status ReadDebugRequest(const json& root, json& debug) {
  CHECK_IPC_ERROR(root, command_t::DEBUG_REQUEST);
  debug = root["debug"];
  return Status::OK();
}

void WriteDebugReply(const json& result, std::string& msg) {
  json root;
  root["type"] = "debug_reply";
  root["result"] = result;
  encode_msg(root, msg);
}

Status ReadDebugReply(const json& root, json& result) {
  CHECK_IPC_ERROR(root, "debug_reply");
  result = root["result"];
  return Status::OK();
}

void WriteTryAcquireLockRequest(const std::string& key, std::string& msg) {
  json root;
  root["type"] = command_t::ACQUIRE_LOCK_REQUEST;
  root["key"] = key;
  encode_msg(root, msg);
}

Status ReadTryAcquireLockRequest(const json& root, std::string& key) {
  CHECK_IPC_ERROR(root, command_t::ACQUIRE_LOCK_REQUEST);
  key = root["key"].get<std::string>();
  return Status::OK();
}

void WriteTryAcquireLockReply(const bool result, const std::string actual_key,
                              std::string& msg) {
  json root;
  root["type"] = command_t::ACQUIRE_LOCK_REPLY;
  root["key"] = actual_key;
  root["result"] = result;
  encode_msg(root, msg);
}

Status ReadTryAcquireLockReply(const json& root, bool& result,
                               std::string& key) {
  CHECK_IPC_ERROR(root, command_t::ACQUIRE_LOCK_REPLY);
  result = root["result"].get<bool>();
  key = root["key"].get<std::string>();
  return Status::OK();
}

void WriteTryReleaseLockRequest(const std::string& key, std::string& msg) {
  json root;
  root["type"] = command_t::RELEASE_LOCK_REQUEST;
  root["key"] = key;
  encode_msg(root, msg);
}

Status ReadTryReleaseLockRequest(const json& root, std::string& key) {
  CHECK_IPC_ERROR(root, command_t::RELEASE_LOCK_REQUEST);
  key = root["key"].get<std::string>();
  return Status::OK();
}

void WriteTryReleaseLockReply(const bool result, std::string& msg) {
  json root;
  root["type"] = command_t::RELEASE_LOCK_REPLY;
  root["result"] = result;
  encode_msg(root, msg);
}

Status ReadTryReleaseLockReply(const json& root, bool& result) {
  CHECK_IPC_ERROR(root, command_t::RELEASE_LOCK_REPLY);
  result = root["result"].get<bool>();
  return Status::OK();
}

}  // namespace vineyard
