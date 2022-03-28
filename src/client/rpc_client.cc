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

#include "client/rpc_client.h"

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "client/ds/blob.h"
#include "client/ds/object_factory.h"
#include "client/io.h"
#include "client/utils.h"
#include "common/util/boost.h"
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
  WriteRegisterRequest(message_out, "Any");
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
  VINEYARD_CHECK_OK(this->GetMetaData(id, meta, true));
  VINEYARD_ASSERT(!meta.MetaData().empty());
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
  std::vector<ObjectMeta> metas;
  VINEYARD_CHECK_OK(this->GetMetaData(ids, metas, true));
  for (auto const& meta : metas) {
    VINEYARD_ASSERT(!meta.MetaData().empty());
  }
  std::vector<std::shared_ptr<Object>> objects;
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

}  // namespace vineyard
