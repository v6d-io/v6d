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

#include <sys/stat.h>
#include <memory>
#include <sstream>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/env.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_docs.h"   // NOLINT(build/include_subdir)
#include "pybind11_utils.h"  // NOLINT(build/include_subdir)

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces_literals)

namespace vineyard {

/**
 * Note [Client Manager]
 *
 * The client manager is designed for hold a reference of client to avoid it
 * being release when python object being deleted.
 *
 * When user wants to close a client, the user must invoke the close method
 * on the client, otherwise the client will still live, even when the variable
 * gets out of the scope.
 *
 * When user connect to a vineyard_socket and if there are existing client
 * instance that connects to the same socket file, it will reuse the client,
 * since the vineyard client itself is thread-safe.
 *
 * The aim is to make python API more pythonic, and make the object lifecycle
 * control easier.
 */
template <typename ClientType>
class ClientManager {
 public:
  ClientManager() {}

  static std::shared_ptr<ClientManager<ClientType>> GetManager() {
    static std::shared_ptr<ClientManager<ClientType>> manager =
        std::make_shared<ClientManager<ClientType>>();
    return manager;
  }

  std::shared_ptr<ClientType> Connect(const std::string& username,
                                      const std::string& password) {
    return Connect("", RootSessionID(), username, password);
  }

  std::shared_ptr<ClientType> Connect(std::string const& endpoint,
                                      const std::string& username,
                                      const std::string& password) {
    return Connect(endpoint, RootSessionID(), username, password);
  }

  std::shared_ptr<ClientType> Connect(std::string const& endpoint,
                                      const SessionID session_id,
                                      const std::string& username,
                                      const std::string& password) {
    std::lock_guard<std::mutex> guard{mtx_};
    std::string endpoint_key = endpoint + ":" + SessionIDToString(session_id);
    auto iter = client_set_.find(endpoint_key);
    if (iter != client_set_.end()) {
      if (iter->second->Connected()) {
        return iter->second;
      }
    }
    std::shared_ptr<ClientType> client = std::make_shared<ClientType>();
    auto connect_status =
        this->ConnectImpl(client, endpoint, session_id, username, password);
    if (PyErr_CheckSignals() != 0) {
      // The method `Connect` will keep retrying, we need to propagate
      // the Ctrl-C when during the C++ code run retries.
      throw py::error_already_set();
    }
    // propagate the KeyboardInterrupt exception correctly before the
    // RuntimeError
    throw_on_error(connect_status);
    client_set_.emplace(endpoint_key, client);
    return client;
  }

  void Disconnect(std::string const& endpoint,
                  const SessionID session_id = RootSessionID()) {
    std::lock_guard<std::mutex> guard{mtx_};
    std::string endpoint_key = endpoint + ":" + SessionIDToString(session_id);
    auto iter = client_set_.find(endpoint_key);
    if (iter != client_set_.end() && iter->second.use_count() == 2) {
      // the python variable is the last reference, another reference
      // lies in the `client_set_`.
      iter->second->Disconnect();
      client_set_.erase(endpoint_key);
    }
  }

 private:
  // See also "Notes" in https://en.cppreference.com/w/cpp/types/enable_if

  template <
      typename ClientPtrType,
      typename std::enable_if<std::is_same<
          ClientPtrType, std::shared_ptr<Client>>::value>::type* = nullptr>
  Status ConnectImpl(ClientPtrType& client, std::string const& endpoint = "",
                     const SessionID session_id = RootSessionID(),
                     const std::string& username = "",
                     const std::string& password = "") {
    return endpoint.empty() ? client->Connect(username, password)
                            : client->Connect(endpoint, username, password);
  }

  template <
      typename ClientPtrType,
      typename std::enable_if<std::is_same<
          ClientPtrType, std::shared_ptr<RPCClient>>::value>::type* = nullptr>
  Status ConnectImpl(ClientPtrType& client, std::string const& endpoint = "",
                     const SessionID session_id = RootSessionID(),
                     const std::string& username = "",
                     const std::string& password = "") {
    return endpoint.empty()
               ? client->Connect(username, password)
               : client->Connect(endpoint, session_id, username, password);
  }

  std::mutex mtx_;
  std::unordered_map<std::string, std::shared_ptr<ClientType>> client_set_;
};

void bind_client(py::module& mod) {
  // InstanceStatus
  py::class_<InstanceStatus, std::shared_ptr<InstanceStatus>>(
      mod, "InstanceStatus", doc::InstanceStatus)
      .def_property_readonly(
          "instance_id",
          [](InstanceStatus* status) { return status->instance_id; },
          doc::InstanceStatus_instance_id)
      .def_property_readonly(
          "deployment",
          [](InstanceStatus* status) { return status->deployment; },
          doc::InstanceStatus_deployment)
      .def_property_readonly(
          "memory_usage",
          [](InstanceStatus* status) { return status->memory_usage; },
          doc::InstanceStatus_memory_usage)
      .def_property_readonly(
          "memory_limit",
          [](InstanceStatus* status) { return status->memory_limit; },
          doc::InstanceStatus_memory_limit)
      .def_property_readonly(
          "deferred_requests",
          [](InstanceStatus* status) { return status->deferred_requests; },
          doc::InstanceStatus_deferred_requests)
      .def_property_readonly(
          "ipc_connections",
          [](InstanceStatus* status) { return status->ipc_connections; },
          doc::InstanceStatus_ipc_connections)
      .def_property_readonly(
          "rpc_connections",
          [](InstanceStatus* status) { return status->rpc_connections; },
          doc::InstanceStatus_rpc_connections)
      .def("__repr__",
           [](InstanceStatus* status) {
             std::stringstream ss;
             ss << "{" << std::endl;
             ss << "    instance_id: " << status->instance_id << ","
                << std::endl;
             ss << "    deployment: " << status->deployment << "," << std::endl;
             ss << "    memory_usage: " << status->memory_usage << ","
                << std::endl;
             ss << "    memory_limit: " << status->memory_limit << ","
                << std::endl;
             ss << "    deferred_requests: " << status->deferred_requests << ","
                << std::endl;
             ss << "    ipc_connections: " << status->ipc_connections << ","
                << std::endl;
             ss << "    rpc_connections: " << status->rpc_connections
                << std::endl;
             ss << "}";
             return ss.str();
           })
      .def("__str__", [](InstanceStatus* status) {
        std::stringstream ss;
        ss << "InstanceStatus:" << std::endl;
        ss << "    instance_id: " << status->instance_id << std::endl;
        ss << "    deployment: " << status->deployment << std::endl;
        ss << "    memory_usage: " << status->memory_usage << std::endl;
        ss << "    memory_limit: " << status->memory_limit << std::endl;
        ss << "    deferred_requests: " << status->deferred_requests
           << std::endl;
        ss << "    ipc_connections: " << status->ipc_connections << std::endl;
        ss << "    rpc_connections: " << status->rpc_connections;
        return ss.str();
      });

  // ClientBase
  py::class_<ClientBase, std::shared_ptr<ClientBase>>(mod, "ClientBase",
                                                      doc::ClientBase)
      .def(
          "create_metadata",
          [](ClientBase* self, ObjectMeta& metadata) -> ObjectMeta& {
            ObjectID object_id;
            throw_on_error(self->CreateMetaData(metadata, object_id));
            return metadata;
          },
          "metadata"_a, doc::ClientBase_create_metadata)
      .def(
          "create_metadata",
          [](ClientBase* self,
             std::vector<ObjectMeta>& metadatas) -> std::vector<ObjectMeta>& {
            std::vector<ObjectID> object_ids;
            throw_on_error(self->CreateMetaData(metadatas, object_ids));
            return metadatas;
          },
          "metadata"_a, doc::ClientBase_create_metadata)
      .def(
          "create_metadata",
          [](ClientBase* self, ObjectMeta& metadata,
             InstanceID const& instance_id) -> ObjectMeta& {
            ObjectID object_id;
            throw_on_error(
                self->CreateMetaData(metadata, instance_id, object_id));
            return metadata;
          },
          "metadata"_a, "instance_id"_a)
      .def(
          "create_metadata",
          [](ClientBase* self, std::vector<ObjectMeta>& metadatas,
             InstanceID const& instance_id) -> std::vector<ObjectMeta>& {
            std::vector<ObjectID> object_ids;
            throw_on_error(
                self->CreateMetaData(metadatas, instance_id, object_ids));
            return metadatas;
          },
          "metadata"_a, "instance_id"_a)
      .def(
          "delete",
          [](ClientBase* self, const ObjectIDWrapper object_id,
             const bool force, const bool deep, const bool memory_trim) {
            throw_on_error(self->DelData(object_id, force, deep, memory_trim));
          },
          "object_id"_a, py::arg("force") = false, py::arg("deep") = true,
          py::arg("memory_trim") = false, doc::ClientBase_delete)
      .def(
          "delete",
          [](ClientBase* self, const std::vector<ObjectIDWrapper>& object_ids,
             const bool force, const bool deep, const bool memory_trim) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            throw_on_error(
                self->DelData(unwrapped_object_ids, force, deep, memory_trim));
          },
          "object_ids"_a, py::arg("force") = false, py::arg("deep") = true,
          py::arg("memory_trim") = false, doc::ClientBase_delete)
      .def(
          "delete",
          [](ClientBase* self, const ObjectMeta& meta, const bool force,
             const bool deep, const bool memory_trim) {
            throw_on_error(
                self->DelData(meta.GetId(), force, deep, memory_trim));
          },
          "object_meta"_a, py::arg("force") = false, py::arg("deep") = true,
          py::arg("memory_trim") = false, doc::ClientBase_delete)
      .def(
          "delete",
          [](ClientBase* self, const Object* object, const bool force,
             const bool deep, const bool memory_trim) {
            throw_on_error(
                self->DelData(object->id(), force, deep, memory_trim));
          },
          "object"_a, py::arg("force") = false, py::arg("deep") = true,
          py::arg("memory_trim") = false, doc::ClientBase_delete)
      .def(
          "create_stream",
          [](ClientBase* self, ObjectID const id) {
            throw_on_error(self->CreateStream(id));
          },
          "stream"_a)
      .def(
          "open_stream",
          [](ClientBase* self, ObjectID const id, std::string const& mode) {
            if (mode == "r") {
              throw_on_error(self->OpenStream(id, StreamOpenMode::read));
            } else if (mode == "w") {
              throw_on_error(self->OpenStream(id, StreamOpenMode::write));
            } else {
              throw_on_error(
                  Status::AssertionFailed("Mode can only be 'r' or 'w'"));
            }
          },
          "stream"_a, "mode"_a)
      .def(
          "push_chunk",
          [](ClientBase* self, ObjectID const stream_id, ObjectID const chunk) {
            throw_on_error(self->PushNextStreamChunk(stream_id, chunk));
          },
          "stream"_a, "chunk"_a, py::call_guard<py::gil_scoped_release>())
      .def(
          "next_chunk_id",
          [](ClientBase* self, ObjectID const stream_id) -> ObjectID {
            ObjectID id;
            throw_on_error(self->PullNextStreamChunk(stream_id, id));
            return id;
          },
          "stream"_a, py::call_guard<py::gil_scoped_release>())
      .def(
          "next_chunk_meta",
          [](ClientBase* self, ObjectID const stream_id) -> ObjectMeta {
            ObjectMeta meta;
            throw_on_error(self->PullNextStreamChunk(stream_id, meta));
            return meta;
          },
          "stream"_a, py::call_guard<py::gil_scoped_release>())
      .def(
          "next_chunk",
          [](ClientBase* self,
             ObjectID const stream_id) -> std::shared_ptr<Object> {
            std::shared_ptr<Object> object;
            throw_on_error(self->PullNextStreamChunk(stream_id, object));
            return object;
          },
          "stream"_a, py::call_guard<py::gil_scoped_release>())
      .def(
          "stop_stream",
          [](ClientBase* self, ObjectID const stream_id, bool failed) {
            throw_on_error(self->StopStream(stream_id, failed));
          },
          "stream"_a, "failed"_a)
      .def(
          "drop_stream",
          [](ClientBase* self, ObjectID const stream_id,
             const bool drop_metadata = true) {
            throw_on_error(self->DropStream(stream_id));
            if (drop_metadata) {
              VINEYARD_SUPPRESS(self->DelData(stream_id));
            }
          },
          "stream"_a, py::arg("drop_metadata") = true)
      .def(
          "persist",
          [](ClientBase* self, const ObjectIDWrapper object_id) {
            throw_on_error(self->Persist(object_id));
          },
          "object_id"_a, doc::ClientBase_persist)
      .def(
          "persist",
          [](ClientBase* self, const ObjectMeta& meta) {
            throw_on_error(self->Persist(meta.GetId()));
          },
          "object_meta"_a)
      .def(
          "persist",
          [](ClientBase* self, const Object* object) {
            throw_on_error(self->Persist(object->id()));
          },
          "object"_a)
      .def(
          "exists",
          [](ClientBase* self, const ObjectIDWrapper object_id) -> bool {
            bool exists;
            throw_on_error(self->Exists(object_id, exists));
            return exists;
          },
          "object_id"_a, doc::ClientBase_exists)
      .def(
          "shallow_copy",
          [](ClientBase* self,
             const ObjectIDWrapper object_id) -> ObjectIDWrapper {
            ObjectID target_id;
            throw_on_error(self->ShallowCopy(object_id, target_id));
            return target_id;
          },
          "object_id"_a, doc::ClientBase_shallow_copy)
      .def(
          "shallow_copy",
          [](ClientBase* self, const ObjectIDWrapper object_id,
             py::dict extra_metadata) -> ObjectIDWrapper {
            ObjectID target_id;
            json meta = detail::to_json(extra_metadata);
            if (meta == json(nullptr)) {
              throw_on_error(self->ShallowCopy(object_id, target_id));
            } else {
              throw_on_error(self->ShallowCopy(object_id, meta, target_id));
            }
            return target_id;
          },
          "object_id"_a, "extra_metadata"_a)
      .def(
          "list_names",
          [](ClientBase* self, std::string const& pattern, const bool regex,
             const size_t limit) -> std::map<std::string, ObjectIDWrapper> {
            std::map<std::string, ObjectID> names;
            std::map<std::string, ObjectIDWrapper> transformed_names;
            throw_on_error(self->ListNames(pattern, regex, limit, names));
            for (auto const& name : names) {
              transformed_names[name.first] = ObjectIDWrapper(name.second);
            }
            return transformed_names;
          },
          py::arg("pattern"), py::arg("regex") = false, py::arg("limit") = 5,
          doc::ClientBase_list_names)
      .def(
          "put_name",
          [](ClientBase* self, const ObjectIDWrapper object_id,
             std::string const& name) {
            throw_on_error(self->PutName(object_id, name));
          },
          "object_id"_a, "name"_a, doc::ClientBase_put_name)
      .def(
          "put_name",
          [](ClientBase* self, const ObjectIDWrapper object_id,
             ObjectNameWrapper const& name) {
            throw_on_error(self->PutName(object_id, name));
          },
          "object_id"_a, "name"_a)
      .def(
          "put_name",
          [](ClientBase* self, const ObjectMeta& meta,
             std::string const& name) {
            throw_on_error(self->PutName(meta.GetId(), name));
          },
          "object_meta"_a, "name"_a)
      .def(
          "put_name",
          [](ClientBase* self, const ObjectMeta& meta,
             ObjectNameWrapper const& name) {
            throw_on_error(self->PutName(meta.GetId(), name));
          },
          "object_meta"_a, "name"_a)
      .def(
          "put_name",
          [](ClientBase* self, const Object* object, std::string const& name) {
            throw_on_error(self->PutName(object->id(), name));
          },
          "object"_a, "name"_a)
      .def(
          "put_name",
          [](ClientBase* self, const Object* object,
             ObjectNameWrapper const& name) {
            throw_on_error(self->PutName(object->id(), name));
          },
          "object"_a, "name"_a)
      .def(
          "get_name",
          [](ClientBase* self, std::string const& name,
             const bool wait) -> ObjectIDWrapper {
            ObjectID object_id;
            throw_on_error(self->GetName(name, object_id, wait));
            return object_id;
          },
          "object_id"_a, py::arg("wait") = false,
          py::call_guard<py::gil_scoped_release>(), doc::ClientBase_get_name)
      .def(
          "get_name",
          [](ClientBase* self, ObjectNameWrapper const& name,
             const bool wait) -> ObjectIDWrapper {
            ObjectID object_id;
            throw_on_error(self->GetName(name, object_id, wait));
            return object_id;
          },
          "object_id"_a, py::arg("wait") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "drop_name",
          [](ClientBase* self, std::string const& name) {
            throw_on_error(self->DropName(name));
          },
          "name"_a, doc::ClientBase_drop_name)
      .def(
          "drop_name",
          [](ClientBase* self, ObjectNameWrapper const& name) {
            throw_on_error(self->DropName(name));
          },
          "name"_a)
      .def(
          "sync_meta",
          [](ClientBase* self) -> void {
            VINEYARD_DISCARD(self->SyncMetaData());
          },
          doc::ClientBase_sync_meta)
      .def(
          "migrate",
          [](ClientBase* self, const ObjectID object_id) -> ObjectIDWrapper {
            ObjectID target_id = InvalidObjectID();
            throw_on_error(self->MigrateObject(object_id, target_id));
            return target_id;
          },
          "object_id"_a)
      .def(
          "clear", [](ClientBase* self) { throw_on_error(self->Clear()); },
          doc::ClientBase_clear)
      .def(
          "memory_trim",
          [](ClientBase* self) -> bool {
            bool trimmed = false;
            throw_on_error(self->MemoryTrim(trimmed));
            return trimmed;
          },
          doc::ClientBase_memory_trim)
      .def(
          "label",
          [](ClientBase* self, ObjectID id, std::string const& key,
             std::string const& value) -> void {
            throw_on_error(self->Label(id, key, value));
          },
          "object"_a, "key"_a, "value"_a)
      .def(
          "label",
          [](ClientBase* self, ObjectID id,
             std::map<std::string, std::string> const& labels) -> void {
            throw_on_error(self->Label(id, labels));
          },
          "object"_a, "labels"_a)
      .def(
          "evict",
          [](ClientBase* self, std::vector<ObjectID> const& objects) -> void {
            throw_on_error(self->Evict(objects));
          },
          "objects"_a)
      .def(
          "load",
          [](ClientBase* self, std::vector<ObjectID> const& objects,
             const bool pin) -> void {
            throw_on_error(self->Load(objects, pin));
          },
          "objects"_a, py::arg("pin") = false)
      .def(
          "unpin",
          [](ClientBase* self, std::vector<ObjectID> const& objects) -> void {
            throw_on_error(self->Unpin(objects));
          },
          "objects"_a)
      .def(
          "reset", [](ClientBase* self) { throw_on_error(self->Clear()); },
          doc::ClientBase_reset)
      .def_property_readonly("connected", &Client::Connected,
                             doc::ClientBase_connected)
      .def_property_readonly("instance_id", &Client::instance_id,
                             doc::ClientBase_instance_id)
      .def_property_readonly(
          "meta",
          [](ClientBase* self)
              -> std::map<uint64_t,
                          std::unordered_map<std::string, py::object>> {
            std::map<uint64_t, json> meta;
            throw_on_error(self->ClusterInfo(meta));
            std::map<uint64_t, std::unordered_map<std::string, py::object>>
                meta_to_return;
            for (auto const& kv : meta) {
              std::unordered_map<std::string, py::object> element;
              if (!kv.second.empty()) {
                for (auto const& elem : kv.second.items()) {
                  element[elem.key()] = detail::from_json(elem.value());
                }
              }
              meta_to_return.emplace(kv.first, std::move(element));
            }
            return meta_to_return;
          },
          doc::ClientBase_meta)
      .def_property_readonly(
          "status",
          [](ClientBase* self) -> std::shared_ptr<InstanceStatus> {
            std::shared_ptr<InstanceStatus> status;
            throw_on_error(self->InstanceStatus(status));
            return status;
          },
          doc::ClientBase_status)
      .def("debug",
           [](ClientBase* self, py::dict debug) {
             json result;
             throw_on_error(self->Debug(detail::to_json(debug), result));
             return detail::from_json(result);
           })
      .def_property("compression", &ClientBase::compression_enabled,
                    &ClientBase::set_compression_enabled)
      .def_property_readonly("ipc_socket", &ClientBase::IPCSocket,
                             doc::ClientBase_ipc_socket)
      .def_property_readonly("rpc_endpoint", &ClientBase::RPCEndpoint,
                             doc::ClientBase_rpc_endpoint)
      .def_property_readonly("version", &ClientBase::Version,
                             doc::ClientBase_version)
      .def_property_readonly("is_ipc", &ClientBase::IsIPC,
                             doc::ClientBase_is_ipc)
      .def_property_readonly("is_rpc", &ClientBase::IsRPC,
                             doc::ClientBase_is_rpc);

  // IPCClient
  py::class_<Client, std::shared_ptr<Client>, ClientBase>(mod, "IPCClient",
                                                          doc::IPCClient)
      .def(
          "create_blob",
          [](Client* self, size_t size) {
            std::unique_ptr<BlobWriter> blob;
            throw_on_error(self->CreateBlob(size, blob));
            return std::shared_ptr<BlobWriter>(blob.release());
          },
          py::return_value_policy::move, "size"_a, doc::IPCClient_create_blob)
      .def(
          "create_blob",
          [](Client* self, std::vector<size_t> const& sizes) {
            std::vector<std::unique_ptr<BlobWriter>> blobs;
            throw_on_error(self->CreateBlobs(sizes, blobs));
            std::vector<std::shared_ptr<BlobWriter>> lived_blobs;
            for (auto& blob : blobs) {
              lived_blobs.emplace_back(blob.release());
            }
            return lived_blobs;
          },
          py::return_value_policy::move, "size"_a, doc::IPCClient_create_blob)
      .def(
          "create_empty_blob",
          [](Client* self) -> std::shared_ptr<Blob> {
            return Blob::MakeEmpty(*self);
          },
          doc::IPCClient_create_empty_blob)
      .def(
          "get_blob",
          [](Client* self, const ObjectIDWrapper object_id, const bool unsafe) {
            std::shared_ptr<Blob> blob;
            throw_on_error(self->GetBlob(object_id, unsafe, blob));
            return blob;
          },
          "object_id"_a, py::arg("unsafe") = false, doc::IPCClient_get_blob)
      .def(
          "get_blobs",
          [](Client* self, std::vector<ObjectIDWrapper> object_ids,
             const bool unsafe) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            std::vector<std::shared_ptr<Blob>> blobs;
            throw_on_error(self->GetBlobs(unwrapped_object_ids, unsafe, blobs));
            return blobs;
          },
          "object_ids"_a, py::arg("unsafe") = false, doc::IPCClient_get_blobs)
      .def(
          "get_object",
          [](Client* self, const ObjectIDWrapper object_id, bool const fetch) {
            // receive the status to throw a more precise exception when failed.
            std::shared_ptr<Object> object;
            if (fetch) {
              throw_on_error(self->FetchAndGetObject(object_id, object));
            } else {
              throw_on_error(self->GetObject(object_id, object));
            }
            return object;
          },
          "object_id"_a, py::arg("fetch") = false, doc::IPCClient_get_object)
      .def(
          "get_objects",
          [](Client* self, const std::vector<ObjectIDWrapper>& object_ids) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            return self->GetObjects(unwrapped_object_ids);
          },
          "object_ids"_a, doc::IPCClient_get_objects)
      .def(
          "get_meta",
          [](Client* self, ObjectIDWrapper const& object_id,
             bool const sync_remote) -> ObjectMeta {
            ObjectMeta meta;
            throw_on_error(self->GetMetaData(object_id, meta, sync_remote));
            return meta;
          },
          "object_id"_a, py::arg("sync_remote") = false,
          doc::IPCClient_get_meta)
      .def(
          "get_metas",
          [](Client* self, std::vector<ObjectIDWrapper> const& object_ids,
             bool const sync_remote) -> std::vector<ObjectMeta> {
            std::vector<ObjectMeta> metas;
            // FIXME: do we really not need to sync from etcd? We assume the
            // object is a local object
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            throw_on_error(
                self->GetMetaData(unwrapped_object_ids, metas, sync_remote));
            return metas;
          },
          "object_ids"_a, py::arg("sync_remote") = false,
          doc::IPCClient_get_metas)
      .def("list_objects", &Client::ListObjects, "pattern"_a,
           py::arg("regex") = false, py::arg("limit") = 5,
           doc::IPCClient_list_objects)
      .def("list_metadatas", &Client::ListObjectMeta, "pattern"_a,
           py::arg("regex") = false, py::arg("limit") = 5,
           py::arg("nobuffer") = false, doc::IPCClient_list_metadatas)
      .def(
          "new_buffer_chunk",
          [](Client* self, ObjectID const stream_id,
             size_t const size) -> py::memoryview {
            std::unique_ptr<MutableBuffer> buffer;
            throw_on_error(self->GetNextStreamChunk(stream_id, size, buffer));
            if (buffer == nullptr) {
              return py::none();
            } else {
              return py::memoryview::from_memory(buffer->mutable_data(),
                                                 buffer->size(), false);
            }
          },
          "stream"_a, "size"_a)
      .def(
          "next_buffer_chunk",
          [](Client* self, ObjectID const stream_id) -> py::memoryview {
            std::unique_ptr<Buffer> buffer;
            throw_on_error(self->PullNextStreamChunk(stream_id, buffer));
            if (buffer == nullptr) {
              return py::none();
            } else {
              return py::memoryview::from_memory(
                  const_cast<uint8_t*>(buffer->data()), buffer->size(), true);
            }
          },
          "stream"_a)
      .def(
          "allocated_size",
          [](Client* self, const ObjectID id) -> size_t {
            size_t size = 0;
            throw_on_error(self->AllocatedSize(id, size));
            return size;
          },
          "target"_a, doc::IPCClient_allocated_size)
      .def(
          "allocated_size",
          [](Client* self, const Object* target) -> size_t {
            size_t size = 0;
            if (target) {
              throw_on_error(self->AllocatedSize(target->id(), size));
            }
            return size;
          },
          "target"_a)
      .def(
          "is_shared_memory",
          [](Client* self, const uintptr_t target) -> bool {
            ObjectID object_id = InvalidObjectID();
            return self->IsSharedMemory(target, object_id);
          },
          doc::IPCClient_is_shared_memory)
      .def(
          "is_shared_memory",
          [](Client* self, py::buffer const& target) -> bool {
            ObjectID object_id = InvalidObjectID();
            return self->IsSharedMemory(target.ptr(), object_id);
          },
          doc::IPCClient_is_shared_memory)
      .def(
          "find_shared_memory",
          [](Client* self, const uintptr_t target) -> py::object {
            ObjectID object_id = InvalidObjectID();
            if (self->IsSharedMemory(target, object_id)) {
              return py::cast(ObjectIDWrapper(object_id));
            } else {
              return py::none();
            }
          },
          doc::IPCClient_find_shared_memory)
      .def(
          "find_shared_memory",
          [](Client* self, py::buffer const& target) -> py::object {
            ObjectID object_id = InvalidObjectID();
            if (self->IsSharedMemory(target.ptr(), object_id)) {
              return py::cast(ObjectIDWrapper(object_id));
            } else {
              return py::none();
            }
          },
          doc::IPCClient_find_shared_memory)
      .def(
          "close",
          [](Client* self) {
            return ClientManager<Client>::GetManager()->Disconnect(
                self->IPCSocket());
          },
          doc::IPCClient_close)
      .def("fork",
           [](Client* self) {
             std::shared_ptr<Client> client(new Client());
             throw_on_error(self->Fork(*client));
             return client;
           })
      .def("__enter__", [](Client* self) { return self; })
      .def("__exit__", [](Client* self, py::object, py::object, py::object) {
        // DO NOTHING
      });

  // RPCClient
  py::class_<RPCClient, std::shared_ptr<RPCClient>, ClientBase>(
      mod, "RPCClient", doc::RPCClient)
      .def(
          "create_remote_blob",
          [](RPCClient* self,
             const std::shared_ptr<RemoteBlobWriter>& remote_blob_builder)
              -> ObjectMeta {
            ObjectMeta blob_meta;
            throw_on_error(
                self->CreateRemoteBlob(remote_blob_builder, blob_meta));
            return blob_meta;
          },
          "remote_blob_builder"_a, doc::RPCClient_create_remote_blob)
      .def(
          "create_remote_blob",
          [](RPCClient* self,
             const std::vector<std::shared_ptr<RemoteBlobWriter>>&
                 remote_blob_builders) -> std::vector<ObjectMeta> {
            std::vector<ObjectMeta> blob_metas;
            throw_on_error(
                self->CreateRemoteBlobs(remote_blob_builders, blob_metas));
            return blob_metas;
          },
          "remote_blob_builder"_a, doc::RPCClient_create_remote_blob)
      .def(
          "get_remote_blob",
          [](RPCClient* self, const ObjectIDWrapper object_id,
             const bool unsafe) {
            std::shared_ptr<RemoteBlob> remote_blob;
            throw_on_error(self->GetRemoteBlob(object_id, unsafe, remote_blob));
            return remote_blob;
          },
          "object_id"_a, py::arg("unsafe") = false,
          doc::RPCClient_get_remote_blob)
      .def(
          "get_remote_blobs",
          [](RPCClient* self, std::vector<ObjectIDWrapper> object_ids,
             const bool unsafe) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            std::vector<std::shared_ptr<RemoteBlob>> remote_blobs;
            throw_on_error(self->GetRemoteBlobs(unwrapped_object_ids, unsafe,
                                                remote_blobs));
            return remote_blobs;
          },
          "object_ids"_a, py::arg("unsafe") = false,
          doc::RPCClient_get_remote_blobs)
      .def(
          "get_object",
          [](RPCClient* self, const ObjectIDWrapper object_id) {
            // receive the status to throw a more precise exception when failed.
            std::shared_ptr<Object> object;
            throw_on_error(self->GetObject(object_id, object));
            return object;
          },
          "object_id"_a, doc::RPCClient_get_object)
      .def(
          "get_objects",
          [](RPCClient* self, std::vector<ObjectIDWrapper> const& object_ids) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            return self->GetObjects(unwrapped_object_ids);
          },
          "object_ids"_a, doc::RPCClient_get_objects)
      .def(
          "get_meta",
          [](RPCClient* self, ObjectIDWrapper const& object_id,
             bool const sync_remote) -> ObjectMeta {
            ObjectMeta meta;
            throw_on_error(self->GetMetaData(object_id, meta, sync_remote));
            return meta;
          },
          "object_id"_a,
          py::arg("sync_remote") =
              true /* rpc client will sync remote meta by default */,
          doc::RPCClient_get_meta)
      .def(
          "get_metas",
          [](RPCClient* self, std::vector<ObjectIDWrapper> const& object_ids,
             bool const sync_remote) -> std::vector<ObjectMeta> {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            std::vector<ObjectMeta> metas;
            throw_on_error(
                self->GetMetaData(unwrapped_object_ids, metas, true));
            return metas;
          },
          "object_ids"_a,
          py::arg("sync_remote") =
              true /* rpc client will sync remote meta by default */,
          doc::RPCClient_get_metas)
      .def("list_objects", &RPCClient::ListObjects, "pattern"_a,
           py::arg("regex") = false, py::arg("limit") = 5,
           doc::RPCClient_list_objects)
      .def("list_metadatas", &RPCClient::ListObjectMeta, "pattern"_a,
           py::arg("regex") = false, py::arg("limit") = 5,
           py::arg("nobuffer") = false, doc::RPCClient_list_metadatas)
      .def(
          "close",
          [](RPCClient* self) {
            return ClientManager<RPCClient>::GetManager()->Disconnect(
                self->RPCEndpoint(), self->session_id());
          },
          doc::RPCClient_close)
      .def("fork",
           [](RPCClient* self) {
             std::shared_ptr<RPCClient> rpc_client(new RPCClient());
             throw_on_error(self->Fork(*rpc_client));
             return rpc_client;
           })
      .def(
          "is_fetchable",
          [](RPCClient* self, ObjectMeta& metadata) -> bool {
            return self->IsFetchable(metadata);
          },
          doc::RPCClient_is_fetchable)
      .def_property_readonly("remote_instance_id",
                             &RPCClient::remote_instance_id,
                             doc::RPCClient_remote_instance_id)
      .def("__enter__", [](RPCClient* self) { return self; })
      .def("__exit__", [](RPCClient* self, py::object, py::object, py::object) {
        // DO NOTHING
      });

  mod.def(
         "_connect",
         [](std::string const& socket, const SessionID session_id,
            const std::string& username,
            const std::string& password) -> py::object {
           return py::cast(ClientManager<Client>::GetManager()->Connect(
               socket, session_id, username, password));
         },
         "socket"_a, py::kw_only(), py::arg("session") = RootSessionID(),
         py::arg("username") = "", py::arg("password") = "")
      .def(
          "_connect",
          [](std::string const& host, const uint32_t port,
             const SessionID session_id, const std::string& username,
             const std::string& password) {
            std::string rpc_endpoint = host + ":" + std::to_string(port);
            return py::cast(ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint, session_id, username, password));
          },
          "host"_a, "port"_a, py::kw_only(),
          py::arg("session") = RootSessionID(), py::arg("username") = "",
          py::arg("password") = "")
      .def(
          "_connect",
          [](std::string const& host, std::string const& port,
             const SessionID session_id, const std::string& username,
             const std::string& password) {
            std::string rpc_endpoint = host + ":" + port;
            return ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint, session_id, username, password);
          },
          "host"_a, "port"_a, py::kw_only(),
          py::arg("session") = RootSessionID(), py::arg("username") = "",
          py::arg("password") = "")
      .def(
          "_connect",
          [](std::pair<std::string, uint32_t> const& endpoint,
             const SessionID session_id, const std::string& username,
             const std::string& password) {
            std::string rpc_endpoint =
                endpoint.first + ":" + std::to_string(endpoint.second);
            return ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint, session_id, username, password);
          },
          "endpoint"_a, py::kw_only(), py::arg("session") = RootSessionID(),
          py::arg("username") = "", py::arg("password") = "")
      .def(
          "_connect",
          [](std::pair<std::string, std::string> const& endpoint,
             const SessionID session_id, const std::string& username,
             const std::string& password) {
            std::string rpc_endpoint = endpoint.first + ":" + endpoint.second;
            return ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint, session_id, username, password);
          },
          "endpoint"_a, py::kw_only(), py::arg("session") = RootSessionID(),
          py::arg("username") = "", py::arg("password") = "");
}  // NOLINT(readability/fn_size)

}  // namespace vineyard
