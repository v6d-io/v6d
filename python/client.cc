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

#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#pragma GCC visibility push(default)
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/rpc_client.h"
#include "common/util/status.h"
#pragma GCC visibility pop

#include "pybind11_utils.h"  // NOLINT(build/include)

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces)

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

  std::shared_ptr<ClientType> Connect(std::string const& endpoint = "") {
    std::lock_guard<std::mutex> guard{mtx_};
    auto iter = client_set_.find(endpoint);
    if (iter != client_set_.end()) {
      if (iter->second->Connected()) {
        return iter->second;
      }
    }
    std::shared_ptr<ClientType> client = std::make_shared<ClientType>();
    auto connect_status =
        endpoint.empty() ? client->Connect() : client->Connect(endpoint);
    if (PyErr_CheckSignals() != 0) {
      // The method `Connect` will keep retrying, we need to propogate
      // the Ctrl-C when during the C++ code run retries.
      throw py::error_already_set();
    }
    // propogate the KeyboardInterrupt exception correctly before the
    // RuntimeError
    throw_on_error(connect_status);
    client_set_.emplace(endpoint, client);
    return client;
  }

  void Disconnect(std::string const& ipc_socket) {
    std::lock_guard<std::mutex> guard{mtx_};
    auto iter = client_set_.find(ipc_socket);
    CHECK(iter != client_set_.end());
    if (iter->second.use_count() == 2) {
      // the python variable is the last reference, another reference
      // lies in the `client_set_`.
      iter->second->Disconnect();
      client_set_.erase(ipc_socket);
    }
  }

 private:
  std::mutex mtx_;
  std::unordered_map<std::string, std::shared_ptr<ClientType>> client_set_;
};

void bind_client(py::module& mod) {
  // ClientBase
  py::class_<ClientBase, std::shared_ptr<ClientBase>>(mod, "ClientBase")
      .def(
          "create_metadata",
          [](ClientBase* self, ObjectMeta& metadata) -> ObjectIDWrapper {
            ObjectID object_id;
            throw_on_error(self->CreateMetaData(metadata, object_id));
            return object_id;
          },
          "metadata"_a)
      .def(
          "delete",
          [](ClientBase* self, const ObjectIDWrapper object_id,
             const bool force, const bool deep) {
            throw_on_error(self->DelData(object_id, force, deep));
          },
          "object_id"_a, py::arg("force") = false, py::arg("deep") = true)
      .def(
          "delete",
          [](ClientBase* self, const std::vector<ObjectIDWrapper>& object_ids,
             const bool force, const bool deep) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            throw_on_error(self->DelData(unwrapped_object_ids, force, deep));
          },
          "object_ids"_a, py::arg("force") = false, py::arg("deep") = true)
      .def(
          "persist",
          [](ClientBase* self, const ObjectIDWrapper object_id) {
            throw_on_error(self->Persist(object_id));
          },
          "object_id"_a)
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
          "object_id"_a)
      .def(
          "shallow_copy",
          [](ClientBase* self, const ObjectIDWrapper object_id) -> bool {
            ObjectID target_id;
            throw_on_error(self->ShallowCopy(object_id, target_id));
            return target_id;
          },
          "object_id"_a)
      .def(
          "put_name",
          [](ClientBase* self, const ObjectIDWrapper object_id,
             std::string const& name) {
            throw_on_error(self->PutName(object_id, name));
          },
          "object_id"_a, "name"_a)
      .def(
          "get_name",
          [](ClientBase* self, std::string const& name,
             const bool wait) -> ObjectIDWrapper {
            ObjectID object_id;
            throw_on_error(self->GetName(name, object_id));
            return object_id;
          },
          "object_id"_a, py::arg("wait") = false)
      .def("drop_name",
           [](ClientBase* self, std::string const& name) {
             throw_on_error(self->DropName(name));
           })
      .def_property_readonly("connected", &Client::Connected)
      .def_property_readonly("instance_id", &Client::instance_id)
      .def_property_readonly(
          "meta",
          [](ClientBase* self)
              -> std::map<uint64_t, std::unordered_map<std::string, py::object>> {
            std::map<uint64_t, json> meta;
            throw_on_error(self->ClusterInfo(meta));
            std::map<uint64_t, std::unordered_map<std::string, py::object>> meta_to_return;
            for (auto const& kv : meta) {
              std::unordered_map<std::string, py::object> element;
              if (!kv.second.empty()) {
                for (auto const& elem : json::iterator_wrapper(kv.second)) {
                  element[elem.key()] = json_to_python(elem.value());
                }
              }
              meta_to_return.emplace(kv.first, std::move(element));
            }
            return meta_to_return;
          })
      .def_property_readonly(
          "status",
          [](ClientBase* self) -> std::shared_ptr<InstanceStatus> {
            std::shared_ptr<InstanceStatus> status;
            throw_on_error(self->InstanceStatus(status));
            return status;
          })
      .def_property_readonly("ipc_socket", &ClientBase::IPCSocket)
      .def_property_readonly("rpc_endpoint", &ClientBase::RPCEndpoint);

  // InstanceStatus
  py::class_<InstanceStatus, std::shared_ptr<InstanceStatus>>(mod,
                                                              "InstanceStatus")
      .def_property_readonly(
          "instance_id",
          [](InstanceStatus* status) { return status->instance_id; })
      .def_property_readonly(
          "deployment",
          [](InstanceStatus* status) { return status->deployment; })
      .def_property_readonly(
          "memory_usage",
          [](InstanceStatus* status) { return status->memory_usage; })
      .def_property_readonly(
          "memory_limit",
          [](InstanceStatus* status) { return status->memory_limit; })
      .def_property_readonly(
          "deferred_requests",
          [](InstanceStatus* status) { return status->deferred_requests; })
      .def_property_readonly(
          "ipc_connections",
          [](InstanceStatus* status) { return status->ipc_connections; })
      .def_property_readonly(
          "rpc_connections",
          [](InstanceStatus* status) { return status->rpc_connections; })
      .def("__repr__", [](InstanceStatus* status) { return "InstanceStatus"; })
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

  // Client
  py::class_<Client, std::shared_ptr<Client>, ClientBase>(mod, "IPCClient")
      .def(
          "create_blob",
          [](Client* self, size_t size) {
            std::unique_ptr<BlobWriter> blob;
            throw_on_error(self->CreateBlob(size, blob));
            return std::shared_ptr<BlobWriter>(blob.release());
          },
          py::return_value_policy::move, "size"_a)
      .def("create_empty_blob",
           [](Client* self) -> std::shared_ptr<Blob> {
             return Blob::MakeEmpty(*self);
           })
      .def(
          "get_object",
          [](Client* self, const ObjectIDWrapper object_id) {
            return self->GetObject(object_id);
          },
          "object_id"_a)
      .def(
          "get_objects",
          [](Client* self, const std::vector<ObjectIDWrapper>& object_ids) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            return self->GetObjects(unwrapped_object_ids);
          },
          "object_ids"_a)
      .def(
          "get_meta",
          [](Client* self, ObjectIDWrapper const& object_id, bool const sync_remote) -> ObjectMeta {
            ObjectMeta meta;
            // FIXME: do we really not need to sync from etcd? We assume the
            // object is a local object
            throw_on_error(self->GetMetaData(object_id, meta, sync_remote));
            return meta;
          },
          "object_id"_a, py::arg("sync_remote") = false)
      .def(
          "get_metas",
          [](Client* self, std::vector<ObjectIDWrapper> const& object_ids, bool const sync_remote)
              -> std::vector<ObjectMeta> {
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
          "object_ids"_a, py::arg("sync_remote") = false)
      .def("list_objects", &Client::ListObjects, "pattern"_a,
           py::arg("regex") = false, py::arg("limit") = 5)
      .def("close",
           [](Client* self) {
             return ClientManager<Client>::GetManager()->Disconnect(
                 self->IPCSocket());
           })
      .def("__enter__", [](Client* self) { return self; })
      .def("__exit__", [](Client* self, py::object, py::object, py::object) {
        // DO NOTHING
      });

  // RPCClient
  py::class_<RPCClient, std::shared_ptr<RPCClient>, ClientBase>(mod,
                                                                "RPCClient")
      .def(
          "get_object",
          [](RPCClient* self, const ObjectIDWrapper object_id) {
            return self->GetObject(object_id);
          },
          "object_id"_a)
      .def(
          "get_objects",
          [](RPCClient* self, std::vector<ObjectIDWrapper> const& object_ids) {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            return self->GetObjects(unwrapped_object_ids);
          },
          "object_ids"_a)
      .def(
          "get_meta",
          [](RPCClient* self, ObjectIDWrapper const& object_id) -> ObjectMeta {
            ObjectMeta meta;
            throw_on_error(self->GetMetaData(object_id, meta, true));
            return meta;
          },
          "object_id"_a)
      .def(
          "get_metas",
          [](RPCClient* self, std::vector<ObjectIDWrapper> const& object_ids)
              -> std::vector<ObjectMeta> {
            std::vector<ObjectID> unwrapped_object_ids(object_ids.size());
            for (size_t idx = 0; idx < object_ids.size(); ++idx) {
              unwrapped_object_ids[idx] = object_ids[idx];
            }
            std::vector<ObjectMeta> metas;
            throw_on_error(
                self->GetMetaData(unwrapped_object_ids, metas, true));
            return metas;
          },
          "object_ids"_a)
      .def("list_objects", &RPCClient::ListObjects, "pattern"_a,
           py::arg("regex") = false, py::arg("limit") = 5)
      .def("close",
           [](RPCClient* self) {
             return ClientManager<RPCClient>::GetManager()->Disconnect(
                 self->RPCEndpoint());
           })
      .def("__enter__", [](RPCClient* self) { return self; })
      .def("__exit__", [](RPCClient* self, py::object, py::object, py::object) {
        // DO NOTHING
      });

  mod.def(
         "connect",
         [](nullptr_t) -> py::object {
           try {
             return py::cast(ClientManager<Client>::GetManager()->Connect());
           } catch (...) {}
           try {
             return py::cast(ClientManager<RPCClient>::GetManager()->Connect());
           } catch (...) {}
           throw_on_error(Status::ConnectionFailed(
               "Failed to resolve IPC socket or RPC endpoint of vineyard "
               "server from environment variables."));
           return py::none();
         },
         py::arg("target") = py::none())
      .def(
          "connect",
          [](std::string const& endpoint) -> py::object {
            try {
              return py::cast(
                  ClientManager<Client>::GetManager()->Connect(endpoint));
            } catch (...) {}
            try {
              return py::cast(
                  ClientManager<RPCClient>::GetManager()->Connect(endpoint));
            } catch (...) {}
            throw_on_error(Status::ConnectionFailed(
                "Failed to resolve IPC socket or RPC endpoint of vineyard "
                "server"));
            return py::none();
          },
          "endpoint"_a)
      .def(
          "connect",
          [](std::string const& host, const uint32_t port) {
            std::string rpc_endpoint = host + ":" + std::to_string(port);
            return ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint);
          },
          "host"_a, "port"_a)
      .def(
          "connect",
          [](std::string const& host, std::string const& port) {
            std::string rpc_endpoint = host + ":" + port;
            return ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint);
          },
          "host"_a, "port"_a)
      .def(
          "connect",
          [](std::pair<std::string, uint32_t> const& endpoint) {
            std::string rpc_endpoint =
                endpoint.first + ":" + std::to_string(endpoint.second);
            return ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint);
          },
          "(host, port)"_a)
      .def(
          "connect",
          [](std::pair<std::string, std::string> const& endpoint) {
            std::string rpc_endpoint = endpoint.first + ":" + endpoint.second;
            return ClientManager<RPCClient>::GetManager()->Connect(
                rpc_endpoint);
          },
          "(host, port)"_a);
}

}  // namespace vineyard
