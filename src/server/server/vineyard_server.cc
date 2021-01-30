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

#include "server/server/vineyard_server.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/util/boost.h"
#include "common/util/callback.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "server/async/ipc_server.h"
#include "server/async/rpc_server.h"
#include "server/services/meta_service.h"
#include "server/util/meta_tree.h"

namespace vineyard {

#ifndef ENSURE_VINEYARDD_READY
#define ENSURE_VINEYARDD_READY()                                      \
  do {                                                                \
    if (!((ready_ & kMeta) && (ready_ & kIPC) && (ready_ & kBulk))) { \
      std::stringstream ss;                                           \
      ss << "{";                                                      \
      ss << "meta: " << (ready_ & kMeta) << ", ";                     \
      ss << "ipc: " << (ready_ & kIPC) << ", ";                       \
      ss << "rpc: " << (ready_ & kRPC) << ", ";                       \
      ss << "bulk store: " << (ready_ & kBulk);                       \
      ss << "}";                                                      \
      return Status::VineyardServerNotReady(ss.str());                \
    }                                                                 \
  } while (0)
#endif  // ENSURE_VINEYARDD_READY

bool DeferredReq::Alive() const { return alive_fn_(); }

bool DeferredReq::TestThenCall(const json& meta) const {
  if (test_fn_(meta)) {
    VINEYARD_SUPPRESS(call_fn_(meta));
    return true;
  }
  return false;
}

VineyardServer::VineyardServer(const json& spec)
    : spec_(spec),
#if BOOST_VERSION >= 106600
      guard_(asio::make_work_guard(context_)),
#else
      guard_(new boost::asio::io_service::work(context_)),
#endif
      ready_(0),
      stopped_(false) {
}

Status VineyardServer::Serve() {
  this->meta_service_ptr_ = IMetaService::Get(shared_from_this());
  RETURN_ON_ERROR(this->meta_service_ptr_->Start());

  bulk_store_ = std::make_shared<BulkStore>();
  RETURN_ON_ERROR(bulk_store_->PreAllocate(
      spec_["bulkstore_spec"]["memory_size"].get<size_t>()));
  stream_store_ = std::make_shared<StreamStore>(
      bulk_store_, spec_["bulkstore_spec"]["stream_threshold"].get<size_t>());
  BulkReady();

  serve_status_ = Status::OK();
  context_.run();
  return serve_status_;
}

Status VineyardServer::Finalize() { return Status::OK(); }

std::shared_ptr<VineyardServer> VineyardServer::Get(const json& spec) {
  return std::shared_ptr<VineyardServer>(new VineyardServer(spec));
}

void VineyardServer::Ready() {}

void VineyardServer::BackendReady() {
  try {
    ipc_server_ptr_ =
        std::unique_ptr<IPCServer>(new IPCServer(shared_from_this()));
    ipc_server_ptr_->Start();
  } catch (std::exception const& ex) {
    LOG(ERROR) << "Failed to start vineyard IPC server: " << ex.what()
               << ", or please try to cleanup existing "
               << spec_["ipc_spec"]["socket"];
    serve_status_ = Status::IOError();
    context_.stop();
    return;
  }
  try {
    rpc_server_ptr_ =
        std::unique_ptr<RPCServer>(new RPCServer(shared_from_this()));
    rpc_server_ptr_->Start();
  } catch (std::exception const& ex) {
    LOG(ERROR) << "Failed to start vineyard RPC server: " << ex.what();
    serve_status_ = Status::IOError();
    context_.stop();
    return;
  }
}

void VineyardServer::MetaReady() {
  VINEYARD_ASSERT(!(ready_ & kMeta), "A component can't be initialized twice!");
  ready_ |= kMeta;
  if (ready_ == kReady) {
    Ready();
  }
  if (ready_ == kBackendReady) {
    BackendReady();
  }
}

void VineyardServer::BulkReady() {
  VINEYARD_ASSERT(!(ready_ & kBulk), "A component can't be initialized twice!");
  ready_ |= kBulk;
  if (ready_ == kReady) {
    Ready();
  }
  if (ready_ == kBackendReady) {
    BackendReady();
  }
}

void VineyardServer::IPCReady() {
  VINEYARD_ASSERT(!(ready_ & kIPC), "A component can't be initialized twice!");
  ready_ |= kIPC;
  if (ready_ == kReady) {
    Ready();
  }
}

void VineyardServer::RPCReady() {
  VINEYARD_ASSERT(!(ready_ & kRPC), "A component can't be initialized twice!");
  ready_ |= kRPC;
  if (ready_ == kReady) {
    Ready();
  }
}

Status VineyardServer::GetData(const std::vector<ObjectID>& ids,
                               const bool sync_remote, const bool wait,
                               std::function<bool()> alive,
                               callback_t<const json&> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToGetData(
      sync_remote, [this, ids, wait, alive, callback](const Status& status,
                                                      const json& meta) {
        if (status.ok()) {
      // When object not exists, we return an empty json, rather than
      // the status to indicate the error.
#if !defined(NDEBUG)
          if (VLOG_IS_ON(10)) {
            VLOG(10) << "Got request from client to get data, dump json:";
            VLOG(10) << meta.dump(4);
            VLOG(10) << "=========================================";
          }
#endif
          auto test_task = [ids](const json& meta) -> bool {
            for (auto const& id : ids) {
              bool exists = false;
              VINEYARD_SUPPRESS(
                  CATCH_JSON_ERROR(meta_tree::Exists(meta, id, exists)));
              if (!exists) {
                return exists;
              }
            }
            return true;
          };
          auto eval_task = [ids, callback](const json& meta) -> Status {
            json sub_tree_group;
            for (auto const& id : ids) {
              json sub_tree;
              VINEYARD_SUPPRESS(
                  CATCH_JSON_ERROR(meta_tree::GetData(meta, id, sub_tree)));
              if (sub_tree.is_object() && !sub_tree.empty()) {
                sub_tree_group[VYObjectIDToString(id)] = sub_tree;
              }
            }
            return callback(Status::OK(), sub_tree_group);
          };
          if (!wait || test_task(meta)) {
            return eval_task(meta);
          } else {
            this->deferred_.emplace_back(alive, test_task, eval_task);
            return Status::OK();
          }
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::ListData(std::string const& pattern, bool const regex,
                                size_t const limit,
                                callback_t<const json&> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToGetData(
      false,  // no need for sync from etcd
      [pattern, regex, limit, callback](const Status& status,
                                        const json& meta) {
        if (status.ok()) {
          json sub_tree_group;
          VINEYARD_CHECK_OK(CATCH_JSON_ERROR(meta_tree::ListData(
              meta, pattern, regex, limit, sub_tree_group)));
          return callback(status, sub_tree_group);
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::CreateData(
    const json& tree,
    callback_t<const ObjectID, const Signature, const InstanceID> callback) {
  ENSURE_VINEYARDD_READY();
#if !defined(NDEBUG)
  if (VLOG_IS_ON(10)) {
    VLOG(10) << "Got request from client to create data:";
    VLOG(10) << tree.dump(4);
    VLOG(10) << "=========================================";
  }
#endif
  // validate typename
  auto type_name_node = tree.value("typename", json(nullptr));
  if (type_name_node.is_null() || !type_name_node.is_string()) {
    RETURN_ON_ERROR(callback(Status::MetaTreeInvalid("No typename field"),
                             InvalidObjectID(), InvalidSignature(),
                             UnspecifiedInstanceID()));
  }
  std::string const& type = type_name_node.get_ref<std::string const&>();

  // generate ObjectID
  ObjectID id;
  if (type == "vineyard::Blob") {  // special codepath for creating Blob
    RETURN_ON_ASSERT(tree.contains("id"));
    id = VYObjectIDFromString(tree["id"].get_ref<std::string const&>());
    // RETURN_ON_ASSERT(IsBlob(id));
  } else {
    id = GenerateObjectID();
  }

  // Check if instance_id information available
  RETURN_ON_ASSERT(tree.contains("instance_id"));

  auto decorated_tree = tree;
  Signature signature = GenerateSignature();
  decorated_tree["signature"] = signature;

  // update meta into json
  meta_service_ptr_->RequestToBulkUpdate(
      [id, decorated_tree](const Status& status, const json& meta,
                           std::vector<IMetaService::op_t>& ops,
                           InstanceID& computed_instance_id) {
        if (status.ok()) {
          return CATCH_JSON_ERROR(meta_tree::PutDataOps(
              meta, id, decorated_tree, ops, computed_instance_id));
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      boost::bind(callback, _1, id, signature, _2));
  return Status::OK();
}

Status VineyardServer::Persist(const ObjectID id, callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToPersist(
      [id](const Status& status, const json& meta,
           std::vector<IMetaService::op_t>& ops) {
        if (status.ok()) {
          return CATCH_JSON_ERROR(meta_tree::PersistOps(meta, id, ops));
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      callback);
  return Status::OK();
}

Status VineyardServer::IfPersist(const ObjectID id,
                                 callback_t<const bool> callback) {
  ENSURE_VINEYARDD_READY();
  // How to decide if an object (an id) is persist:
  //
  // 1. every object has a `persist` field in meta
  // 2. if the object has been persist by other client that connects to
  //    the same vineyardd: just read the meta
  // 3. if the object has been persist on other vineyardd: that is impossible,
  //    since we cannot get a remote object before it has been persisted.
  //
  // Thus we just need to read from the metadata in vineyardd, without
  // touching etcd.
  meta_service_ptr_->RequestToGetData(
      false, [id, callback](const Status& status, const json& meta) {
        if (status.ok()) {
          bool persist = false;
          auto s = CATCH_JSON_ERROR(meta_tree::IfPersist(meta, id, persist));
          return callback(s, persist);
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::Exists(const ObjectID id,
                              callback_t<const bool> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToGetData(
      true, [id, callback](const Status& status, const json& meta) {
        if (status.ok()) {
          bool exists = false;
          auto s = CATCH_JSON_ERROR(meta_tree::Exists(meta, id, exists));
          return callback(s, exists);
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::ShallowCopy(const ObjectID id,
                                   callback_t<const ObjectID> callback) {
  ENSURE_VINEYARDD_READY();
  RETURN_ON_ASSERT(!IsBlob(id), "The blobs cannot be shallow copied");
  ObjectID target_id = GenerateObjectID();
  meta_service_ptr_->RequestToShallowCopy(
      [id, target_id](const Status& status, const json& meta,
                      std::vector<IMetaService::op_t>& ops, bool& transient) {
        if (status.ok()) {
          return CATCH_JSON_ERROR(
              meta_tree::ShallowCopyOps(meta, id, target_id, ops, transient));
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      [target_id, callback](const Status& status) {
        return callback(status, target_id);
      });
  return Status::OK();
}

Status VineyardServer::DelData(const std::vector<ObjectID>& ids,
                               const bool force, const bool deep,
                               callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToDelete(
      ids, force, deep,
      [](const Status& status, const json& meta,
         std::set<ObjectID> const& ids_to_delete,
         std::vector<IMetaService::op_t>& ops) {
        if (status.ok()) {
          auto status =
              CATCH_JSON_ERROR(meta_tree::DelDataOps(meta, ids_to_delete, ops));
          if (status.IsMetaTreeSubtreeNotExists()) {
            return Status::ObjectNotExists();
          }
          return status;
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      callback);
  return Status::OK();
}

Status VineyardServer::DeleteBlobBatch(const std::set<ObjectID>& ids) {
  for (auto object_id : ids) {
    VINEYARD_SUPPRESS(this->bulk_store_->ProcessDeleteRequest(object_id));
  }
  return Status::OK();
}

Status VineyardServer::DeleteAllAt(const json& meta,
                                   InstanceID const instance_id) {
  std::vector<ObjectID> objects_to_cleanup;
  auto status = CATCH_JSON_ERROR(
      meta_tree::FilterAtInstance(meta, instance_id, objects_to_cleanup));
  RETURN_ON_ERROR(status);
  return this->DelData(
      objects_to_cleanup, true, true, [](Status const& status) -> Status {
        if (!status.ok()) {
          LOG(ERROR) << "Error happens on cleanup: " << status.ToString();
        }
        return Status::OK();
      });
}

Status VineyardServer::PutName(const ObjectID object_id,
                               const std::string& name, callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToPersist(
      [object_id, name](const Status& status, const json& meta,
                        std::vector<IMetaService::op_t>& ops) {
        if (status.ok()) {
          // TODO: do proper validation:
          // 1. global objects can have name, local ones cannot.
          // 2. the name-object_id mapping shouldn't be overwrite.
          ops.emplace_back(
              IMetaService::op_t::Put("/names/" + name, object_id));
          return Status::OK();
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      callback);
  return Status::OK();
}

Status VineyardServer::GetName(const std::string& name, const bool wait,
                               DeferredReq::alive_t alive,
                               callback_t<const ObjectID&> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToGetData(
      true, [this, name, wait, alive, callback](const Status& status,
                                                const json& meta) {
        if (status.ok()) {
          auto test_task = [name](const json& meta) -> bool {
            auto names = meta.value("names", json(nullptr));
            if (names.is_object()) {
              return names.contains(name);
            }
            return false;
          };
          auto eval_task = [name, callback](const json& meta) -> Status {
            auto names = meta.value("names", json(nullptr));
            if (names.is_object() && names.contains(name)) {
              auto entry = names[name];
              if (!entry.is_null()) {
                return callback(Status::OK(), entry.get<ObjectID>());
              }
            }
            return callback(Status::ObjectNotExists(), InvalidObjectID());
          };
          if (!wait || test_task(meta)) {
            return eval_task(meta);
          } else {
            deferred_.emplace_back(alive, test_task, eval_task);
            return Status::OK();
          }
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::DropName(const std::string& name,
                                callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToPersist(
      [name](const Status& status, const json& meta,
             std::vector<IMetaService::op_t>& ops) {
        if (status.ok()) {
          auto names = meta.value("names", json(nullptr));
          if (names.is_object() && names.contains(name)) {
            ops.emplace_back(IMetaService::op_t::Del("/names/" + name));
          }
          return Status::OK();
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      callback);
  return Status::OK();
}

Status VineyardServer::ClusterInfo(callback_t<const json&> callback) {
  ENSURE_VINEYARDD_READY();
  meta_service_ptr_->RequestToGetData(
      true, [callback](const Status& status, const json& meta) {
        if (status.ok()) {
          return callback(status, meta["instances"]);
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::InstanceStatus(callback_t<const json&> callback) {
  ENSURE_VINEYARDD_READY();

  json status;
  status["instance_id"] = instance_id_;
  status["deployment"] = GetDeployment();
  status["memory_usage"] = bulk_store_->Footprint();
  status["memory_limit"] = bulk_store_->FootprintLimit();
  status["deferred_requests"] = deferred_.size();
  if (ipc_server_ptr_) {
    status["ipc_connections"] = ipc_server_ptr_->AliveConnections();
  } else {
    status["ipc_connections"] = 0;
  }
  if (rpc_server_ptr_) {
    status["rpc_connections"] = rpc_server_ptr_->AliveConnections();
  } else {
    status["rpc_connections"] = 0;
  }

  return callback(Status::OK(), status);
}

Status VineyardServer::ProcessDeferred(const json& meta) {
  auto iter = deferred_.begin();
  while (iter != deferred_.end()) {
    if (!iter->Alive() || iter->TestThenCall(meta)) {
      deferred_.erase(iter++);
    } else {
      ++iter;
    }
  }
  return Status::OK();
}

const std::string VineyardServer::IPCSocket() {
  return ipc_server_ptr_->Socket();
}

const std::string VineyardServer::RPCEndpoint() {
  if (this->rpc_server_ptr_) {
    return rpc_server_ptr_->Endpoint();
  } else {
    return "0.0.0.0:0";
  }
}

void VineyardServer::Stop() {
  if (stopped_) {
    return;
  }
  stopped_ = true;
  guard_.reset();
  if (this->ipc_server_ptr_) {
    this->ipc_server_ptr_->Stop();
    this->ipc_server_ptr_.reset(nullptr);
  }
  if (this->rpc_server_ptr_) {
    this->rpc_server_ptr_->Stop();
    this->rpc_server_ptr_.reset(nullptr);
  }

  meta_service_ptr_->Stop();

  // stop the asio context at last
  context_.stop();
}

VineyardServer::~VineyardServer() { this->Stop(); }

}  // namespace vineyard
