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

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/util/boost.h"
#include "common/util/callback.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "server/async/ipc_server.h"
#include "server/async/rpc_server.h"
#include "server/services/meta_service.h"
#include "server/util/kubectl.h"
#include "server/util/meta_tree.h"
#include "server/util/proc.h"

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
      concurrency_(std::thread::hardware_concurrency()),
      context_(concurrency_),
      meta_context_(),
#if BOOST_VERSION >= 106600
      guard_(asio::make_work_guard(context_)),
      meta_guard_(asio::make_work_guard(meta_context_)),
#else
      guard_(new boost::asio::io_service::work(context_)),
      meta_guard_(new boost::asio::io_service::work(context_)),
#endif
      ready_(0) {
}

Status VineyardServer::Serve() {
  stopped_.store(false);

  // Initialize the ipc/rpc server ptr first to get self endpoints when
  // initializing the metadata service.
  ipc_server_ptr_ =
      std::unique_ptr<IPCServer>(new IPCServer(shared_from_this()));
  rpc_server_ptr_ =
      std::unique_ptr<RPCServer>(new RPCServer(shared_from_this()));

  this->meta_service_ptr_ = IMetaService::Get(shared_from_this());
  RETURN_ON_ERROR(this->meta_service_ptr_->Start());

  bulk_store_ = std::make_shared<BulkStore>();
  RETURN_ON_ERROR(bulk_store_->PreAllocate(
      spec_["bulkstore_spec"]["memory_size"].get<size_t>()));
  stream_store_ = std::make_shared<StreamStore>(
      bulk_store_, spec_["bulkstore_spec"]["stream_threshold"].get<size_t>());
  BulkReady();

  serve_status_ = Status::OK();

  for (unsigned int idx = 0; idx < concurrency_; ++idx) {
#if BOOST_VERSION >= 106600
    workers_.emplace_back(
        boost::bind(&boost::asio::io_context::run, &context_));
#else
    workers_.emplace_back(
        boost::bind(&boost::asio::io_service::run, &context_));
#endif
  }
  meta_context_.run();

  return serve_status_;
}

Status VineyardServer::Finalize() { return Status::OK(); }

std::shared_ptr<VineyardServer> VineyardServer::Get(const json& spec) {
  return std::shared_ptr<VineyardServer>(new VineyardServer(spec));
}

void VineyardServer::Ready() {}

void VineyardServer::BackendReady() {
  try {
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
            std::cerr << meta.dump(4) << std::endl;
            VLOG(10) << "=========================================";
          }
#endif

          auto test_task = [this, ids](const json& meta) -> bool {
            for (auto const& id : ids) {
              bool exists = false;
              if (IsBlob(id)) {
                exists = this->bulk_store_->Exists(id);
              } else {
                VINEYARD_SUPPRESS(
                    CATCH_JSON_ERROR(meta_tree::Exists(meta, id, exists)));
              }
              if (!exists) {
                return exists;
              }
            }
            return true;
          };
          auto eval_task = [this, ids, callback](const json& meta) -> Status {
            json sub_tree_group;
            for (auto const& id : ids) {
              json sub_tree;
              if (IsBlob(id)) {
                std::shared_ptr<Payload> object;
                if (this->bulk_store_->Get(id, object).ok()) {
                  sub_tree["id"] = VYObjectIDToString(id);
                  sub_tree["typename"] = "vineyard::Blob";
                  sub_tree["length"] = object->data_size;
                  sub_tree["nbytes"] = object->data_size;
                  sub_tree["transient"] = true;
                  sub_tree["instance_id"] = this->instance_id();
                }
              } else {
                VINEYARD_SUPPRESS(CATCH_JSON_ERROR(
                    meta_tree::GetData(meta, id, sub_tree, instance_id_)));
#if !defined(NDEBUG)
                if (VLOG_IS_ON(10)) {
                  VLOG(10) << "Got request response:";
                  std::cerr << sub_tree.dump(4) << std::endl;
                  VLOG(10) << "=========================================";
                }
#endif
              }
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
    // NB: glog has limit on maximum lines.
    std::cerr << tree.dump(4) << std::endl;
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

  RETURN_ON_ASSERT(type != "vineyard::Blob", "Blob has no metadata");

  ObjectID id = GenerateObjectID();
  // Check if instance_id information available
  RETURN_ON_ASSERT(tree.contains("instance_id"),
                   "The instance_id filed must be presented");

  auto decorated_tree = tree;
  Signature signature = GenerateSignature();
  if (decorated_tree.find("signature") != decorated_tree.end()) {
    signature = decorated_tree["signature"].get<Signature>();
  } else {
    decorated_tree["signature"] = signature;
  }

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
  RETURN_ON_ASSERT(!IsBlob(id), "The blobs cannot be persisted");
  auto self(shared_from_this());
  meta_service_ptr_->RequestToPersist(
      [self, id](const Status& status, const json& meta,
                 std::vector<IMetaService::op_t>& ops) {
        if (status.ok()) {
          auto status = CATCH_JSON_ERROR(meta_tree::PersistOps(meta, id, ops));
          if (self->spec_["sync_crds"].get<bool>() && status.ok() &&
              !ops.empty()) {
            json tree;
            VINEYARD_SUPPRESS(
                CATCH_JSON_ERROR(meta_tree::GetData(meta, id, tree)));
            if (tree.is_object() && !tree.empty()) {
              auto kube = std::make_shared<Kubectl>(self->GetMetaContext());
              kube->ApplyObject(meta["instances"], tree);
              kube->Finish();
            }
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
  if (IsBlob(id)) {
    context_.post(boost::bind(callback, Status::OK(), false));
    return Status::OK();
  }
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
  if (IsBlob(id)) {
    context_.post([this, id, callback] {
      VINEYARD_DISCARD(callback(Status::OK(), bulk_store_->Exists(id)));
    });
    return Status::OK();
  }
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
                               const bool fastpath, callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  if (fastpath) {
    // forcely delete the given blobs: used for allocators
    for (auto const id : ids) {
      RETURN_ON_ASSERT(IsBlob(id),
                       "Fastpath deletion can only be applied to blobs");
    }
    context_.post([this, ids, callback] {
      for (auto const id : ids) {
        VINEYARD_DISCARD(bulk_store_->Delete(id));
      }
      VINEYARD_DISCARD(callback(Status::OK()));
    });
    return Status::OK();
  }
  meta_service_ptr_->RequestToDelete(
      ids, force, deep,
      [](const Status& status, const json& meta,
         std::vector<ObjectID> const& ids_to_delete,
         std::vector<IMetaService::op_t>& ops, bool& sync_remote) {
        if (status.ok()) {
          auto status = CATCH_JSON_ERROR(
              meta_tree::DelDataOps(meta, ids_to_delete, ops, sync_remote));
          if (status.IsMetaTreeSubtreeNotExists()) {
            return Status::ObjectNotExists(status.ToString());
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
    VINEYARD_SUPPRESS(this->bulk_store_->Delete(object_id));
  }
  return Status::OK();
}

Status VineyardServer::DeleteAllAt(const json& meta,
                                   InstanceID const instance_id) {
  std::vector<ObjectID> objects_to_cleanup;
  auto status = CATCH_JSON_ERROR(
      meta_tree::FilterAtInstance(meta, instance_id, objects_to_cleanup));
  RETURN_ON_ERROR(status);
  return this->DelData(objects_to_cleanup, true, true, false /* fastpath */,
                       [](Status const& status) -> Status {
                         if (!status.ok()) {
                           LOG(ERROR) << "Error happens on cleanup: "
                                      << status.ToString();
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

Status VineyardServer::MigrateObject(const ObjectID object_id, const bool local,
                                     const std::string& peer,
                                     const std::string& peer_rpc_endpoint,
                                     callback_t<const ObjectID&> callback) {
  ENSURE_VINEYARDD_READY();
  RETURN_ON_ASSERT(!IsBlob(object_id), "The blobs cannot be migrated");
  auto self(shared_from_this());
  static const std::string migrate_process = "vineyard-migrate";

  if (!local) {
    std::vector<std::string> args = {
        "--client",       "true",
        "--ipc_socket",   IPCSocket(),
        "--rpc_endpoint", peer_rpc_endpoint,
        "--host",         peer,
        "--id",           VYObjectIDToString(object_id)};
    auto proc = std::make_shared<Process>(context_);
    proc->Start(
        migrate_process, args,
        [self, callback, proc, object_id](Status const& status,
                                          std::string const& line) {
          if (status.ok()) {
            proc->Wait();
            if (proc->ExitCode() != 0) {
              return callback(
                  Status::IOError("The migration client exit abnormally"),
                  InvalidObjectID());
            }

            ObjectID result_id = VYObjectIDFromString(line);

            // associate the signature
            self->meta_service_ptr_->RequestToBulkUpdate(
                [object_id, result_id](const Status& status, const json& meta,
                                       std::vector<IMetaService::op_t>& ops,
                                       InstanceID&) {
                  // get signature
                  json tree;
                  VINEYARD_SUPPRESS(CATCH_JSON_ERROR(
                      meta_tree::GetData(meta, object_id, tree)));
                  Signature sig = tree["signature"].get<Signature>();
                  VLOG(2) << "original " << ObjectIDToString(object_id)
                          << " -> " << SignatureToString(sig);
                  // put signature
                  ops.emplace_back(IMetaService::op_t::Put(
                      "/signatures/" + SignatureToString(sig),
                      ObjectIDToString(result_id)));
                  VLOG(2) << "becomes " << ObjectIDToString(object_id) << " -> "
                          << SignatureToString(sig);
                  return Status::OK();
                },
                [callback, result_id](const Status& status, InstanceID const&) {
                  return callback(Status::OK(), result_id);
                });
          } else {
            proc->Terminate();
            return callback(status, InvalidObjectID());
          }
          return Status::OK();
        });
  } else {
    std::vector<std::string> args = {"--server",       "true",
                                     "--ipc_socket",   IPCSocket(),
                                     "--rpc_endpoint", peer_rpc_endpoint,
                                     "--host",         "0.0.0.0"};
    auto proc = std::make_shared<Process>(context_);
    proc->Start(
        migrate_process, args,
        [callback, proc, object_id](Status const& status,
                                    std::string const& line) {
          if (status.ok()) {
            proc->Wait();
            if (proc->ExitCode() != 0) {
              return callback(
                  Status::IOError("The migration server exit abnormally"),
                  InvalidObjectID());
            } else {
              return callback(Status::OK(), object_id);
            }
          } else {
            proc->Terminate();
            return callback(status, InvalidObjectID());
          }
        });
  }
  return Status::OK();
}

Status VineyardServer::MigrateStream(const ObjectID stream_id, const bool local,
                                     const std::string& peer,
                                     const std::string& peer_rpc_endpoint,
                                     callback_t<const ObjectID&> callback) {
  ENSURE_VINEYARDD_READY();
  RETURN_ON_ASSERT(!IsBlob(stream_id), "The blobs cannot be migrated");
  auto self(shared_from_this());
  static const std::string migrate_process = "vineyard-migrate-stream";

  if (local) {
    std::vector<std::string> args = {
        "--client",       "true",
        "--ipc_socket",   IPCSocket(),
        "--rpc_endpoint", peer_rpc_endpoint,
        "--host",         peer,
        "--id",           VYObjectIDToString(stream_id)};
    auto proc = std::make_shared<Process>(context_);
    proc->Start(
        migrate_process, args,
        [self, callback, proc, stream_id](Status const& status,
                                          std::string const& line) {
          if (status.ok()) {
            RETURN_ON_ERROR(callback(Status::OK(), stream_id));
            if (!proc->Running() && proc->ExitCode() != 0) {
              return Status::IOError("The migration client exit abnormally");
            }
            proc->Detach();
            return Status::OK();
          } else {
            proc->Terminate();
            return callback(status, InvalidObjectID());
          }
          return Status::OK();
        });
  } else {
    std::vector<std::string> args = {
        "--server",       "true",
        "--ipc_socket",   IPCSocket(),
        "--rpc_endpoint", peer_rpc_endpoint,
        "--host",         "0.0.0.0",
        "--id",           VYObjectIDToString(stream_id)};
    auto proc = std::make_shared<Process>(context_);
    proc->Start(
        migrate_process, args,
        [callback, proc](Status const& status, std::string const& line) {
          if (status.ok()) {
            ObjectID result_id = VYObjectIDFromString(line);
            RETURN_ON_ERROR(callback(Status::OK(), result_id));
            if (!proc->Running() && proc->ExitCode() != 0) {
              return Status::IOError("The migration server exit abnormally");
            }
            proc->Detach();
            return Status::OK();
          } else {
            proc->Terminate();
            return callback(status, InvalidObjectID());
          }
        });
  }
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
  if (this->ipc_server_ptr_) {
    return ipc_server_ptr_->Socket();
  } else {
    return "-";
  }
}

const std::string VineyardServer::RPCEndpoint() {
  if (this->rpc_server_ptr_) {
    return rpc_server_ptr_->Endpoint();
  } else {
    return "0.0.0.0:0";
  }
}

void VineyardServer::Stop() {
  if (stopped_.exchange(true)) {
    return;
  }

  guard_.reset();
  meta_guard_.reset();
  if (this->ipc_server_ptr_) {
    this->ipc_server_ptr_->Stop();
  }
  if (this->rpc_server_ptr_) {
    this->rpc_server_ptr_->Stop();
  }
  if (this->meta_service_ptr_) {
    this->meta_service_ptr_->Stop();
  }

  // stop the asio context at last
  context_.stop();
  meta_context_.stop();

  // cleanup
  this->ipc_server_ptr_.reset(nullptr);
  this->rpc_server_ptr_.reset(nullptr);
  this->meta_service_ptr_.reset();

  // wait for the IO context finishes.
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

bool VineyardServer::Running() const { return !stopped_.load(); }

VineyardServer::~VineyardServer() { this->Stop(); }

}  // namespace vineyard
