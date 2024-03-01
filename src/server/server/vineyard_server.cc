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

#include "server/server/vineyard_server.h"

#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "gulrak/filesystem.hpp"

#include "common/util/callback.h"
#include "common/util/json.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "server/async/ipc_server.h"
#include "server/async/rpc_server.h"
#include "server/services/meta_service.h"
#include "server/util/kubectl.h"
#include "server/util/meta_tree.h"
#include "server/util/metrics.h"
#include "server/util/proc.h"
#include "server/util/remote.h"

namespace vineyard {

#ifndef ENSURE_VINEYARDD_READY
#define ENSURE_VINEYARDD_READY()                       \
  do {                                                 \
    if (!((ready_ & kMeta) && (ready_ & kBulk))) {     \
      std::stringstream ss;                            \
      ss << "{";                                       \
      ss << "meta: " << (ready_ & kMeta) << ", ";      \
      ss << "bulk store: " << (ready_ & kBulk);        \
      ss << "}";                                       \
      return Status::VineyardServerNotReady(ss.str()); \
    }                                                  \
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

VineyardServer::VineyardServer(const json& spec, const SessionID& session_id,
                               std::shared_ptr<VineyardRunner> runner,
                               asio::io_context& context,
                               asio::io_context& meta_context,
                               asio::io_context& io_context,
                               callback_t<std::string const&> callback)
    : spec_(spec),
      session_id_(session_id),
      context_(context),
      meta_context_(meta_context),
      io_context_(io_context),
      callback_(callback),
      runner_(runner),
      ready_(0) {}

template <>
std::shared_ptr<BulkStore> VineyardServer::GetBulkStore<ObjectID>() {
  return bulk_store_;
}

template <>
std::shared_ptr<PlasmaBulkStore> VineyardServer::GetBulkStore<PlasmaID>() {
  return plasma_bulk_store_;
}

Status VineyardServer::Serve(StoreType const& bulk_store_type) {
  stopped_.store(false);
  this->bulk_store_type_ = bulk_store_type;

  // Initialize the ipc/rpc server ptr first to get self endpoints when
  // initializing the metadata service.
  ipc_server_ptr_ = std::make_shared<IPCServer>(shared_from_this());
  if (session_id_ == RootSessionID() && spec_["rpc_spec"]["rpc"].get<bool>()) {
    // N.B: the rpc won't be enabled for child sessions. In the handler
    // of "Register" request in RPC server the session will be set as the
    // request session as expected.
    rpc_server_ptr_ = std::make_shared<RPCServer>(shared_from_this());
  }

  this->meta_service_ptr_ = IMetaService::Get(shared_from_this());
  RETURN_ON_ERROR(this->meta_service_ptr_->Start());

  auto memory_limit = spec_["bulkstore_spec"]["memory_size"].get<size_t>();
  auto allocator = spec_["bulkstore_spec"]["allocator"].get<std::string>();

  // the allocator behind `BulkAllocator` is a singleton
  static std::once_flag allocator_init_flag;
  Status allocator_init_error;

  if (bulk_store_type_ == StoreType::kPlasma) {
    plasma_bulk_store_ = std::make_shared<PlasmaBulkStore>();
    std::call_once(allocator_init_flag,
                   [this, memory_limit, allocator, &allocator_init_error]() {
                     allocator_init_error = plasma_bulk_store_->PreAllocate(
                         memory_limit, allocator);
                   });
    RETURN_ON_ERROR(allocator_init_error);

    // TODO(mengke.mk): Currently we do not allow streaming in plasma
    // bulkstore, anyway, we can templatize stream store to solve this.
    stream_store_ = nullptr;
  } else if (bulk_store_type_ == StoreType::kDefault) {
    bulk_store_ = std::make_shared<BulkStore>();
    auto spill_lower_bound_rate =
        spec_["bulkstore_spec"]["spill_lower_bound_rate"].get<double>();
    auto spill_upper_bound_rate =
        spec_["bulkstore_spec"]["spill_upper_bound_rate"].get<double>();
    std::call_once(allocator_init_flag, [this, memory_limit, allocator,
                                         &allocator_init_error]() {
      allocator_init_error = bulk_store_->PreAllocate(memory_limit, allocator);
    });
    RETURN_ON_ERROR(allocator_init_error);

    // setup spill
    bulk_store_->SetMemSpillUpBound(memory_limit * spill_upper_bound_rate);
    bulk_store_->SetMemSpillLowBound(memory_limit * spill_lower_bound_rate);
    bulk_store_->SetSpillPath(
        spec_["bulkstore_spec"]["spill_path"].get<std::string>());

    // setup stream store
    stream_store_ = std::make_shared<StreamStore>(
        shared_from_this(), bulk_store_,
        spec_["bulkstore_spec"]["stream_threshold"].get<size_t>());
  }

  BulkReady();

  serve_status_ = Status::OK();
  return serve_status_;
}

Status VineyardServer::Finalize() { return Status::OK(); }

void VineyardServer::Ready() {
  VINEYARD_DISCARD(callback_(Status::OK(), IPCSocket()));
}

void VineyardServer::BackendReady() {
  try {
    if (ipc_server_ptr_) {
      ipc_server_ptr_->Start();
      LOG_SUMMARY("ipc_connection_total", this->instance_id(), 1);
    }
  } catch (std::exception const& ex) {
    LOG(ERROR) << "Failed to start vineyard IPC server: " << ex.what()
               << ", or please try to cleanup existing "
               << spec_["ipc_spec"]["socket"];
    serve_status_ = Status::IOError(
        std::string("Failed to start vineyard RPC server: ") + ex.what());
    VINEYARD_DISCARD(callback_(serve_status_, IPCSocket()));
    context_.stop();
    return;
  }

  try {
    if (rpc_server_ptr_) {
      rpc_server_ptr_->Start();
      LOG_SUMMARY("rpc_connection_total", this->instance_id(), 1);
    } else {
      RPCReady();
    }
  } catch (std::exception const& ex) {
    LOG(ERROR) << "Failed to start vineyard RPC server: " << ex.what();
    serve_status_ = Status::IOError(
        std::string("Failed to start vineyard RPC server: ") + ex.what());
    VINEYARD_DISCARD(callback_(serve_status_, IPCSocket()));
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
  auto self(shared_from_this());
  meta_service_ptr_->RequestToGetData(
      sync_remote, [self, ids, wait, alive, callback](const Status& status,
                                                      const json& meta) {
        if (status.ok()) {
      // When object not exists, we return an empty json, rather than
      // the status to indicate the error.
#if !defined(NDEBUG)
          if (VLOG_IS_ON(100)) {
            DVLOG(100) << "Got request from client to get data, dump json:";
            std::cerr << meta.dump(4) << std::endl;
            DVLOG(100) << "=========================================";
            std::stringstream ss;
            for (auto const& id : ids) {
              ss << id << "(" << ObjectIDToString(id) << "), ";
            }
            DVLOG(100) << "Requesting objects: " << ss.str();
            DVLOG(100) << "=========================================";
          }
#endif
          auto test_task = [self, ids](const json& meta) -> bool {
            for (auto const& id : ids) {
              bool exists = false;
              if (IsBlob(id)) {
                exists = self->bulk_store_->Exists(id);
              } else {
                Status status;
                VCATCH_JSON_ERROR(meta, status,
                                  meta_tree::Exists(meta, id, exists));
                VINEYARD_SUPPRESS(status);
              }
              if (!exists) {
                return exists;
              }
            }
            return true;
          };
          auto eval_task = [self, ids, callback](const json& meta) -> Status {
            json sub_tree_group;
            for (auto const& id : ids) {
              json sub_tree;
              if (IsBlob(id)) {
                std::shared_ptr<Payload> object;
                auto status = self->bulk_store_->Get(id, object);
                if (status.ok()) {
                  sub_tree["id"] = ObjectIDToString(id);
                  sub_tree["typename"] = "vineyard::Blob";
                  sub_tree["length"] = object->data_size;
                  sub_tree["nbytes"] = object->data_size;
                  sub_tree["transient"] = true;
                  sub_tree["instance_id"] = self->instance_id();
                } else {
                  VLOG(10) << "Failed to find payload for blob: "
                           << ObjectIDToString(id)
                           << ", reason: " << status.ToString();
                }
              } else {
                Status s;
                VCATCH_JSON_ERROR(
                    meta, s,
                    meta_tree::GetData(meta, self->instance_name(), id,
                                       sub_tree, self->instance_id_));
                if (s.IsMetaTreeInvalid()) {
                  LOG(WARNING) << "Found errors in metadata: " << s;
                }
#if !defined(NDEBUG)
                if (VLOG_IS_ON(100)) {
                  DVLOG(100) << "Got request response:";
                  std::cerr << sub_tree.dump(4) << std::endl;
                  DVLOG(100) << "=========================================";
                }
#endif
              }
              if (sub_tree.is_object() && !sub_tree.empty()) {
                sub_tree_group[ObjectIDToString(id)] = sub_tree;
              }
            }
            return callback(Status::OK(), sub_tree_group);
          };
          if (!wait || test_task(meta)) {
            return eval_task(meta);
          } else {
            self->deferred_.emplace_back(alive, test_task, eval_task);
            return Status::OK();
          }
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::ListData(std::string const& pattern, bool const regex,
                                size_t const limit,
                                callback_t<const json&> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  meta_service_ptr_->RequestToGetData(
      false,  // no need for sync from etcd
      [self, pattern, regex, limit, callback](const Status& status,
                                              const json& meta) {
        if (status.ok()) {
          json sub_tree_group;
          Status s;
          VCATCH_JSON_ERROR(
              meta, s,
              meta_tree::ListData(meta, self->instance_name(), pattern, regex,
                                  limit, sub_tree_group));
          if (!s.ok()) {
            return callback(s, sub_tree_group);
          }
          size_t current = sub_tree_group.size();
          if (current < limit &&
              meta_tree::MatchTypeName(false, pattern, "vineyard::Blob")) {
            // consider returns blob when not reach the limit
            auto& blobs = self->bulk_store_->List();
            {
              auto locked = blobs.lock_table();
              for (auto const& item : locked) {
                if (current >= limit) {
                  break;
                }
                if (!item.second->IsSealed()) {
                  // skip unsealed blobs, otherwise `GetBuffers()` will fail on
                  // client after `ListData()`.
                  continue;
                }
                if (item.first ==
                    GenerateBlobID(std::numeric_limits<uintptr_t>::max())) {
                  // skip the dummy blob with the initialized blob id
                  continue;
                }
                std::string sub_tree_key = ObjectIDToString(item.first);
                json sub_tree;
                {
                  sub_tree["id"] = sub_tree_key;
                  sub_tree["typename"] = "vineyard::Blob";
                  sub_tree["length"] = item.second->data_size;
                  sub_tree["nbytes"] = item.second->data_size;
                  sub_tree["transient"] = true;
                  sub_tree["instance_id"] = self->instance_id();
                }
                sub_tree_group[sub_tree_key] = sub_tree;
                current += 1;
              }
            }
          }
          return callback(status, sub_tree_group);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return callback(status, json{});
        }
      });
  return Status::OK();
}

Status VineyardServer::ListAllData(
    callback_t<std::vector<ObjectID> const&> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  meta_service_ptr_->RequestToGetData(
      false,  // no need for sync from etcd
      [self, callback](const Status& status, const json& meta) {
        if (status.ok()) {
          std::vector<ObjectID> objects;
          Status s;
          VCATCH_JSON_ERROR(meta, s, meta_tree::ListAllData(meta, objects));
          if (!s.ok()) {
            return callback(s, objects);
          }
          auto& blobs = self->bulk_store_->List();
          {
            auto locked = blobs.lock_table();
            for (auto const& item : locked) {
              objects.emplace_back(item.first);
            }
          }
          return callback(status, objects);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return callback(status, {});
        }
      });
  return Status::OK();
}

Status VineyardServer::ListName(
    std::string const& pattern, bool const regex, size_t const limit,
    callback_t<const std::map<std::string, ObjectID>&> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  meta_service_ptr_->RequestToGetData(
      true, [pattern, regex, limit, callback](const Status& status,
                                              const json& meta) {
        if (status.ok()) {
          std::map<std::string, ObjectID> names;
          Status s;
          VCATCH_JSON_ERROR(
              meta, s, meta_tree::ListName(meta, pattern, regex, limit, names));
          return callback(s, names);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

namespace detail {

Status validate_metadata(const json& tree, json& result, Signature& signature) {
  // validate typename
  auto type_name_node = tree.value("typename", json(nullptr));
  if (type_name_node.is_null() || !type_name_node.is_string()) {
    return Status::MetaTreeInvalid("No 'typename' field in incoming metadata");
  }
  std::string const& type = type_name_node.get_ref<std::string const&>();

  RETURN_ON_ASSERT(type != "vineyard::Blob", "Blob has no metadata");

  // Check if instance_id information available
  RETURN_ON_ASSERT(tree.contains("instance_id"),
                   "The instance_id filed must be presented");
  result = tree;
  signature = GenerateSignature();
  if (result.find("signature") != result.end()) {
    signature = result["signature"].get<Signature>();
  } else {
    result["signature"] = signature;
  }

  // generate timestamp, in milliseconds.
  result["__timestamp"] =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  return Status::OK();
}

Status put_members_recursively(
    std::shared_ptr<IMetaService> metadata_service_ptr, const json& meta,
    json& tree, std::string const& instance_name) {
  for (auto& item : tree.items()) {
    if (item.value().is_object()) {
      auto& sub_tree = item.value();
      if (!sub_tree.contains("id")) {
        Signature signature;
        RETURN_ON_ERROR(validate_metadata(sub_tree, sub_tree, signature));

        // recursively create members
        RETURN_ON_ERROR(put_members_recursively(metadata_service_ptr, meta,
                                                sub_tree, instance_name));

        Status s;
        ObjectID id = GenerateObjectID();
        InstanceID computed_instance_id = 0;
        std::vector<meta_tree::op_t> ops;
        VCATCH_JSON_ERROR(
            sub_tree, s,
            meta_tree::PutDataOps(meta, instance_name, id, sub_tree, ops,
                                  computed_instance_id));
        RETURN_ON_ERROR(s);
        metadata_service_ptr->RequestToDirectUpdate(ops);

        // annotate id into the subtree
        sub_tree["id"] = ObjectIDToString(id);
      }
    }
  }
  return Status::OK();
}

}  // namespace detail

Status VineyardServer::CreateData(
    const json& tree,
    callback_t<const ObjectID, const Signature, const InstanceID> callback) {
  return CreateData(tree, false, callback);
}

Status VineyardServer::CreateData(
    const std::vector<json>& trees,
    callback_t<const std::vector<ObjectID>, const std::vector<Signature>,
               const std::vector<InstanceID>>
        callback) {
  return CreateData(trees, false, callback);
}

Status VineyardServer::CreateData(
    const json& tree, bool recursive,
    callback_t<const ObjectID, const Signature, const InstanceID> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  // update meta into json
  meta_service_ptr_->RequestToBulkUpdate(
      [self, tree, recursive](const Status& status, const json& meta,
                              std::vector<meta_tree::op_t>& ops, ObjectID& id,
                              Signature& signature,
                              InstanceID& computed_instance_id) {
        if (status.ok()) {
          auto decorated_tree = json::object();
          RETURN_ON_ERROR(
              detail::validate_metadata(tree, decorated_tree, signature));

          // expand trees: for putting many metadatas in a single call
          if (recursive) {
            RETURN_ON_ERROR(detail::put_members_recursively(
                self->meta_service_ptr_, meta, decorated_tree,
                self->instance_name_));
          }

          Status s;
          id = GenerateObjectID();
          VCATCH_JSON_ERROR(
              meta, s,
              meta_tree::PutDataOps(meta, self->instance_name(), id,
                                    decorated_tree, ops, computed_instance_id));
          return s;
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      },
      boost::bind(callback, _1, _2, _3, _4));
  return Status::OK();
}

Status VineyardServer::CreateData(
    const std::vector<json>& trees, bool recursive,
    callback_t<const std::vector<ObjectID>, const std::vector<Signature>,
               const std::vector<InstanceID>>
        callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  // update meta into json
  meta_service_ptr_->RequestToBulkUpdate(
      [self, trees, recursive](const Status& status, const json& meta,
                               std::vector<meta_tree::op_t>& ops,
                               std::vector<ObjectID>& ids,
                               std::vector<Signature>& signatures,
                               std::vector<InstanceID>& computed_instance_ids) {
        if (status.ok()) {
          std::vector<json> decorated_trees;
          for (auto const& tree : trees) {
            Signature signature;
            auto decorated_tree = json::object();
            RETURN_ON_ERROR(
                detail::validate_metadata(tree, decorated_tree, signature));
            signatures.emplace_back(signature);
            decorated_trees.emplace_back(decorated_tree);
          }

          // expand trees: for putting many metadatas in a single call
          if (recursive) {
            for (auto& decorated_tree : decorated_trees) {
              RETURN_ON_ERROR(detail::put_members_recursively(
                  self->meta_service_ptr_, meta, decorated_tree,
                  self->instance_name_));
            }
          }

          for (auto& decorated_tree : decorated_trees) {
            ObjectID id = GenerateObjectID();
            InstanceID computed_instance_id = UnspecifiedInstanceID();
            Status s;
            VCATCH_JSON_ERROR(meta, s,
                              meta_tree::PutDataOps(meta, self->instance_name(),
                                                    id, decorated_tree, ops,
                                                    computed_instance_id));
            RETURN_ON_ERROR(s);
            ids.emplace_back(id);
            computed_instance_ids.emplace_back(computed_instance_id);
          }
          return Status::OK();
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      },
      boost::bind(callback, _1, _2, _3, _4));
  return Status::OK();
}

Status VineyardServer::Persist(const ObjectID id, callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  RETURN_ON_ASSERT(!IsBlob(id), "The blobs cannot be persisted");
  meta_service_ptr_->RequestToPersist(
      [self, id](const Status& status, const json& meta,
                 std::vector<meta_tree::op_t>& ops) {
        if (status.ok()) {
          Status s;
          VCATCH_JSON_ERROR(
              meta, s,
              meta_tree::PersistOps(meta, self->instance_name(), id, ops));
          if (status.ok() && !ops.empty() &&
              self->spec_["sync_crds"].get<bool>()) {
            json tree;
            Status s;
            VCATCH_JSON_ERROR(
                meta, s,
                meta_tree::GetData(meta, self->instance_name(), id, tree));
            if (s.ok() && tree.is_object() && !tree.empty()) {
              auto kube = std::make_shared<Kubectl>(self->GetMetaContext());
              kube->CreateObject(meta["instances"], tree);
              kube->Finish();
            }
          }
          return s;
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      },
      callback);
  return Status::OK();
}

Status VineyardServer::IfPersist(const ObjectID id,
                                 callback_t<const bool> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
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
          Status s;
          VCATCH_JSON_ERROR(meta, s, meta_tree::IfPersist(meta, id, persist));
          return callback(s, persist);
        } else {
          VLOG(100) << "Error: " << status.ToString();
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
  auto self(shared_from_this());
  meta_service_ptr_->RequestToGetData(
      true, [id, callback](const Status& status, const json& meta) {
        if (status.ok()) {
          bool exists = false;
          Status s;
          VCATCH_JSON_ERROR(meta, s, meta_tree::Exists(meta, id, exists));
          return callback(s, exists);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::ShallowCopy(const ObjectID id,
                                   const json& extra_metadata,
                                   callback_t<const ObjectID> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  RETURN_ON_ASSERT(!IsBlob(id), "The blobs cannot be shallow copied");
  ObjectID target_id = GenerateObjectID();
  meta_service_ptr_->RequestToShallowCopy(
      [id, extra_metadata, target_id](const Status& status, const json& meta,
                                      std::vector<meta_tree::op_t>& ops,
                                      bool& transient) {
        if (status.ok()) {
          Status s;

          VCATCH_JSON_ERROR(
              meta, s,
              meta_tree::ShallowCopyOps(meta, id, extra_metadata, target_id,
                                        ops, transient));
          return s;
        } else {
          VLOG(100) << "Error: " << status.ToString();
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
                               const bool memory_trim, const bool fastpath,
                               callback_t<> callback) {
  return DelData(ids, force, deep, memory_trim, fastpath,
                 [callback](Status const& status,
                            std::vector<ObjectID> const& deleted_ids) {
                   return callback(status);
                 });
}

Status VineyardServer::DelData(
    const std::vector<ObjectID>& ids, const bool force, const bool deep,
    const bool memory_trim, const bool fastpath,
    callback_t<std::vector<ObjectID> const&> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  if (fastpath) {
    // forcely delete the given blobs: used for allocators
    for (auto const id : ids) {
      RETURN_ON_ASSERT(IsBlob(id),
                       "Fastpath deletion can only be applied to blobs");
    }
    context_.post([this, memory_trim, ids, callback] {
      for (auto const id : ids) {
        VINEYARD_DISCARD(bulk_store_->OnDelete(id, memory_trim));
      }
      VINEYARD_DISCARD(callback(Status::OK(), ids));
    });
    return Status::OK();
  }
  meta_service_ptr_->RequestToDelete(
      ids, force, deep, memory_trim,
      [self](const Status& status, const json& meta,
             std::vector<ObjectID> const& ids_to_delete,
             std::vector<meta_tree::op_t>& ops, bool& sync_remote) {
        if (status.ok()) {
          Status s;
          VCATCH_JSON_ERROR(
              meta, s,
              meta_tree::DelDataOps(meta, ids_to_delete, ops, sync_remote));
          if (status.ok() && !ops.empty() &&
              self->spec_["sync_crds"].get<bool>()) {
            for (auto const& id : ids_to_delete) {
              if (IsBlob(id)) {
                continue;
              }
              json tree;
              Status s;
              VCATCH_JSON_ERROR(
                  meta, s,
                  meta_tree::GetData(meta, self->instance_name(), id, tree));
              if (s.ok() && tree.is_object() && !tree.empty() &&
                  tree.value("persist", false)) {
                auto kube = std::make_shared<Kubectl>(self->GetMetaContext());
                kube->DeleteObject(tree);
                kube->Finish();
              }
            }
          }
          if (s.IsMetaTreeSubtreeNotExists()) {
            // ignore non-exist objects
            return Status::OK();
          }
          return s;
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      },
      callback);
  return Status::OK();
}

Status VineyardServer::DeleteBlobBatch(const std::set<ObjectID>& ids,
                                       const bool memory_trim) {
  for (auto object_id : ids) {
    VINEYARD_SUPPRESS(this->bulk_store_->OnDelete(object_id, memory_trim));
  }
  return Status::OK();
}

Status VineyardServer::DeleteAllAt(const json& meta,
                                   InstanceID const instance_id) {
  std::vector<ObjectID> objects_to_cleanup;
  Status status;
  VCATCH_JSON_ERROR(
      meta, status,
      meta_tree::FilterAtInstance(meta, instance_id, objects_to_cleanup));
  RETURN_ON_ERROR(status);
  return DelData(objects_to_cleanup, true, true, true, false /* fastpath */,
                 [](Status const& status) -> Status {
                   if (!status.ok()) {
                     VLOG(100) << "Error: failed during cleanup: "
                               << status.ToString();
                   }
                   return Status::OK();
                 });
}

Status VineyardServer::PutName(const ObjectID object_id,
                               const std::string& name, callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  meta_service_ptr_->RequestToPersist(
      [object_id, name](const Status& status, const json& meta,
                        std::vector<meta_tree::op_t>& ops) {
        if (status.ok()) {
          // TODO: do proper validation:
          // 1. global objects can have name, local ones cannot.
          // 2. the name-object_id mapping shouldn't be overwrite.

          // blob cannot have name
          if (IsBlob(object_id)) {
            return Status::Invalid("blobs cannot have name");
          }

          bool exists = false;
          {
            Status s;
            VCATCH_JSON_ERROR(meta, s,
                              meta_tree::Exists(meta, object_id, exists));
            VINEYARD_DISCARD(s);
          }
          if (!exists) {
            return Status::ObjectNotExists("failed to put name: object " +
                                           ObjectIDToString(object_id) +
                                           " doesn't exist");
          }

          bool persist = false;
          {
            Status s;
            VCATCH_JSON_ERROR(meta, s,
                              meta_tree::IfPersist(meta, object_id, persist));
            VINEYARD_DISCARD(s);
          }
          if (!persist) {
            return Status::Invalid(
                "transient objects cannot have name, please persist it first");
          }

          ops.emplace_back(meta_tree::op_t::Put("/names/" + name, object_id));
          ops.emplace_back(meta_tree::op_t::Put(
              "/data/" + ObjectIDToString(object_id) + "/__name",
              meta_tree::EncodeValue(name)));
          return Status::OK();
        } else {
          VLOG(100) << "Error: " << status.ToString();
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
  auto self(shared_from_this());
  meta_service_ptr_->RequestToGetData(true, [self, name, wait, alive, callback](
                                                const Status& status,
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
        return callback(Status::ObjectNotExists("failed to find name: " + name),
                        InvalidObjectID());
      };
      if (!wait || test_task(meta)) {
        return eval_task(meta);
      } else {
        self->deferred_.emplace_back(alive, test_task, eval_task);
        return Status::OK();
      }
    } else {
      VLOG(100) << "Error: " << status.ToString();
      return status;
    }
  });
  return Status::OK();
}

Status VineyardServer::DropName(const std::string& name,
                                callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  meta_service_ptr_->RequestToPersist(
      [name](const Status& status, const json& meta,
             std::vector<meta_tree::op_t>& ops) {
        if (status.ok()) {
          auto names = meta.value("names", json(nullptr));
          if (names.is_object()) {
            auto iter = names.find(name);
            if (iter != names.end()) {
              ops.emplace_back(
                  meta_tree::op_t::Del("/names/" + escape_json_pointer(name)));
              auto object_id = iter->get<ObjectID>();
              // delete the name in the object meta as well.
              bool exists = false;
              {
                Status s;
                VCATCH_JSON_ERROR(meta, s,
                                  meta_tree::Exists(meta, object_id, exists));
                VINEYARD_DISCARD(s);
              }

              if (exists) {
                ops.emplace_back(meta_tree::op_t::Del(
                    "/data/" + ObjectIDToString(object_id) + "/__name"));
              }
            }
          }
          return Status::OK();
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      },
      callback);
  return Status::OK();
}

Status VineyardServer::MigrateObject(const ObjectID object_id,
                                     callback_t<const ObjectID&> callback) {
  ENSURE_VINEYARDD_READY();
  if (IsBlob(object_id)) {
    return callback(Status::Invalid("blobs cannot be directly migrated"),
                    InvalidObjectID());
  }
  auto self(shared_from_this());
  meta_service_ptr_->RequestToGetData(
      true /* sync remote */,
      [self, callback, object_id](const Status& status, const json& meta) {
        if (status.ok()) {
          Status s;
          json metadata;
          VCATCH_JSON_ERROR(
              meta, s,
              meta_tree::GetData(meta, self->instance_name(), object_id,
                                 metadata, self->instance_id_));
          if (!s.ok()) {
            return callback(s, InvalidObjectID());
          }
          if (metadata.value("global", false)) {
            return callback(
                Status::Invalid("global objects cannot be directly migrated"),
                InvalidObjectID());
          }
          InstanceID remote_instance_id =
              metadata.value("instance_id", UnspecifiedInstanceID());
          if (remote_instance_id == self->instance_id_) {
            // no need for migration
            return callback(Status::OK(), object_id);
          }
          if (remote_instance_id == UnspecifiedInstanceID()) {
            return callback(
                Status::Invalid("the location of object " +
                                ObjectIDToString(object_id) +
                                " cannot be resolved from metadata"),
                InvalidObjectID());
          }

          // find the remote instance endpoint
          json const& instances = meta["instances"];
          std::string key = "i" + std::to_string(remote_instance_id);
          json::const_iterator instance = instances.find(key);
          if (instance == instances.end()) {
            return callback(
                Status::Invalid("the remote instances doesn't exist"),
                InvalidObjectID());
          }

          std::string remote_endpoint =
              (*instance)["rpc_endpoint"].get_ref<std::string const&>();
          // push to the async queues
          boost::asio::post(
              self->GetIOContext(),
              [self, callback, remote_endpoint, object_id, metadata]() {
                auto remote = std::make_shared<RemoteClient>(self);
                RETURN_ON_ERROR(
                    remote->Connect(remote_endpoint, self->session_id()));
                return remote->MigrateObject(
                    object_id, metadata,
                    [self, remote, callback](const Status& status,
                                             const ObjectID result) {
                      return callback(status, result);
                    });
              });
          return Status::OK();
        } else {
          VLOG(100) << "Error: " << status.ToString();
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::TryAcquireLock(std::string& key,
                                      callback_t<bool, std::string> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  meta_service_ptr_->TryAcquireLock(
      key, [self, callback](const Status& status, bool result,
                            std::string actual_key) {
        if (status.ok()) {
          return callback(status, result, actual_key);
        } else {
          return callback(status, result, actual_key);
        }
      });

  return Status::OK();
}

Status VineyardServer::TryReleaseLock(std::string& key,
                                      callback_t<bool> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  meta_service_ptr_->TryReleaseLock(
      key, [self, callback](const Status& status, bool result) {
        if (status.ok()) {
          return callback(status, result);
        } else {
          return status;
        }
      });
  return Status::OK();
}

Status VineyardServer::LabelObjects(const ObjectID object_id,
                                    const std::vector<std::string>& keys,
                                    const std::vector<std::string>& values,
                                    callback_t<> callback) {
  ENSURE_VINEYARDD_READY();
  auto self(shared_from_this());
  return GetData(
      std::vector<ObjectID>{object_id}, false, false,
      [self]() -> bool {
        return self->ready_ == kReady && (!self->stopped_.load());
      },
      [self, callback, object_id, keys, values](const Status& status,
                                                const json& tree) {
        if (!status.ok()) {
          return callback(status);
        }
        if (!tree.contains(ObjectIDToString(object_id))) {
          return callback(Status::ObjectNotExists(
              "object " + ObjectIDToString(object_id) + " doesn't exist"));
        }
        auto const& metadata = tree[ObjectIDToString(object_id)];
        bool is_transient = metadata.value("transient", true);
        const std::string labels = metadata.value("__labels", "{}");
        json labels_object;
        Status s;
        CATCH_JSON_ERROR(labels_object, s, json::parse(labels));
        if (!s.ok()) {
          return callback(s);
        }
        for (size_t i = 0; i < keys.size(); ++i) {
          labels_object[keys[i]] = values[i];
        }
        std::string label_string = meta_tree::EncodeValue(labels_object.dump());
        if (is_transient) {
          self->meta_service_ptr_->RequestToBulkUpdate(
              [callback, object_id, label_string](
                  const Status& status, const json&,
                  std::vector<meta_tree::op_t>& ops, ObjectID&, Signature&,
                  InstanceID&) {
                if (!status.ok()) {
                  return callback(status);
                }
                ops.emplace_back(meta_tree::op_t::Put(
                    "/data/" + ObjectIDToString(object_id) + "/__labels",
                    label_string));
                return Status::OK();
              },
              [callback](const Status& status, const ObjectID, const Signature,
                         const InstanceID) { return callback(status); });
        } else {
          self->meta_service_ptr_->RequestToPersist(
              [callback, object_id, label_string](
                  const Status& status, const json&,
                  std::vector<meta_tree::op_t>& ops) {
                if (!status.ok()) {
                  return callback(status);
                }
                ops.emplace_back(meta_tree::op_t::Put(
                    "/data/" + ObjectIDToString(object_id) + "/__labels",
                    label_string));
                return Status::OK();
              },
              [callback](const Status& status) { return callback(status); });
        }
        return Status::OK();
      });
}

namespace detail {
static void traverse_local_blobs(const json& tree, const InstanceID instance,
                                 std::set<ObjectID>& objects) {
  if (!tree.is_object() || tree.empty()) {
    return;
  }
  std::string invalid_object_id = ObjectIDToString(InvalidObjectID());
  ObjectID member_id = ObjectIDFromString(tree.value("id", invalid_object_id));
  if (IsBlob(member_id)) {
    if (instance == UnspecifiedInstanceID() ||
        instance == tree.value("instance_id", UnspecifiedInstanceID())) {
      objects.emplace(member_id);
    }
  } else {
    for (auto const& item : tree) {
      if (item.is_object()) {
        traverse_local_blobs(item, instance, objects);
      }
    }
  }
}
}  // namespace detail

Status VineyardServer::EvictObjects(const std::vector<ObjectID>& ids,
                                    callback_t<> callback) {
  auto self(shared_from_this());
  return GetData(
      ids, false, false,
      [self]() -> bool {
        return self->ready_ == kReady && (!self->stopped_.load());
      },
      [self](const Status& status, const json& tree) {
        std::set<ObjectID> objects;
        for (auto const& item : tree) {
          if (item.is_object()) {
            detail::traverse_local_blobs(item, self->instance_id_, objects);
          }
        }
        std::map<ObjectID, std::shared_ptr<Payload>> payloads;
        for (auto const& id : objects) {
          std::shared_ptr<Payload> payload;
          if (self->bulk_store_->Get(id, payload).ok()) {
            payloads.emplace(id, payload);
          }
        }
        return self->bulk_store_->SpillColdObjects(payloads);
      });
}

Status VineyardServer::LoadObjects(const std::vector<ObjectID>& ids,
                                   const bool pin, callback_t<> callback) {
  auto self(shared_from_this());
  return GetData(
      ids, false, false,
      [self]() -> bool {
        return self->ready_ == kReady && (!self->stopped_.load());
      },
      [self, pin](const Status& status, const json& tree) {
        std::set<ObjectID> objects;
        for (auto const& item : tree) {
          if (item.is_object()) {
            detail::traverse_local_blobs(item, self->instance_id_, objects);
          }
        }
        std::map<ObjectID, std::shared_ptr<Payload>> payloads;
        for (auto const& id : objects) {
          std::shared_ptr<Payload> payload;
          if (self->bulk_store_->Get(id, payload).ok()) {
            payloads.emplace(id, payload);
          }
        }
        return self->bulk_store_->ReloadColdObjects(payloads, pin);
      });
}

Status VineyardServer::UnpinObjects(const std::vector<ObjectID>& ids,
                                    callback_t<> callback) {
  auto self(shared_from_this());
  return GetData(
      ids, false, false,
      [self]() -> bool {
        return self->ready_ == kReady && (!self->stopped_.load());
      },
      [self](const Status& status, const json& tree) {
        std::set<ObjectID> objects;
        for (auto const& item : tree) {
          if (item.is_object()) {
            detail::traverse_local_blobs(item, self->instance_id_, objects);
          }
        }
        std::map<ObjectID, std::shared_ptr<Payload>> payloads;
        for (auto const& id : objects) {
          std::shared_ptr<Payload> payload;
          if (self->bulk_store_->Get(id, payload).ok()) {
            payload->Unpin();
          }
        }
        return Status::OK();
      });
}

Status VineyardServer::ClusterInfo(callback_t<const json&> callback) {
  ENSURE_VINEYARDD_READY();
  auto self = shared_from_this();
  meta_service_ptr_->RequestToGetData(
      true, [callback](const Status& status, const json& meta) {
        if (status.ok()) {
          return callback(status, meta["instances"]);
        } else {
          VLOG(100) << "Error: " << status.ToString();
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

Status VineyardServer::Verify(const std::string& username,
                              const std::string& password,
                              callback_t<> callback) {
  const std::string htpasswd = spec_.value("htpasswd", /* default */ "");
  if (htpasswd == "") {
    if (!username.empty() || !password.empty()) {
      LOG(WARNING) << "Authentication is not enabled, ignored";
    }
    return callback(Status::OK());
  }
  if (!ghc::filesystem::exists(htpasswd)) {
    return callback(
        Status::IOError("Failed to find the htpasswd database for verifying"));
  }
  std::shared_ptr<Process> proc = std::make_shared<Process>(context_);
  proc->Start("htpasswd", std::vector<std::string>{
                              "-v",
                              "-b",
                              htpasswd,
                              username,
                              password,
                          });
  proc->Wait();
  if (proc->ExitCode() == 0) {
    return callback(Status::OK());
  }
  std::stringstream m;
  m << "Incorrect username and password, or, htpasswd is not "
       "installed as expected: \n";
  m << "\n";
  m << "  - ubuntu: apt-get install apache2-utils\n";
  m << "  - centos: yum install httpd-tools\n";
  m << "\n";
  if (!proc->Diagnostic().empty()) {
    m << "Detail error: ";
    for (auto const& s : proc->Diagnostic()) {
      m << s << "\n";
    }
  }
  return callback(Status::IOError(m.str()));
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

  if (this->ipc_server_ptr_) {
    this->ipc_server_ptr_->Stop();
  }
  if (this->rpc_server_ptr_) {
    this->rpc_server_ptr_->Stop();
  }
  if (this->meta_service_ptr_) {
    this->meta_service_ptr_->Stop();
  }

  // cleanup
  this->ipc_server_ptr_.reset();
  this->rpc_server_ptr_.reset();
  this->meta_service_ptr_.reset();
  this->stream_store_.reset();
  this->bulk_store_.reset();
  this->plasma_bulk_store_.reset();
}

bool VineyardServer::Running() const { return !stopped_.load(); }

VineyardServer::~VineyardServer() { this->Stop(); }

}  // namespace vineyard
