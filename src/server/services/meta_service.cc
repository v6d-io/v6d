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

#include "server/services/meta_service.h"

#include <algorithm>
#include <memory>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/trim.hpp"

#include "common/util/env.h"
#include "common/util/functions.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "server/services/local_meta_service.h"
#if defined(BUILD_VINEYARDD_ETCD)
#include "server/services/etcd_meta_service.h"
#endif  // BUILD_VINEYARDD_ETCD
#if defined(BUILD_VINEYARDD_REDIS)
#include "server/services/redis_meta_service.h"
#endif  // BUILD_VINEYARDD_REDIS
#include "server/util/meta_tree.h"
#include "server/util/metrics.h"

namespace vineyard {

std::shared_ptr<IMetaService> IMetaService::Get(
    std::shared_ptr<VineyardServer> server_ptr) {
  std::string meta = server_ptr->GetSpec()["metastore_spec"]["meta"]
                         .get_ref<const std::string&>();
  VINEYARD_ASSERT(false
#if defined(BUILD_VINEYARDD_ETCD)
                      || meta == "etcd"
#endif  // BUILD_VINEYARDD_ETCD
#if defined(BUILD_VINEYARDD_REDIS)
                      || meta == "redis"
#endif  // BUILD_VINEYARDD_REDIS
                      || meta == "local",
                  "Invalid metastore: " + meta);
#if defined(BUILD_VINEYARDD_ETCD)
  if (meta == "etcd") {
    return std::shared_ptr<IMetaService>(new EtcdMetaService(server_ptr));
  }
#endif  // BUILD_VINEYARDD_ETCD
#if defined(BUILD_VINEYARDD_REDIS)
  if (meta == "redis") {
    return std::shared_ptr<IMetaService>(new RedisMetaService(server_ptr));
  }
#endif  // BUILD_VINEYARDD_REDIS
  if (meta == "local") {
    return std::shared_ptr<IMetaService>(new LocalMetaService(server_ptr));
  }
  return nullptr;
}

IMetaService::~IMetaService() { this->Stop(); }

void IMetaService::Stop() {
  if (this->stopped_.exchange(true)) {
    return;
  }
  LOG(INFO) << "meta service is stopping ...";
}

Status IMetaService::Start() {
  LOG(INFO) << "meta service is starting, waiting the metadata backend "
               "service becoming ready ...";
  RETURN_ON_ERROR(this->preStart());
  auto current = std::chrono::system_clock::now();
  auto timeout =
      std::chrono::seconds(server_ptr_->GetSpec()["metastore_spec"].value(
          "meta_timeout", 60 /* 1 minutes */));
  Status s;
  while (std::chrono::system_clock::now() - current < timeout) {
    if (this->stopped_.load()) {
      return Status::AlreadyStopped("etcd metadata service");
    }
    s = this->probe();
    if (s.ok()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  RETURN_ON_ERROR(s);
  auto self(shared_from_this());
  requestValues(
      "", [self](const Status& status, const json& meta, unsigned rev) {
        if (self->stopped_.load()) {
          return Status::AlreadyStopped("etcd metadata service");
        }
        if (status.ok()) {
          // start the watcher.
          self->startDaemonWatch("", self->rev_,
                                 boost::bind(&IMetaService::daemonWatchHandler,
                                             self, _1, _2, _3, _4));

          // register self info.
          self->registerToEtcd();
        } else {
          Status s = status;
          s << "Failed to get initial value";
          // Abort: since the probe has succeeded but the etcd
          // doesn't work, we have no idea about what happened.
          s.Abort();
        }
        return status;
      });
  return Status::OK();
}

void IMetaService::RequestToBulkUpdate(
    callback_t<const json&, std::vector<op_t>&, ObjectID&, Signature&,
               InstanceID&>
        callback_after_ready,
    callback_t<const ObjectID, const Signature, const InstanceID>
        callback_after_finish) {
  auto self(shared_from_this());
  server_ptr_->GetMetaContext().post(
      [self, callback_after_ready, callback_after_finish]() {
        ObjectID object_id;
        Signature signature;
        InstanceID computed_instance_id;
        if (self->stopped_.load()) {
          VINEYARD_SUPPRESS(callback_after_finish(
              Status::AlreadyStopped("etcd metadata service"),
              InvalidObjectID(), InvalidSignature(), UnspecifiedInstanceID()));
          return;
        }
        std::vector<op_t> ops;
        auto status =
            callback_after_ready(Status::OK(), self->meta_, ops, object_id,
                                 signature, computed_instance_id);
        if (status.ok()) {
#ifndef NDEBUG
          // debugging
          self->printDepsGraph();
#endif
          self->metaUpdate(ops, false);
        } else {
          VLOG(100) << "Error: failed to generated ops to update metadata: "
                    << status.ToString();
        }
        VINEYARD_SUPPRESS(callback_after_finish(status, object_id, signature,
                                                computed_instance_id));
      });
}

void IMetaService::RequestToBulkUpdate(
    callback_t<const json&, std::vector<op_t>&, std::vector<ObjectID>&,
               std::vector<Signature>&, std::vector<InstanceID>&>
        callback_after_ready,
    callback_t<const std::vector<ObjectID>, const std::vector<Signature>,
               const std::vector<InstanceID>>
        callback_after_finish) {
  auto self(shared_from_this());
  server_ptr_->GetMetaContext().post(
      [self, callback_after_ready, callback_after_finish]() {
        std::vector<ObjectID> object_ids;
        std::vector<Signature> signatures;
        std::vector<InstanceID> computed_instance_ids;

        if (self->stopped_.load()) {
          VINEYARD_SUPPRESS(callback_after_finish(
              Status::AlreadyStopped("etcd metadata service"), object_ids,
              signatures, computed_instance_ids));
          return;
        }
        std::vector<op_t> ops;
        auto status =
            callback_after_ready(Status::OK(), self->meta_, ops, object_ids,
                                 signatures, computed_instance_ids);
        if (status.ok()) {
#ifndef NDEBUG
          // debugging
          self->printDepsGraph();
#endif
          self->metaUpdate(ops, false);
        } else {
          VLOG(100) << "Error: failed to generated ops to update metadata: "
                    << status.ToString();
        }
        VINEYARD_SUPPRESS(callback_after_finish(status, object_ids, signatures,
                                                computed_instance_ids));
      });
}

// When requesting direct update, we already worked inside the meta context.
void IMetaService::RequestToDirectUpdate(std::vector<op_t> const& ops,
                                         const bool from_remote) {
  this->metaUpdate(ops, from_remote);
}

void IMetaService::RequestToPersist(
    callback_t<const json&, std::vector<op_t>&> callback_after_ready,
    callback_t<> callback_after_finish) {
  // NB: when persist local meta to etcd, we needs the meta_sync_lock_ to
  // avoid contention between other vineyard instances.
  auto self(shared_from_this());
  this->requestLock(
      meta_sync_lock_, [self, callback_after_ready, callback_after_finish](
                           const Status& status, std::shared_ptr<ILock> lock) {
        if (self->stopped_.load()) {
          return Status::AlreadyStopped("etcd metadata service");
        }
        if (status.ok()) {
          self->requestValues(
              "", [self, callback_after_ready, callback_after_finish, lock](
                      const Status& status, const json& meta, unsigned rev) {
                if (self->stopped_.load()) {
                  return Status::AlreadyStopped("etcd metadata service");
                }
                std::vector<op_t> ops;
                auto s = callback_after_ready(status, meta, ops);
                if (s.ok()) {
                  if (ops.empty()) {
                    unsigned rev_after_unlock = 0;
                    VINEYARD_DISCARD(lock->Release(rev_after_unlock));
                    return callback_after_finish(Status::OK());
                  }
                  // apply changes locally before committing to etcd
                  self->metaUpdate(ops, false);
                  // commit to etcd
                  self->commitUpdates(ops, [self, callback_after_finish, lock](
                                               const Status& status,
                                               unsigned rev) {
                    if (self->stopped_.load()) {
                      return Status::AlreadyStopped("etcd metadata service");
                    }
                    // update rev_ to the revision after unlock.
                    unsigned rev_after_unlock = 0;
                    VINEYARD_DISCARD(lock->Release(rev_after_unlock));
                    return callback_after_finish(status);
                  });
                  return Status::OK();
                } else {
                  unsigned rev_after_unlock = 0;
                  VINEYARD_DISCARD(lock->Release(rev_after_unlock));
                  return callback_after_finish(s);  // propagate the error
                }
              });
          return Status::OK();
        } else {
          VLOG(100) << "Error: failed to request metadata lock: "
                    << status.ToString();
          return callback_after_finish(status);  // propagate the error
        }
      });
}

void IMetaService::RequestToGetData(const bool sync_remote,
                                    callback_t<const json&> callback) {
  if (sync_remote) {
    requestValues(
        "", [callback](const Status& status, const json& meta, unsigned rev) {
          return callback(status, meta);
        });
  } else {
    // post the task to asio queue as well for well-defined processing order.
    //
    // Note that we need to pass `meta_` as reference, see also:
    //
    //    https://www.boost.org/doc/libs/1_73_0/libs/bind/doc/html/bind.html
    server_ptr_->GetMetaContext().post(
        boost::bind(callback, Status::OK(), std::ref(meta_)));
  }
}

void IMetaService::RequestToDelete(
    const std::vector<ObjectID>& object_ids, const bool force, const bool deep,
    const bool memory_trim,
    callback_t<const json&, std::vector<ObjectID> const&, std::vector<op_t>&,
               bool&>
        callback_after_ready,
    callback_t<std::vector<ObjectID> const&> callback_after_finish) {
  auto self(shared_from_this());
  server_ptr_->GetMetaContext().post([self, object_ids, force, deep,
                                      memory_trim, callback_after_ready,
                                      callback_after_finish]() {
    if (self->stopped_.load()) {
      VINEYARD_DISCARD(callback_after_finish(
          Status::AlreadyStopped("etcd metadata service"), {}));
    }

    // generated ops.
    std::vector<op_t> ops;

    bool sync_remote = false;
    std::vector<ObjectID> processed_delete_set;
    self->findDeleteSet(object_ids, processed_delete_set, force, deep);

#ifndef NDEBUG
    if (VLOG_IS_ON(10)) {
      for (auto const& item : processed_delete_set) {
        VLOG(10) << "deleting object: " << ObjectIDToString(item);
      }
    }
#endif

    auto s = callback_after_ready(Status::OK(), self->meta_,
                                  processed_delete_set, ops, sync_remote);
    if (!s.ok()) {
      VINEYARD_DISCARD(callback_after_finish(s, {}));
      return;
    }

    // apply changes locally (before committing to etcd)
    self->metaUpdate(ops, false, memory_trim);

    if (!sync_remote) {
      VINEYARD_DISCARD(callback_after_finish(s, processed_delete_set));
      return;
    }

    // apply remote updates
    //
    // NB: when persist local meta to etcd, we needs the meta_sync_lock_ to
    // avoid contention between other vineyard instances.
    self->requestLock(
        self->meta_sync_lock_,
        [self, ops /* by copy */, callback_after_ready, processed_delete_set,
         callback_after_finish](const Status& status,
                                std::shared_ptr<ILock> lock) {
          if (self->stopped_.load()) {
            return Status::AlreadyStopped("etcd metadata service");
          }
          if (status.ok()) {
            // commit to etcd
            self->commitUpdates(
                ops, [self, processed_delete_set, callback_after_finish, lock](
                         const Status& status, unsigned rev) {
                  if (self->stopped_.load()) {
                    return Status::AlreadyStopped("etcd metadata service");
                  }
                  // update rev_ to the revision after unlock.
                  unsigned rev_after_unlock = 0;
                  VINEYARD_DISCARD(lock->Release(rev_after_unlock));
                  return callback_after_finish(status, processed_delete_set);
                });
            return Status::OK();
          } else {
            VLOG(100) << "Error: failed to request metadata lock: "
                      << status.ToString();
            return callback_after_finish(status, {});  // propagate the error.
          }
        });
  });
}

void IMetaService::RequestToShallowCopy(
    callback_t<const json&, std::vector<op_t>&, bool&> callback_after_ready,
    callback_t<> callback_after_finish) {
  auto self(shared_from_this());
  requestValues("", [self, callback_after_ready, callback_after_finish](
                        const Status& status, const json& meta, unsigned rev) {
    if (self->stopped_.load()) {
      return Status::AlreadyStopped("etcd metadata service");
    }
    if (status.ok()) {
      std::vector<op_t> ops;
      bool transient = true;
      auto status = callback_after_ready(Status::OK(), meta, ops, transient);
      if (status.ok()) {
        if (transient) {
          // Already trim physical memory for remote deletion events
          self->metaUpdate(ops, false, true);
          return callback_after_finish(Status::OK());
        } else {
          self->RequestToPersist(
              [self, ops](const Status& status, const json& meta,
                          std::vector<op_t>& persist_ops) {
                if (self->stopped_.load()) {
                  return Status::AlreadyStopped("etcd metadata service");
                }
                persist_ops.insert(persist_ops.end(), ops.begin(), ops.end());
                return Status::OK();
              },
              callback_after_finish);
          return Status::OK();
        }
      } else {
        VLOG(100) << "Error: failed to generated ops to update metadata: "
                  << status.ToString();
        return callback_after_finish(status);
      }
    } else {
      VLOG(100) << "Error: request values failed: " << status.ToString();
      return callback_after_finish(status);
    }
  });
}

/** Note [Deleting objects and blobs]
 *
 * Blob is special: suppose A -> B and A -> C, where A is an object, B is an
 * object as well, but C is a blob, when delete(A, deep=False), B will won't be
 * touched but C will be checked (and possible deleted) as well.
 *
 * That is because, for remote object, when it being delete (with deep), the
 * deletion of the blobs cannot be reflected in watched etcd events.
 */

/**
 * Note [Deleting global object and members]
 *
 * Because of migration deleting member of global objects is tricky, as, assume
 * we have
 *
 *    A -> B      -- h1
 *      -> C      -- h2
 *
 * then migration happens,
 *
 *    A -> B      -- h1
 *      -> C      -- h2
 *      -> C'     -- h1
 *
 * and if we delete C with (deep=true, force=false), A, B and C' should still be
 * kept as they still construct a complete object.
 *
 * To archive this, we
 *
 * - in `incRef(A, B)`: if `B` is a signature, we just record the object id
 *   instead.
 *
 * - in `findDeleteSet` (in `deleteable`), an object is deleteable, when
 *
 *      * it is not used, or
 *      * it has an *equivalent* object in the metatree, whether it is a local
 *        or remote object. Here *equivalent* means they has the same signature.
 *
 *   Note that in the second case, the signature of *equivalent* object must has
 *   already been pushed into "/signatures/", as we have another assumption that
 *   only member of global objects can be migrated, see also `MigrateObject` in
 *   vineyard_server.cc.
 */

void IMetaService::IncRef(std::string const& instance_name,
                          std::string const& key, std::string const& value,
                          const bool from_remote) {
  std::vector<std::string> vs;
  boost::algorithm::split(vs, key, [](const char c) { return c == '/'; });
  if (vs[0].empty()) {
    vs.erase(vs.begin());
  }
  if (vs.size() < 2 || vs[0] != "data") {
    // The key is not an object id: data.id
    return;
  }
  ObjectID key_obj, value_obj;
  if (meta_tree::DecodeObjectID(meta_, instance_name, value, value_obj).ok()) {
    key_obj = ObjectIDFromString(vs[1]);
    if (from_remote && IsBlob(value_obj)) {
      // don't put remote blob refs into deps graph, since two blobs may share
      // the same object id.
      return;
    }
    {
      // validate the dependency graph
      decltype(subobjects_.begin()) iter;
      auto range = subobjects_.equal_range(key_obj);
      for (iter = range.first; iter != range.second; ++iter) {
        if (iter->second == value_obj) {
          break;
        }
      }
      if (iter == range.second) {
        subobjects_.emplace(key_obj, value_obj);
      }
    }
    {
      // validate the dependency graph
      decltype(supobjects_.begin()) iter;
      auto range = supobjects_.equal_range(value_obj);
      for (iter = range.first; iter != range.second; ++iter) {
        if (iter->second == key_obj) {
          break;
        }
      }
      if (iter == range.second) {
        supobjects_.emplace(value_obj, key_obj);
      }
    }
  }
}

void IMetaService::CloneRef(ObjectID const target, ObjectID const mirror) {
  // avoid repeatedly clone
  VLOG(10) << "[" << server_ptr_->instance_id() << "] clone ref: "
           << ": " << ObjectIDToString(target) << " -> "
           << ObjectIDToString(mirror);
  if (supobjects_.find(mirror) != supobjects_.end()) {
    return;
  }
  auto range = supobjects_.equal_range(target);
  std::vector<ObjectID> suprefs;
  // n.b.: avoid traverse & modify at the same time (in the same loop).
  for (auto iter = range.first; iter != range.second; ++iter) {
    suprefs.emplace_back(iter->second);
  }
  for (auto const supref : suprefs) {
    supobjects_.emplace(mirror, supref);
    subobjects_.emplace(supref, mirror);
  }
}

void IMetaService::registerToEtcd() {
  auto self(shared_from_this());
  RequestToPersist(
      [self](const Status& status, const json& tree, std::vector<op_t>& ops) {
        if (self->stopped_.load()) {
          return Status::AlreadyStopped("etcd metadata service");
        }
        if (status.ok()) {
          std::string hostname = get_hostname(), nodename = get_nodename();

          int64_t timestamp = GetTimestamp();

          self->instances_list_.clear();
          uint64_t self_host_id =
              static_cast<uint64_t>(gethostid()) | detail::cycleclock::now();
          if (tree.contains("instances") && !tree["instances"].is_null()) {
            for (auto& instance : tree["instances"].items()) {
              auto id =
                  static_cast<InstanceID>(std::stoul(instance.key().substr(1)));
              self->instances_list_.emplace(id);
            }
          }
          InstanceID rank = 0;
          if (tree.contains("next_instance_id") &&
              !tree["next_instance_id"].is_null()) {
            rank = tree["next_instance_id"].get<InstanceID>();
          }

          self->server_ptr_->set_instance_id(rank);
          self->server_ptr_->set_hostname(hostname);
          self->server_ptr_->set_nodename(nodename);

          // store an entry in the meta tree
          self->meta_["my_instance_id"] = rank;
          self->meta_["my_hostname"] = hostname;
          self->meta_["my_nodename"] = nodename;

          self->instances_list_.emplace(rank);
          std::string key = "/instances/" + self->server_ptr_->instance_name();
          ops.emplace_back(op_t::Put(key + "/hostid", self_host_id));
          ops.emplace_back(op_t::Put(key + "/hostname", hostname));
          ops.emplace_back(op_t::Put(key + "/nodename", nodename));
          ops.emplace_back(op_t::Put(key + "/rpc_endpoint",
                                     self->server_ptr_->RPCEndpoint()));
          ops.emplace_back(
              op_t::Put(key + "/ipc_socket", self->server_ptr_->IPCSocket()));
          ops.emplace_back(op_t::Put(key + "/timestamp", timestamp));
          ops.emplace_back(op_t::Put("/next_instance_id", rank + 1));
          LOG(INFO) << "Decide to set rank as " << rank;
          return status;
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      [self](const Status& status) {
        if (self->stopped_.load()) {
          return Status::AlreadyStopped("etcd metadata service");
        }
        if (status.ok()) {
          // start heartbeat
          VINEYARD_DISCARD(startHeartbeat(self, Status::OK()));
          // mark meta service as ready
          self->Ready();
        } else {
          self->server_ptr_->set_instance_id(UINT64_MAX);
          LOG(ERROR) << "compute instance_id error.";
        }
        return status;
      });
}

void IMetaService::checkInstanceStatus(
    std::shared_ptr<IMetaService> const& self,
    callback_t<> callback_after_finish) {
  self->RequestToPersist(
      [self](const Status& status, const json& tree, std::vector<op_t>& ops) {
        if (self->stopped_.load()) {
          return Status::AlreadyStopped("etcd metadata service");
        }
        if (status.ok()) {
          ops.emplace_back(op_t::Put(
              "/instances/" + self->server_ptr_->instance_name() + "/timestamp",
              GetTimestamp()));
          return status;
        } else {
          LOG(ERROR) << status.ToString();
          return status;
        }
      },
      [self, callback_after_finish](const Status& status) {
        if (self->stopped_.load()) {
          return Status::AlreadyStopped("etcd metadata service");
        }
        if (!status.ok()) {
          LOG(ERROR) << "Failed to refresh self: " << status.ToString();
          return callback_after_finish(status);
        }
        auto the_next =
            self->instances_list_.upper_bound(self->server_ptr_->instance_id());
        if (the_next == self->instances_list_.end()) {
          the_next = self->instances_list_.begin();
        }
        InstanceID target_inst = *the_next;
        if (target_inst == self->server_ptr_->instance_id()) {
          return callback_after_finish(status);
        }
        VLOG(10) << "Instance size " << self->instances_list_.size()
                 << ", target instance is " << target_inst;
        auto target =
            self->meta_["instances"]["i" + std::to_string(target_inst)];
        // The subtree might be empty, when the etcd been resumed with another
        // data directory but the same endpoint. that leads to a crash here
        // but we just let it crash to help us diagnosis the error.
        if (!target.is_null() /* && !target.empty() */) {
          int64_t ts = target["timestamp"].get<int64_t>();
          if (ts == self->target_latest_time_) {
            ++self->timeout_count_;
          } else {
            self->timeout_count_ = 0;
          }
          self->target_latest_time_ = ts;
          if (self->timeout_count_ >= MAX_TIMEOUT_COUNT) {
            LOG(ERROR) << "Instance " << target_inst << " timeout";
            self->timeout_count_ = 0;
            self->target_latest_time_ = 0;
            self->RequestToPersist(
                [self, target_inst](const Status& status, const json& tree,
                                    std::vector<op_t>& ops) {
                  if (self->stopped_.load()) {
                    return Status::AlreadyStopped("etcd metadata service");
                  }
                  if (status.ok()) {
                    std::string key =
                        "/instances/i" + std::to_string(target_inst);
                    ops.emplace_back(op_t::Del(key + "/hostid"));
                    ops.emplace_back(op_t::Del(key + "/timestamp"));
                    ops.emplace_back(op_t::Del(key + "/hostname"));
                    ops.emplace_back(op_t::Del(key + "/nodename"));
                    ops.emplace_back(op_t::Del(key + "/rpc_endpoint"));
                    ops.emplace_back(op_t::Del(key + "/ipc_socket"));
                  } else {
                    LOG(ERROR) << status.ToString();
                  }
                  return status;
                },
                [self, callback_after_finish](const Status& status) {
                  if (self->stopped_.load()) {
                    return Status::AlreadyStopped("etcd metadata service");
                  }
                  return callback_after_finish(status);
                });
            VINEYARD_SUPPRESS(
                self->server_ptr_->DeleteAllAt(self->meta_, target_inst));
            return status;
          } else {
            return callback_after_finish(status);
          }
        } else {
          return callback_after_finish(status);
        }
      });
}

Status IMetaService::startHeartbeat(std::shared_ptr<IMetaService> const& self,
                                    Status const&) {
  self->heartbeat_timer_.reset(
      new asio::steady_timer(self->server_ptr_->GetMetaContext(),
                             std::chrono::seconds(HEARTBEAT_TIME)));
  self->heartbeat_timer_->async_wait(
      [self](const boost::system::error_code& error) {
        if (self->stopped_.load()) {
          return;
        }
        if (error) {
          LOG(ERROR) << "heartbeat timer error: " << error << ", "
                     << error.message();
        }
        if (!error || error != boost::system::errc::operation_canceled) {
          // run check, and start the next round in the finish callback.
          checkInstanceStatus(
              self, boost::bind(&IMetaService::startHeartbeat, self, _1));
        }
      });
  return Status::OK();
}

void IMetaService::requestValues(const std::string& prefix,
                                 callback_t<const json&, unsigned> callback) {
  // We still need to run a `etcdctl get` for the first time. With a
  // long-running and no compact Etcd, watching from revision 0 may
  // lead to a super huge amount of events, which is unacceptable.
  auto self(shared_from_this());
  if (rev_ == 0) {
    requestAll(prefix, rev_,
               [self, callback](const Status& status,
                                const std::vector<op_t>& ops, unsigned rev) {
                 if (self->stopped_.load()) {
                   return Status::AlreadyStopped("etcd metadata service");
                 }
                 if (status.ok()) {
                   self->metaUpdate(ops, true);
                   self->rev_ = rev;
                 }
                 return callback(status, self->meta_, self->rev_);
               });
  } else {
    requestUpdates(
        prefix, rev_,
        [self, callback](const Status& status, const std::vector<op_t>& ops,
                         unsigned rev) {
          if (self->stopped_.load()) {
            return Status::AlreadyStopped("etcd metadata service");
          }
          if (status.ok()) {
            self->metaUpdate(ops, true);
            self->rev_ = rev;
          }
          return callback(status, self->meta_, self->rev_);
        });
  }
}

bool IMetaService::deleteable(ObjectID const object_id) {
  if (object_id == InvalidObjectID()) {
    return true;
  }
  if (supobjects_.find(object_id) == supobjects_.end()) {
    return true;
  }
  ObjectID equivalent = InvalidObjectID();
  return meta_tree::HasEquivalent(meta_, object_id, equivalent);
}

void IMetaService::traverseToDelete(std::set<ObjectID>& initial_delete_set,
                                    std::set<ObjectID>& delete_set,
                                    int32_t depth,
                                    std::map<ObjectID, int32_t>& depthes,
                                    const ObjectID object_id, const bool force,
                                    const bool deep) {
  // emulate a topological sort to ensure the correctness when deleting multiple
  // objects at the same time.
  if (delete_set.find(object_id) != delete_set.end()) {
    // already been processed
    if (depthes[depth] < depth) {
      depthes[depth] = depth;
    }
    return;
  }
  // process the "initial_delete_set" in topo-sort order.
  auto sup_target_range = supobjects_.equal_range(object_id);
  std::set<ObjectID> sup_target_to_preprocess;
  for (auto it = sup_target_range.first; it != sup_target_range.second; ++it) {
    if (initial_delete_set.find(it->second) != initial_delete_set.end()) {
      sup_target_to_preprocess.emplace(it->second);
    }
  }
  for (ObjectID const& sup_target : sup_target_to_preprocess) {
    traverseToDelete(initial_delete_set, delete_set, depth + 1, depthes,
                     sup_target, force, deep);
  }
  if (force || deleteable(object_id)) {
    delete_set.emplace(object_id);
    depthes[object_id] = depth;
    {
      // delete downwards
      std::set<ObjectID> to_delete;
      {
        // delete sup-edges of subobjects
        auto range = subobjects_.equal_range(object_id);
        for (auto it = range.first; it != range.second; ++it) {
          // remove dependency edge
          auto suprange = supobjects_.equal_range(it->second);
          decltype(suprange.first) p;
          for (p = suprange.first; p != suprange.second; /* no self-inc */) {
            if (p->second == object_id) {
              supobjects_.erase(p++);
            } else {
              ++p;
            }
          }
          if (deep || IsBlob(it->second)) {
            // blob is special: see Note [Deleting objects and blobs].
            to_delete.emplace(it->second);
          }
        }
      }

      {
        // delete sub-edges of supobjects
        auto range = supobjects_.equal_range(object_id);
        for (auto it = range.first; it != range.second; ++it) {
          // remove dependency edge
          auto subrange = subobjects_.equal_range(it->second);
          decltype(subrange.first) p;
          for (p = subrange.first; p != subrange.second; /* no self-inc */) {
            if (p->second == object_id) {
              subobjects_.erase(p++);
            } else {
              ++p;
            }
          }
        }
      }

      for (auto const& target : to_delete) {
        traverseToDelete(initial_delete_set, delete_set, depth - 1, depthes,
                         target, false, true);
      }
    }
    if (force) {
      // delete upwards
      std::set<ObjectID> to_delete;
      auto range = supobjects_.equal_range(object_id);
      for (auto it = range.first; it != range.second; ++it) {
        // remove dependency edge
        auto subrange = subobjects_.equal_range(it->second);
        decltype(subrange.first) p;
        for (p = subrange.first; p != subrange.second; /* no self-inc */) {
          if (p->second == object_id) {
            subobjects_.erase(p++);
          } else {
            ++p;
          }
        }
        if (force) {
          to_delete.emplace(it->second);
        }
      }
      if (force) {
        for (auto const& target : to_delete) {
          traverseToDelete(initial_delete_set, delete_set, depth + 1, depthes,
                           target, true, false);
        }
      }
    }
    subobjects_.erase(object_id);
    supobjects_.erase(object_id);
  }
  initial_delete_set.erase(object_id);
}

void IMetaService::findDeleteSet(std::vector<ObjectID> const& object_ids,
                                 std::vector<ObjectID>& processed_delete_set,
                                 bool force, bool deep) {
  // implements dependent-based (usage-based) lifecycle: find the delete set.
  std::set<ObjectID> initial_delete_set{object_ids.begin(), object_ids.end()};
  std::set<ObjectID> delete_set;
  std::map<ObjectID, int32_t> depthes;
  for (auto const object_id : object_ids) {
    traverseToDelete(initial_delete_set, delete_set, 0, depthes, object_id,
                     force, deep);
  }
  postProcessForDelete(delete_set, depthes, processed_delete_set);
}

/**
 * N.B.: all object ids are guaranteed to exist in `depthes`.
 */
void IMetaService::postProcessForDelete(
    const std::set<ObjectID>& delete_set,
    const std::map<ObjectID, int32_t>& depthes,
    std::vector<ObjectID>& delete_objects) {
  delete_objects.assign(delete_set.begin(), delete_set.end());
  std::stable_sort(delete_objects.begin(), delete_objects.end(),
                   [&depthes](const ObjectID& x, const ObjectID& y) {
                     return depthes.at(x) > depthes.at(y);
                   });
}

void IMetaService::printDepsGraph() {
  if (!VLOG_IS_ON(100)) {
    return;
  }
  std::stringstream ss;
  ss << "object top -> down dependencies: " << std::endl;
  for (auto const& kv : subobjects_) {
    ss << ObjectIDToString(kv.first) << " -> " << ObjectIDToString(kv.second)
       << std::endl;
  }
  ss << "object down <- top dependencies: " << std::endl;
  for (auto const& kv : supobjects_) {
    ss << ObjectIDToString(kv.first) << " <- " << ObjectIDToString(kv.second)
       << std::endl;
  }
  VLOG(100) << "Dependencies graph on " << server_ptr_->instance_name()
            << ": \n"
            << ss.str();
}

void IMetaService::putVal(const kv_t& kv, bool const from_remote) {
  // don't crash the server for any reason (any potential garbage value)
  auto upsert_to_meta = [&]() -> Status {
    json value = json::parse(kv.value);
    if (value.is_string()) {
      IncRef(server_ptr_->instance_name(), kv.key,
             value.get_ref<std::string const&>(), from_remote);
    } else if (value.is_object() && !value.empty()) {
      for (auto const& item : value.items()) {
        if (item.value().is_string()) {
          IncRef(server_ptr_->instance_name(), kv.key,
                 item.value().get_ref<std::string const&>(), from_remote);
        }
      }
    }
    // NB: inserting (with `operator[]`) using json pointer is truly unsafe.
    Status status;
    CATCH_JSON_ERROR_STATEMENT(status,
                               meta_[json::json_pointer(kv.key)] = value);
    if (!status.ok()) {
      return Status::Invalid("Failed to insert to metadata: key = '" + kv.key +
                             "', value = '" + value.dump(4) +
                             "', reason: " + status.ToString());
    }
    return Status::OK();
  };

  auto upsert_sig_to_meta = [&]() -> Status {
    json value = json::parse(kv.value);
    if (value.is_string()) {
      ObjectID object_id =
          ObjectIDFromString(value.get_ref<std::string const&>());
      ObjectID equivalent = InvalidObjectID();
      std::string signature_key = kv.key.substr(kv.key.find_last_of("/") + 1);
      if (meta_tree::HasEquivalentWithSignature(
              meta_, SignatureFromString(signature_key), object_id,
              equivalent)) {
        CloneRef(equivalent, object_id);
      }
    } else {
      LOG(ERROR) << "Invalid signature record: " << kv.key << " -> "
                 << kv.value;
    }
    return Status::OK();
  };

  // update signatures
  if (boost::algorithm::starts_with(kv.key, "/signatures/")) {
    if (!from_remote || !meta_.contains(json::json_pointer(kv.key))) {
      Status status;
      CATCH_JSON_ERROR(status, upsert_to_meta());
      VINEYARD_LOG_ERROR(status);
    }
    Status status;
    CATCH_JSON_ERROR(status, upsert_sig_to_meta());
    VINEYARD_LOG_ERROR(status);
    return;
  }

  // update names
  if (boost::algorithm::starts_with(kv.key, "/names/")) {
    if (!from_remote && meta_.contains(json::json_pointer(kv.key))) {
      LOG(WARNING) << "Warning: name got overwritten: " << kv.key;
    }
    Status status;
    CATCH_JSON_ERROR(status, upsert_to_meta());
    VINEYARD_LOG_ERROR(status);
    return;
  }

  // update ordinary data
  Status status;
  CATCH_JSON_ERROR(status, upsert_to_meta());
  VINEYARD_LOG_ERROR(status);
}

void IMetaService::delVal(std::string const& key) {
  auto path = json::json_pointer(key);
  if (meta_.contains(path)) {
    auto ppath = path.parent_pointer();
    meta_[ppath].erase(path.back());
    if (meta_[ppath].empty()) {
      meta_[ppath.parent_pointer()].erase(ppath.back());
    }
  }
}

void IMetaService::delVal(const kv_t& kv) { delVal(kv.key); }

void IMetaService::delVal(ObjectID const& target, std::set<ObjectID>& blobs) {
  if (target == InvalidObjectID()) {
    return;
  }
  auto targetkey = json::json_pointer("/data/" + ObjectIDToString(target));
  if (deleteable(target)) {
    // if deletable blob: delete blob
    if (IsBlob(target)) {
      blobs.emplace(target);
    }
    delVal(targetkey);
  } else if (target != InvalidObjectID()) {
    // mark as transient
    if (meta_.contains(targetkey)) {
      meta_[targetkey]["transient"] = true;
    } else {
      LOG(ERROR) << "invalid metatree state: '" << targetkey << "' not found";
    }
  }
}

template <class RangeT>
void IMetaService::metaUpdate(const RangeT& ops, const bool from_remote,
                              const bool memory_trim) {
  std::set<ObjectID> blobs_to_delete;

  std::vector<op_t> add_sigs, drop_sigs;
  std::vector<op_t> add_objects, drop_objects;
  std::vector<op_t> add_others, drop_others;

  // group-by all changes
  for (const op_t& op : ops) {
    if (op.kv.rev != 0 && op.kv.rev <= rev_) {
#ifndef NDEBUG
      if (from_remote && op.kv.rev <= rev_) {
        LOG(WARNING) << "skip updates: " << op.ToString();
      }
#endif
      // revision resolution: means this revision has already been updated
      // the revision value 0 means local update ops.
      continue;
    }
    if (boost::algorithm::trim_copy(op.kv.key).empty()) {
      // skip unprintable keys
      continue;
    }
    if (boost::algorithm::starts_with(op.kv.key, meta_sync_lock_)) {
      // skip the update of etcd lock
      continue;
    }

    // update instance status
    if (boost::algorithm::starts_with(op.kv.key, "/instances/")) {
      instanceUpdate(op, from_remote);
    }

#ifndef NDEBUG
    if (from_remote) {
      VLOG(11) << "update op in meta tree: " << op.ToString();
    }
#endif

    if (boost::algorithm::starts_with(op.kv.key, "/signatures/")) {
      if (op.op == op_t::op_type_t::kPut) {
        add_sigs.emplace_back(op);
      } else if (op.op == op_t::op_type_t::kDel) {
        drop_sigs.emplace_back(op);
      } else {
        LOG(ERROR) << "warn: unknown op type for signatures: " << op.op;
      }
    } else if (boost::algorithm::starts_with(op.kv.key, "/data/")) {
      if (op.op == op_t::op_type_t::kPut) {
        add_objects.emplace_back(op);
      } else if (op.op == op_t::op_type_t::kDel) {
        drop_objects.emplace_back(op);
      } else {
        LOG(ERROR) << "warn: unknown op type for objects: " << op.op;
      }
    } else {
      if (op.op == op_t::op_type_t::kPut) {
        add_others.emplace_back(op);
      } else if (op.op == op_t::op_type_t::kDel) {
        drop_others.emplace_back(op);
      } else {
        LOG(ERROR) << "warn: unknown op type for others: " << op.op;
      }
    }
  }

  // apply adding signature mappings first.
  for (const op_t& op : add_sigs) {
    putVal(op.kv, from_remote);
  }

  // apply adding others
  for (const op_t& op : add_others) {
    putVal(op.kv, from_remote);
  }

  // apply adding objects
  for (const op_t& op : add_objects) {
    putVal(op.kv, from_remote);
  }

  // apply drop objects
  {
    // 1. collect all ids
    std::set<ObjectID> initial_delete_set;
    std::vector<std::string> vs;
    for (const op_t& op : drop_objects) {
      vs.clear();
      boost::algorithm::split(vs, op.kv.key,
                              [](const char c) { return c == '/'; });
      if (vs[0].empty()) {
        vs.erase(vs.begin());
      }
      // `__name` is our injected properties, and will be erased during
      // `DropName`.
      if (vs.size() >= 3 && vs[2] == "__name") {
        // move the key to `drop_others` to drop
        drop_others.emplace_back(op);
      } else {
        initial_delete_set.emplace(ObjectIDFromString(vs[1]));
      }
    }
    std::vector<ObjectID> object_ids{initial_delete_set.begin(),
                                     initial_delete_set.end()};

    // 2. traverse to find the delete set
    std::vector<ObjectID> processed_delete_set;
    findDeleteSet(object_ids, processed_delete_set, false, false);

#ifndef NDEBUG
    if (VLOG_IS_ON(10)) {
      for (auto const& item : processed_delete_set) {
        VLOG(10) << "deleting object (in meta update): "
                 << ObjectIDToString(item);
      }
    }
#endif

    // 3. execute delete for every object
    for (auto const target : processed_delete_set) {
      delVal(target, blobs_to_delete);
    }
  }

  // apply drop others
  for (const op_t& op : drop_others) {
    delVal(op.kv);
  }

  // apply drop signatures
  for (const op_t& op : drop_sigs) {
    delVal(op.kv);
  }

#ifndef NDEBUG
  // debugging
  printDepsGraph();
  for (auto const& id : blobs_to_delete) {
    LOG(INFO) << "blob to delete: " << ObjectIDToString(id);
  }
#endif

  VINEYARD_SUPPRESS(server_ptr_->DeleteBlobBatch(blobs_to_delete, memory_trim));
  VINEYARD_SUPPRESS(server_ptr_->ProcessDeferred(meta_));
}

void IMetaService::instanceUpdate(const op_t& op, const bool from_remote) {
  std::vector<std::string> key_segments;
  boost::split(key_segments, op.kv.key, boost::is_any_of("/"));
  if (key_segments[0].empty()) {
    key_segments.erase(key_segments.begin());
  }
  if (key_segments[2] == "hostid") {
    uint64_t instance_id = std::stoul(key_segments[1].substr(1));
    if (op.op == op_t::op_type_t::kPut) {
      if (from_remote) {
        LOG(INFO) << "Instance join: " << instance_id;
      }
      instances_list_.emplace(instance_id);
    } else if (op.op == op_t::op_type_t::kDel) {
      if (from_remote) {
        LOG(INFO) << "Instance exit: " << instance_id;
      }
      instances_list_.erase(instance_id);
    } else {
      if (from_remote) {
        LOG(ERROR) << "Unknown op type: " << op.ToString();
      }
    }
    LOG_SUMMARY("instances_total", "", instances_list_.size());
  }
}

Status IMetaService::daemonWatchHandler(
    std::shared_ptr<IMetaService> self, const Status& status,
    const std::vector<op_t>& ops, unsigned rev,
    callback_t<unsigned> callback_after_update) {
  // `this` must be non-stopped in this handler, as the {Etcd}WatchHandler
  // keeps a reference of `this` (std::shared_ptr<EtcdMetaService>).
  if (self->stopped_.load()) {
    return Status::AlreadyStopped("etcd metadata service");
  }
  // Guarantee: all kvs inside a txn reaches the client at the same time,
  // which is guaranteed by the implementation of etcd.
  //
  // That means, every time this handler is called, we just need to response
  // for one type of change.
  if (!status.ok()) {
    LOG(ERROR) << "Error in daemon watching: " << status.ToString();
    return callback_after_update(status, rev);
  }
  if (ops.empty()) {
    return callback_after_update(Status::OK(), rev);
  }
  // process events grouped by revision
  size_t idx = 0;
  std::vector<op_t> op_batch;
  while (idx < ops.size()) {
    unsigned head_index = ops[idx].kv.rev;
    while (idx < ops.size() && ops[idx].kv.rev == head_index) {
      op_batch.emplace_back(ops[idx]);
      idx += 1;
    }
    self->metaUpdate(op_batch, true);
    op_batch.clear();
    self->rev_ = head_index;
  }
  return callback_after_update(Status::OK(), rev);
}

}  // namespace vineyard
