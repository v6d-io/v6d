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

#ifndef SRC_SERVER_SERVICES_META_SERVICE_H_
#define SRC_SERVER_SERVICES_META_SERVICE_H_

#include <sys/param.h>

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "boost/asio.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/bind.hpp"
#include "boost/range/iterator_range.hpp"

#include "common/util/boost.h"
#include "common/util/callback.h"
#include "common/util/functions.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "server/server/vineyard_server.h"

#define HEARTBEAT_TIME 20
#define MAX_TIMEOUT_COUNT 3

namespace vineyard {

namespace asio = boost::asio;

/**
 * @brief ILock is the base class of EtcdLock
 *
 */
class ILock {
 public:
  explicit ILock(unsigned rev) : rev_(rev) {}
  virtual ~ILock() {}
  // return revision after unlock
  virtual Status Release(unsigned&) = 0;
  inline unsigned GetRev() { return rev_; }

 private:
  const unsigned rev_;
};

/**
 * @brief IMetaService is the base class of EtcdMetaService
 *
 */
class IMetaService {
 public:
  struct kv_t {
    std::string key;
    std::string value;
    unsigned rev;
  };

  struct op_t {
    enum op_type_t : unsigned { kPut = 0, kDel = 1 } op;
    kv_t kv;
    std::string ToString() const {
      std::stringstream ss;
      ss.str("");
      ss.clear();
      ss << ((op == kPut) ? "put " : "del ");
      ss << "[" << kv.rev << "] " << kv.key << " -> " << kv.value;
      return ss.str();
    }

    static op_t Del(std::string const& key) {
      return op_t{.op = op_type_t::kDel,
                  .kv = kv_t{.key = key, .value = "", .rev = 0}};
    }
    static op_t Del(std::string const& key, unsigned const rev) {
      return op_t{.op = op_type_t::kDel,
                  .kv = kv_t{.key = key, .value = "", .rev = rev}};
    }
    // send to etcd
    template <typename T>
    static op_t Put(std::string const& key, T const& value) {
      return op_t{
          .op = op_type_t::kPut,
          .kv =
              kv_t{.key = key, .value = json_to_string(json(value)), .rev = 0}};
    }
    template <typename T>
    static op_t Put(std::string const& key, json const& value) {
      return op_t{
          .op = op_type_t::kPut,
          .kv = kv_t{.key = key, .value = json_to_string(value), .rev = 0}};
    }
    // receive from etcd
    static op_t Put(std::string const& key, std::string const& value,
                    unsigned const rev) {
      return op_t{.op = op_type_t::kPut,
                  .kv = kv_t{.key = key, .value = value, .rev = rev}};
    }
  };

  struct watcher_t {
    watcher_t(callback_t<const json&, const std::string&> w,
              const std::string& t)
        : watcher(w), tag(t) {}
    callback_t<const json&, const std::string&> watcher;
    std::string tag;
  };
  virtual ~IMetaService() {}
  explicit IMetaService(vs_ptr_t& server_ptr)
      : server_ptr_(server_ptr), rev_(0), meta_sync_lock_("/meta_sync_lock") {}

  static std::shared_ptr<IMetaService> Get(vs_ptr_t);

  inline Status Start() {
    LOG(INFO) << "start!";
    RETURN_ON_ERROR(this->probe());
    rev_ = 0;
    requestValues("",
                  [this](const Status& status, const json& meta, unsigned rev) {
                    if (status.ok()) {
                      this->registerToEtcd();
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

  virtual inline void Stop() { LOG(INFO) << "stop!"; }

 public:
  inline void RequestToBulkUpdate(
      callback_t<const json&, std::vector<op_t>&, InstanceID&>
          callback_after_ready,
      callback_t<const InstanceID> callback_after_finish) {
    server_ptr_->GetIOContext().post([this, callback_after_ready,
                                      callback_after_finish]() {
      std::vector<op_t> ops;
      InstanceID computed_instance_id;
      auto status =
          callback_after_ready(Status::OK(), meta_, ops, computed_instance_id);
      if (status.ok()) {
#ifndef NDEBUG
        // debugging
        if (VLOG_IS_ON(10)) {
          printDepsGraph();
        }
#endif
        this->metaUpdate(ops, false);
      } else {
        LOG(ERROR) << status.ToString();
      }
      VINEYARD_SUPPRESS(callback_after_finish(status, computed_instance_id));
    });
  }

  inline void RequestToPersist(
      callback_t<const json&, std::vector<op_t>&> callback_after_ready,
      callback_t<> callback_after_finish) {
    // NB: when persist local meta to etcd, we needs the meta_sync_lock_ to
    // avoid contention between other vineyard instances.
    this->requestLock(
        meta_sync_lock_,
        [this, callback_after_ready, callback_after_finish](
            const Status& status, std::shared_ptr<ILock> lock) {
          if (status.ok()) {
            requestValues(
                "", [this, callback_after_ready, callback_after_finish, lock](
                        const Status& status, const json& meta, unsigned rev) {
                  std::vector<op_t> ops;
                  auto s = callback_after_ready(status, meta, ops);
                  if (s.ok()) {
                    if (ops.empty()) {
                      unsigned rev_after_unlock = 0;
                      if (lock->Release(rev_after_unlock).ok()) {
                        rev_ = rev_after_unlock;
                      }
                      return callback_after_finish(Status::OK());
                    }
                    // apply changes locally before committing to etcd
                    this->metaUpdate(ops, true);
                    // commit to etcd
                    this->commitUpdates(
                        ops, [this, callback_after_finish, lock](
                                 const Status& status, unsigned rev) {
                          // update rev_ to the revision after unlock.
                          unsigned rev_after_unlock = 0;
                          if (lock->Release(rev_after_unlock).ok()) {
                            rev_ = rev_after_unlock;
                          }
                          return callback_after_finish(status);
                        });
                    return Status::OK();
                  } else {
                    unsigned rev_after_unlock = 0;
                    if (lock->Release(rev_after_unlock).ok()) {
                      rev_ = rev_after_unlock;
                    }
                    return callback_after_finish(s);  // propogate the error
                  }
                });
            return Status::OK();
          } else {
            LOG(ERROR) << status.ToString();
            return callback_after_finish(status);  // propogate the error
          }
        });
  }

  inline void RequestToGetData(const bool sync_remote,
                               callback_t<const json&> callback) {
    if (sync_remote) {
      requestValues(
          "", [callback](const Status& status, const json& meta, unsigned rev) {
            return callback(status, meta);
          });
    } else {
      // post the task to asio queue as well for well-defined processing order.
      server_ptr_->GetIOContext().post(
          boost::bind(callback, Status::OK(), meta_));
    }
  }

  inline void RequestToDelete(
      const std::vector<ObjectID>& ids, const bool force, const bool deep,
      callback_t<const json&, std::set<ObjectID> const&, std::vector<op_t>&>
          callback_after_ready,
      callback_t<> callback_after_finish) {
    // NB: when persist local meta to etcd, we needs the meta_sync_lock_ to
    // avoid contention between other vineyard instances.
    this->requestLock(
        meta_sync_lock_,
        [this, ids, force, deep, callback_after_ready, callback_after_finish](
            const Status& status, std::shared_ptr<ILock> lock) {
          if (status.ok()) {
            rev_ = lock->GetRev();
            requestValues(
                "", [this, ids, force, deep, callback_after_ready,
                     callback_after_finish, lock](
                        const Status& status, const json& meta, unsigned rev) {
                  // Implements dependent-based (usage-based) lifecycle.
                  std::set<ObjectID> initial_delete_set{ids.begin(), ids.end()};
                  std::set<ObjectID> delete_set;
                  for (auto const object_id : ids) {
                    traverseToDelete(initial_delete_set, delete_set, object_id,
                                     force, deep);
                  }
                  std::vector<op_t> ops;
                  auto s = callback_after_ready(status, meta, delete_set, ops);
                  if (s.ok()) {
                    // apply changes locally before committing to etcd
                    this->metaUpdate(ops, true);
                    // commit to etcd
                    this->commitUpdates(
                        ops, [this, callback_after_finish, lock](
                                 const Status& status, unsigned rev) {
                          // update rev_ to the revision after unlock.
                          unsigned rev_after_unlock = 0;
                          if (lock->Release(rev_after_unlock).ok()) {
                            rev_ = rev_after_unlock;
                          }
                          return callback_after_finish(status);
                        });
                    return Status::OK();
                  } else {
                    unsigned rev_after_unlock = 0;
                    if (lock->Release(rev_after_unlock).ok()) {
                      rev_ = rev_after_unlock;
                    }
                    return callback_after_finish(s);  // propogate the error.
                  }
                });
            return Status::OK();
          } else {
            LOG(ERROR) << status.ToString();
            return callback_after_finish(status);  // propogate the error.
          }
        });
  }

  inline void RequestToShallowCopy(
      callback_t<const json&, std::vector<op_t>&, bool&> callback_after_ready,
      callback_t<> callback_after_finish) {
    requestValues("", [this, callback_after_ready, callback_after_finish](
                          const Status& status, const json& meta,
                          unsigned rev) {
      if (status.ok()) {
        std::vector<op_t> ops;
        bool transient = true;
        auto status = callback_after_ready(Status::OK(), meta, ops, transient);
        if (status.ok()) {
          if (transient) {
            this->metaUpdate(ops, true);
            return callback_after_finish(Status::OK());
          } else {
            this->RequestToPersist(
                [ops](const Status& status, const json& meta,
                      std::vector<IMetaService::op_t>& persist_ops) {
                  persist_ops.insert(persist_ops.end(), ops.begin(), ops.end());
                  return Status::OK();
                },
                callback_after_finish);
            return Status::OK();
          }
        } else {
          LOG(ERROR) << status.ToString();
          return callback_after_finish(status);
        }
      } else {
        LOG(ERROR) << "request values failed: " << status.ToString();
        return callback_after_finish(status);
      }
    });
  }

 private:
  inline void registerToEtcd() {
    RequestToPersist(
        [&](const Status& status, const json& tree, std::vector<op_t>& ops) {
          if (status.ok()) {
            char hostname_value[MAXHOSTNAMELEN];
            gethostname(&hostname_value[0], MAXHOSTNAMELEN);
            std::string hostname = std::string(hostname_value);
            int64_t timestamp = GetTimestamp();

            instances_list_.clear();
            uint64_t self_host_id = static_cast<uint64_t>(gethostid()) |
                                    static_cast<uint64_t>(__rdtsc());
            if (tree.contains("instances") && !tree["instances"].is_null()) {
              for (auto& instance : json::iterator_wrapper(tree["instances"])) {
                auto id = static_cast<InstanceID>(
                    std::stoul(instance.key().substr(1)));
                instances_list_.emplace(id);
              }
            }
            InstanceID rank = 0;
            if (tree.contains("next_instance_id") &&
                !tree["next_instance_id"].is_null()) {
              rank = tree["next_instance_id"].get<InstanceID>();
            }
            instances_list_.emplace(rank);
            ops.emplace_back(op_t::Put(
                "/instances/i" + std::to_string(rank) + "/" + "hostid",
                self_host_id));
            ops.emplace_back(op_t::Put(
                "/instances/i" + std::to_string(rank) + "/" + "hostname",
                hostname));
            ops.emplace_back(op_t::Put(
                "/instances/i" + std::to_string(rank) + "/" + "timestamp",
                timestamp));
            ops.emplace_back(op_t::Put("/next_instance_id", rank + 1));
            this->server_ptr_->set_instance_id(rank);
            LOG(INFO) << "Decide to set rank as " << rank;
            return status;
          } else {
            LOG(ERROR) << status.ToString();
            return status;
          }
        },
        [&](const Status& status) {
          if (status.ok()) {
            // start the watcher.
            LOG(INFO) << "start background etcd watch, since " << rev_;
            this->startDaemonWatch(
                "", rev_,
                boost::bind(&IMetaService::daemonWatchHandler, this, _1, _2,
                            _3));
            // start heartbeat
            this->startHeartbeat();
            // mark meta service as ready
            Ready();
          } else {
            this->server_ptr_->set_instance_id(UINT64_MAX);
            LOG(ERROR) << "compute instance_id error.";
          }
          return status;
        });
  }

  /**
   * Watch rules:
   *
   *  - every instance checks the status of the "NEXT" instance;
   *  - the last one watches for the first one;
   *  - if there's only one instance, it does nothing.
   */
  void checkInstanceStatus() {
    RequestToPersist(
        [&](const Status& status, const json& tree, std::vector<op_t>& ops) {
          if (status.ok()) {
            ops.emplace_back(op_t::Put(
                "/instances/i" + std::to_string(server_ptr_->instance_id()) +
                    "/" + "timestamp",
                GetTimestamp()));
            return status;
          } else {
            LOG(ERROR) << status.ToString();
            return status;
          }
        },
        [&](const Status& status) {
          if (!status.ok()) {
            LOG(ERROR) << "Failed to refresh self: " << status.ToString();
            return status;
          }
          auto the_next =
              instances_list_.upper_bound(server_ptr_->instance_id());
          if (the_next == instances_list_.end()) {
            the_next = instances_list_.begin();
          }
          InstanceID target_inst = *the_next;
          if (target_inst == server_ptr_->instance_id()) {
            return Status::OK();
          }
          VLOG(10) << "Instance size " << instances_list_.size()
                   << ", target instance is " << target_inst;
          auto target = meta_["instances"]["i" + std::to_string(target_inst)];
          // The subtree might be empty, when the etcd been resumed with another
          // data directory but the same endpoint. that leads to a crash here
          // but we just let it crash to help us diagnosis the error.
          if (!target.is_null() /* && !target.empty() */) {
            int64_t ts = target["timestamp"].get<int64_t>();
            if (ts == target_latest_time_) {
              ++timeout_count_;
            } else {
              timeout_count_ = 0;
            }
            target_latest_time_ = ts;
            if (timeout_count_ >= MAX_TIMEOUT_COUNT) {
              LOG(ERROR) << "Instance " << target_inst << " timeout";
              timeout_count_ = 0;
              target_latest_time_ = 0;
              RequestToPersist(
                  [&, target_inst](const Status& status, const json& tree,
                                   std::vector<op_t>& ops) {
                    if (status.ok()) {
                      std::string key =
                          "/instances/i" + std::to_string(target_inst);
                      ops.emplace_back(op_t::Del(key + "/hostid"));
                      ops.emplace_back(op_t::Del(key + "/timestamp"));
                      ops.emplace_back(op_t::Del(key + "/hostname"));
                    } else {
                      LOG(ERROR) << status.ToString();
                    }
                    return status;
                  },
                  [&](const Status& status) { return status; });
              VINEYARD_SUPPRESS(server_ptr_->DeleteAllAt(meta_, target_inst));
            }
          }
          return status;
        });
  }

  void startHeartbeat() {
    heartbeat_timer_.reset(new asio::steady_timer(
        server_ptr_->GetIOContext(), std::chrono::seconds(HEARTBEAT_TIME)));
    heartbeat_timer_->async_wait([&](const boost::system::error_code& error) {
      if (error) {
        LOG(ERROR) << "heartbeat timer error: " << error << ", "
                   << error.message();
      }
      // run check
      checkInstanceStatus();
      // run the next round
      startHeartbeat();
    });
  }

 protected:
  // invoke when everything is ready (after Start() and ready for invoking)
  inline void Ready() {
    server_ptr_->MetaReady();  // notify server the meta svc is ready
  }

  virtual void commitUpdates(const std::vector<op_t>&,
                             callback_t<unsigned> callback_after_updated) = 0;

  void requestValues(const std::string& prefix,
                     callback_t<const json&, unsigned> callback) {
    // We still need to run a `etcdctl get` for the first time. With a
    // long-running and no compact Etcd, watching from revision 0 may
    // lead to a super huge amount of events, which is unacceptable.
    if (rev_ == 0) {
      requestAll(prefix, rev_,
                 [this, callback](const Status& status,
                                  const std::vector<op_t>& ops, unsigned rev) {
                   if (status.ok()) {
                     this->metaUpdate(ops, true);
                     rev_ = rev;
                   }
                   return callback(status, meta_, rev_);
                 });
    } else {
      requestUpdates(
          prefix, rev_,
          [this, callback](const Status& status, const std::vector<op_t>& ops,
                           unsigned rev) {
            if (status.ok()) {
              this->metaUpdate(ops, true);
              rev_ = rev;
            }
            return callback(status, meta_, rev_);
          });
    }
  }

  virtual void requestLock(
      std::string lock_name,
      callback_t<std::shared_ptr<ILock>> callback_after_locked) = 0;

  virtual void requestAll(
      const std::string& prefix, unsigned base_rev,
      callback_t<const std::vector<op_t>&, unsigned> callback) = 0;

  virtual void requestUpdates(
      const std::string& prefix, unsigned since_rev,
      callback_t<const std::vector<op_t>&, unsigned> callback) = 0;

  virtual void startDaemonWatch(
      const std::string& prefix, unsigned since_rev,
      callback_t<const std::vector<op_t>&, unsigned> callback) = 0;

  // validate the liveness of the underlying meta service.
  virtual Status probe() = 0;

  void incRef(std::string const& key, std::string const& value);
  void printDepsGraph();

  json meta_;
  vs_ptr_t server_ptr_;

  unsigned rev_;
  bool backend_retrying_;

  std::string meta_sync_lock_;

 private:
  bool deleteable(ObjectID const object_id);

  void traverseToDelete(std::set<ObjectID>& initial_delete_set,
                        std::set<ObjectID>& delete_set,
                        const ObjectID object_id, const bool force,
                        const bool deep);

  inline void putVal(const kv_t& kv, bool const from_remote) {
    // don't crash the server for any reason (any potential garbage value)
    auto upsert_to_meta = [&]() {
      json value = json::parse(kv.value);
      if (value.is_string()) {
        incRef(kv.key, value.get_ref<std::string const&>());
      } else if (value.is_object() && !value.empty()) {
        for (auto const& item : json::iterator_wrapper(value)) {
          if (item.value().is_string()) {
            incRef(kv.key, item.value().get_ref<std::string const&>());
          }
        }
      }
      meta_[json::json_pointer(kv.key)] = value;
      return Status::OK();
    };
    // update signatures
    if (boost::algorithm::starts_with(kv.key, "/signatures/")) {
      auto json_path = json::json_pointer(kv.key);
      if (!from_remote || !meta_.contains(json_path)) {
        VINEYARD_SUPPRESS(CATCH_JSON_ERROR(upsert_to_meta()));
      }
      return;
    }

    // update names
    if (boost::algorithm::starts_with(kv.key, "/names/")) {
      auto json_path = json::json_pointer(kv.key);
      if (meta_.contains(json_path)) {
        LOG(INFO) << "Warning: name got overwritten: " << kv.key;
      }
      VINEYARD_SUPPRESS(CATCH_JSON_ERROR(upsert_to_meta()));
      return;
    }

    // update ordinary data
    VINEYARD_SUPPRESS(CATCH_JSON_ERROR(upsert_to_meta()));
  }

  inline void delVal(const kv_t& kv, std::set<ObjectID>& blobs) {
    size_t last_dot = kv.key.find_last_of('/');
    size_t parent_end = last_dot;
    size_t child_start = last_dot + 1;
    if (static_cast<int>(last_dot) == -1) {
      child_start = parent_end = 0;
    }
    const std::string parent_path(kv.key.cbegin(),
                                  kv.key.cbegin() + parent_end);
    const std::string child_name(kv.key.cbegin() + child_start, kv.key.cend());

    std::vector<std::string> vs;
    boost::algorithm::split(vs, kv.key, [](const char c) { return c == '/'; });
    if (vs[0].empty()) {
      vs.erase(vs.begin());
    }

    ObjectID id_in_key = InvalidObjectID();
    if (vs[0] == "data" && vs.size() > 1) {
      id_in_key = VYObjectIDFromString(vs[1]);
    }
    if (vs[0] != "data" || deleteable(id_in_key)) {
      // delete metadata: a delete operation might be applied multiple times
      auto parent_json_path = json::json_pointer(parent_path);
      if (!meta_.contains(parent_json_path)) {
        return;
      }
      auto& parent_node = meta_[parent_json_path];
      parent_node.erase(child_name);
      if (parent_node.empty()) {
        size_t last_dot_in_path = parent_path.find_last_of('/');
        if (last_dot_in_path == 0) {
          meta_.erase(parent_path.substr(last_dot_in_path + 1));
        } else {
          meta_[json::json_pointer(parent_path.substr(0, last_dot_in_path))]
              .erase(parent_path.substr(last_dot_in_path + 1));
        }
      }

      // if deletable blob: delete blob
      if (id_in_key != InvalidObjectID() && IsBlob(id_in_key)) {
        blobs.emplace(id_in_key);
      }
    } else {
      if (vs[0] == "data" && id_in_key != InvalidObjectID()) {
        // mark as transient
        if ("transient" == child_name) {
          meta_[json::json_pointer(parent_path)]["transient"] = true;
        }
      }
    }
  }

  template <class RangeT>
  void metaUpdate(const RangeT& ops, bool const from_remote) {
    std::set<ObjectID> blobs_to_delete;
    for (const op_t& op : ops) {
      if (op.kv.rev != 0 && op.kv.rev <= rev_) {
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
      if (boost::algorithm::starts_with(op.kv.key,
                                        "/instances" + std::string("/"))) {
        instanceUpdate(op);
      }
#ifndef NDEBUG
      VLOG(10) << "update op in meta tree: " << op.ToString();
#endif
      const kv_t& kv = op.kv;
      if (op.op == op_t::op_type_t::kPut) {
        putVal(kv, from_remote);
      } else if (op.op == op_t::op_type_t::kDel) {
        delVal(kv, blobs_to_delete);
      }
    }
    VINEYARD_SUPPRESS(server_ptr_->DeleteBlobBatch(blobs_to_delete));
    VINEYARD_SUPPRESS(server_ptr_->ProcessDeferred(meta_));
  }

  void instanceUpdate(const op_t& op) {
    std::vector<std::string> key_segments;
    boost::split(key_segments, op.kv.key, boost::is_any_of("/"));
    if (key_segments[0].empty()) {
      key_segments.erase(key_segments.begin());
    }
    if (key_segments[2] == "hostid") {
      uint64_t instance_id = std::stoul(key_segments[1].substr(1));
      if (op.op == op_t::op_type_t::kPut) {
        LOG(INFO) << "Instance join: " << instance_id;
        instances_list_.emplace(instance_id);
      } else if (op.op == op_t::op_type_t::kDel) {
        LOG(INFO) << "Instance exit: " << instance_id;
        instances_list_.erase(instance_id);
      } else {
        LOG(ERROR) << "Unknown op type: " << op.ToString();
      }
    }
  }

  Status daemonWatchHandler(const Status& status, const std::vector<op_t>& ops,
                            unsigned rev) {
    // Guarantee: all kvs inside a txn reaches the client at the same time,
    // which is guaranteed by the implementation of etcd.
    //
    // That means, every time this handler is called, we just need to reponse
    // for one type of change.
    if (!status.ok()) {
      LOG(ERROR) << "Error in daemon watching: " << status.ToString();
      return status;
    }
    if (ops.empty()) {
      return Status::OK();
    }
    // process events grouped by revision
    size_t idx = 0;
    std::vector<op_t> op_batch;
    while (idx < ops.size()) {
      unsigned index = ops[idx].kv.rev;
      while (idx < ops.size() && ops[idx].kv.rev == index) {
        op_batch.emplace_back(ops[idx]);
        idx += 1;
      }
      metaUpdate(op_batch, true);
      op_batch.clear();
    }
    return Status::OK();
  }

  std::unique_ptr<asio::steady_timer> heartbeat_timer_;
  std::set<InstanceID> instances_list_;
  int64_t target_latest_time_ = 0;
  size_t timeout_count_ = 0;

  // dependency: object id -> members' object id
  std::multimap<ObjectID, ObjectID> subobjects_;
  // dependency: object id -> ancestors' object id
  std::multimap<ObjectID, ObjectID> supobjects_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_SERVICES_META_SERVICE_H_
