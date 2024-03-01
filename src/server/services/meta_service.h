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

#ifndef SRC_SERVER_SERVICES_META_SERVICE_H_
#define SRC_SERVER_SERVICES_META_SERVICE_H_

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/trim.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/bind.hpp"  // IWYU pragma: keep

#include "common/util/asio.h"  // IWYU pragma: keep
#include "common/util/callback.h"
#include "common/util/env.h"
#include "common/util/functions.h"
#include "common/util/json.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "common/util/status.h"
#include "server/server/vineyard_server.h"
#include "server/util/meta_tree.h"
#include "server/util/metrics.h"

#define HEARTBEAT_TIME 60
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
class IMetaService : public std::enable_shared_from_this<IMetaService> {
 public:
  using kv_t = meta_tree::kv_t;
  using op_t = meta_tree::op_t;

  struct watcher_t {
    watcher_t(callback_t<const json&, const std::string&> w,
              const std::string& t)
        : watcher(w), tag(t) {}
    callback_t<const json&, const std::string&> watcher;
    std::string tag;
  };

  explicit IMetaService(std::shared_ptr<VineyardServer>& server_ptr)
      : server_ptr_(server_ptr), rev_(0), meta_sync_lock_("/meta_sync_lock") {
    stopped_.store(false);
  }

  virtual ~IMetaService();

  static std::shared_ptr<IMetaService> Get(
      std::shared_ptr<VineyardServer> vs_ptr);

  Status Start();

  virtual void Stop();

 public:
  void RequestToBulkUpdate(
      callback_t<const json&, std::vector<op_t>&, ObjectID&, Signature&,
                 InstanceID&>
          callback_after_ready,
      callback_t<const ObjectID, const Signature, const InstanceID>
          callback_after_finish);

  void RequestToBulkUpdate(
      callback_t<const json&, std::vector<op_t>&, std::vector<ObjectID>&,
                 std::vector<Signature>&, std::vector<InstanceID>&>
          callback_after_ready,
      callback_t<const std::vector<ObjectID>, const std::vector<Signature>,
                 const std::vector<InstanceID>>
          callback_after_finish);

  // When requesting direct update, we already worked inside the meta context.
  void RequestToDirectUpdate(std::vector<op_t> const& ops,
                             const bool from_remote = false);

  void RequestToPersist(
      callback_t<const json&, std::vector<op_t>&> callback_after_ready,
      callback_t<> callback_after_finish);

  void RequestToGetData(const bool sync_remote,
                        callback_t<const json&> callback);

  void RequestToDelete(
      const std::vector<ObjectID>& object_ids, const bool force,
      const bool deep, const bool memory_trim,
      callback_t<const json&, std::vector<ObjectID> const&, std::vector<op_t>&,
                 bool&>
          callback_after_ready,
      callback_t<std::vector<ObjectID> const&> callback_after_finish);

  void RequestToShallowCopy(
      callback_t<const json&, std::vector<op_t>&, bool&> callback_after_ready,
      callback_t<> callback_after_finish);

  void IncRef(std::string const& instance_name, std::string const& key,
              std::string const& value, const bool from_remote);

  void CloneRef(ObjectID const target, ObjectID const mirror);

  bool stopped() const { return this->stopped_.load(); }

  virtual void TryAcquireLock(std::string key,
                              callback_t<bool, std::string> callback) = 0;

  virtual void TryReleaseLock(std::string key, callback_t<bool> callback) = 0;

 private:
  void registerToEtcd();

  /**
   * Watch rules:
   *
   *  - every instance checks the status of the "NEXT" instance;
   *  - the last one watches for the first one;
   *  - if there's only one instance, it does nothing.
   */
  static void checkInstanceStatus(std::shared_ptr<IMetaService> const& self,
                                  callback_t<> callback_after_finish);

  static Status startHeartbeat(std::shared_ptr<IMetaService> const& self,
                               Status const&);

 protected:
  // invoke when everything is ready (after Start() and ready for invoking)
  inline void Ready() {
    server_ptr_->MetaReady();  // notify server the meta svc is ready
  }

  virtual void commitUpdates(const std::vector<op_t>&,
                             callback_t<unsigned> callback_after_updated) = 0;

  void requestValues(const std::string& prefix,
                     callback_t<const json&, unsigned> callback);

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
      callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
          callback) = 0;

  // validate the liveness of the underlying meta service.
  virtual Status probe() = 0;

  void printDepsGraph();

  std::atomic<bool> stopped_;
  json meta_;
  std::shared_ptr<VineyardServer> server_ptr_;

  unsigned rev_;
  bool backend_retrying_;

  std::string meta_sync_lock_;

 private:
  virtual Status preStart() { return Status::OK(); }

  bool deleteable(ObjectID const object_id);

  void traverseToDelete(std::set<ObjectID>& initial_delete_set,
                        std::set<ObjectID>& delete_set, int32_t depth,
                        std::map<ObjectID, int32_t>& depthes,
                        const ObjectID object_id, const bool force,
                        const bool deep);

  void findDeleteSet(std::vector<ObjectID> const& object_ids,
                     std::vector<ObjectID>& processed_delete_set, bool force,
                     bool deep);

  void postProcessForDelete(const std::set<ObjectID>& delete_set,
                            const std::map<ObjectID, int32_t>& depthes,
                            std::vector<ObjectID>& delete_objects);

  void putVal(const kv_t& kv, bool const from_remote);
  void delVal(std::string const& key);
  void delVal(const kv_t& kv);
  void delVal(ObjectID const& target, std::set<ObjectID>& blobs);

  template <class RangeT>
  void metaUpdate(const RangeT& ops, bool const from_remote,
                  const bool memory_trim = false);

  void instanceUpdate(const op_t& op, const bool from_remote = true);

  static Status daemonWatchHandler(std::shared_ptr<IMetaService> self,
                                   const Status& status,
                                   const std::vector<op_t>& ops, unsigned rev,
                                   callback_t<unsigned> callback_after_update);

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
