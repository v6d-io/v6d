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

#ifndef SRC_SERVER_SERVER_VINEYARD_SERVER_H_
#define SRC_SERVER_SERVER_VINEYARD_SERVER_H_

#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/util/asio.h"  // IWYU pragma: keep
#include "common/util/callback.h"
#include "common/util/json.h"
#include "common/util/protocols.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "server/memory/memory.h"

#include "server/memory/stream_store.h"
#include "server/server/vineyard_runner.h"

namespace vineyard {

namespace asio = boost::asio;

class IMetaService;

class IPCServer;
class RPCServer;

/**
 * @brief DeferredReq aims to defer a socket request such that the request
 * is executed only when the metadata satisfies some specific condition.
 *
 */
class DeferredReq {
 public:
  using alive_t = std::function<bool()>;
  using test_t = std::function<bool(const json& meta)>;
  using call_t = std::function<Status(const json& meta)>;

  DeferredReq(alive_t alive_fn, test_t test_fn, call_t call_fn)
      : alive_fn_(alive_fn), test_fn_(test_fn), call_fn_(call_fn) {}

  bool Alive() const;

  bool TestThenCall(const json& meta) const;

 private:
  alive_t alive_fn_;
  test_t test_fn_;
  call_t call_fn_;
};

/**
 * @brief VineyardServer is the main server of vineyard
 *
 */
class VineyardServer : public std::enable_shared_from_this<VineyardServer> {
 public:
  explicit VineyardServer(const json& spec, const SessionID& session_id,
                          std::shared_ptr<VineyardRunner> runner,
                          asio::io_context& context,
                          asio::io_context& meta_context,
                          asio::io_context& io_context,
                          callback_t<std::string const&> callback);
  Status Serve(StoreType const& bulk_store_type,
               const bool create_new_instance = true);
  Status Finalize();
  inline const json& GetSpec() { return spec_; }
  inline const std::string GetDeployment() {
    return spec_["deployment"].get_ref<std::string const&>();
  }

  inline asio::io_context& GetContext() { return context_; }
  inline asio::io_context& GetMetaContext() { return meta_context_; }
  inline asio::io_context& GetIOContext() { return io_context_; }
  inline StoreType GetBulkStoreType() { return bulk_store_type_; }

  template <typename ObjectIDType = ObjectID>
  std::shared_ptr<typename detail::bulk_store_t<ObjectIDType>::type>
  GetBulkStore();

  inline std::shared_ptr<StreamStore> GetStreamStore() { return stream_store_; }
  inline std::shared_ptr<VineyardRunner> GetRunner() { return runner_; }

  void MetaReady();
  void BulkReady();
  void IPCReady();
  void RPCReady();
  void BackendReady();
  void Ready();

  Status GetData(const std::vector<ObjectID>& ids, const bool sync_remote,
                 const bool wait,
                 DeferredReq::alive_t alive,  // if connection is still alive
                 callback_t<const json&> callback);

  Status ListData(std::string const& pattern, bool const regex,
                  size_t const limit, callback_t<const json&> callback);

  Status ListAllData(callback_t<std::vector<ObjectID> const&> callback);

  Status ListName(std::string const& pattern, bool const regex,
                  size_t const limit,
                  callback_t<const std::map<std::string, ObjectID>&> callback);

  Status CreateData(
      const json& tree,
      callback_t<const ObjectID, const Signature, const InstanceID> callback);

  Status CreateData(
      const std::vector<json>& trees,
      callback_t<const std::vector<ObjectID>, const std::vector<Signature>,
                 const std::vector<InstanceID>>
          callback);

  Status CreateData(
      const json& tree, bool recursive,
      callback_t<const ObjectID, const Signature, const InstanceID> callback);

  Status CreateData(
      const std::vector<json>& trees, bool recursive,
      callback_t<const std::vector<ObjectID>, const std::vector<Signature>,
                 const std::vector<InstanceID>>
          callback);

  Status Persist(const ObjectID id, callback_t<> callback);

  Status IfPersist(const ObjectID id, callback_t<const bool> callback);

  Status Exists(const ObjectID id, callback_t<const bool> callback);

  Status ShallowCopy(const ObjectID id, json const& extra_metadata,
                     callback_t<const ObjectID> callback);

  Status DelData(const std::vector<ObjectID>& id, const bool force,
                 const bool deep, const bool memory_trim, const bool fastpath,
                 callback_t<> callback);

  Status DelData(const std::vector<ObjectID>& id, const bool force,
                 const bool deep, const bool memory_trim, const bool fastpath,
                 callback_t<std::vector<ObjectID> const&> callback);

  Status DeleteBlobBatch(const std::set<ObjectID>& blobs,
                         const bool memory_trim = false);

  Status DeleteAllAt(const json& meta, InstanceID const instance_id);

  Status PutName(const ObjectID object_id, const std::string& name,
                 callback_t<> callback);

  Status GetName(const std::string& name, const bool wait,
                 DeferredReq::alive_t alive,  // if connection is still alive
                 callback_t<const ObjectID&> callback);

  Status DropName(const std::string& name, callback_t<> callback);

  Status MigrateObject(
      const ObjectID object_id,
      DeferredReq::alive_t alive,  // if connection is still alive
      callback_t<const ObjectID&> callback);

  Status LabelObjects(const ObjectID object_id,
                      const std::vector<std::string>& keys,
                      const std::vector<std::string>& values,
                      callback_t<> callback);

  Status EvictObjects(const std::vector<ObjectID>& ids, callback_t<> callback);

  Status LoadObjects(const std::vector<ObjectID>& ids, const bool pin,
                     callback_t<> callback);

  Status UnpinObjects(const std::vector<ObjectID>& ids, callback_t<> callback);

  Status ClusterInfo(callback_t<const json&> callback);

  Status InstanceStatus(callback_t<const json&> callback);

  Status ProcessDeferred(const json& meta);

  Status Verify(const std::string& username, const std::string& password,
                callback_t<> callback);

  Status TryAcquireLock(std::string& key,
                        callback_t<bool, std::string> callback);

  Status TryReleaseLock(std::string& key, callback_t<bool> callback);

  inline SessionID session_id() const { return session_id_; }
  inline InstanceID instance_id() { return instance_id_; }
  inline std::string instance_name() { return instance_name_; }
  inline void set_instance_id(InstanceID id) {
    instance_id_ = id;
    instance_name_ = "i" + std::to_string(instance_id_);
  }

  inline std::string const& hostname() { return hostname_; }
  inline void set_hostname(std::string const& hostname) {
    hostname_ = hostname;
  }

  inline std::string const& nodename() { return nodename_; }
  inline void set_nodename(std::string const& nodename) {
    nodename_ = nodename;
  }

  inline bool store_matched(StoreType const& store_type) {
    return bulk_store_type_ == store_type;
  }
  inline bool compression_enabled() const {
    return spec_.value("compression", true);
  }

  const std::string IPCSocket();

  const std::string RPCEndpoint();

  const std::string RDMAEndpoint();

  void LockTransmissionObjects(std::vector<ObjectID> const& ids) {
    std::lock_guard<std::mutex> lock(transmission_objects_mutex_);
    for (auto const& id : ids) {
      if (transmission_objects_.find(id) == transmission_objects_.end()) {
        transmission_objects_[id] = 1;
      } else {
        ++transmission_objects_[id];
      }
    }
  }

  void UnlockTransmissionObjects(std::vector<ObjectID> const& ids) {
    {
      std::lock_guard<std::mutex> lock(transmission_objects_mutex_);
      for (auto const& id : ids) {
        if (transmission_objects_.find(id) != transmission_objects_.end()) {
          if (--transmission_objects_[id] == 0) {
            transmission_objects_.erase(id);
          }
        }
      }
    }
    DeletePendingObjects();
  }

  std::unique_lock<std::mutex> FindTransmissionObjects(
      std::vector<ObjectID> const& ids, std::vector<ObjectID>& transmissions,
      std::vector<ObjectID>& non_transmissions) {
    std::unique_lock<std::mutex> lock(transmission_objects_mutex_);
    for (auto const& id : ids) {
      if (transmission_objects_.find(id) != transmission_objects_.end()) {
        transmissions.push_back(id);
      } else {
        non_transmissions.push_back(id);
      }
    }
    return lock;
  }

  void Stop();

  bool Running() const;

  void PrintTransmissionList() {
    std::lock_guard<std::mutex> lock(transmission_objects_mutex_);
    LOG(INFO) << "print transmission objects, size:"
              << transmission_objects_.size();
    for (auto const& pair : transmission_objects_) {
      LOG(INFO) << "Object " << pair.first
                << " is in transmission, refcnt: " << pair.second;
    }
  }

  void RemoveFromMigrationList(std::vector<ObjectID>& ids) {
    std::lock_guard<std::mutex> lock_origin(migrations_target_to_origin_mutex_);
    std::lock_guard<std::mutex> lock_target(migrations_origin_to_target_mutex_);
    for (auto const id : ids) {
      if (migrations_target_to_origin_.find(id) !=
          migrations_target_to_origin_.end()) {
        ObjectID remoteID = migrations_target_to_origin_[id];
        migrations_origin_to_target_.erase(remoteID);
        migrations_target_to_origin_.erase(id);
      }
    }
  }

  void DeletePendingObjects() {
    std::vector<ObjectID> ids;
    {
      std::lock_guard<std::mutex> lock(pendding_to_delete_objects_mutex_);
      if (pendding_to_delete_objects_.empty()) {
        return;
      }
      for (auto const& id : pendding_to_delete_objects_) {
        ids.push_back(id);
      }
      pendding_to_delete_objects_.clear();
    }
    VINEYARD_DISCARD(
        this->DelData(ids, false, false, false, false,
                      [](const Status& status) { return Status::OK(); }));
  }

  void AddPendingObjects(std::vector<ObjectID> const& ids) {
    std::lock_guard<std::mutex> lock(pendding_to_delete_objects_mutex_);
    for (auto const& id : ids) {
      pendding_to_delete_objects_.insert(id);
    }
  }

  ~VineyardServer();

 private:
  json spec_;
  SessionID session_id_;

  asio::io_context& context_;
  asio::io_context& meta_context_;
  asio::io_context& io_context_;
  callback_t<std::string const&> callback_;

  std::shared_ptr<IMetaService> meta_service_ptr_;
  std::shared_ptr<IPCServer> ipc_server_ptr_;
  std::shared_ptr<RPCServer> rpc_server_ptr_;

  std::list<DeferredReq> deferred_;

  StoreType bulk_store_type_;
  std::shared_ptr<BulkStore> bulk_store_;
  std::shared_ptr<PlasmaBulkStore> plasma_bulk_store_;
  std::shared_ptr<StreamStore> stream_store_;
  std::shared_ptr<VineyardRunner> runner_;

  Status serve_status_;

  enum ready_t {
    kMeta = 0b1,
    kBulk = 0b10,
    kIPC = 0b100,
    kRPC = 0b1000,
    kBackendReady = 0b11,  // then we can serve ipc/rpc.
    kReady = 0b1111,
  };
  unsigned char ready_;
  std::atomic_bool stopped_;  // avoid invoke Stop() twice.

  InstanceID instance_id_;
  std::string instance_name_;
  std::string hostname_;
  std::string nodename_;
  std::map<InstanceID, std::string> instance_id_to_member_id_;

  std::mutex migrations_origin_to_target_mutex_;
  std::mutex migrations_target_to_origin_mutex_;
  // Record the migration status of objects to avoid duplicated migration.
  std::unordered_map<ObjectID, ObjectID> migrations_origin_to_target_;
  std::unordered_map<ObjectID, ObjectID> migrations_target_to_origin_;

  std::unordered_map<ObjectID, int> transmission_objects_;
  std::mutex transmission_objects_mutex_;
  // It must be blob.
  std::unordered_set<ObjectID> pendding_to_delete_objects_;
  std::mutex pendding_to_delete_objects_mutex_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_SERVER_VINEYARD_SERVER_H_
