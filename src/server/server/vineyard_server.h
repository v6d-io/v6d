/** Copyright 2020-2022 Alibaba Group Holding Limited.

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
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/util/asio.h"
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
  Status Serve(StoreType const& bulk_store_type);
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
      const json& tree, bool recursive,
      callback_t<const ObjectID, const Signature, const InstanceID> callback);

  Status Persist(const ObjectID id, callback_t<> callback);

  Status IfPersist(const ObjectID id, callback_t<const bool> callback);

  Status Exists(const ObjectID id, callback_t<const bool> callback);

  Status ShallowCopy(const ObjectID id, json const& extra_metadata,
                     callback_t<const ObjectID> callback);

  Status DelData(const std::vector<ObjectID>& id, const bool force,
                 const bool deep, const bool fastpath, callback_t<> callback);

  Status DelData(const std::vector<ObjectID>& id, const bool force,
                 const bool deep, const bool fastpath,
                 callback_t<std::vector<ObjectID> const&> callback);

  Status DeleteBlobBatch(const std::set<ObjectID>& blobs);

  Status DeleteAllAt(const json& meta, InstanceID const instance_id);

  Status PutName(const ObjectID object_id, const std::string& name,
                 callback_t<> callback);

  Status GetName(const std::string& name, const bool wait,
                 DeferredReq::alive_t alive,  // if connection is still alive
                 callback_t<const ObjectID&> callback);

  Status DropName(const std::string& name, callback_t<> callback);

  Status MigrateObject(const ObjectID object_id,
                       callback_t<const ObjectID&> callback);

  Status ClusterInfo(callback_t<const json&> callback);

  Status InstanceStatus(callback_t<const json&> callback);

  Status ProcessDeferred(const json& meta);

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

  const std::string IPCSocket();

  const std::string RPCEndpoint();

  void Stop();

  bool Running() const;

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
};

}  // namespace vineyard

#endif  // SRC_SERVER_SERVER_VINEYARD_SERVER_H_
