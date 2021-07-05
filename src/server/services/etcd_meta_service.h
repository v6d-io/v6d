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

#ifndef SRC_SERVER_SERVICES_ETCD_META_SERVICE_H_
#define SRC_SERVER_SERVICES_ETCD_META_SERVICE_H_

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "etcd/Client.hpp"
#include "etcd/Watcher.hpp"

#include "server/services/meta_service.h"
#include "server/util/etcd_launcher.h"

namespace vineyard {
using registered_callback_type = std::queue<std::pair<
    unsigned, callback_t<const std::vector<IMetaService::op_t>&, unsigned>>>;

/**
 * @brief EtcdWatchHandler manages the watch on etcd
 *
 */
class EtcdWatchHandler {
 public:
  EtcdWatchHandler(
#if BOOST_VERSION >= 106600
      asio::io_context& ctx,
#else
      asio::io_service& ctx,
#endif
      callback_t<const std::vector<IMetaService::op_t>&, unsigned,
                 callback_t<unsigned>>
          callback,
      std::string const& prefix, std::string const& filter_prefix,
      registered_callback_type& registered_callbacks,
      std::atomic<unsigned>& handled_rev,
      std::mutex& registered_callbacks_mutex)
      : ctx_(ctx),
        callback_(callback),
        prefix_(prefix),
        filter_prefix_(filter_prefix),
        registered_callbacks_(registered_callbacks),
        handled_rev_(handled_rev),
        registered_callbacks_mutex_(registered_callbacks_mutex) {
  }

  void operator()(pplx::task<etcd::Response> const& resp_task);
  void operator()(etcd::Response const& task);

 private:
#if BOOST_VERSION >= 106600
  asio::io_context& ctx_;
#else
  asio::io_service& ctx_;
#endif
  const callback_t<const std::vector<IMetaService::op_t>&, unsigned,
                   callback_t<unsigned>>
      callback_;
  std::string const prefix_, filter_prefix_;

  registered_callback_type& registered_callbacks_;
  std::atomic<unsigned>& handled_rev_;
  std::mutex& registered_callbacks_mutex_;
};

/**
 * @brief EtcdLock is designed as the lock for accessing etcd
 *
 */
class EtcdLock : public ILock {
 public:
  Status Release(unsigned& rev) override {
    return callback_(Status::OK(), rev);
  }
  ~EtcdLock() override {}

  explicit EtcdLock(const callback_t<unsigned&>& callback, unsigned rev)
      : ILock(rev), callback_(callback) {}

 protected:
  const callback_t<unsigned&> callback_;
};

/**
 * @brief EtcdMetaService provides meta services in regards to etcd, e.g.
 * requesting and committing udpates
 *
 */
class EtcdMetaService : public IMetaService {
 public:
  inline void Stop() override {
    if (watcher_) {
      try {
        watcher_->Cancel();
      } catch (...) {}
    }
    if (etcd_proc_) {
      std::error_code err;
      etcd_proc_->terminate(err);
      kill(etcd_proc_->id(), SIGTERM);
      etcd_proc_->wait(err);
    }
  }

 protected:
  explicit EtcdMetaService(vs_ptr_t& server_ptr)
      : IMetaService(server_ptr),
        etcd_spec_(server_ptr_->GetSpec()["metastore_spec"]),
        prefix_(etcd_spec_["prefix"].get_ref<std::string const&>()) {
    this->handled_rev_.store(0);
  }

  void requestLock(
      std::string lock_name,
      callback_t<std::shared_ptr<ILock>> callback_after_locked) override;

  void requestAll(
      const std::string& prefix, unsigned base_rev,
      callback_t<const std::vector<op_t>&, unsigned> callback) override;

  void requestUpdates(
      const std::string& prefix, unsigned since_rev,
      callback_t<const std::vector<op_t>&, unsigned> callback) override;

  void commitUpdates(const std::vector<op_t>&,
                     callback_t<unsigned> callback_after_updated) override;

  void startDaemonWatch(
      const std::string& prefix, unsigned since_rev,
      callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
          callback) override;

  void retryDaeminWatch(
      const std::string& prefix, unsigned since_rev,
      callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
          callback);

  Status probe() override {
    if (EtcdLauncher::probeEtcdServer(etcd_, prefix_)) {
      return Status::OK();
    } else {
      return Status::Invalid(
          "Failed to startup meta service, please check your etcd");
    }
  }

  const json etcd_spec_;
  const std::string prefix_;

 private:
  Status preStart() override;

  std::unique_ptr<etcd::Client> etcd_;
  std::shared_ptr<etcd::Watcher> watcher_;
  std::unique_ptr<asio::steady_timer> backoff_timer_;
  std::unique_ptr<boost::process::child> etcd_proc_;

  registered_callback_type registered_callbacks_;
  std::atomic<unsigned> handled_rev_;
  std::mutex registered_callbacks_mutex_;

  friend class IMetaService;
};
}  // namespace vineyard

#endif  // SRC_SERVER_SERVICES_ETCD_META_SERVICE_H_
