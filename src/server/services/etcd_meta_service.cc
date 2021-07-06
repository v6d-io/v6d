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

#include "server/services/etcd_meta_service.h"

#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "boost/asio/steady_timer.hpp"

#include "etcd/v3/Transaction.hpp"

#include "common/util/boost.h"
#include "common/util/logging.h"

#define BACKOFF_RETRY_TIME 10

namespace vineyard {

void EtcdWatchHandler::operator()(pplx::task<etcd::Response> const& resp_task) {
  this->operator()(resp_task.get());
}

void EtcdWatchHandler::operator()(etcd::Response const& resp) {
  VLOG(10) << "etcd watch use " << resp.duration().count()
           << " microseconds, event size = " << resp.events().size();

  // NB: the head rev is not the latest rev in those events.
  unsigned head_rev = static_cast<unsigned>(resp.index());
  if (resp.error_code() == 0 && !resp.events().empty()) {
    head_rev = resp.events().back().kv().modified_index();
  }

  std::vector<EtcdMetaService::op_t> ops;
  ops.reserve(resp.events().size());
  for (auto const& event : resp.events()) {
    std::string const& key = event.kv().key();
    if (!filter_prefix_.empty() &&
        boost::algorithm::starts_with(key, filter_prefix_)) {
      // FIXME: for simplicity, we don't care the instance-lock related keys.
      continue;
    }
    if (!boost::algorithm::starts_with(key, prefix_ + "/")) {
      // ignore garbage values
      continue;
    }
    EtcdMetaService::op_t op;
    std::string op_key = boost::algorithm::erase_head_copy(key, prefix_.size());
    switch (event.event_type()) {
    case etcd::Event::EventType::PUT: {
      auto op = EtcdMetaService::op_t::Put(op_key, event.kv().as_string(),
                                           event.kv().modified_index());
      ops.emplace_back(op);
      break;
    }
    case etcd::Event::EventType::DELETE_: {
      auto op = EtcdMetaService::op_t::Del(op_key, event.kv().modified_index());
      ops.emplace_back(op);
      break;
    }
    default: {
      // invalid event type.
      break;
    }
    }
  }
  // Notes on [Execute order in boost asio context]
  //
  // The execution order is guaranteed to be the same with the post order.
  //
  // Ref: https://www.boost.org/doc/libs/1_75_0/doc/html/boost_asio/reference/
  //      io_context__strand.html#boost_asio.reference.io_context__strand.orde
  //      r_of_handler_invocation

#ifndef NDEBUG
  static unsigned processed = 0;
#endif

  auto status = Status::EtcdError(resp.error_code(), resp.error_message());

  // NB: update the `handled_rev_` after we have truely applied the update ops.
  ctx_.post(boost::bind(callback_, status, ops, head_rev,
                        [this, status](Status const&, unsigned rev) -> Status {
                          std::lock_guard<std::mutex> scope_lock(
                              this->registered_callbacks_mutex_);
                          this->handled_rev_.store(rev);

                          // handle registered callbacks
                          while (!this->registered_callbacks_.empty()) {
                            auto iter = this->registered_callbacks_.front();
#ifndef NDEBUG
                            VINEYARD_ASSERT(iter.first >= processed);
                            processed = iter.first;
#endif
                            if (iter.first > rev) {
                              break;
                            }
                            this->ctx_.post(boost::bind(
                                iter.second, status,
                                std::vector<EtcdMetaService::op_t>{}, rev));
                            this->registered_callbacks_.pop();
                          }
                          return Status::OK();
                        }));
}

void EtcdMetaService::requestLock(
    std::string lock_name,
    callback_t<std::shared_ptr<ILock>> callback_after_locked) {
  etcd_->lock(prefix_ + lock_name)
      .then([this, callback_after_locked](
                pplx::task<etcd::Response> const& resp_task) {
        auto const& resp = resp_task.get();
        VLOG(10) << "etcd lock use " << resp.duration().count()
                 << " microseconds";
        auto lock_key = resp.lock_key();
        auto lock_ptr = std::make_shared<EtcdLock>(
            [this, lock_key](const Status& status, unsigned& rev) {
              // ensure the lock get released.
              auto unlock_resp = this->etcd_->unlock(lock_key).get();
              if (unlock_resp.is_ok()) {
                rev = unlock_resp.index();
              }
              return Status::EtcdError(unlock_resp.error_code(),
                                       unlock_resp.error_message());
            },
            resp.index());
        auto status =
            Status::EtcdError(resp.error_code(), resp.error_message());
        server_ptr_->GetMetaContext().post(
            boost::bind(callback_after_locked, status, lock_ptr));
      });
}

void EtcdMetaService::commitUpdates(
    const std::vector<op_t>& changes,
    callback_t<unsigned> callback_after_updated) {
  // Split to many small txns to conform the requirement of max-txn-ops
  // limitation (128) from etcd.
  //
  // The first n segments will be performed synchronously while the last
  // txn will still be executed in a asynchronous manner.
  size_t offset = 0;
  while (offset + 127 < changes.size()) {
    etcdv3::Transaction tx;
    for (size_t idx = offset; idx < offset + 127; ++idx) {
      auto const& op = changes[idx];
      if (op.op == op_t::kPut) {
        tx.setup_put(prefix_ + op.kv.key, op.kv.value);
      } else if (op.op == op_t::kDel) {
        tx.setup_delete(prefix_ + op.kv.key);
      }
    }
    auto resp = etcd_->txn(tx).get();
    if (resp.is_ok()) {
      offset += 127;
    } else {
      auto status = Status::EtcdError(resp.error_code(), resp.error_message());
      server_ptr_->GetMetaContext().post(
          boost::bind(callback_after_updated, status, resp.index()));
      return;
    }
  }
  etcdv3::Transaction tx;
  for (size_t idx = offset; idx < changes.size(); ++idx) {
    auto const& op = changes[idx];
    if (op.op == op_t::kPut) {
      tx.setup_put(prefix_ + op.kv.key, op.kv.value);
    } else if (op.op == op_t::kDel) {
      tx.setup_delete(prefix_ + op.kv.key);
    }
  }
  etcd_->txn(tx).then([this, callback_after_updated](
                          pplx::task<etcd::Response> const& resp_task) {
    auto resp = resp_task.get();
    VLOG(10) << "etcd (last) txn use " << resp.duration().count()
             << " microseconds";
    auto status = Status::EtcdError(resp.error_code(), resp.error_message());
    server_ptr_->GetMetaContext().post(
        boost::bind(callback_after_updated, status, resp.index()));
  });
}

void EtcdMetaService::requestAll(
    const std::string& prefix, unsigned base_rev,
    callback_t<const std::vector<IMetaService::op_t>&, unsigned> callback) {
  etcd_->ls(prefix_ + prefix)
      .then([this, callback](pplx::task<etcd::Response> resp_task) {
        auto resp = resp_task.get();
        VLOG(10) << "etcd ls use " << resp.duration().count()
                 << " microseconds for " << resp.keys().size() << " keys";
        std::vector<IMetaService::op_t> ops(resp.keys().size());
        for (size_t i = 0; i < resp.keys().size(); ++i) {
          if (resp.key(i).empty()) {
            continue;
          }
          if (!boost::algorithm::starts_with(resp.key(i), prefix_ + "/")) {
            // ignore garbage values
            continue;
          }
          std::string op_key =
              boost::algorithm::erase_head_copy(resp.key(i), prefix_.size());
          auto op = EtcdMetaService::op_t::Put(
              op_key, resp.value(i).as_string(), resp.index());
          ops.emplace_back(op);
        }
        auto status =
            Status::EtcdError(resp.error_code(), resp.error_message());
        server_ptr_->GetMetaContext().post(
            boost::bind(callback, status, ops, resp.index()));
      });
}

void EtcdMetaService::requestUpdates(
    const std::string& prefix, unsigned,
    callback_t<const std::vector<op_t>&, unsigned> callback) {
  etcd_->head().then([this, prefix,
                      callback](pplx::task<etcd::Response> resp_task) {
    auto resp = resp_task.get();
    auto head_rev = static_cast<unsigned>(resp.index());
    {
      std::lock_guard<std::mutex> scope_lock(this->registered_callbacks_mutex_);
      auto handled_rev = this->handled_rev_.load();
      if (head_rev <= handled_rev) {
        server_ptr_->GetMetaContext().post(
            boost::bind(callback, Status::OK(), std::vector<op_t>{},
                        this->handled_rev_.load()));
        return;
      }
      // We still choose to wait event there's no watchers, as if the watcher
      // fails, a explict watch action will fail as well.
      this->registered_callbacks_.emplace(std::make_pair(head_rev, callback));
    }
  });
}

void EtcdMetaService::startDaemonWatch(
    const std::string& prefix, unsigned since_rev,
    callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
        callback) {
  try {
    this->handled_rev_.store(since_rev);
    if (!handler_) {
      handler_.reset(new EtcdWatchHandler(
          server_ptr_->GetMetaContext(), callback, prefix_,
          prefix_ + meta_sync_lock_, this->registered_callbacks_,
          this->handled_rev_, this->registered_callbacks_mutex_));
    }
    this->watcher_.reset(new etcd::Watcher(
        *etcd_, prefix_ + prefix, since_rev + 1, std::ref(*handler_), true)),
        this->watcher_->Wait([this, prefix, callback](bool cancalled) {
          if (cancalled) {
            return;
          }
          this->retryDaeminWatch(prefix, callback);
        });
  } catch (std::runtime_error& e) {
    LOG(ERROR) << "Failed to create daemon etcd watcher: " << e.what();
    this->watcher_.reset();
    this->retryDaeminWatch(prefix, callback);
  }
}

void EtcdMetaService::retryDaeminWatch(
    const std::string& prefix,
    callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
        callback) {
  backoff_timer_.reset(new asio::steady_timer(
      server_ptr_->GetMetaContext(), std::chrono::seconds(BACKOFF_RETRY_TIME)));
  backoff_timer_->async_wait([this, prefix, callback](
                                 const boost::system::error_code& error) {
    if (error) {
      LOG(ERROR) << "backoff timer error: " << error << ", " << error.message();
    }
    // retry
    LOG(INFO) << "retrying to connect etcd...";
    this->startDaemonWatch(prefix, handled_rev_.load(), callback);
  });
}

Status EtcdMetaService::preStart() {
  auto launcher = EtcdLauncher(etcd_spec_);
  return launcher.LaunchEtcdServer(etcd_, meta_sync_lock_, etcd_proc_);
}

}  // namespace vineyard
