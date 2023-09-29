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

#include "server/services/redis_meta_service.h"

#include <chrono>
#include <string>
#include <vector>

#if defined(BUILD_VINEYARDD_REDIS)

#include "pplx/pplxtasks.h"

#include "common/util/logging.h"
#include "server/util/metrics.h"
#include "server/util/redis_launcher.h"

#define BACKOFF_RETRY_TIME 10

namespace vineyard {

void RedisWatchHandler::operator()(std::unique_ptr<redis::Redis>& redis,
                                   std::string rev) {
  // need to ensure handled_rev_ updates before next publish is handled
  std::lock_guard<std::mutex> scope_lock(registered_callbacks_mutex_);
  if (this->meta_service_ptr_->stopped()) {
    return;
  }

  unsigned irev = 0;
  try {
    irev = static_cast<unsigned>(std::stol(rev));
  } catch (...) {
    stateCode = UNNAMED_ERROR;
    errMsg = "redis watchHandler error:";
    errType += " resolve redis_revision error";
  }

  if (irev < handled_rev_.load()) {
    return;
  }

  std::vector<std::string> operations;
  try {
    redis->lrange("opslist", handled_rev_.load(), irev,
                  std::back_inserter(operations));
  } catch (...) {
    stateCode = UNNAMED_ERROR;
    errMsg = "redis watchHandler error:";
    errType += " lrange error";
  }

  std::vector<IMetaService::op_t> ops;
  try {
    for (auto const& item : operations) {
      std::vector<std::string> puts;
      std::vector<std::string> dels;
      std::unordered_map<std::string, std::string> kvs;
      redis->hgetall(item, std::inserter(kvs, kvs.begin()));
      for (auto const& kv : kvs) {
        if (kv.second == kPut) {
          // keys need to Put
          puts.emplace_back(kv.first);
        } else if (kv.second == kDel) {
          // keys need to Del
          ops.emplace_back(IMetaService::op_t::Del(kv.first, irev + 1));
        }
      }

      if (puts.size() > 0) {
        std::vector<redis::OptionalString> vals;
        try {
          redis->mget(puts.begin(), puts.end(), std::back_inserter(vals));
          for (size_t i = 0; i < vals.size(); ++i) {
            if (vals[i]) {
              ops.emplace_back(IMetaService::op_t::Put(
                  boost::algorithm::erase_head_copy(puts[i], prefix_.size()),
                  *vals[i], irev + 1));
            }
          }
        } catch (...) {
          stateCode = UNNAMED_ERROR;
          errMsg = "redis watchHandler error:";
          errType += " mget error";
        }
      }
    }
  } catch (...) {
    stateCode = UNNAMED_ERROR;
    errMsg = "redis watchHandler error:";
    errType += " hgetall error";
  }

#ifndef NDEBUG
  static unsigned processed = 0;
#endif

  auto status = Status::RedisError(stateCode, errMsg, errType);
  ctx_.post(boost::bind(
      callback_, status, ops, irev + 1,
      [this, status](Status const&, unsigned rev) -> Status {
        if (this->meta_service_ptr_->stopped()) {
          return Status::AlreadyStopped("redis metadata service");
        }
        std::lock_guard<std::mutex> scope_lock(
            this->registered_callbacks_mutex_);
        if (status.ok()) {
          this->handled_rev_.store(rev);
        }
        // handle registered callbacks
        while (!this->registered_callbacks_.empty()) {
          auto iter = this->registered_callbacks_.top();
#ifndef NDEBUG
          VINEYARD_ASSERT(iter.first >= processed);
          processed = iter.first;
#endif
          if (iter.first > rev) {
            break;
          }
          this->ctx_.post(boost::bind(iter.second, status,
                                      std::vector<IMetaService::op_t>{}, rev));
          this->registered_callbacks_.pop();
        }
        return Status::OK();
      }));
}

void RedisMetaService::Stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  // invoke parent's stop method
  IMetaService::Stop();
  if (backoff_timer_) {
    boost::system::error_code ec;
    backoff_timer_->cancel(ec);
  }
  if (watcher_) {
    try {
      watcher_->unsubscribe();
    } catch (...) {}
  }
  if (redis_launcher_) {
    redis_launcher_.reset();
  }
}

void RedisMetaService::requestLock(
    std::string lock_name,
    callback_t<std::shared_ptr<ILock>> callback_after_locked) {
  auto self(shared_from_base());
  pplx::create_task([self]() {
    unsigned irev = 0;
    try {
      while (!self->redlock_->try_lock(std::chrono::seconds(600))) {}
      auto val = *(self->redis_->get("redis_revision").get());
      irev = static_cast<unsigned>(std::stol(val));
    } catch (...) {
      self->rLStateCode = UNNAMED_ERROR;
      self->rLErrMsg = "redis requestLock error:";
      self->rLErrType += " get redis_revision error";
    }
    return irev;
  }).then([self, callback_after_locked](unsigned val) {
    auto lock_ptr = std::make_shared<RedisLock>(
        self,
        [self](const Status& status, unsigned& rev) {
          // ensure the lock gets released.
          try {
            self->redlock_->unlock();
          } catch (...) {
            self->rLStateCode = UNNAMED_ERROR;
            self->rLErrMsg = "redis requestLock error:";
            self->rLErrType += " unlock error";
          }
          if (self->stopped_.load()) {
            return Status::AlreadyStopped("redis metadata service");
          }
          return Status::RedisError(self->rLStateCode, self->rLErrMsg,
                                    self->rLErrType);
        },
        val);
    Status status;
    if (self->stopped_.load()) {
      status = Status::AlreadyStopped("redis metadata service");
    } else {
      status = Status::RedisError(self->rLStateCode, self->rLErrMsg,
                                  self->rLErrType);
    }
    self->server_ptr_->GetMetaContext().post(
        boost::bind(callback_after_locked, status, lock_ptr));
  });
}

void RedisMetaService::requestAll(
    const std::string& prefix, unsigned base_rev,
    callback_t<const std::vector<op_t>&, unsigned> callback) {
  auto self(shared_from_base());
  // We must ensure that the redis_revision matches the local data.
  // But we're not getting kvs at the same time, redis_revision can be changed,
  // when getting kvs in two steps.
  // So, get redis_revision first.
  // Local data behind revision is fine. They can match when publish coming.
  std::string val;
  try {
    val = *redis_->get("redis_revision").get();
  } catch (...) {
    rAStateCode = UNNAMED_ERROR;
    rAErrMsg = "redis requestAll error:";
    rAErrType += " get redis_revision error";
  }
  redis_->command<std::vector<std::string>>(
      "KEYS", "vineyard/*",
      [self, callback, val](redis::Future<std::vector<std::string>>&& resp) {
        std::vector<std::string> keys;
        unsigned irev = 0;
        try {
          irev = static_cast<unsigned>(std::stol(val));
          auto const& vec = resp.get();
          keys.emplace_back("MGET");
          for (size_t i = 0; i < vec.size(); ++i) {
            if (!boost::algorithm::starts_with(vec[i], self->prefix_ + "/")) {
              // ignore garbage values
              continue;
            }
            keys.emplace_back(vec[i]);
          }
        } catch (...) {
          self->rAStateCode = UNNAMED_ERROR;
          self->rAErrMsg = "redis requestAll error:";
          self->rAErrType += " keys* error";
        }
        if (keys.size() > 1) {
          // mget
          self->redis_->command<std::vector<redis::OptionalString>>(
              keys.begin(), keys.end(),
              [self, keys, callback,
               irev](redis::Future<std::vector<redis::OptionalString>>&& resp) {
                std::string op_key;
                std::vector<op_t> ops;
                try {
                  auto const& vals = resp.get();
                  ops.reserve(vals.size());
                  // collect kvs
                  for (size_t i = 1; i < keys.size(); ++i) {
                    if (vals[i - 1]) {
                      op_key = boost::algorithm::erase_head_copy(
                          keys[i], self->prefix_.size());
                      ops.emplace_back(op_t::Put(op_key, *vals[i - 1], irev));
                    }
                  }
                } catch (...) {
                  self->rAStateCode = UNNAMED_ERROR;
                  self->rAErrMsg = "redis requestAll error:";
                  self->rAErrType += " mget error";
                }
                auto status = Status::RedisError(
                    self->rAStateCode, self->rAErrMsg, self->rAErrType);
                self->server_ptr_->GetMetaContext().post(
                    boost::bind(callback, status, ops, irev));
              });
        } else {
          std::vector<op_t> ops;
          auto status = Status::RedisError(self->rAStateCode, self->rAErrMsg,
                                           self->rAErrType);
          self->server_ptr_->GetMetaContext().post(
              boost::bind(callback, status, ops, irev));
        }
      });
}

void RedisMetaService::requestUpdates(
    const std::string& prefix, unsigned,
    callback_t<const std::vector<op_t>&, unsigned> callback) {
  auto self(shared_from_base());
  redis_->get(
      "redis_revision",
      [self, callback](redis::Future<redis::OptionalString>&& resp) {
        if (self->stopped_.load()) {
          return;
        }
        auto head_rev = static_cast<unsigned>(std::stol(*resp.get()));
        {
          std::lock_guard<std::mutex> scope_lock(
              self->registered_callbacks_mutex_);
          auto handled_rev = self->handled_rev_.load();
          if (head_rev <= handled_rev + 1) {
            self->server_ptr_->GetMetaContext().post(boost::bind(
                callback, Status::OK(), std::vector<op_t>{}, handled_rev));
            return;
          }
          // all updates through publish
          self->registered_callbacks_.emplace(
              std::make_pair(head_rev, callback));
        }
      });
}

void RedisMetaService::commitUpdates(
    const std::vector<op_t>& changes,
    callback_t<unsigned> callback_after_updated) {
  // Just a reminder: When the number of hash entries exceeds 500, hash tables
  // are used instead of ZipList, which occupies a large memory.

  // If rev or op_prefix doesn't initialize, which means errors have already
  // happened.
  std::string rev;
  std::string op_prefix;
  unsigned irev = 0;
  try {
    // the operation number is ready to publish
    rev = *redis_->get("redis_revision").get();
    redis_->incr("redis_revision").get();
    op_prefix = "op" + rev;
    irev = static_cast<unsigned>(std::stol(rev));
  } catch (...) {
    cUStateCode = UNNAMED_ERROR;
    cUErrMsg = "redis commitUpdates error:";
    cUErrType += " get redis_revision error";
  }

  std::vector<std::string> kvs;
  kvs.emplace_back("MSET");
  std::unordered_map<std::string, unsigned> ops;
  std::vector<std::string> keys;
  for (auto const& op : changes) {
    if (op.op == op_t::kPut) {
      kvs.emplace_back(prefix_ + op.kv.key);
      kvs.emplace_back(op.kv.value);
      ops.insert({prefix_ + op.kv.key, op_t::kPut});
    } else if (op.op == op_t::kDel) {
      keys.emplace_back(prefix_ + op.kv.key);
      ops.insert({op.kv.key, op_t::kDel});
    }
  }

  auto self(shared_from_base());
  // delete keys
  if (keys.size() > 0) {
    redis_->del(
        keys.begin(), keys.end(),
        [self](redis::Future<long long>&& resp) {  // NOLINT(runtime/int)
          try {
            resp.get();
          } catch (...) {
            self->cUStateCode = UNNAMED_ERROR;
            self->cUErrMsg = "redis commitUpdates error:";
            self->cUErrType += " del keys error";
          }
        });
  }

  // mset kvs
  if (kvs.size() > 2) {
    redis_->command<void>(kvs.begin(), kvs.end(),
                          [self, callback_after_updated, ops, op_prefix,
                           irev](redis::Future<void>&& resp) {
                            try {
                              resp.get();
                            } catch (...) {
                              self->cUStateCode = UNNAMED_ERROR;
                              self->cUErrMsg = "redis commitUpdates error:";
                              self->cUErrType += " mset error";
                            }
                            self->operationUpdates(callback_after_updated, ops,
                                                   op_prefix, irev);
                          });
  } else {
    operationUpdates(callback_after_updated, ops, op_prefix, irev);
  }
}

void RedisMetaService::startDaemonWatch(
    const std::string& prefix, unsigned since_rev,
    callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
        callback) {
  LOG(INFO) << "start background redis watch, since " << rev_;
  try {
    this->handled_rev_.store(since_rev);
    if (!handler_) {
      handler_.reset(new RedisWatchHandler(
          shared_from_base(), server_ptr_->GetMetaContext(), callback, prefix_,
          this->registered_callbacks_, this->handled_rev_,
          this->registered_callbacks_mutex_));
    }
    auto self(shared_from_base());
    this->watcher_.reset(new redis::AsyncSubscriber(redis_->subscriber()));
    this->watcher_->on_message([self](std::string channel, std::string msg) {
      self->server_ptr_->GetMetaContext().post(boost::bind<void>(
          std::ref(*(self->handler_)), std::ref(self->watch_client_), msg));
    });
    this->watcher_->subscribe("operations");
  } catch (std::runtime_error& e) {
    LOG(ERROR) << "Failed to create daemon redis watcher: " << e.what();
    this->watcher_.reset();
    this->retryDaeminWatch(prefix, callback);
  }
}

void RedisMetaService::retryDaeminWatch(
    const std::string& prefix,
    callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
        callback) {
  auto self(shared_from_base());
  backoff_timer_.reset(new asio::steady_timer(
      server_ptr_->GetMetaContext(), std::chrono::seconds(BACKOFF_RETRY_TIME)));
  backoff_timer_->async_wait([self, prefix, callback](
                                 const boost::system::error_code& error) {
    if (self->stopped_.load()) {
      return;
    }
    if (error) {
      LOG(ERROR) << "backoff timer error: " << error << ", " << error.message();
    }
    if (!error || error != boost::asio::error::operation_aborted) {
      // retry
      LOG(INFO) << "retrying to connect redis ...";
      self->startDaemonWatch(prefix, self->handled_rev_.load(), callback);
    }
  });
}

Status RedisMetaService::probe() {
  if (RedisLauncher::probeRedisServer(redis_, syncredis_, watch_client_)) {
    return Status::OK();
  } else {
    return Status::Invalid(
        "Failed to startup meta service, please check your redis");
  }
}

Status RedisMetaService::preStart() {
  redis_launcher_ =
      std::unique_ptr<RedisLauncher>(new RedisLauncher(redis_spec_));
  return redis_launcher_->LaunchRedisServer(redis_, syncredis_, watch_client_,
                                            mtx_, redlock_);
}

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_REDIS
