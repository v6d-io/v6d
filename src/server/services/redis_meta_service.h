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

/**
 * @brief Vineyard implements the metadata service using redis as the backend
 * now. Actually redis_meta_service takes etcd_meta_service as a reference.
 * But due to the differences of commands between redis and etcd and
 * the limits of the redis client API, threre are still some differences of
 * implementing the metadata service. For example, we used a hash list to save
 * all changes of key-value(put or delete) from every vineyard server rather
 * than getting changes directly through etcd watch. In redis_meta_sevrice, we
 * used a key redis_revision to record every change after every commitUpdate by
 * increasing redis_revision. To further introduce the implementation of redis
 * metadata service, let's explain some important keys meaning first. They play
 * a key role to make redis as backend come true.
 *
 * - redis_revision: the key represented the latest revision of commitUpdates
 * - op + redis_revision: the key for hash data structure, which is about the
 * every operation to commitUpdate
 * - opslist: the key for list data structure, we can get any operation from the
 * list
 *
 *                    hashlist for saving operations
 *                   -------------------------------
 *                   | ops | key  | operation type |
 *                   -------------------------------
 *                   |     | key1 |       PUT      |
 *                   | op0 | key2 |       PUT      |
 *                   |     | key3 |       DEL      |
 *                   -------------------------------
 *                   |     | key1 |       PUT      |
 *                   | op1 | key2 |       DEL      |
 *                   |     | key3 |       PUT      |
 *                   -------------------------------
 *                   | ... | ...  |       ...      |
 *                   -------------------------------
 *                               opslist
 *            ----------------------------------------------
 *            | op0 | op1 | op2 | op3 | op4 | op5 | ...... |
 *            ----------------------------------------------
 *
 * - operations: the key is used to publish and subscribe, publish to every
 * server who subscribed the key when the operation is updated
 * - resource: the name of distributed lock
 *
 * Let's see how they collaborate together to work.
 * First, the redis_revision is set to zero when the first server starts up.
 * Other servers won't change it, because we use 'SETNX' command. The vineyard
 * server who wants to commit it's kvs changes to redis have to get the
 * distributed lock named resource. It gets to know it's operation revision by
 * redis_revision and it increased the redis_revision before it puts down the
 * lock, which prevents the changes disappearing when other server tried to
 * commit it's changes. The reason why we used a lot of callbacks in
 * commitUpdates function is that redis client API now doesn't support txn for
 * async_clinet and we must ensure operations are already in hashlist when ops
 * appear in opslist. So PUBLISH can notice other servers after all changes are
 * ready in redis.
 *
 */

#ifndef SRC_SERVER_SERVICES_REDIS_META_SERVICE_H_
#define SRC_SERVER_SERVICES_REDIS_META_SERVICE_H_

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(BUILD_VINEYARDD_REDIS)

#include "redis++/async_redis++.h"
#include "redis++/redis++.h"
#include "redis-plus-plus-shim/recipes/redlock.h"

#include "server/services/etcd_meta_service.h"
#include "server/services/meta_service.h"
#include "server/util/redis_launcher.h"

namespace vineyard {

namespace redis = sw::redis;

class RedisMetaService;
/**
 * @brief RedisWatchHandler manages the watch on redis
 *
 */
class RedisWatchHandler {
 public:
  RedisWatchHandler(const std::shared_ptr<RedisMetaService>& meta_service_ptr,
#if BOOST_VERSION >= 106600
                    asio::io_context& ctx,
#else
                    asio::io_service& ctx,
#endif
                    callback_t<const std::vector<IMetaService::op_t>&, unsigned,
                               callback_t<unsigned>>
                        callback,
                    std::string const& prefix,
                    callback_task_queue_t& registered_callbacks,
                    std::atomic<unsigned>& handled_rev,
                    std::mutex& registered_callbacks_mutex)
      : meta_service_ptr_(meta_service_ptr),
        ctx_(ctx),
        callback_(callback),
        prefix_(prefix),
        registered_callbacks_(registered_callbacks),
        handled_rev_(handled_rev),
        registered_callbacks_mutex_(registered_callbacks_mutex) {
  }

  void operator()(std::unique_ptr<redis::Redis>&, std::string);

 private:
  const std::shared_ptr<RedisMetaService> meta_service_ptr_;
#if BOOST_VERSION >= 106600
  asio::io_context& ctx_;
#else
  asio::io_service& ctx_;
#endif
  const callback_t<const std::vector<IMetaService::op_t>&, unsigned,
                   callback_t<unsigned>>
      callback_;
  std::string const prefix_;

  callback_task_queue_t& registered_callbacks_;
  std::atomic<unsigned>& handled_rev_;
  std::mutex& registered_callbacks_mutex_;

  const std::string kPut = "0";
  const std::string kDel = "1";

  // TODO: more error codes
  enum { OK = 0, UNNAMED_ERROR = 1 };
  std::string errMsg;
  int stateCode = OK;
  std::string errType;
};
/**
 * @brief RedisLock is designed as the lock for accessing redis
 *
 */
class RedisLock : public ILock {
 public:
  Status Release(unsigned& rev) override {
    return callback_(Status::OK(), rev);
  }
  ~RedisLock() override {}

  explicit RedisLock(std::shared_ptr<RedisMetaService> meta_service_ptr,
                     const callback_t<unsigned&>& callback, unsigned rev)
      : ILock(rev), meta_service_ptr_(meta_service_ptr), callback_(callback) {}

 protected:
  const std::shared_ptr<RedisMetaService> meta_service_ptr_;
  const callback_t<unsigned&> callback_;
};

/**
 * @brief RedisMetaService provides meta services in regards to redis, e.g.
 * requesting and committing udpates
 */
class RedisMetaService : public IMetaService {
 public:
  inline void Stop() override;
  ~RedisMetaService() override {}

 protected:
  explicit RedisMetaService(std::shared_ptr<VineyardServer>& server_ptr)
      : IMetaService(server_ptr),
        redis_spec_(server_ptr_->GetSpec()["metastore_spec"]),
        prefix_(redis_spec_["redis_prefix"].get<std::string>() + "/" +
                SessionIDToString(server_ptr->session_id())) {
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
      const std::string& prefix,
      callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
          callback);

  Status probe() override;

  const json redis_spec_;
  const std::string prefix_;

 private:
  std::shared_ptr<RedisMetaService> shared_from_base() {
    return std::static_pointer_cast<RedisMetaService>(shared_from_this());
  }

  Status preStart() override;

  std::unique_ptr<redis::AsyncRedis> redis_;
  std::unique_ptr<redis::Redis> syncredis_;
  std::unique_ptr<redis::Redis> watch_client_;
  std::shared_ptr<redis::RedMutex> mtx_;
  std::shared_ptr<redis::RedLock<redis::RedMutex>> redlock_;
  std::shared_ptr<redis::AsyncSubscriber> watcher_;
  std::shared_ptr<RedisWatchHandler> handler_;
  std::unique_ptr<asio::steady_timer> backoff_timer_;
  std::unique_ptr<RedisLauncher> redis_launcher_;

  callback_task_queue_t registered_callbacks_;
  std::atomic<unsigned> handled_rev_;
  std::mutex registered_callbacks_mutex_;

  // TODO: more error codes
  enum { OK = 0, UNNAMED_ERROR = 1 };
  // requestAll error
  int rAStateCode = OK;
  std::string rAErrMsg;
  std::string rAErrType;
  // commitUpdates error
  int cUStateCode = OK;
  std::string cUErrMsg;
  std::string cUErrType;
  // requestLock error
  int rLStateCode = OK;
  std::string rLErrMsg;
  std::string rLErrType;

  friend class IMetaService;

  inline void operationUpdates(
      callback_t<unsigned> callback_after_updated,
      const std::unordered_map<std::string, unsigned> ops,
      std::string op_prefix, unsigned irev) {
    auto self(shared_from_base());
    redis_->hset(
        op_prefix, ops.begin(), ops.end(),
        [self, callback_after_updated, op_prefix,
         irev](redis::Future<long long>&& resp) {  // NOLINT(runtime/int)
          try {
            resp.get();
          } catch (...) {
            self->cUStateCode = UNNAMED_ERROR;
            self->cUErrMsg = "redis commitUpdates error:";
            self->cUErrType += " hset error";
          }
          self->redis_->command<long long>(  // NOLINT(runtime/int)
              "RPUSH", "opslist", op_prefix,
              [self, callback_after_updated,
               irev](redis::Future<long long>&& resp) {  // NOLINT(runtime/int)
                try {
                  resp.get();
                } catch (...) {
                  self->cUStateCode = UNNAMED_ERROR;
                  self->cUErrMsg = "redis commitUpdates error:";
                  self->cUErrType += " rpush error";
                }
                self->redis_->command<long long>(  // NOLINT(runtime/int)
                    "PUBLISH", "operations", irev,
                    [self, callback_after_updated,
                     irev](redis::Future<long long>&&  // NOLINT(runtime/int)
                               resp) {
                      try {
                        resp.get();
                      } catch (...) {
                        self->cUStateCode = UNNAMED_ERROR;
                        self->cUErrMsg = "redis commitUpdates error:";
                        self->cUErrType += " publish error";
                      }
                      Status status;
                      if (self->stopped_.load()) {
                        status =
                            Status::AlreadyStopped("redis metadata service");
                      } else {
                        status = Status::RedisError(
                            self->cUStateCode, self->cUErrMsg, self->cUErrType);
                      }
                      self->server_ptr_->GetMetaContext().post(boost::bind(
                          callback_after_updated, status, irev + 1));
                    });
              });
        });
  }
};

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_REDIS

#endif  // SRC_SERVER_SERVICES_REDIS_META_SERVICE_H_
