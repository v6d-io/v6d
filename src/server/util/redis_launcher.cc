/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include "server/util/redis_launcher.h"

#if defined(BUILD_VINEYARDD_REDIS)

#include "redis-plus-plus-shim/recipes/redlock.h"

#include <netdb.h>

#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"

#include "common/util/env.h"

namespace vineyard {

Status RedisLauncher::LaunchRedisServer(
    std::unique_ptr<redis::AsyncRedis>& redis_client,
    std::unique_ptr<redis::Redis>& syncredis_client,
    std::shared_ptr<redis::RedMutex>& mtx,
    std::shared_ptr<redis::RedLock<redis::RedMutex>>& redlock,
    std::unique_ptr<boost::process::child>& redis_proc) {
  redis::ConnectionOptions opts;
  std::string url = redis_spec_["redis_endpoint"].get_ref<std::string const&>();
  int end = url.size() - 1;
  int mid = 0;
  for (int i = end; i >= 0; --i) {
    if (url[i] == ':') {
      try {
        opts.port = std::stoi(url.substr(i + 1, end - i));
      } catch (...) {
        return Status::Invalid(
            "Invalid redis_endpoint. Check it out and try again.");
      }
      mid = i - 1;
    } else if (url[i] == '/') {
      opts.host = url.substr(i + 1, mid - i);
      break;
    } else if (i == 0) {
      opts.host = url.substr(i, mid - i + 1);
    }
  }
  if (mid == 0) {
    return Status::Invalid(
        "Invalid redis_endpoint. Check it out and try again.");
  }

  redis::ConnectionPoolOptions pool_opts;
  pool_opts.size = 3;
  // async redis client
  redis_client.reset(new redis::AsyncRedis(opts, pool_opts));
  // sync redis client
  syncredis_client.reset(new redis::Redis(opts, pool_opts));
  mtx.reset(new redis::RedMutex(*syncredis_client, "resource"));
  redlock.reset(new redis::RedLock<redis::RedMutex>(*mtx, std::defer_lock));

  if (probeRedisServer(redis_client, syncredis_client)) {
    return Status::OK();
  }

  // TODO: launch redis server when redis doesn't startup.
  return Status::RedisError(
      "Failed to connect to redis, startup redis manually first, or check your "
      "redis_endpoint.");
}

void RedisLauncher::parseEndpoint() {}

void RedisLauncher::initHostInfo() {}

bool RedisLauncher::probeRedisServer(
    std::unique_ptr<redis::AsyncRedis>& redis_client,
    std::unique_ptr<redis::Redis>& syncredis_client) {
  try {
    auto task = redis_client->ping();
    auto response = task.get();
    auto sync_response = syncredis_client->ping();
    redis_client->command<long long>("SETNX", "redis_revision", 0).get();
    return redis_client && syncredis_client && (response == "PONG") &&
           (sync_response == "PONG");
  } catch (...) { return false; }
}

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_REDIS
