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

#ifndef SRC_SERVER_UTIL_REDIS_LAUNCHER_H_
#define SRC_SERVER_UTIL_REDIS_LAUNCHER_H_

#include <memory>
#include <set>
#include <string>

#include "boost/filesystem.hpp"
#include "boost/process.hpp"

#if defined(BUILD_VINEYARDD_REDIS)

#include "redis++/async_redis++.h"
#include "redis++/redis++.h"
#include "redis-plus-plus-shim/recipes/redlock.h"

#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

namespace redis = sw::redis;

class RedisLauncher {
 public:
  explicit RedisLauncher(const json& redis_spec);
  ~RedisLauncher();

  Status LaunchRedisServer(
      std::unique_ptr<redis::AsyncRedis>& redis_client,
      std::unique_ptr<redis::Redis>& syncredis_client,
      std::unique_ptr<redis::Redis>& watch_client,
      std::shared_ptr<redis::RedMutex>& mtx,
      std::shared_ptr<redis::RedLock<redis::RedMutex>>& lock);

  static bool probeRedisServer(std::unique_ptr<redis::AsyncRedis>& redis_client,
                               std::unique_ptr<redis::Redis>& syncredis_client,
                               std::unique_ptr<redis::Redis>& watch_client);

 private:
  Status parseEndpoint();

  Status initHostInfo();

  const json redis_spec_;
  std::string endpoint_host_;
  int endpoint_port_;
  std::set<std::string> local_hostnames_;
  std::set<std::string> local_ip_addresses_;

  std::unique_ptr<boost::process::child> redis_proc_;
};
}  // namespace vineyard

#endif  // BUILD_VINEYARDD_REDIS

#endif  // SRC_SERVER_UTIL_REDIS_LAUNCHER_H_
