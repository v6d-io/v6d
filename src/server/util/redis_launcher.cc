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

#include "server/util/redis_launcher.h"

#if defined(BUILD_VINEYARDD_REDIS)

#include <netdb.h>

#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "redis-plus-plus-shim/recipes/redlock.h"

#include "common/util/env.h"

namespace vineyard {

constexpr int max_probe_retries = 10;

RedisLauncher::RedisLauncher(const json& redis_spec)
    : redis_spec_(redis_spec) {}

RedisLauncher::~RedisLauncher() {
  if (redis_proc_) {
    std::error_code err;
    redis_proc_->terminate(err);
    kill(redis_proc_->id(), SIGTERM);
    redis_proc_->wait(err);
  }
}

Status RedisLauncher::LaunchRedisServer(
    std::unique_ptr<redis::AsyncRedis>& redis_client,
    std::unique_ptr<redis::Redis>& syncredis_client,
    std::unique_ptr<redis::Redis>& watch_client,
    std::shared_ptr<redis::RedMutex>& mtx,
    std::shared_ptr<redis::RedLock<redis::RedMutex>>& redlock) {
  RETURN_ON_ERROR(parseEndpoint());
  redis::ConnectionOptions opts;
  opts.host = endpoint_host_;
  opts.port = endpoint_port_;

  redis::ConnectionPoolOptions pool_opts;
  pool_opts.size = 3;
  // async redis client
  redis_client.reset(new redis::AsyncRedis(opts, pool_opts));
  // sync redis client
  syncredis_client.reset(new redis::Redis(opts, pool_opts));
  // watch sync redis client
  watch_client.reset(new redis::Redis(opts, pool_opts));

  mtx.reset(new redis::RedMutex(*syncredis_client, "resource"));
  redlock.reset(new redis::RedLock<redis::RedMutex>(*mtx, std::defer_lock));

  if (probeRedisServer(redis_client, syncredis_client, watch_client)) {
    return Status::OK();
  }

  RETURN_ON_ERROR(initHostInfo());
  bool try_launch = false;
  if (local_hostnames_.find(endpoint_host_) != local_hostnames_.end() ||
      local_ip_addresses_.find(endpoint_host_) != local_ip_addresses_.end()) {
    try_launch = true;
  }

  if (!try_launch) {
    return Status::OK();
  }

  LOG(INFO) << "Starting the redis server";

  // resolve redis binary
  std::string redis_cmd =
      redis_spec_["redis_cmd"].get_ref<std::string const&>();
  if (redis_cmd.empty()) {
    setenv("LC_ALL", "C", 1);  // makes boost's path works as expected.
    redis_cmd = boost::process::search_path("redis-server").string();
  }
  LOG(INFO) << "Found redis at: " << redis_cmd;

  std::vector<std::string> args;
  args.emplace_back("--port");
  args.emplace_back(std::to_string(endpoint_port_));

  auto env = boost::this_process::environment();

  DLOG(INFO) << "Launching redis with: " << boost::algorithm::join(args, " ");
  std::error_code ec;
  redis_proc_ = std::unique_ptr<boost::process::child>(
      new boost::process::child(redis_cmd, boost::process::args(args),
                                boost::process::std_out > stdout,
                                boost::process::std_err > stderr, env, ec));
  if (ec) {
    LOG(ERROR) << "Failed to launch redis: " << ec.message();
    return Status::RedisError("Failed to launch redis: " + ec.message());
  } else {
    LOG(INFO) << "redis launched: pid = " << redis_proc_->id() << ", listen on "
              << endpoint_port_;

    int retries = 0;
    std::error_code err;
    while (redis_proc_ && redis_proc_->running(err) && !err &&
           retries < max_probe_retries) {
      if (probeRedisServer(redis_client, syncredis_client, watch_client)) {
        break;
      }
      retries += 1;
      sleep(1);
    }
    if (!redis_proc_) {
      return Status::IOError(
          "Failed to wait until redis ready: operation has been interrupted");
    } else if (err) {
      return Status::IOError("Failed to check the process status: " +
                             err.message());
    } else if (retries >= max_probe_retries) {
      return Status::RedisError(
          "Redis has been launched but failed to connect to it");
    } else {
      return Status::OK();
    }
  }
}

Status RedisLauncher::initHostInfo() {
  local_hostnames_.emplace("localhost");
  local_ip_addresses_.emplace("127.0.0.1");
  local_ip_addresses_.emplace("0.0.0.0");
  std::string hostname_value = get_hostname();
  local_hostnames_.emplace(hostname_value);
  struct hostent* host_entry = gethostbyname(hostname_value.c_str());
  if (host_entry == nullptr) {
    LOG(WARNING) << "Failed in gethostbyname: " << hstrerror(h_errno);
    return Status::OK();
  }
  {
    size_t index = 0;
    while (true) {
      if (host_entry->h_aliases[index] == nullptr) {
        break;
      }
      local_hostnames_.emplace(host_entry->h_aliases[index++]);
    }
  }
  {
    size_t index = 0;
    while (true) {
      if (host_entry->h_addr_list[index] == nullptr) {
        break;
      }
      local_ip_addresses_.emplace(
          inet_ntoa(*((struct in_addr*) host_entry->h_addr_list[index++])));
    }
  }
  return Status::OK();
}

Status RedisLauncher::parseEndpoint() {
  std::string url = redis_spec_["redis_endpoint"].get_ref<std::string const&>();
  int end = url.size() - 1;
  int mid = 0;
  for (int i = end; i >= 0; --i) {
    if (url[i] == ':') {
      try {
        endpoint_port_ = std::stoi(url.substr(i + 1, end - i));
      } catch (...) {
        return Status::Invalid("Invalid redis_endpoint '" + url +
                               "', please check it out and try again.");
      }
      mid = i - 1;
    } else if (url[i] == '/') {
      endpoint_host_ = url.substr(i + 1, mid - i);
      break;
    } else if (i == 0) {
      endpoint_host_ = url.substr(i, mid - i + 1);
    }
  }
  if (mid == 0) {
    return Status::Invalid("Invalid redis_endpoint '" + url +
                           "', please check it out and try again.");
  }
  return Status::OK();
}

bool RedisLauncher::probeRedisServer(
    std::unique_ptr<redis::AsyncRedis>& redis_client,
    std::unique_ptr<redis::Redis>& syncredis_client,
    std::unique_ptr<redis::Redis>& watch_client) {
  LOG(INFO) << "Waiting for the redis server response ...";
  try {
    auto watch_response = watch_client->ping();
    auto task = redis_client->ping();
    auto response = task.get();
    auto sync_response = syncredis_client->ping();
    redis_client
        ->command<long long>("SETNX",  // NOLINT(runtime/int)
                             "redis_revision", 0)
        .get();
    return redis_client && syncredis_client && (response == "PONG") &&
           (sync_response == "PONG") && (watch_response == "PONG");
  } catch (...) { return false; }
}

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_REDIS
