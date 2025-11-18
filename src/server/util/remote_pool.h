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

#ifndef SRC_SERVER_UTIL_REMOTE_POOL_H_
#define SRC_SERVER_UTIL_REMOTE_POOL_H_

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>

#include "common/util/status.h"
#include "server/async/rpc_server.h"
#include "server/server/vineyard_server.h"
#include "server/util/remote.h"

namespace vineyard {

class RemoteClientPool {
 public:
  explicit RemoteClientPool(std::shared_ptr<VineyardServer> server_ptr,
                            std::shared_ptr<RPCServer> rpc_server_ptr)
      : server_ptr_(server_ptr), rpc_server_ptr_(rpc_server_ptr) {}

  ~RemoteClientPool() = default;

  Status BorrowClient(std::string endpoint,
                      std::shared_ptr<RemoteClient>& client);

  Status ReleaseClient(std::string endpoint,
                       std::shared_ptr<RemoteClient> client);

  size_t AvailableClientNum(std::string endpoint);

 private:
  std::unordered_map<std::string, std::queue<std::shared_ptr<RemoteClient>>>
      clients_;
  uint64_t total_clients_ = 0;

  std::recursive_mutex clients_mutex_;
  std::shared_ptr<VineyardServer> server_ptr_;
  std::shared_ptr<RPCServer> rpc_server_ptr_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_REMOTE_POOL_H_
