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

#include <memory>
#include <string>

#include "common/util/status.h"
#include "server/async/rpc_server.h"
#include "server/server/vineyard_server.h"
#include "server/util/remote.h"
#include "server/util/remote_pool.h"

namespace vineyard {

Status RemoteClientPool::BorrowClient(std::string endpoint,
                                      std::shared_ptr<RemoteClient>& client) {
  VLOG(2) << "Borrow client from pool, endpoint: " << endpoint;
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  auto iter = clients_.find(endpoint);
  if (iter != clients_.end() && !iter->second.empty()) {
    VLOG(2) << "Get client from pool";
    client = iter->second.front();
    iter->second.pop();
    return Status::OK();
  } else {
    VLOG(2) << "Client is not enough, create a new one";
    client = std::make_shared<RemoteClient>(server_ptr_);
    RETURN_ON_ERROR(client->Connect(endpoint, server_ptr_->session_id(), ""));
    total_clients_++;
    return Status::OK();
  }
}

Status RemoteClientPool::ReleaseClient(std::string endpoint,
                                       std::shared_ptr<RemoteClient> client) {
  VLOG(2) << "Release client to pool, endpoint: " << endpoint;
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  if (!client->connected_) {
    LOG(WARNING) << "Client is not connected, discard...";
    LOG(INFO) << "Pool size of " << endpoint << " is "
              << clients_[endpoint].size();
    total_clients_--;
    return Status::OK();
  }
  auto iter = clients_.find(endpoint);
  if (iter != clients_.end()) {
    iter->second.push(client);
  } else {
    std::queue<std::shared_ptr<RemoteClient>> q;
    q.push(client);
    clients_[endpoint] = q;
  }
  VLOG(2) << "Release client to pool. Pool size of " << endpoint << " is "
          << clients_[endpoint].size();
  static uint64_t last_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  if (now - last_time > SECOND_TO_MILLISECOND(3)) {
    LOG(INFO) << "Currently remote client pool size is:" << clients_.size();
    for (auto item : clients_) {
      LOG(INFO) << "endpoint: " << item.first
                << " client num: " << item.second.size();
    }
    LOG(INFO) << "Total client:" << total_clients_;
    last_time = now;
  }
  return Status::OK();
}

size_t RemoteClientPool::AvailableClientNum(std::string endpoint) {
  size_t num = 0;
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  auto iter = clients_.find(endpoint);
  if (iter != clients_.end()) {
    num = iter->second.size();
  }
  return num;
}

}  // namespace vineyard
