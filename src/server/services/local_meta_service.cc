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

#include "server/services/local_meta_service.h"

#include <string>
#include <vector>

namespace vineyard {

inline void LocalMetaService::Stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  IMetaService::Stop();
}

void LocalMetaService::requestLock(
    std::string lock_name,
    callback_t<std::shared_ptr<ILock>> callback_after_locked) {
  auto lock_ptr = std::make_shared<LocalLock>(
      [](const Status& status, unsigned& rev) { return Status::OK(); }, 0);
  VINEYARD_SUPPRESS(callback_after_locked(Status::OK(), lock_ptr));
}

void LocalMetaService::commitUpdates(
    const std::vector<op_t>& changes,
    callback_t<unsigned> callback_after_updated) {
  server_ptr_->GetMetaContext().post(
      boost::bind(callback_after_updated, Status::OK(), 0));
}

void LocalMetaService::requestAll(
    const std::string& prefix, unsigned base_rev,
    callback_t<const std::vector<op_t>&, unsigned> callback) {
  server_ptr_->GetMetaContext().post(
      boost::bind(callback, Status::OK(), std::vector<op_t>{}, 0));
}

void LocalMetaService::requestUpdates(
    const std::string& prefix, unsigned since_rev,
    callback_t<const std::vector<op_t>&, unsigned> callback) {
  server_ptr_->GetMetaContext().post(
      boost::bind(callback, Status::OK(), std::vector<op_t>{}, 0));
}

void LocalMetaService::startDaemonWatch(
    const std::string& prefix, unsigned since_rev,
    callback_t<const std::vector<op_t>&, unsigned, callback_t<unsigned>>
        callback) {}

}  // namespace vineyard
