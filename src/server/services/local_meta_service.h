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

#ifndef SRC_SERVER_SERVICES_LOCAL_META_SERVICE_H_
#define SRC_SERVER_SERVICES_LOCAL_META_SERVICE_H_

#include <memory>
#include <string>
#include <vector>

#include "server/services/meta_service.h"

namespace vineyard {

/**
 * @brief LocalLock is designed as the lock for dummy local meta service.
 *
 */
class LocalLock : public ILock {
 public:
  Status Release(unsigned& rev) override {
    return callback_(Status::OK(), rev);
  }
  ~LocalLock() override {}

  explicit LocalLock(const callback_t<unsigned&>& callback, unsigned rev)
      : ILock(rev), callback_(callback) {}

 protected:
  const callback_t<unsigned&> callback_;
};

/**
 * @brief LocalMetaService provides meta services in regards to local, e.g.
 * requesting and committing udpates
 *
 */
class LocalMetaService : public IMetaService {
 public:
  inline void Stop() override;

  ~LocalMetaService() override {}

 protected:
  explicit LocalMetaService(std::shared_ptr<VineyardServer>& server_ptr)
      : IMetaService(server_ptr) {}

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

  Status probe() override { return Status::OK(); }

 private:
  std::shared_ptr<LocalMetaService> shared_from_base() {
    return std::static_pointer_cast<LocalMetaService>(shared_from_this());
  }

  friend class IMetaService;
};
}  // namespace vineyard

#endif  // SRC_SERVER_SERVICES_LOCAL_META_SERVICE_H_
