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

#ifndef SRC_SERVER_SERVER_VINEYARD_RUNNER_H_
#define SRC_SERVER_SERVER_VINEYARD_RUNNER_H_

#include <atomic>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "oneapi/tbb/concurrent_hash_map.h"

#include "common/util/asio.h"
#include "common/util/callback.h"
#include "common/util/json.h"
#include "common/util/protocols.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "server/memory/memory.h"

namespace vineyard {

namespace asio = boost::asio;

class VineyardServer;

using session_dict_t =
    tbb::concurrent_hash_map<SessionID, std::shared_ptr<VineyardServer>>;

class VineyardRunner : public std::enable_shared_from_this<VineyardRunner> {
 public:
  static std::shared_ptr<VineyardRunner> Get(const json& spec);
  Status Serve();
  Status Finalize();
  Status GetRootSession(std::shared_ptr<VineyardServer>& vs_ptr);
  Status CreateNewSession(StoreType const& bulk_store_type,
                          callback_t<std::string const&> callback);
  Status Delete(SessionID const& sid);
  Status Get(SessionID const& sid, std::shared_ptr<VineyardServer>& session);
  bool Exists(SessionID const& sid);
  void Stop();
  bool Running() const;

 private:
  explicit VineyardRunner(const json& spec);

  json spec_template_;

  unsigned int concurrency_;

  asio::io_context context_, meta_context_, io_context_;
#if BOOST_VERSION >= 106600
  using ctx_guard = asio::executor_work_guard<asio::io_context::executor_type>;
#else
  using ctx_guard = std::unique_ptr<asio::io_context::work>;
#endif

  ctx_guard guard_, meta_guard_, io_guard_;
  std::vector<std::thread> workers_, io_workers_;

  session_dict_t sessions_;
  std::atomic_bool stopped_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_SERVER_VINEYARD_RUNNER_H_
