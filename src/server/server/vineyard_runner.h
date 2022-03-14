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

#ifndef SRC_SERVER_SERVER_VINEYARD_RUNNER_H_
#define SRC_SERVER_SERVER_VINEYARD_RUNNER_H_

#include <atomic>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "boost/asio.hpp"

#include "common/util/callback.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

#include "server/memory/memory.h"
#include "server/memory/stream_store.h"

#include "oneapi/tbb/concurrent_hash_map.h"

namespace vineyard {

namespace asio = boost::asio;
using session_map_t = tbb::concurrent_hash_map<SessionId, vs_ptr_t>;

class VineyardServer;

class VineyardRunner : public std::enable_shared_from_this<VineyardRunner> {
 public:
  static std::shared_ptr<VineyardRunner> Get(const json& spec);
  Status Serve();
  Status Finalize();
  Status GetRootSession(vs_ptr_t& vs_ptr);
  Status CreateNewSession(std::string& ipc_socket);
  Status Delete(SessionId const& sid);
  Status Get(SessionId const& sid, vs_ptr_t& session);
  bool Exists(SessionId const& sid);
  void Stop();
  bool Running() const;

 private:
  explicit VineyardRunner(const json& spec);

  json spec_template_;

  unsigned int concurrency_;

#if BOOST_VERSION >= 106600
  asio::io_context context_, meta_context_;
  using ctx_guard = asio::executor_work_guard<asio::io_context::executor_type>;
#else
  asio::io_service context_, meta_context_;
  using ctx_guard = std::unique_ptr<boost::asio::io_service::work>;
#endif

  ctx_guard guard_, meta_guard_;
  std::vector<std::thread> workers_;

  session_map_t sessions_;
  std::atomic_bool stopped_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_SERVER_VINEYARD_RUNNER_H_
