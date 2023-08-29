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

#include "server/server/vineyard_runner.h"

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "boost/bind/bind.hpp"  // IWYU pragma: keep

#include "common/util/json.h"
#include "common/util/likely.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "server/server/vineyard_server.h"

namespace vineyard {

VineyardRunner::VineyardRunner(const json& spec)
    : spec_template_(spec),
      concurrency_(std::thread::hardware_concurrency()),
      context_(concurrency_),
      meta_context_(),
      io_context_(concurrency_),
#if BOOST_VERSION < 106600
      guard_(new asio::io_service::work(context_)),
      meta_guard_(new asio::io_service::work(context_)),
      io_guard_(new asio::io_service::work(context_))
#else
      guard_(asio::make_work_guard(context_)),
      meta_guard_(asio::make_work_guard(meta_context_)),
      io_guard_(asio::make_work_guard(io_context_))
#endif
{
}

std::shared_ptr<VineyardRunner> VineyardRunner::Get(const json& spec) {
  return std::shared_ptr<VineyardRunner>(new VineyardRunner(spec));
}

bool VineyardRunner::Running() const { return !stopped_.load(); }

Status VineyardRunner::Serve() {
  stopped_.store(false);

  VINEYARD_ASSERT(sessions_.empty(), "Vineyard runner already started");
  auto root_vs = std::make_shared<VineyardServer>(
      spec_template_, RootSessionID(), shared_from_this(), context_,
      meta_context_, io_context_,
      [](Status const& s, std::string const&) { return s; });
  sessions_.insert(RootSessionID(), root_vs);

  // start a root session
  VINEYARD_CHECK_OK(root_vs->Serve(StoreType::kDefault));

  for (unsigned int idx = 0; idx < concurrency_; ++idx) {
    workers_.emplace_back(
        boost::bind(&boost::asio::io_context::run, &context_));
  }

  for (unsigned int idx = 0; idx < concurrency_ / 2 + 1; ++idx) {
    io_workers_.emplace_back(
        boost::bind(&boost::asio::io_context::run, &io_context_));
  }

  meta_context_.run();
  return Status::OK();
}

Status VineyardRunner::Finalize() { return Status::OK(); }

Status VineyardRunner::GetRootSession(std::shared_ptr<VineyardServer>& vs_ptr) {
  if (sessions_.find(RootSessionID(), vs_ptr)) {
    return Status::OK();
  } else {
    return Status::Invalid("Cannot find the root session.");
  }
}

Status VineyardRunner::CreateNewSession(
    StoreType const& bulk_store_type, callback_t<std::string const&> callback) {
  SessionID session_id = GenerateSessionID();
  json spec(spec_template_);

  std::string default_ipc_socket =
      spec["ipc_spec"]["socket"].get<std::string>();

  spec["ipc_spec"]["socket"] =
      default_ipc_socket + "." + SessionIDToString(session_id);

  auto vs_ptr = std::make_shared<VineyardServer>(
      spec, session_id, shared_from_this(), context_, meta_context_,
      io_context_, callback);
  sessions_.insert(session_id, vs_ptr);
  LOG(INFO) << "Vineyard creates a new session with ID '"
            << SessionIDToString(session_id) << "'";
  return vs_ptr->Serve(bulk_store_type);
}

Status VineyardRunner::Delete(SessionID const& sid) {
  sessions_.erase_fn(sid, [](std::shared_ptr<VineyardServer>& vs_ptr) -> bool {
    vs_ptr->Stop();
    return true;
  });
  if (unlikely(sid != RootSessionID())) {
    LOG(INFO) << "Deleting session: '" << SessionIDToString(sid) << "'";
  }
  return Status::OK();
}

Status VineyardRunner::Get(SessionID const& sid,
                           std::shared_ptr<VineyardServer>& session) {
  if (sessions_.find(sid, session)) {
    return Status::OK();
  } else {
    return Status::Invalid("Session (sid = " + SessionIDToString(sid) +
                           ") not exit");
  }
}

bool VineyardRunner::Exists(SessionID const& sid) {
  return sessions_.contains(sid);
}

void VineyardRunner::Stop() {
  // Stop creating.
  if (stopped_.exchange(true)) {
    return;
  }

  std::vector<SessionID> session_ids;
  session_ids.reserve(sessions_.size());
  {
    auto locked = sessions_.lock_table();
    for (auto iter = locked.begin(); iter != locked.end(); iter++) {
      session_ids.emplace_back(iter->first);
    }
  }
  for (auto const& item : session_ids) {
    VINEYARD_DISCARD(Delete(item));  // trigger item->stop()
  }

  io_guard_.reset();
  meta_guard_.reset();
  guard_.reset();

  // stop the asio context at last
  io_context_.stop();
  meta_context_.stop();
  context_.stop();

  // wait for the IO context finishes.
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

}  // namespace vineyard
