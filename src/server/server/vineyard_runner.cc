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

#include "server/server/vineyard_runner.h"

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "boost/bind.hpp"

#include "common/util//logging.h"
#include "server/server/vineyard_server.h"

namespace vineyard {

VineyardRunner::VineyardRunner(const json& spec)
    : spec_template_(spec),
      concurrency_(std::thread::hardware_concurrency()),
      context_(concurrency_),
      meta_context_(),
#if BOOST_VERSION >= 106600
      guard_(asio::make_work_guard(context_)),
      meta_guard_(asio::make_work_guard(meta_context_))
#else
      guard_(new boost::asio::io_service::work(context_)),
      meta_guard_(new boost::asio::io_service::work(context_))
#endif
{
}

std::shared_ptr<VineyardRunner> VineyardRunner::Get(const json& spec) {
  return std::shared_ptr<VineyardRunner>(new VineyardRunner(spec));
}

bool VineyardRunner::Running() const { return !stopped_.load(); }

Status VineyardRunner::Serve() {
  stopped_.store(false);

  VINEYARD_ASSERT(sessions_.empty(), "Vineyard Runner already started");
  auto root_vs = std::make_shared<VineyardServer>(
      spec_template_, RootSessionID(), shared_from_this(), context_,
      meta_context_, [](Status const& s, std::string const&) { return s; });
  sessions_.emplace(RootSessionID(), root_vs);

  // start a root session
  VINEYARD_CHECK_OK(root_vs->Serve("Normal"));

  for (unsigned int idx = 0; idx < concurrency_; ++idx) {
#if BOOST_VERSION >= 106600
    workers_.emplace_back(
        boost::bind(&boost::asio::io_context::run, &context_));
#else
    workers_.emplace_back(
        boost::bind(&boost::asio::io_service::run, &context_));
#endif
  }

  meta_context_.run();
  return Status::OK();
}

Status VineyardRunner::Finalize() { return Status::OK(); }

Status VineyardRunner::GetRootSession(vs_ptr_t& vs_ptr) {
  session_map_t::const_accessor accessor;
  if (!sessions_.find(accessor, RootSessionID())) {
    return Status::Invalid("No root session.");
  }
  vs_ptr = accessor->second;
  return Status::OK();
}

Status VineyardRunner::CreateNewSession(
    std::string const& bulk_store_type,
    callback_t<std::string const&> callback) {
  SessionID session_id = GenerateSessionID();
  json spec(spec_template_);

  std::string default_ipc_socket =
      spec["ipc_spec"]["socket"].get<std::string>();

  spec["ipc_spec"]["socket"] =
      default_ipc_socket + "." + SessionIDToString(session_id);

  auto vs_ptr = std::make_shared<VineyardServer>(
      spec, session_id, shared_from_this(), context_, meta_context_, callback);
  sessions_.emplace(session_id, vs_ptr);
  LOG(INFO) << "Vineyard creates a new session with SessionID = "
            << SessionIDToString(session_id) << std::endl;
  return vs_ptr->Serve(bulk_store_type);
}

Status VineyardRunner::Delete(SessionID const& sid) {
  session_map_t::const_accessor accessor;
  if (!sessions_.find(accessor, sid)) {
    return Status::OK();
  }
  accessor->second->Stop();
  sessions_.erase(accessor);
  LOG(INFO) << "Delete session : " << SessionIDToString(sid) << std::endl;
  return Status::OK();
}

Status VineyardRunner::Get(SessionID const& sid, vs_ptr_t& session) {
  session_map_t::const_accessor accessor;
  if (sessions_.find(accessor, sid)) {
    session = accessor->second;
    return Status::OK();
  } else {
    return Status::Invalid("Session (sid = " + SessionIDToString(sid) +
                           ") not exit");
  }
}

bool VineyardRunner::Exists(SessionID const& sid) {
  session_map_t::const_accessor accessor;
  return sessions_.find(accessor, sid);
}

void VineyardRunner::Stop() {
  // Stop creating.
  if (stopped_.exchange(true)) {
    return;
  }

  std::vector<SessionID> session_ids;
  session_ids.reserve(sessions_.size());
  for (auto iter = sessions_.begin(); iter != sessions_.end(); iter++) {
    session_ids.emplace_back(iter->first);
  }
  for (auto const& item : session_ids) {
    VINEYARD_DISCARD(Delete(item));  // trigger item->stop()
  }

  guard_.reset();
  meta_guard_.reset();

  // stop the asio context at last
  context_.stop();
  meta_context_.stop();

  // wait for the IO context finishes.
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

}  // namespace vineyard
