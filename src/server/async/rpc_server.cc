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

#include "server/async/rpc_server.h"

#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "common/util/json.h"

namespace vineyard {

RPCServer::RPCServer(vs_ptr_t vs_ptr)
    : SocketServer(vs_ptr),
      rpc_spec_(vs_ptr_->GetSpec()["rpc_spec"]),
      acceptor_(vs_ptr_->GetContext()),
      socket_(vs_ptr_->GetContext()) {
  auto endpoint = getEndpoint(vs_ptr_->GetContext());
  acceptor_.open(endpoint.protocol());
  using reuse_port =
      asio::detail::socket_option::boolean<SOL_SOCKET, SO_REUSEPORT>;
  // reuse address and port for rpc service.
  acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
  acceptor_.set_option(reuse_port(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();
}

RPCServer::~RPCServer() {
  if (acceptor_.is_open()) {
    acceptor_.close();
  }
}

void RPCServer::Start() {
  SocketServer::Start();
  LOG(INFO) << "Vineyard will listen on 0.0.0.0:"
            << rpc_spec_["port"].get<uint32_t>() << " for RPC";
  vs_ptr_->RPCReady();
}

#if BOOST_VERSION >= 106600
asio::ip::tcp::endpoint RPCServer::getEndpoint(asio::io_context&) {
#else
asio::ip::tcp::endpoint RPCServer::getEndpoint(asio::io_service&) {
#endif
  uint32_t port = rpc_spec_["port"].get<uint32_t>();
  return asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port);
}

void RPCServer::doAccept() {
  if (!acceptor_.is_open()) {
    return;
  }
  acceptor_.async_accept(socket_, [this](boost::system::error_code ec) {
    if (!ec) {
      std::shared_ptr<SocketConnection> conn =
          std::make_shared<SocketConnection>(std::move(this->socket_), vs_ptr_,
                                             this, next_conn_id_);
      conn->Start();
      std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutx_);
      connections_.emplace(next_conn_id_, conn);
      ++next_conn_id_;
    }
    // don't continue when the iocontext being cancelled.
    if (!stopped_.load()) {
      doAccept();
    }
  });
}

}  // namespace vineyard
