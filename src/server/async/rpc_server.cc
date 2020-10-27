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

#include "common/util/boost.h"

namespace vineyard {

RPCServer::RPCServer(vs_ptr_t vs_ptr)
    : SocketServer(vs_ptr),
      rpc_spec_(vs_ptr_->GetSpec().get_child("rpc_spec")),
      acceptor_(vs_ptr_->GetIOContext()) {
  auto endpoint = getEndpoint(vs_ptr_->GetIOContext());
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
  vs_ptr_->RPCReady();
}

asio::ip::tcp::endpoint RPCServer::getEndpoint(asio::io_context&) {
  uint32_t port = rpc_spec_.get<uint32_t>("port");
  return asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port);
}

void RPCServer::doAccept() {
  if (!acceptor_.is_open()) {
    return;
  }
  acceptor_.async_accept(
      [this](boost::system::error_code ec, stream_protocol::socket socket) {
        if (!ec) {
          std::shared_ptr<SocketConnection> conn =
              std::make_shared<SocketConnection>(std::move(socket), vs_ptr_,
                                                 this, next_conn_id_);
          conn->Start();
          std::lock_guard<std::mutex> scope_lock(this->connections_mutx_);
          connections_.emplace(next_conn_id_, conn);
          ++next_conn_id_;
        }
        doAccept();
      });
}

}  // namespace vineyard
