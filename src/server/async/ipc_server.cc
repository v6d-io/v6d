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

#include "server/async/ipc_server.h"

#include <mutex>
#include <string>
#include <utility>

#include "common/util/boost.h"

namespace vineyard {

IPCServer::IPCServer(vs_ptr_t vs_ptr)
    : SocketServer(vs_ptr),
      ipc_spec_(vs_ptr_->GetSpec().get_child("ipc_spec")),
      acceptor_(vs_ptr_->GetIOContext(), getEndpoint(vs_ptr_->GetIOContext())) {
}

IPCServer::~IPCServer() {
  if (acceptor_.is_open()) {
    acceptor_.close();
  }
  std::string ipc_socket = ipc_spec_.get<std::string>("socket");
  ::unlink(ipc_socket.c_str());
}

void IPCServer::Start() {
  SocketServer::Start();
  vs_ptr_->IPCReady();
}

asio::local::stream_protocol::endpoint IPCServer::getEndpoint(
    asio::io_context& context) {
  std::string ipc_socket = ipc_spec_.get<std::string>("socket");
  auto endpoint = asio::local::stream_protocol::endpoint(ipc_socket);
  // first check if the socket file is used by another process, if not, unlink
  // it first, otherwise raise an exception.
  asio::local::stream_protocol::socket socket(context);
  boost::system::error_code ec;
  socket.connect(endpoint, ec);
  if (!ec) {
    throw boost::system::system_error(
        asio::error::make_error_code(asio::error::address_in_use));
  }
  ::unlink(ipc_socket.c_str());
  return endpoint;
}

void IPCServer::doAccept() {
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
