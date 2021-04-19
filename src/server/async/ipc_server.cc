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

#include "common/util/json.h"

namespace vineyard {

IPCServer::IPCServer(vs_ptr_t vs_ptr)
    : SocketServer(vs_ptr),
      ipc_spec_(vs_ptr_->GetSpec()["ipc_spec"]),
      acceptor_(vs_ptr_->GetContext(), getEndpoint(vs_ptr_->GetContext())),
      socket_(vs_ptr_->GetContext()) {}

IPCServer::~IPCServer() {
  if (acceptor_.is_open()) {
    acceptor_.close();
  }
  std::string const& ipc_socket =
      ipc_spec_["socket"].get_ref<std::string const&>();
  ::unlink(ipc_socket.c_str());
}

void IPCServer::Start() {
  SocketServer::Start();
  LOG(INFO) << "Vineyard will listen on " << ipc_spec_["socket"] << " for IPC";
  vs_ptr_->IPCReady();
}

#if BOOST_VERSION >= 106600
asio::local::stream_protocol::endpoint IPCServer::getEndpoint(
    asio::io_context& context) {
#else
asio::local::stream_protocol::endpoint IPCServer::getEndpoint(
    asio::io_service& context) {
#endif
  std::string const& ipc_socket =
      ipc_spec_["socket"].get_ref<std::string const&>();
  auto endpoint = asio::local::stream_protocol::endpoint(ipc_socket);
  if (access(ipc_socket.c_str(), F_OK) == 0) {
    // first check if the socket file is writable
    if (access(ipc_socket.c_str(), W_OK) != 0) {
      throw std::invalid_argument("Cannot launch vineyardd on " + ipc_socket +
                                  ": " + strerror(errno));
    }
    // then check if the socket file is used by another process, if not, unlink
    // it first, otherwise raise an exception.
    asio::local::stream_protocol::socket socket(context);
    boost::system::error_code ec;
    socket.connect(endpoint, ec);
    if (!ec) {
      throw boost::system::system_error(
          asio::error::make_error_code(asio::error::address_in_use));
    }
  }
  ::unlink(ipc_socket.c_str());
  return endpoint;
}

void IPCServer::doAccept() {
  if (!acceptor_.is_open()) {
    return;
  }
  acceptor_.async_accept(socket_, [this](boost::system::error_code ec) {
    if (!ec) {
      std::shared_ptr<SocketConnection> conn =
          std::make_shared<SocketConnection>(std::move(this->socket_), vs_ptr_,
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
