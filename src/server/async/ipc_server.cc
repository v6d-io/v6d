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

#include "server/async/ipc_server.h"

#include <mutex>
#include <string>
#include <utility>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

#include "common/util/json.h"
#include "common/util/logging.h"
#include "server/server/vineyard_server.h"

namespace vineyard {

IPCServer::IPCServer(std::shared_ptr<VineyardServer> vs_ptr)
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
  std::string const& ipc_socket =
      ipc_spec_["socket"].get_ref<std::string const&>();
  chmod(ipc_socket.c_str(),
        S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);

  vs_ptr_->IPCReady();
  SocketServer::Start();
  LOG(INFO) << "Vineyard will listen on " << ipc_spec_["socket"] << " for IPC";
}

void IPCServer::Close() {
  SocketServer::Close();
  boost::system::error_code ec;
  acceptor_.cancel(ec);
  if (ec) {
    LOG(ERROR) << "Failed to close session : " << ec.message();
  }
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
      std::string reason = strerror(errno);
      if (errno == EACCES) {
        reason +=
            ",\n\n  - please run vineyardd as root using 'sudo',\n"
            "  - or use another IPC socket path to start vineyardd,\n\n"
            "\te.g., vineyardd --socket=/tmp/vineyard.sock\n\n"
            "  for more vineyardd options, see also: vineyard --help";
      }
      throw std::invalid_argument("cannot launch vineyardd on '" + ipc_socket +
                                  "': " + reason);
    }
    // then check if the socket file is used by another process, if not, unlink
    // it first, otherwise raise an exception.
    asio::local::stream_protocol::socket socket(context);
    boost::system::error_code ec;
    socket.connect(endpoint, ec);
    if (!ec) {
      std::string message =
          "the UNIX-domain socket '" + ipc_socket +
          "' has already been listened on,\n\n"
          "  - please use another IPC socket path to start vineyardd,\n\n"
          "\te.g., vineyardd --socket=/tmp/vineyard.sock\n\n"
          "  for more vineyardd options, see also: vineyard --help\n\n";
      throw boost::system::system_error(
          asio::error::make_error_code(asio::error::address_in_use), message);
    }
  } else if (errno == ENOENT) {
    // create parent directory
    auto socket_path =
        boost::filesystem::absolute(boost::filesystem::path(ipc_socket));
    boost::system::error_code ec;
    boost::filesystem::create_directories(socket_path.parent_path(), ec);
    if (ec) {
      std::string message = "Failed to create parent directory '" +
                            socket_path.parent_path().string() +
                            "' for specific socket path";
      throw boost::system::system_error(ec, message);
    }
  } else {
    throw boost::system::system_error(
        boost::system::errc::make_error_code(boost::system::errc::io_error),
        strerror(errno));
  }
  ::unlink(ipc_socket.c_str());
  return endpoint;
}

void IPCServer::doAccept() {
  if (!acceptor_.is_open()) {
    return;
  }
  auto self(shared_from_this());
  acceptor_.async_accept(socket_, [self](boost::system::error_code ec) {
    if (!ec) {
      std::lock_guard<std::recursive_mutex> scope_lock(
          self->connections_mutex_);
      if (self->stopped_.load() || self->closable_.load()) {
        return;
      }
      std::shared_ptr<SocketConnection> conn =
          std::make_shared<SocketConnection>(std::move(self->socket_),
                                             self->vs_ptr_, self,
                                             self->next_conn_id_);
      conn->Start();
      self->connections_.emplace(self->next_conn_id_, conn);
      ++self->next_conn_id_;
    }
    // don't continue when the iocontext being cancelled or the session is going
    // to close.
    if (!ec || ec != boost::system::errc::operation_canceled) {
      if (!self->stopped_.load() || !self->closable_.load()) {
        self->doAccept();
      }
    }
  });
}

}  // namespace vineyard
