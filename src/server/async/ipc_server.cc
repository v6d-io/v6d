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

#include "server/async/ipc_server.h"

#include <mutex>
#include <string>
#include <utility>

#include "gulrak/filesystem.hpp"

#include "common/util/env.h"
#include "common/util/json.h"
#include "server/server/vineyard_server.h"

namespace vineyard {

namespace detail {

static bool check_connectable(asio::io_context& context,
                              std::string const& path) {
  asio::local::stream_protocol::socket socket(context);
  boost::system::error_code ec;
  ec = socket.connect(path, ec);
  return !ec;
}

static bool check_listenable(asio::io_context& context, std::string const& path,
                             std::string& error_message) {
  // create parent directory
  std::error_code ec;
  auto socket_path = ghc::filesystem::absolute(ghc::filesystem::path(path), ec);
  if (ec) {
    error_message =
        "Failed to resolve the absolute path for specified socket path '" +
        path + "': " + ec.message();
    return false;
  }
  ghc::filesystem::create_directories(socket_path.parent_path(), ec);
  if (ec) {
    error_message = "Failed to create parent directory '" +
                    socket_path.parent_path().string() +
                    "' for specified socket path '" + path +
                    "': " + ec.message();
    return false;
  }
  if (ghc::filesystem::exists(socket_path, ec)) {
    if (ghc::filesystem::is_socket(socket_path, ec)) {
      if (check_connectable(context, path)) {
        error_message =
            "the UNIX-domain socket '" + path +
            "' is already inuse and has been listened on,\n\n"
            "  - please use another IPC socket path to start vineyardd,\n\n"
            "\te.g., vineyardd --socket=/tmp/vineyard.sock\n\n"
            "  for more vineyardd options, see also: vineyard --help\n\n";
        return false;
      }
      if (!ghc::filesystem::remove(socket_path, ec) &&
          ec != std::errc::no_such_file_or_directory) {
        error_message =
            "Permission error when attempting to write the UNIX-domain socket "
            "'" +
            path + "': " + strerror(errno) +
            ",\n\n"
            "  - please use another IPC socket path to start vineyardd,\n\n"
            "\te.g., vineyardd --socket=/tmp/vineyard.sock\n\n"
            "  for more vineyardd options, see also: vineyard --help\n\n";
        return false;
      }
      return true;
    } else {
      error_message =
          "the UNIX-domain socket '" + path +
          "' is not a named socket,\n\n"
          "  - please use another IPC socket path to start vineyardd,\n\n"
          "\te.g., vineyardd --socket=/tmp/vineyard.sock\n\n"
          "  for more vineyardd options, see also: vineyard --help\n\n";
      return false;
    }
  }
  // check if we have permission to create the new file
  std::ofstream ofs(socket_path.string(), std::ios::out | std::ios::binary);
  if (ofs.fail()) {
    error_message =
        "Permission error when attempting to create the UNIX-domain socket "
        "'" +
        path + "': " + strerror(errno) +
        ",\n\n"
        "  - please use another IPC socket path to start vineyardd,\n\n"
        "\te.g., vineyardd --socket=/tmp/vineyard.sock\n\n"
        "  for more vineyardd options, see also: vineyard --help\n\n";
    return false;
  } else {
    ofs.close();
    ghc::filesystem::remove(socket_path, ec);
    return true;
  }
}

}  // namespace detail

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
  chmod(ipc_socket.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP |
                                S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH);

  vs_ptr_->IPCReady();
  SocketServer::Start();
  LOG(INFO) << "Vineyard will listen on " << ipc_spec_["socket"] << " for IPC";
}

void IPCServer::Close() {
  SocketServer::Close();
  boost::system::error_code ec;
  ec = acceptor_.cancel(ec);
  if (ec) {
    LOG(ERROR) << "Failed to close the IPC server: " << ec.message();
  }
}

asio::local::stream_protocol::endpoint IPCServer::getEndpoint(
    asio::io_context& context) {
  std::string ipc_socket = ipc_spec_["socket"].get<std::string>();
  std::string error_message;
  if (ipc_socket.empty()) {
    ipc_socket = "/var/run/vineyard.sock";
    if (detail::check_listenable(context, ipc_socket, error_message)) {
      ::unlink(ipc_socket.c_str());
      ipc_spec_["socket"] = ipc_socket;
      return asio::local::stream_protocol::endpoint(ipc_socket);
    }
    ipc_socket = read_env("HOME") + "/.local/vineyard/vineyard.sock";
    LOG(WARNING)
        << "Failed to listen on default socket '/var/run/vineyard.sock'";
    LOG(INFO) << "Falling back to '" << ipc_socket << "' ...";
    if (detail::check_listenable(context, ipc_socket, error_message)) {
      ::unlink(ipc_socket.c_str());
      ipc_spec_["socket"] = ipc_socket;
      return asio::local::stream_protocol::endpoint(ipc_socket);
    }
  } else {
    if (detail::check_listenable(context, ipc_socket, error_message)) {
      ::unlink(ipc_socket.c_str());
      return asio::local::stream_protocol::endpoint(ipc_socket);
    }
  }
  throw std::invalid_argument(error_message);
}

Status IPCServer::Register(std::shared_ptr<SocketConnection> conn,
                           const SessionID session_id) {
  conn->registered_.store(true);
  return Status::OK();
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
