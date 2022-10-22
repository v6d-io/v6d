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

#ifndef SRC_SERVER_ASYNC_IPC_SERVER_H_
#define SRC_SERVER_ASYNC_IPC_SERVER_H_

#include <memory>
#include <string>

#include "common/util/asio.h"
#include "common/util/protocols.h"
#include "server/async/socket_server.h"
#include "server/memory/memory.h"

namespace vineyard {

class VineyardServer;

/**
 * @brief The server for inter-process communication (IPC)
 *
 */
class IPCServer : public SocketServer,
                  public std::enable_shared_from_this<IPCServer> {
 public:
  explicit IPCServer(std::shared_ptr<VineyardServer> vs_ptr);

  ~IPCServer() override;

  void Start() override;

  void Close() override;

  std::string Socket() {
    return ipc_spec_["socket"].get_ref<std::string const&>();
  }

  Status Register(std::shared_ptr<SocketConnection> conn,
                  const SessionID session_id) override;

 private:
  asio::local::stream_protocol::endpoint getEndpoint(asio::io_context&);

  void doAccept() override;

  const json ipc_spec_;
  asio::local::stream_protocol::acceptor acceptor_;
  asio::local::stream_protocol::socket socket_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_ASYNC_IPC_SERVER_H_
