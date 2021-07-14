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

#ifndef SRC_SERVER_ASYNC_RPC_SERVER_H_
#define SRC_SERVER_ASYNC_RPC_SERVER_H_

#include <string>

#include "boost/asio.hpp"

#include "common/util/env.h"
#include "server/async/socket_server.h"
#include "server/server/vineyard_server.h"

namespace vineyard {

/**
 * @brief A kind of server that supports remote procedure call (RPC)
 *
 */
class RPCServer : public SocketServer {
 public:
  explicit RPCServer(vs_ptr_t vs_ptr);

  ~RPCServer() override;

  void Start() override;

  std::string Endpoint() {
    return get_hostname() + ":" + json_to_string(rpc_spec_["port"]);
  }

 private:
#if BOOST_VERSION >= 106600
  asio::ip::tcp::endpoint getEndpoint(asio::io_context&);
#else
  asio::ip::tcp::endpoint getEndpoint(asio::io_service&);
#endif

  void doAccept() override;

  const json rpc_spec_;
  asio::ip::tcp::acceptor acceptor_;
  asio::ip::tcp::socket socket_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_ASYNC_RPC_SERVER_H_
