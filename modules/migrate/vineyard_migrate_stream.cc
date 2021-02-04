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

#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>

#include "boost/asio.hpp"
#include "gflags/gflags.h"

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/rpc_client.h"
#include "common/util/boost.h"
#include "common/util/flags.h"
#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

namespace asio = boost::asio;
using boost::asio::generic::stream_protocol;

DEFINE_bool(client, false, "Run as the client");
DEFINE_bool(server, false, "Run as the server");
DEFINE_string(host, "", "RPC host for migration");
DEFINE_uint64(port, 9601, "RPC port for migration");
DEFINE_string(ipc_socket, "/tmp/vineyard/vineyard.sock",
              "IPC socket of vineyard server");
DEFINE_string(
    rpc_endpoint, "",
    "RPC endpoint of the peer vineyard server for fetching complete metadata");
DEFINE_string(id, VYObjectIDToString(InvalidObjectID()),
              "Object to migrate to local");

Status Serve(Client& client, RPCClient& rpc_client, asio::ip::tcp::socket&& socket) {
  ObjectMeta metadata;
  RETURN_ON_ERROR(
      rpc_client.GetMetaData(VYObjectIDFromString(FLAGS_id), metadata, true));

  ObjectID target_id;
  RETURN_ON_ERROR(client.CreateMetaData(metadata, target_id));
  RETURN_ON_ERROR(client.CreateStream(target_id));
  RETURN_ON_ERROR(client.OpenStream(target_id, OpenStreamMode::write));

  // print the result object id to stdout
  std::cout << VYObjectIDToString(target_id) << std::endl;

  while (true) {
    size_t buffer_size;
    asio::read(socket, asio::buffer(&buffer_size, sizeof(size_t)));
    VLOG(10) << "Recieve buffer size " << buffer_size;
    if (buffer_size == 0) {
      client.StopStream(target_id, false);
      LOG(INFO) << "The server finishes its job, exit normally";
      return Status::OK();
    } else if (buffer_size == std::numeric_limits<size_t>::max()) {
      client.StopStream(target_id, true);
      LOG(ERROR) << "The server exit unnormally as the source stream corrupted";
      return Status::StreamFailed();
    } else {
      std::unique_ptr<arrow::MutableBuffer> buffer;
      client.GetNextStreamChunk(target_id, buffer_size, buffer);
      asio::read(socket, asio::buffer(buffer->mutable_data(), buffer_size));
    }
  }
  return Status::OK();
}

Status RunServer() {
#if BOOST_VERSION >= 106600
  asio::io_context context;
#else
  asio::io_service context;
#endif
  asio::ip::tcp::acceptor acceptor(context);
  auto endpoint = asio::ip::tcp::endpoint(asio::ip::tcp::v4(), FLAGS_port);
  acceptor.open(endpoint.protocol());
  using reuse_port =
      asio::detail::socket_option::boolean<SOL_SOCKET, SO_REUSEPORT>;
  // reuse address and port for rpc service.
  acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(true));
  acceptor.set_option(reuse_port(true));
  acceptor.bind(endpoint);
  acceptor.listen();

  Client client;
  RETURN_ON_ERROR(client.Connect(FLAGS_ipc_socket));
  
  RPCClient rpc_client;
  RETURN_ON_ERROR(rpc_client.Connect(FLAGS_rpc_endpoint));

  LOG(INFO) << "Starting server for migration ...";
  asio::ip::tcp::socket socket(context);
  acceptor.accept(socket);
  return Serve(client, rpc_client, std::move(socket));
}


Status Work(Client& client,
            asio::ip::tcp::socket& socket) {
  ObjectID stream_id = VYObjectIDFromString(FLAGS_id);
  client.OpenStream(stream_id, OpenStreamMode::read);
  while (true) {
    std::unique_ptr<arrow::Buffer> buffer;
    Status status = client.PullNextStreamChunk(stream_id, buffer);
    if (status.IsStreamDrained()) {
      size_t buffer_size = 0;
      asio::write(socket, asio::buffer(&buffer_size, sizeof(size_t)));
      return Status::OK();
    } else if (status.ok()) {
      size_t buffer_size = buffer->size();
      asio::write(socket, asio::buffer(&buffer_size, sizeof(size_t)));
      asio::write(socket, asio::buffer(buffer->data(), buffer_size));
    } else {
      size_t buffer_size = std::numeric_limits<size_t>::max();
      asio::write(socket, asio::buffer(&buffer_size, sizeof(size_t)));
      return status;
    }
  }

  return Status::OK();
}

Status RunClient() {
#if BOOST_VERSION >= 106600
  asio::io_context context;
#else
  asio::io_service context;
#endif
  asio::ip::tcp::socket socket(context);
  if (FLAGS_host.empty()) {
    LOG(ERROR)
        << "Host parameter is required to request blob from remote server";
  }
  asio::ip::tcp::resolver resolver(context);
  int retries = 0, max_connect_retries = 10;
  boost::system::error_code ec;
  while (retries < max_connect_retries) {
#if BOOST_VERSION >= 106600
    asio::connect(socket,
                  resolver.resolve(FLAGS_host, std::to_string(FLAGS_port)), ec);
#else
    asio::connect(socket,
                  resolver.resolve(asio::ip::tcp::resolver::query(
                      FLAGS_host, std::to_string(FLAGS_port))),
                  ec);
#endif
    if (ec) {
      LOG(ERROR) << "Failed to connect to migration peer: " << ec.message();
      usleep(static_cast<int>(1 * 1000));
      retries += 1;
    } else {
      break;
    }
  }
  if (ec) {
    LOG(ERROR) << "Failed to connect to migration peer after "
               << max_connect_retries << " retries: " << ec.message();
    exit(-1);
  }

  Client client;
  RETURN_ON_ERROR(client.Connect(FLAGS_ipc_socket));

  auto status = Work(client, socket);

  // ensure stopping the server
  size_t buffer_size = 0;
  asio::write(socket, asio::buffer(&buffer_size, sizeof(size_t)));

  return status;
}

}  // namespace vineyard

DECLARE_bool(help);
DECLARE_string(helpmatch);

int main(int argc, char** argv) {
  sigset(SIGINT, SIG_DFL);
  FLAGS_stderrthreshold = 0;
  vineyard::logging::InitGoogleLogging("vineyard");
  vineyard::flags::SetUsageMessage("Usage: vineyard-migrate [options]");
  vineyard::flags::ParseCommandLineNonHelpFlags(&argc, &argv, false);
  if (FLAGS_help) {
    FLAGS_help = false;
    FLAGS_helpmatch = "vineyard";
  }
  vineyard::flags::HandleCommandLineHelpFlags();

  if (vineyard::FLAGS_client && vineyard::FLAGS_server) {
    LOG(ERROR)
        << "A process cannot be serve as client and server at the same time";
    vineyard::flags::ShowUsageWithFlagsRestrict(argv[0], "vineyard");
    exit(1);
  }
  if (!vineyard::FLAGS_client && !vineyard::FLAGS_server) {
    LOG(ERROR) << "A process must be serve as either client or server";
    vineyard::flags::ShowUsageWithFlagsRestrict(argv[0], "vineyard");
    exit(1);
  }

  vineyard::Status status;
  if (vineyard::FLAGS_client) {
    status = vineyard::RunClient();
  }
  if (vineyard::FLAGS_server) {
    status = vineyard::RunServer();
  }
  if (!status.ok()) {
    LOG(ERROR) << "Migration failed: " << status.ToString();
    return static_cast<int>(status.code());
  }
  return 0;
}
