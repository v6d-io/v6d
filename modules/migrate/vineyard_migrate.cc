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

#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>

#include "boost/asio.hpp"

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/rpc_client.h"
#include "common/util/flags.h"
#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

namespace asio = boost::asio;

DEFINE_bool(client, false, "Run as the client");
DEFINE_bool(server, false, "Run as the server");
DEFINE_string(host, "", "RPC host for migration");
DEFINE_uint64(port, 9601, "RPC port for migration");
DEFINE_string(ipc_socket, "/tmp/vineyard/vineyard.sock",
              "IPC socket of vineyard server");
DEFINE_string(
    rpc_endpoint, "",
    "RPC endpoint of the peer vineyard server for fetching complete metadata");
DEFINE_string(id, ObjectIDToString(InvalidObjectID()),
              "Object to migrate to local");
DEFINE_bool(local_copy, false, "Make a copy of blobs even on the same machine");

static void find_blobs_on_remote_instance(const InstanceID remote_instance_id,
                                          const json& tree,
                                          std::set<ObjectID>& blobs) {
  if (tree.empty()) {
    return;
  }
  ObjectID member_id =
      ObjectIDFromString(tree["id"].get_ref<std::string const&>());
  if (IsBlob(member_id)) {
    if (FLAGS_local_copy) {
      blobs.emplace(member_id);
    } else {
      if (tree["instance_id"].get<InstanceID>() == remote_instance_id) {
        blobs.emplace(member_id);
      }
    }
  } else {
    for (auto& item : tree) {
      if (item.is_object()) {
        find_blobs_on_remote_instance(remote_instance_id, item, blobs);
      }
    }
  }
}

Status Serve(Client& client, asio::ip::tcp::socket&& socket) {
  while (true) {
    ObjectID blob_to_send = InvalidObjectID();
    asio::read(socket, asio::buffer(&blob_to_send, sizeof(ObjectID)));
    VLOG(10) << "Receive object id " << ObjectIDToString(blob_to_send);
    if (blob_to_send == InvalidObjectID()) {
      LOG(INFO) << "The server finishes its job, exit normally";
      return Status::OK();
    }
    size_t blob_size = 0;
    std::shared_ptr<Blob> blob = nullptr;
    if (blob_to_send != EmptyBlobID()) {
      blob = client.GetObject<Blob>(blob_to_send);
      blob_size = blob->size();
    }
    asio::write(socket, asio::buffer(&blob_size, sizeof(size_t)));
    if (blob_size > 0) {
      VLOG(10) << "Sending blob payload of size " << blob_size << " ...";
      asio::write(socket, asio::buffer(blob->data(), blob_size));
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

  LOG(INFO) << "Starting server for migration ...";
  asio::ip::tcp::socket socket(context);
  acceptor.accept(socket);
  return Serve(client, std::move(socket));
}

Status Rebuild(Client& client, ObjectMeta const& metadata, ObjectMeta& target,
               std::map<ObjectID, std::shared_ptr<Blob>> const& target_blobs) {
  for (auto const& kv : metadata) {
    if (kv.value().is_object()) {
      ObjectMeta member = metadata.GetMemberMeta(kv.key());
      if (member.GetTypeName() == type_name<Blob>()) {
        target.AddMember(kv.key(), target_blobs.at(member.GetId()));
      } else {
        ObjectMeta subtarget;
        RETURN_ON_ERROR(Rebuild(client, member, subtarget, target_blobs));
        target.AddMember(kv.key(), subtarget);
      }
    } else {
      if (kv.value().is_string()) {
        target.AddKeyValue(kv.key(), kv.value().get_ref<std::string const&>());
      } else if (kv.value().is_boolean()) {
        target.AddKeyValue(kv.key(), kv.value().get<bool>());
      } else if (kv.value().is_number_integer()) {
        target.AddKeyValue(kv.key(), kv.value().get<int64_t>());
      } else if (kv.value().is_number_float()) {
        target.AddKeyValue(kv.key(), kv.value().get<double>());
      } else {
        target.AddKeyValue(kv.key(), kv.value());
      }
    }
  }
  ObjectID target_id = InvalidObjectID();
  RETURN_ON_ERROR(client.CreateMetaData(target, target_id));
  target.SetId(target_id);
  return Status::OK();
}

Status Work(Client& client, RPCClient& rpc_client,
            asio::ip::tcp::socket& socket) {
  // ping to ensure server works as expected
  ObjectID empty_blob_id = EmptyBlobID();
  asio::write(socket, asio::buffer(&empty_blob_id, sizeof(ObjectID)));
  size_t received_size = std::numeric_limits<size_t>::max();
  asio::read(socket, asio::buffer(&received_size, sizeof(size_t)));
  RETURN_ON_ASSERT(received_size == 0,
                   "The remote server work as unexpectedly, abort");
  ObjectMeta metadata;
  RETURN_ON_ERROR(
      rpc_client.GetMetaData(ObjectIDFromString(FLAGS_id), metadata, true));

  // step 1: collect blob set
  std::set<ObjectID> remote_blobs;
  find_blobs_on_remote_instance(rpc_client.remote_instance_id(),
                                metadata.MetaData(), remote_blobs);
  metadata.PrintMeta();
  VLOG(10) << "blob sizes to migrate: " << remote_blobs.size();
  std::map<ObjectID, std::shared_ptr<Blob>> target_blobs;

  // step 2: migrate blobs to local
  for (auto const& blob : remote_blobs) {
    VLOG(10) << "Will migrate blob " << ObjectIDToString(blob) << " to local";
    asio::write(socket, asio::buffer(&blob, sizeof(ObjectID)));
    size_t size_of_blob = std::numeric_limits<size_t>::max();
    asio::read(socket, asio::buffer(&size_of_blob, sizeof(size_t)));
    if (size_of_blob > 0) {
      std::unique_ptr<BlobWriter> buffer;
      RETURN_ON_ERROR(client.CreateBlob(size_of_blob, buffer));
      VLOG(10) << "Receiving blob payload of size " << size_of_blob << " ...";
      asio::read(socket, asio::buffer(buffer->data(), size_of_blob));
      target_blobs.emplace(
          blob, std::dynamic_pointer_cast<Blob>(buffer->Seal(client)));
    } else {
      target_blobs.emplace(blob, Blob::MakeEmpty(client));
    }
  }

  // step 3: rebuild metadata and object
  ObjectMeta target;
  RETURN_ON_ERROR(Rebuild(client, metadata, target, target_blobs));
  RETURN_ON_ERROR(client.Persist(target.GetId()));
  VLOG(10) << "Test: get local object meta ...";
  {
    ObjectMeta migrated;
    RETURN_ON_ERROR(client.GetMetaData(target.GetId(), migrated, false));
    VLOG(10) << "Target object type is " << target.GetTypeName();
  }

  // print the result object id to stdout
  std::cout << ObjectIDToString(target.GetId()) << std::endl;

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
      usleep(static_cast<int>(1 * 1000000));
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

  RPCClient rpc_client;
  RETURN_ON_ERROR(rpc_client.Connect(FLAGS_rpc_endpoint));

  auto status = Work(client, rpc_client, socket);

  // ensure stopping the server
  ObjectID target = InvalidObjectID();
  asio::write(socket, asio::buffer(&target, sizeof(ObjectID)));

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
