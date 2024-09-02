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
#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/compression/compressor.h"
#include "common/rdma/rdma_client.h"
#include "common/util/asio.h"
#include "common/util/protocols.h"
#include "server/server/vineyard_server.h"
#include "server/util/remote.h"

namespace vineyard {

RemoteClient::RemoteClient(const std::shared_ptr<VineyardServer> server_ptr)
    : server_ptr_(server_ptr),
      context_(server_ptr->GetIOContext()),
      remote_tcp_socket_(context_),
      socket_(context_),
      connected_(false),
      rdma_connected_(false) {}

RemoteClient::~RemoteClient() {
  boost::system::error_code ec;
  ec = socket_.close(ec);
  Status status = StopRDMA();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to stop RDMA client: " << status.message()
               << ". May "
                  "cause memory leak.";
  }
}

Status RemoteClient::StopRDMA() {
  if (!rdma_connected_) {
    return Status::OK();
  }
  rdma_connected_ = false;

  void* msg;
  RETURN_ON_ERROR(rdma_client_->GetTXFreeMsgBuffer(msg));
  VineyardMsg* vmsg = reinterpret_cast<VineyardMsg*>(msg);
  vmsg->type = VINEYARD_MSG_CLOSE;
  RETURN_ON_ERROR(rdma_client_->Send(msg, sizeof(VineyardMsg), nullptr));
  RETURN_ON_ERROR(rdma_client_->GetTXCompletion(-1, nullptr));

  RETURN_ON_ERROR(rdma_client_->Stop());
  RETURN_ON_ERROR(rdma_client_->Close());
  RETURN_ON_ERROR(RDMAClientCreator::Release(rdma_endpoint_));
  return Status::OK();
}

Status RemoteClient::Connect(const std::string& rpc_endpoint,
                             const SessionID session_id,
                             const std::string& rdma_endpoint) {
  size_t pos = rpc_endpoint.find(":");
  std::string host, port;
  if (pos == std::string::npos) {
    host = rpc_endpoint;
    port = "9600";
  } else {
    host = rpc_endpoint.substr(0, pos);
    port = rpc_endpoint.substr(pos + 1);
  }

  RETURN_ON_ERROR(
      Connect(host, static_cast<uint32_t>(std::stoul(port)), session_id));

  std::string rdma_host, rdma_port;
  pos = rdma_endpoint.find(":");
  if (pos == std::string::npos) {
    VLOG(100) << "No RDMA endpoint provided. Fall back to TCP.";
  } else {
    rdma_host = rdma_endpoint.substr(0, pos);
    rdma_port = rdma_endpoint.substr(pos + 1);
  }

  Status status = ConnectRDMAServer(rdma_host, std::atoi(rdma_port.c_str()));
  if (status.ok()) {
    rdma_endpoint_ = rdma_host + ":" + rdma_port;
    VLOG(100) << "Connect to RDMA server successfully. RDMA host:" << rdma_host
              << ", port:" << rdma_port;
  } else {
    VLOG(100) << "Failed to connect to RDMA server. Fall back to TCP. Error:"
              << status.message();
  }

  return Status::OK();
}

Status RemoteClient::RDMARequestMemInfo(RegisterMemInfo& remote_info) {
  void* buffer;
  VINEYARD_DISCARD(this->rdma_client_->GetTXFreeMsgBuffer(buffer));
  VineyardMsg* msg = reinterpret_cast<VineyardMsg*>(buffer);
  msg->type = VINEYARD_MSG_REQUEST_MEM;
  msg->remoteMemInfo.remote_address = (uint64_t) remote_info.address;
  msg->remoteMemInfo.len = remote_info.size;
  VLOG(100) << "Request remote addr: "
            << reinterpret_cast<void*>(msg->remoteMemInfo.remote_address);
  void* remoteMsg;
  VINEYARD_DISCARD(this->rdma_client_->GetRXFreeMsgBuffer(remoteMsg));
  memset(remoteMsg, 0, 64);
  VINEYARD_CHECK_OK(
      this->rdma_client_->Recv(remoteMsg, sizeof(VineyardMsg), nullptr));
  VINEYARD_CHECK_OK(
      this->rdma_client_->Send(buffer, sizeof(VineyardMsg), nullptr));
  VINEYARD_CHECK_OK(rdma_client_->GetTXCompletion(-1, nullptr));

  VINEYARD_CHECK_OK(rdma_client_->GetRXCompletion(-1, nullptr));

  VineyardMsg* vmsg = reinterpret_cast<VineyardMsg*>(remoteMsg);
  if (vmsg->type == VINEYARD_MSG_REQUEST_MEM) {
    remote_info.address = vmsg->remoteMemInfo.remote_address;
    remote_info.rkey = vmsg->remoteMemInfo.key;
    remote_info.size = vmsg->remoteMemInfo.len;
    VLOG(100) << "Get remote address: "
              << reinterpret_cast<void*>(remote_info.address)
              << ", rkey: " << remote_info.rkey
              << ", size: " << remote_info.size;
  } else {
    LOG(ERROR) << "Unknown message type: " << vmsg->type;
  }
  return Status::OK();
}

Status RemoteClient::RDMAReleaseMemInfo(RegisterMemInfo& remote_info) {
  void* buffer;
  VINEYARD_DISCARD(this->rdma_client_->GetTXFreeMsgBuffer(buffer));
  VineyardMsg* msg = reinterpret_cast<VineyardMsg*>(buffer);
  msg->type = VINEYARD_MSG_RELEASE_MEM;
  msg->remoteMemInfo.remote_address = (uint64_t) remote_info.address;
  msg->remoteMemInfo.len = remote_info.size;
  VLOG(100) << "Send remote addr: "
            << reinterpret_cast<void*>(msg->remoteMemInfo.remote_address)
            << ", rkey: " << msg->remoteMemInfo.key;

  RETURN_ON_ERROR(
      this->rdma_client_->Send(buffer, sizeof(VineyardMsg), nullptr));
  VINEYARD_CHECK_OK(rdma_client_->GetTXCompletion(-1, nullptr));

  return Status::OK();
}

Status RemoteClient::ConnectRDMAServer(const std::string& host,
                                       const uint32_t port) {
  if (this->rdma_connected_) {
    return Status::OK();
  }

  RETURN_ON_ERROR(RDMAClientCreator::Create(this->rdma_client_, host, port));

  VLOG(100) << "Try to connect to RDMA server " << host << ":" << port << "...";
  RETURN_ON_ERROR(this->rdma_client_->Connect());
  this->rdma_connected_ = true;
  return Status::OK();
}

Status RemoteClient::Connect(const std::string& host, const uint32_t port,
                             const SessionID session_id) {
  if (this->connected_) {
    return Status::OK();
  }

  asio::ip::tcp::resolver resolver(context_);
  int retries = 0, max_connect_retries = 10;
  boost::system::error_code ec;
  while (retries < max_connect_retries) {
#if BOOST_VERSION >= 106600
    asio::connect(remote_tcp_socket_,
                  resolver.resolve(host, std::to_string(port)), ec);
#else
    asio::connect(remote_tcp_socket_,
                  resolver.resolve(asio::ip::tcp::resolver::query(
                      host, std::to_string(port))),
                  ec);
#endif
    if (ec) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      retries += 1;
    } else {
      break;
    }
  }
  if (ec) {
    return Status::IOError("Failed to connect to peer after " +
                           std::to_string(max_connect_retries) +
                           " retries: " + ec.message());
  }
  socket_ = std::move(remote_tcp_socket_);

  std::string message_out;
  WriteRegisterRequest(message_out, StoreType::kDefault, session_id);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  std::string ipc_socket_value, rpc_endpoint_value;
  bool store_match, support_rpc_compression;
  SessionID session_id_;
  std::string server_version_;
  RETURN_ON_ERROR(ReadRegisterReply(
      message_in, ipc_socket_value, rpc_endpoint_value, remote_instance_id_,
      session_id_, server_version_, store_match, support_rpc_compression));
  this->connected_ = true;
  return Status::OK();
}

Status RemoteClient::MigrateObject(const ObjectID object_id, const json& meta,
                                   callback_t<const ObjectID> callback) {
  if (meta.value("global", false)) {
    VINEYARD_DISCARD(callback(Status::OK(), object_id));
    return Status::OK();
  }
  InstanceID remote_instance_id =
      meta.value("instance_id", UnspecifiedInstanceID());
  if (remote_instance_id == this->server_ptr_->instance_id()) {
    VINEYARD_DISCARD(callback(Status::OK(), object_id));
    return Status::OK();
  }

  if (remote_instance_id == UnspecifiedInstanceID()) {
    return Status::Invalid(
        "The object is not local, and been distributed across instances");
  }
  if (remote_instance_id != this->remote_instance_id_) {
    return Status::Invalid(
        "The object is not local, and the remote instance id is not does't "
        "match");
  }

  DLOG(INFO) << "Migrating object " << ObjectIDToString(object_id)
             << " from instance " << remote_instance_id << " to instance "
             << server_ptr_->instance_id();

  // inspect the metadata to collect buffers
  std::set<ObjectID> blobs;
  RETURN_ON_ERROR(this->collectRemoteBlobs(meta, blobs));

  // migrate all the blobs from remote server
  auto self(shared_from_this());
  RETURN_ON_ERROR(this->migrateBuffers(
      blobs,
      [self, callback, meta](const Status& status,
                             std::map<ObjectID, ObjectID> const& result_blobs) {
        if (status.ok()) {
          json result = json::object();
          auto s = self->recreateMetadata(meta, result, result_blobs);
          if (s.ok()) {
            return self->server_ptr_->CreateData(
                result, true,
                [self, callback, meta](
                    const Status& status, const ObjectID object_id,
                    const Signature signature, const InstanceID instance_id) {
                  RETURN_ON_ASSERT(
                      signature == meta.value("signature", InvalidSignature()),
                      "Signature after migration doesn't match");
                  RETURN_ON_ASSERT(
                      instance_id == self->server_ptr_->instance_id(),
                      "Instance id after migration doesn't match");
                  return self->server_ptr_->Persist(
                      object_id,
                      [self, callback, object_id](const Status& status) {
                        return callback(status, object_id);
                      });
                });
          } else {
            return callback(status, InvalidObjectID());
          }
          return Status::OK();
        } else {
          return callback(status, InvalidObjectID());
        }
      }));
  return Status::OK();
}

Status RemoteClient::collectRemoteBlobs(const json& tree,
                                        std::set<ObjectID>& blobs) {
  if (tree.empty()) {
    return Status::OK();
  }
  ObjectID member_id =
      ObjectIDFromString(tree["id"].get_ref<std::string const&>());
  if (IsBlob(member_id)) {
    DLOG(INFO) << "Migration: instance id for blob " << member_id << " is "
               << tree["instance_id"].get<InstanceID>() << std::endl;
    blobs.emplace(member_id);
  } else {
    for (auto& item : tree) {
      if (item.is_object()) {
        RETURN_ON_ERROR(collectRemoteBlobs(item, blobs));
      }
    }
  }
  return Status::OK();
}

Status RemoteClient::recreateMetadata(
    json const& metadata, json& target,
    std::map<ObjectID, ObjectID> const& result_blobs) {
  for (auto const& kv : metadata.items()) {
    if (kv.key() == "id") {
      continue;
    }
    if (kv.value().is_object()) {
      json member = kv.value();
      if (member.value("typename", "") == "vineyard::Blob") {
        json blob_meta;
        blob_meta["id"] = ObjectIDToString(result_blobs.at(
            ObjectIDFromString(member["id"].get_ref<std::string const&>())));
        blob_meta["typename"] = "vineyard::Blob";
        blob_meta["nbytes"] = member.value("nbytes", 0);
        blob_meta["instance_id"] = this->server_ptr_->instance_id();
        target[kv.key()] = blob_meta;
      } else {
        json sub_result;
        RETURN_ON_ERROR(recreateMetadata(member, sub_result, result_blobs));
        target[kv.key()] = sub_result;
      }
    } else {
      target[kv.key()] = kv.value();
    }
  }
  target["instance_id"] = this->server_ptr_->instance_id();
  target["transient"] = true;
  return Status::OK();
}

Status RemoteClient::migrateBuffers(
    const std::set<ObjectID> blobs,
    callback_t<const std::map<ObjectID, ObjectID>&> callback) {
  std::vector<Payload> payloads;
  std::vector<int> fd_sent;
  bool compress = server_ptr_->GetSpec().value(
      "compression", true);  // enable compression for migration

  std::string message_out;
  WriteGetRemoteBuffersRequest(blobs, false, compress, rdma_connected_,
                               message_out);
  RETURN_ON_ERROR(doWrite(message_out));
  json message_in;
  RETURN_ON_ERROR(doRead(message_in));
  RETURN_ON_ERROR(ReadGetBuffersReply(message_in, payloads, fd_sent, compress));
  RETURN_ON_ASSERT(payloads.size() == blobs.size(),
                   "The result size doesn't match with the requested sizes: " +
                       std::to_string(payloads.size()) + " vs. " +
                       std::to_string(blobs.size()));

  std::vector<std::shared_ptr<Payload>> results;
  Status status = Status::OK();
  for (auto const& payload : payloads) {
    if (payload.data_size == 0) {
      results.emplace_back(Payload::MakeEmpty());
    } else {
      ObjectID object_id;
      std::shared_ptr<Payload> object;
      status = this->server_ptr_->GetBulkStore()->Create(payload.data_size,
                                                         object_id, object);
      if (!status.ok()) {
        break;
      }
      results.emplace_back(object);
    }
  }
  if (!status.ok()) {
    for (auto const& object : results) {
      if (object && object->data_size > 0) {
        VINEYARD_DISCARD(
            this->server_ptr_->GetBulkStore()->Delete(object->object_id));
      }
    }
    return status;
  }

  auto self(shared_from_this());
  if (rdma_connected_) {
    for (size_t i = 0; i < payloads.size(); i++) {
      if (payloads[i].data_size == 0) {
        continue;
      }
      size_t remain_blob_bytes = payloads[i].data_size;
      uint8_t* local_blob_data = results[i]->pointer;
      size_t max_register_size = payloads[i].data_size;

      do {
        size_t blob_data_offset = payloads[i].data_size - remain_blob_bytes;
        void* server_pointer = payloads[i].pointer;

        // Register mem
        RegisterMemInfo local_info;
        local_info.address =
            reinterpret_cast<uint64_t>(local_blob_data + blob_data_offset);
        local_info.size = std::min(remain_blob_bytes, max_register_size);
        Status status;
        while (true) {
          status = rdma_client_->RegisterMemory(local_info);
          if (status.ok()) {
            break;
          }
          if (status.IsIOError()) {
            // probe the max register size again
            VLOG(100) << "Probe the max register size again.";
            while (true) {
              size_t size = rdma_client_->GetClientMaxRegisterSize();
              if (size > 0) {
                max_register_size = size;
                break;
              }
              // Maybe the registered size is too large. There is no enough
              // memory to register. Wait for next time.
              usleep(1000);
            }
            local_info.size = std::min(remain_blob_bytes, max_register_size);
          } else {
            return status;
          }
        }

        // Exchange mem info
        RegisterMemInfo remote_info;
        while (true) {
          remote_info.address = (uint64_t) server_pointer + blob_data_offset;
          remote_info.size = local_info.size;
          VLOG(100) << "Request remote address: "
                    << reinterpret_cast<void*>(remote_info.address)
                    << ", size: " << remote_info.size;
          RETURN_ON_ERROR(RDMARequestMemInfo(remote_info));
          if (remote_info.size > 0) {
            break;
          }
          usleep(1000);
        }
        size_t receive_size = remote_info.size;

        // Read data
        size_t remain_bytes = receive_size;
        do {
          size_t read_bytes =
              std::min(remain_bytes, rdma_client_->GetMaxTransferBytes());
          size_t read_data_offset = receive_size - remain_bytes;
          VLOG(100) << "blob data offset: " << blob_data_offset
                    << ", read data offset: " << read_data_offset
                    << ", read bytes: " << read_bytes;
          VLOG(100) << "Read to address: "
                    << reinterpret_cast<void*>(
                           reinterpret_cast<uint64_t>(local_blob_data) +
                           blob_data_offset + read_data_offset)
                    << ", size: " << read_bytes << ", remote address: "
                    << reinterpret_cast<void*>(
                           reinterpret_cast<uint64_t>(server_pointer) +
                           blob_data_offset + read_data_offset)
                    << ", rkey: " << remote_info.rkey;
          RETURN_ON_ERROR(rdma_client_->Read(
              local_blob_data + blob_data_offset + read_data_offset, read_bytes,
              reinterpret_cast<uint64_t>(server_pointer) + blob_data_offset +
                  read_data_offset,
              remote_info.rkey, local_info.mr_desc, nullptr));
          RETURN_ON_ERROR(rdma_client_->GetTXCompletion(-1, nullptr));
          remain_bytes -= read_bytes;
        } while (remain_bytes > 0);

        remain_blob_bytes -= receive_size;
        RETURN_ON_ERROR(rdma_client_->DeregisterMemory(local_info));
        RETURN_ON_ERROR(RDMAReleaseMemInfo(remote_info));
      } while (remain_blob_bytes > 0);
    }
    std::map<ObjectID, ObjectID> result_blobs;
    for (size_t i = 0; i < payloads.size(); i++) {
      RETURN_ON_ERROR(
          self->server_ptr_->GetBulkStore()->Seal(results[i]->object_id));
      result_blobs.emplace(payloads[i].object_id, results[i]->object_id);
    }
    return callback(Status::OK(), result_blobs);
  } else {
    ReceiveRemoteBuffers(
        socket_, results, compress,
        [self, callback, payloads, results](const Status& status) {
          std::map<ObjectID, ObjectID> result_blobs;
          if (status.ok()) {
            for (size_t i = 0; i < payloads.size(); ++i) {
              VINEYARD_DISCARD(self->server_ptr_->GetBulkStore()->Seal(
                  results[i]->object_id));
              result_blobs.emplace(payloads[i].object_id,
                                   results[i]->object_id);
            }
          }
          return callback(status, result_blobs);
        });
  }
  return Status::OK();
}

Status RemoteClient::doWrite(const std::string& message_out) {
  boost::system::error_code ec;
  size_t length = message_out.length();
  asio::write(socket_, asio::const_buffer(&length, sizeof(size_t)), ec);
  RETURN_ON_ASIO_ERROR(ec);
  asio::write(socket_,
              asio::const_buffer(message_out.data(), message_out.length()), ec);
  RETURN_ON_ASIO_ERROR(ec);
  return Status::OK();
}

Status RemoteClient::doRead(std::string& message_in) {
  boost::system::error_code ec;
  size_t length = std::numeric_limits<size_t>::max();
  asio::read(socket_, asio::buffer(&length, sizeof(size_t)), ec);
  RETURN_ON_ASIO_ERROR(ec);
  if (length > 64 * 1024 * 1024) {  // 64M bytes
    return Status::IOError("Invalid message header value: " +
                           std::to_string(length));
  }
  message_in.resize(length);
  asio::read(socket_,
             asio::mutable_buffer(const_cast<char*>(message_in.data()), length),
             ec);
  RETURN_ON_ASIO_ERROR(ec);
  return Status::OK();
}

Status RemoteClient::doRead(json& root) {
  std::string message_in;
  RETURN_ON_ERROR(doRead(message_in));
  Status status;
  CATCH_JSON_ERROR(root, status, json::parse(message_in));
  return status;
}

void SendRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                       std::vector<std::shared_ptr<Payload>> const& objects,
                       size_t index, std::shared_ptr<Compressor> compressor,
                       callback_t<> callback_after_finish);

namespace detail {

static void send_chunk(asio::generic::stream_protocol::socket& socket,
                       std::vector<std::shared_ptr<Payload>> const& objects,
                       size_t index, callback_t<> callback_after_finish) {
  asio::async_write(
      socket, asio::buffer(objects[index]->pointer, objects[index]->data_size),
      [callback_after_finish](boost::system::error_code ec, std::size_t) {
        if (ec) {
          VINEYARD_DISCARD(callback_after_finish(Status::IOError(
              "Failed to write buffer to client: " + ec.message())));
        } else {
          VINEYARD_DISCARD(callback_after_finish(Status::OK()));
        }
      });
}

static void send_chunk_compressed(
    asio::generic::stream_protocol::socket& socket,
    std::vector<std::shared_ptr<Payload>> const& objects, size_t index,
    std::shared_ptr<Compressor> compressor, std::shared_ptr<size_t> chunk_size,
    callback_t<> callback_after_finish) {
  void* data = nullptr;
  size_t size = 0;
  Status s;
  do {
    size = 0;
    s = compressor->Pull(data, size);
    if (!s.ok() || size != 0) {
      break;
    }
  } while (true);
  if (s.IsStreamDrained()) {
    VINEYARD_DISCARD(callback_after_finish(Status::OK()));
    return;
  }
  *chunk_size = size;
  asio::async_write(
      socket, asio::buffer(chunk_size.get(), sizeof(size_t)),
      [&socket, objects, index, compressor, callback_after_finish, data, size,
       chunk_size](boost::system::error_code ec, std::size_t) {
        if (ec) {
          VINEYARD_DISCARD(callback_after_finish(Status::IOError(
              "Failed to write buffer size to client: " + ec.message())));
          return;
        }
        asio::async_write(
            socket, asio::buffer(data, size),
            [&socket, objects, index, compressor, chunk_size,
             callback_after_finish](boost::system::error_code ec, std::size_t) {
              if (ec) {
                VINEYARD_DISCARD(callback_after_finish(Status::IOError(
                    "Failed to write buffer to client: " + ec.message())));
                return;
              }
              // continue on the next loop
              send_chunk_compressed(socket, objects, index, compressor,
                                    chunk_size, callback_after_finish);
            });
      });
}

}  // namespace detail

void SendRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                       std::vector<std::shared_ptr<Payload>> const& objects,
                       size_t index, std::shared_ptr<Compressor> compressor,
                       callback_t<> callback_after_finish) {
  while (index < objects.size() && objects[index]->data_size == 0) {
    index += 1;
  }
  if (index >= objects.size()) {
    VINEYARD_DISCARD(callback_after_finish(Status::OK()));
    return;
  }
  auto callback = [&socket, objects, index, compressor,
                   callback_after_finish](const Status& status) {
    if (!status.ok()) {
      return callback_after_finish(status);
    }
    SendRemoteBuffers(socket, objects, index + 1, compressor,
                      callback_after_finish);
    return Status::OK();
  };
  if (compressor) {
    auto s = compressor->Compress(objects[index]->pointer,
                                  objects[index]->data_size);
    if (!s.ok()) {
      VINEYARD_DISCARD(callback_after_finish(s));
      return;
    }
    // we need the `size` leave in heap to keep it alive inside callback
    std::shared_ptr<size_t> chunk_size = std::make_shared<size_t>(0);
    detail::send_chunk_compressed(socket, objects, index, compressor,
                                  chunk_size, callback);
  } else {
    detail::send_chunk(socket, objects, index, callback);
  }
}

void SendRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                       std::vector<std::shared_ptr<Payload>> const& objects,
                       size_t index, const bool compress,
                       callback_t<> callback_after_finish) {
  std::shared_ptr<Compressor> compressor;
  if (compress) {
    compressor = std::make_shared<Compressor>();
  }
  SendRemoteBuffers(socket, objects, index, compressor, callback_after_finish);
}

namespace detail {

static size_t decompress_chunk(size_t read_size,
                               std::shared_ptr<Payload> const& object,
                               size_t offset,
                               std::shared_ptr<Decompressor> decompressor,
                               callback_t<> callback_after_finish) {
  auto s = decompressor->Decompress(read_size);
  if (!s.ok()) {
    VINEYARD_DISCARD(callback_after_finish(s));
    return read_size;
  }
  size_t decompressed_size = 0, decompressed_offset = offset;
  while (true) {
    if (decompressed_offset >= static_cast<size_t>(object->data_size)) {
      break;
    }
    uint8_t* data = object->pointer + decompressed_offset;
    size_t size = 0;
    s = decompressor->Pull(data, object->data_size - decompressed_offset, size);
    if (s.IsStreamDrained()) {
      break;
    }
    if (size > 0) {
      decompressed_offset += size;
      decompressed_size += size;
    }
  }
  // the decompressor is expected to be "finished"
  while (true) {
    char data;
    size_t size = 0;
    if (decompressor->Pull(&data, 1, size).IsStreamDrained()) {
      break;
    }
    assert(s.ok() && size == 0);
  }
  return decompressed_size;
}

static void read_chunk_util(asio::generic::stream_protocol::socket& socket,
                            uint8_t* data, size_t size,
                            callback_t<> callback_after_finish) {
  asio::mutable_buffer buffer = asio::buffer(data, size);
  // In our test, socket::async_receive outperforms asio::async_read.
  socket.async_receive(buffer, [&socket, data, size, callback_after_finish](
                                   boost::system::error_code ec,
                                   std::size_t read_size) {
    if (ec) {
      if (ec != asio::error::eof || read_size != size) {
        auto status = Status::IOError(
            "Failed to read enough bytes from client: " + ec.message() +
            ", expected " + std::to_string(size - read_size) + " more bytes");
        VINEYARD_DISCARD(callback_after_finish(status));
      } else {
        VINEYARD_DISCARD(callback_after_finish(Status::OK()));
      }
    } else {
      if (read_size < size) {
        read_chunk_util(socket, data + read_size, size - read_size,
                        callback_after_finish);
      } else {
        VINEYARD_DISCARD(callback_after_finish(Status::OK()));
      }
    }
  });
}

}  // namespace detail

void ReceiveRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                          std::vector<std::shared_ptr<Payload>> const& objects,
                          bool decompress, callback_t<> callback_after_finish) {
  std::shared_ptr<Decompressor> decompressor;
  if (decompress) {
    decompressor = std::make_shared<Decompressor>();
  }
  struct RemoteState : public std::enable_shared_from_this<RemoteState> {
    std::vector<std::shared_ptr<Payload>> objects;
    std::shared_ptr<Decompressor> decompressor;
    callback_t<> callback_after_finish;
    std::queue<std::pair<size_t, size_t>> pending_payloads;

    RemoteState(const std::vector<std::shared_ptr<Payload>>& objs,
                callback_t<>&& cb, std::shared_ptr<Decompressor> decomp)
        : objects(objs),
          decompressor(std::move(decomp)),
          callback_after_finish(std::move(cb)) {
      for (size_t i = 0; i < objs.size(); ++i) {
        pending_payloads.emplace(i, 0);
      }
    }

    void read_sized_chunk(asio::generic::stream_protocol::socket& socket,
                          std::shared_ptr<Payload> const& object, size_t offset,
                          std::shared_ptr<Decompressor> decompressor,
                          uint8_t* data,
                          // `expected_size` not used, as the size of compressed
                          // buffer comes from the incoming stream.
                          size_t expected_size,
                          callback_t<> callback_after_finish) {
      std::shared_ptr<size_t> chunk_size = std::make_shared<size_t>(0);
      asio::async_read(
          socket, asio::buffer(chunk_size.get(), sizeof(size_t)),
          [me = shared_from_this(), &socket, object, offset, decompressor, data,
           callback_after_finish,
           chunk_size](boost::system::error_code ec, std::size_t) {
            if (ec) {
              auto status = Status::IOError(
                  "Failed to read buffer size from client: " + ec.message());
              return;
            }
            me->read_chunk(socket, object, offset, decompressor, data,
                           *chunk_size, callback_after_finish);
          });
    }
    void read_compressed_chunk(asio::generic::stream_protocol::socket& socket,
                               std::shared_ptr<Payload> const& object,
                               size_t offset,
                               std::shared_ptr<Decompressor> decompressor,
                               callback_t<> callback_after_finish) {
      if (offset >= static_cast<size_t>(object->data_size)) {
        VINEYARD_DISCARD(callback_after_finish(Status::OK()));
      } else {
        void* data = nullptr;
        size_t size;
        auto s = decompressor->Buffer(data, size);
        if (!s.ok()) {
          VINEYARD_DISCARD(callback_after_finish(s));
          return;
        }
        read_sized_chunk(socket, object, offset, decompressor,
                         reinterpret_cast<uint8_t*>(data), size,
                         callback_after_finish);
      }
    }
    void read_chunk(asio::generic::stream_protocol::socket& socket,
                    std::shared_ptr<Payload> const& object, size_t offset,
                    std::shared_ptr<Decompressor> decompressor, uint8_t* data,
                    size_t expected_size, callback_t<> callback_after_finish) {
      detail::read_chunk_util(
          socket, data, expected_size,
          [me = shared_from_this(), &socket, object, offset, decompressor,
           expected_size,
           callback_after_finish](const Status& status) -> Status {
            if (!status.ok()) {
              return callback_after_finish(status);
            }
            size_t read_size =
                detail::decompress_chunk(expected_size, object, offset,
                                         decompressor, callback_after_finish);
            me->read_compressed_chunk(socket, object, offset + read_size,
                                      decompressor, callback_after_finish);
            return Status::OK();
          });
    }

    void process_next(asio::generic::stream_protocol::socket& socket) {
      if (pending_payloads.empty()) {
        VINEYARD_DISCARD(callback_after_finish(Status::OK()));
        return;
      }

      auto [index, offset] = pending_payloads.front();
      pending_payloads.pop();

      const auto& object = objects[index];
      if (!object) {
        VINEYARD_DISCARD(
            callback_after_finish(Status::IOError("Object is null")));
        return;
      }

      if (decompressor) {
        read_compressed_chunk(
            socket, object, offset, decompressor,
            [me = shared_from_this(), &socket](const Status& status) -> Status {
              if (!status.ok()) {
                return me->callback_after_finish(status);
              }
              me->process_next(socket);
              return Status::OK();
            });
      } else {
        detail::read_chunk_util(
            socket, object->pointer + offset, object->data_size - offset,
            [me = shared_from_this(), &socket](const Status& status) -> Status {
              if (!status.ok()) {
                return me->callback_after_finish(status);
              }
              me->process_next(socket);
              return Status::OK();
            });
      }
    }
  };

  std::make_shared<RemoteState>(objects, std::move(callback_after_finish),
                                std::move(decompressor))
      ->process_next(socket);
}

}  // namespace vineyard
