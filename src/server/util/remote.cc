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

#include <chrono>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/compression/compressor.h"
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
      connected_(false) {}

RemoteClient::~RemoteClient() {
  boost::system::error_code ec;
  ec = socket_.close(ec);
}

Status RemoteClient::Connect(const std::string& rpc_endpoint,
                             const SessionID session_id) {
  size_t pos = rpc_endpoint.find(":");
  std::string host, port;
  if (pos == std::string::npos) {
    host = rpc_endpoint;
    port = "9600";
  } else {
    host = rpc_endpoint.substr(0, pos);
    port = rpc_endpoint.substr(pos + 1);
  }
  return Connect(host, static_cast<uint32_t>(std::stoul(port)), session_id);
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
  WriteGetRemoteBuffersRequest(blobs, false, compress, message_out);
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
  ReceiveRemoteBuffers(
      socket_, results, 0, 0, compress,
      [self, callback, payloads, results](const Status& status) {
        std::map<ObjectID, ObjectID> result_blobs;
        if (status.ok()) {
          for (size_t i = 0; i < payloads.size(); ++i) {
            VINEYARD_DISCARD(
                self->server_ptr_->GetBulkStore()->Seal(results[i]->object_id));
            result_blobs.emplace(payloads[i].object_id, results[i]->object_id);
          }
        }
        return callback(status, result_blobs);
      });
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

void ReceiveRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                          std::vector<std::shared_ptr<Payload>> const& objects,
                          size_t index, size_t offset,
                          std::shared_ptr<Decompressor> decompressor,
                          callback_t<> callback_after_finish);

namespace detail {

static size_t decompress_chunk(
    size_t read_size, std::vector<std::shared_ptr<Payload>> const& objects,
    size_t index, size_t offset, std::shared_ptr<Decompressor> decompressor,
    callback_t<> callback_after_finish) {
  auto s = decompressor->Decompress(read_size);
  if (!s.ok()) {
    VINEYARD_DISCARD(callback_after_finish(s));
    return read_size;
  }
  size_t decompressed_size = 0, decompressed_offset = offset;
  while (true) {
    if (decompressed_offset >= static_cast<size_t>(objects[index]->data_size)) {
      break;
    }
    uint8_t* data = objects[index]->pointer + decompressed_offset;
    size_t size = 0;
    s = decompressor->Pull(
        data, objects[index]->data_size - decompressed_offset, size);
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

static void read_chunk(asio::generic::stream_protocol::socket& socket,
                       std::vector<std::shared_ptr<Payload>> const& objects,
                       size_t index, size_t offset,
                       std::shared_ptr<Decompressor> decompressor,
                       uint8_t* data, size_t expected_size,
                       callback_t<> callback_after_finish) {
  read_chunk_util(
      socket, data, expected_size,
      [&socket, objects, index, offset, decompressor, expected_size,
       callback_after_finish](const Status& status) -> Status {
        if (!status.ok()) {
          return callback_after_finish(status);
        }
        size_t read_size = expected_size;
        if (decompressor) {
          read_size =
              detail::decompress_chunk(read_size, objects, index, offset,
                                       decompressor, callback_after_finish);
        }
        ReceiveRemoteBuffers(socket, objects, index, read_size + offset,
                             decompressor, callback_after_finish);
        return Status::OK();
      });
}

static void read_sized_chunk(
    asio::generic::stream_protocol::socket& socket,
    std::vector<std::shared_ptr<Payload>> const& objects, size_t index,
    size_t offset, std::shared_ptr<Decompressor> decompressor, uint8_t* data,
    // `expected_size` not used, as the size of compressed buffer comes from
    // the incoming stream.
    size_t expected_size, callback_t<> callback_after_finish) {
  // we need the `size` leave in heap to keep it alive inside callback
  std::shared_ptr<size_t> chunk_size = std::make_shared<size_t>(0);
  asio::async_read(
      socket, asio::buffer(chunk_size.get(), sizeof(size_t)),
      [&socket, objects, index, offset, decompressor, data,
       callback_after_finish,
       chunk_size](boost::system::error_code ec, std::size_t) {
        if (ec) {
          auto status = Status::IOError(
              "Failed to read buffer size from client: " + ec.message());
          return;
        }
        read_chunk(socket, objects, index, offset, decompressor, data,
                   *chunk_size, callback_after_finish);
      });
}

}  // namespace detail

void ReceiveRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                          std::vector<std::shared_ptr<Payload>> const& objects,
                          size_t index, size_t offset,
                          std::shared_ptr<Decompressor> decompressor,
                          callback_t<> callback_after_finish) {
  while (index < objects.size() &&
         offset >= static_cast<size_t>(objects[index]->data_size)) {
    offset = 0;
    index += 1;
  }
  if (index >= objects.size()) {
    VINEYARD_DISCARD(callback_after_finish(Status::OK()));
    return;
  }
  if (decompressor) {
    void* data = nullptr;
    size_t size;
    auto s = decompressor->Buffer(data, size);
    if (!s.ok()) {
      VINEYARD_DISCARD(callback_after_finish(s));
      return;
    }
    detail::read_sized_chunk(socket, objects, index, offset, decompressor,
                             reinterpret_cast<uint8_t*>(data), size,
                             callback_after_finish);
  } else {
    detail::read_chunk(socket, objects, index, offset, decompressor,
                       objects[index]->pointer + offset,
                       objects[index]->data_size - offset,
                       callback_after_finish);
  }
}

void ReceiveRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                          std::vector<std::shared_ptr<Payload>> const& objects,
                          size_t index, size_t offset, const bool decompress,
                          callback_t<> callback_after_finish) {
  std::shared_ptr<Decompressor> decompressor;
  if (decompress) {
    decompressor = std::make_shared<Decompressor>();
  }
  ReceiveRemoteBuffers(socket, objects, index, offset, decompressor,
                       callback_after_finish);
}

}  // namespace vineyard
