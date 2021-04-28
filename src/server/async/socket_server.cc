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

#include "server/async/socket_server.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/memory/fling.h"
#include "common/util/callback.h"
#include "common/util/json.h"

namespace vineyard {

SocketConnection::SocketConnection(stream_protocol::socket socket,
                                   vs_ptr_t server_ptr,
                                   SocketServer* socket_server_ptr, int conn_id)
    : socket_(std::move(socket)),
      server_ptr_(server_ptr),
      socket_server_ptr_(socket_server_ptr),
      conn_id_(conn_id) {}

void SocketConnection::Start() {
  running_.store(true);
  doReadHeader();
}

void SocketConnection::Stop() { doStop(); }

void SocketConnection::doReadHeader() {
  auto self(this->shared_from_this());
  asio::async_read(socket_, asio::buffer(&read_msg_header_, sizeof(size_t)),
                   [this, self](boost::system::error_code ec, std::size_t) {
                     if (!ec && running_.load()) {
                       doReadBody();
                     } else {
                       doStop();
                     }
                   });
}

void SocketConnection::doReadBody() {
  if (read_msg_header_ > 64 * 1024 * 1024) {  // 64M bytes
    // We set a hard limit for the message buffer size, since an evil client,
    // e.g., telnet.
    //
    // We don't revise the structure of protocol, for backwards compatible, as
    // we already released wheel packages on pypi.
    doStop();
    socket_server_ptr_->RemoveConnection(conn_id_);
    return;
  }
  read_msg_body_.resize(read_msg_header_);
  auto self(shared_from_this());
  asio::async_read(socket_, asio::buffer(&read_msg_body_[0], read_msg_header_),
                   [this, self](boost::system::error_code ec, std::size_t) {
                     if ((!ec || ec == asio::error::eof) && running_.load()) {
                       bool exit = processMessage(read_msg_body_);
                       if (exit || ec == asio::error::eof) {
                         doStop();
                         return;
                       }
                     } else {
                       doStop();
                       return;
                     }
                     // start next-round read
                     doReadHeader();
                   });
}

#ifndef TRY_READ_REQUEST
#define TRY_READ_REQUEST(operation)                    \
  do {                                                 \
    auto read_status = (operation);                    \
    if (!read_status.ok()) {                           \
      std::string error_message_out;                   \
      WriteErrorReply(read_status, error_message_out); \
      self->doWrite(error_message_out);                \
      return false;                                    \
    }                                                  \
  } while (0)
#endif  // TRY_READ_REQUEST

#ifndef RESPONSE_ON_ERROR
#define RESPONSE_ON_ERROR(status)                                       \
  do {                                                                  \
    auto exec_status = (status);                                        \
    if (!exec_status.ok()) {                                            \
      LOG(ERROR) << "Unexpected error occurs during message handling: " \
                 << exec_status.ToString();                             \
      std::string error_message_out;                                    \
      WriteErrorReply(exec_status, error_message_out);                  \
      self->doWrite(error_message_out);                                 \
      return false;                                                     \
    }                                                                   \
  } while (0)
#endif  // RESPONSE_ON_ERROR

bool SocketConnection::processMessage(const std::string& message_in) {
  json root;
  std::istringstream is(message_in);

  // DON'T let vineyardd crash when the client is malicious.
  try {
    root = json::parse(message_in);
  } catch (std::out_of_range const& err) {
    LOG(ERROR) << "json: " << err.what();
    std::string message_out;
    WriteErrorReply(Status::Invalid(err.what()), message_out);
    this->doWrite(message_out);
    return false;
  } catch (json::exception const& err) {
    LOG(ERROR) << "json: " << err.what();
    std::string message_out;
    WriteErrorReply(Status::Invalid(err.what()), message_out);
    this->doWrite(message_out);
    return false;
  }

  std::string const& type = root["type"].get_ref<std::string const&>();
  CommandType cmd = ParseCommandType(type);
  switch (cmd) {
  case CommandType::RegisterRequest: {
    return doRegister(root);
  }
  case CommandType::GetBuffersRequest: {
    return doGetBuffers(root);
  }
  case CommandType::GetRemoteBuffersRequest: {
    return doGetRemoteBuffers(root);
  }
  case CommandType::CreateBufferRequest: {
    return doCreateBuffer(root);
  }
  case CommandType::CreateRemoteBufferRequest: {
    return doCreateRemoteBuffer(root);
  }
  case CommandType::DropBufferRequest: {
    return doDropBuffer(root);
  }
  case CommandType::GetDataRequest: {
    return doGetData(root);
  }
  case CommandType::ListDataRequest: {
    return doListData(root);
  }
  case CommandType::CreateDataRequest: {
    return doCreateData(root);
  }
  case CommandType::PersistRequest: {
    return doPersist(root);
  }
  case CommandType::IfPersistRequest: {
    return doIfPersist(root);
  }
  case CommandType::ExistsRequest: {
    return doExists(root);
  }
  case CommandType::ShallowCopyRequest: {
    return doShallowCopy(root);
  }
  case CommandType::DelDataRequest: {
    return doDelData(root);
  }
  case CommandType::CreateStreamRequest: {
    return doCreateStream(root);
  }
  case CommandType::OpenStreamRequest: {
    return doOpenStream(root);
  }
  case CommandType::GetNextStreamChunkRequest: {
    return doGetNextStreamChunk(root);
  }
  case CommandType::PullNextStreamChunkRequest: {
    return doPullNextStreamChunk(root);
  }
  case CommandType::StopStreamRequest: {
    return doStopStream(root);
  }
  case CommandType::PutNameRequest: {
    return doPutName(root);
  }
  case CommandType::GetNameRequest: {
    return doGetName(root);
  }
  case CommandType::DropNameRequest: {
    return doDropName(root);
  }
  case CommandType::MigrateObjectRequest: {
    return doMigrateObject(root);
  }
  case CommandType::ClusterMetaRequest: {
    return doClusterMeta(root);
  }
  case CommandType::InstanceStatusRequest: {
    return doInstanceStatus(root);
  }
  case CommandType::MakeArenaRequest: {
    return doMakeArena(root);
  }
  case CommandType::FinalizeArenaRequest: {
    return doFinalizeArena(root);
  }
  case CommandType::ExitRequest: {
    return true;
  }
  default: {
    LOG(ERROR) << "Got unexpected command: " << type;
    return false;
  }
  }
}

bool SocketConnection::doRegister(const json& root) {
  auto self(shared_from_this());
  std::string client_version, message_out;
  TRY_READ_REQUEST(ReadRegisterRequest(root, client_version));
  WriteRegisterReply(server_ptr_->IPCSocket(), server_ptr_->RPCEndpoint(),
                     server_ptr_->instance_id(), message_out);
  doWrite(message_out);
  return false;
}

bool SocketConnection::doGetBuffers(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;
  std::string message_out;

  TRY_READ_REQUEST(ReadGetBuffersRequest(root, ids));
  RESPONSE_ON_ERROR(server_ptr_->GetBulkStore()->Get(ids, objects));
  WriteGetBuffersReply(objects, message_out);

  /* NOTE: Here we send the file descriptor after the objects.
   *       We are using sendmsg to send the file descriptor
   *       which is a sync method. In theory, this might cause
   *       the server to block, but currently this seems to be
   *       the only method that are widely used in practice, e.g.,
   *       boost and Plasma, and actually the file descriptor is
   *       a very short message.
   *
   *       We will examine other methods later, such as using
   *       explicit file descritors.
   */
  this->doWrite(message_out, [self, objects](const Status& status) {
    for (auto object : objects) {
      int store_fd = object->store_fd;
      int data_size = object->data_size;
      if (data_size > 0 &&
          self->used_fds_.find(store_fd) == self->used_fds_.end()) {
        self->used_fds_.emplace(store_fd);
        send_fd(self->nativeHandle(), store_fd);
      }
    }
    return Status::OK();
  });
  return false;
}

void SocketConnection::sendBufferHelper(
    std::vector<std::shared_ptr<Payload>> const objects, size_t index,
    boost::system::error_code const ec, callback_t<> callback_after_finish) {
  auto self(shared_from_this());
  if (!ec && index < objects.size()) {
    async_write(
        socket_,
        boost::asio::buffer(objects[index]->pointer, objects[index]->data_size),
        [this, self, callback_after_finish, objects, index](
            boost::system::error_code ec, std::size_t) {
          sendBufferHelper(objects, index + 1, ec, callback_after_finish);
        });
  } else {
    if (ec) {
      VINEYARD_DISCARD(callback_after_finish(Status::IOError(
          "Failed to write buffer to client: " + ec.message())));
    } else {
      VINEYARD_DISCARD(callback_after_finish(Status::OK()));
    }
  }
}

bool SocketConnection::doGetRemoteBuffers(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;
  std::string message_out;

  TRY_READ_REQUEST(ReadGetBuffersRequest(root, ids));
  RESPONSE_ON_ERROR(server_ptr_->GetBulkStore()->Get(ids, objects));
  WriteGetBuffersReply(objects, message_out);

  this->doWrite(message_out, [this, self, objects](const Status& status) {
    boost::system::error_code ec;
    sendBufferHelper(objects, 0, ec, [self](const Status& status) {
      if (!status.ok()) {
        LOG(ERROR) << "Failed to send buffers to remote client: "
                   << status.ToString();
      }
      return Status::OK();
    });
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doCreateBuffer(const json& root) {
  auto self(shared_from_this());
  size_t size;
  std::shared_ptr<Payload> object;
  std::string message_out;

  TRY_READ_REQUEST(ReadCreateBufferRequest(root, size));
  ObjectID object_id;
  RESPONSE_ON_ERROR(
      server_ptr_->GetBulkStore()->Create(size, object_id, object));
  WriteCreateBufferReply(object_id, object, message_out);

  int store_fd = object->store_fd;
  int data_size = object->data_size;
  this->doWrite(message_out, [self, store_fd, data_size](const Status& status) {
    if (data_size > 0 &&
        self->used_fds_.find(store_fd) == self->used_fds_.end()) {
      self->used_fds_.emplace(store_fd);
      send_fd(self->nativeHandle(), store_fd);
    }
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doCreateRemoteBuffer(const json& root) {
  auto self(shared_from_this());
  size_t size;
  std::shared_ptr<Payload> object;

  TRY_READ_REQUEST(ReadCreateBufferRequest(root, size));
  ObjectID object_id;
  RESPONSE_ON_ERROR(
      server_ptr_->GetBulkStore()->Create(size, object_id, object));

  asio::async_read(
      socket_, asio::buffer(object->pointer, size),
      [this, self, &object](boost::system::error_code ec, std::size_t size) {
        std::string message_out;
        if (static_cast<size_t>(object->data_size) == size &&
            (!ec || ec == asio::error::eof)) {
          WriteCreateBufferReply(object->object_id, object, message_out);
        } else {
          VINEYARD_DISCARD(
              server_ptr_->GetBulkStore()->Delete(object->object_id));
          if (static_cast<size_t>(object->data_size) == size) {
            WriteErrorReply(
                Status::IOError("Failed to read buffer's content from client"),
                message_out);
          } else {
            WriteErrorReply(Status::IOError(ec.message()), message_out);
          }
        }
        self->doWrite(message_out);
      });
  return false;
}

bool SocketConnection::doDropBuffer(const json& root) {
  auto self(shared_from_this());
  ObjectID object_id = InvalidObjectID();
  TRY_READ_REQUEST(ReadDropBufferRequest(root, object_id));
  auto status = server_ptr_->GetBulkStore()->Delete(object_id);
  std::string message_out;
  if (status.ok()) {
    WriteDropBufferReply(message_out);
  } else {
    WriteErrorReply(status, message_out);
  }
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doGetData(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool sync_remote = false, wait = false;
  TRY_READ_REQUEST(ReadGetDataRequest(root, ids, sync_remote, wait));
  json tree;
  RESPONSE_ON_ERROR(server_ptr_->GetData(
      ids, sync_remote, wait, [self]() { return self->running_.load(); },
      [self](const Status& status, const json& tree) {
        std::string message_out;
        if (status.ok()) {
          WriteGetDataReply(tree, message_out);
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doListData(const json& root) {
  auto self(shared_from_this());
  std::string pattern;
  bool regex;
  size_t limit;
  TRY_READ_REQUEST(ReadListDataRequest(root, pattern, regex, limit));
  RESPONSE_ON_ERROR(server_ptr_->ListData(
      pattern, regex, limit, [self](const Status& status, const json& tree) {
        std::string message_out;
        if (status.ok()) {
          WriteGetDataReply(tree, message_out);
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doCreateData(const json& root) {
  auto self(shared_from_this());
  json tree;
  TRY_READ_REQUEST(ReadCreateDataRequest(root, tree));
  RESPONSE_ON_ERROR(server_ptr_->CreateData(
      tree, [self](const Status& status, const ObjectID id,
                   const Signature signature, const InstanceID instance_id) {
        std::string message_out;
        if (status.ok()) {
          WriteCreateDataReply(id, signature, instance_id, message_out);
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doPersist(const json& root) {
  auto self(shared_from_this());
  ObjectID id;
  TRY_READ_REQUEST(ReadPersistRequest(root, id));
  RESPONSE_ON_ERROR(server_ptr_->Persist(id, [self](const Status& status) {
    std::string message_out;
    if (status.ok()) {
      WritePersistReply(message_out);
    } else {
      LOG(ERROR) << status.ToString();
      WriteErrorReply(status, message_out);
    }
    self->doWrite(message_out);
    return Status::OK();
  }));
  return false;
}

bool SocketConnection::doIfPersist(const json& root) {
  auto self(shared_from_this());
  ObjectID id;
  TRY_READ_REQUEST(ReadIfPersistRequest(root, id));
  RESPONSE_ON_ERROR(server_ptr_->IfPersist(
      id, [self](const Status& status, bool const persist) {
        std::string message_out;
        if (status.ok()) {
          WriteIfPersistReply(persist, message_out);
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doExists(const json& root) {
  auto self(shared_from_this());
  ObjectID id;
  TRY_READ_REQUEST(ReadExistsRequest(root, id));
  RESPONSE_ON_ERROR(
      server_ptr_->Exists(id, [self](const Status& status, bool const exists) {
        std::string message_out;
        if (status.ok()) {
          WriteExistsReply(exists, message_out);
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doShallowCopy(const json& root) {
  auto self(shared_from_this());
  ObjectID id;
  TRY_READ_REQUEST(ReadShallowCopyRequest(root, id));
  RESPONSE_ON_ERROR(server_ptr_->ShallowCopy(
      id, [self](const Status& status, const ObjectID target) {
        std::string message_out;
        if (status.ok()) {
          WriteShallowCopyReply(target, message_out);
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doDelData(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool force, deep, fastpath;
  TRY_READ_REQUEST(ReadDelDataRequest(root, ids, force, deep, fastpath));
  RESPONSE_ON_ERROR(server_ptr_->DelData(
      ids, force, deep, fastpath, [self](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WriteDelDataReply(message_out);
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doCreateStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadCreateStreamRequest(root, stream_id));
  auto status = server_ptr_->GetStreamStore()->Create(stream_id);
  std::string message_out;
  if (status.ok()) {
    WriteCreateStreamReply(message_out);
  } else {
    LOG(ERROR) << status.ToString();
    WriteErrorReply(status, message_out);
  }
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doOpenStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  int64_t mode;
  TRY_READ_REQUEST(ReadOpenStreamRequest(root, stream_id, mode));
  auto status = server_ptr_->GetStreamStore()->Open(stream_id, mode);
  std::string message_out;
  if (status.ok()) {
    WriteOpenStreamReply(message_out);
  } else {
    LOG(ERROR) << status.ToString();
    WriteErrorReply(status, message_out);
  }
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doGetNextStreamChunk(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  size_t size;
  TRY_READ_REQUEST(ReadGetNextStreamChunkRequest(root, stream_id, size));
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Get(
      stream_id, size, [self](const Status& status, const ObjectID chunk) {
        std::string message_out;
        if (status.ok()) {
          std::shared_ptr<Payload> object;
          RETURN_ON_ERROR(
              self->server_ptr_->GetBulkStore()->Get(chunk, object));
          WriteGetNextStreamChunkReply(object, message_out);
          int store_fd = object->store_fd;
          int data_size = object->data_size;
          self->doWrite(
              message_out, [self, store_fd, data_size](const Status& status) {
                if (data_size > 0 &&
                    self->used_fds_.find(store_fd) == self->used_fds_.end()) {
                  self->used_fds_.emplace(store_fd);
                  send_fd(self->nativeHandle(), store_fd);
                }
                return Status::OK();
              });
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
          self->doWrite(message_out);
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doPullNextStreamChunk(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadPullNextStreamChunkRequest(root, stream_id));
  this->associated_streams_.emplace(stream_id);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Pull(
      stream_id, [self](const Status& status, const ObjectID chunk) {
        std::string message_out;
        if (status.ok()) {
          std::shared_ptr<Payload> object;
          RETURN_ON_ERROR(
              self->server_ptr_->GetBulkStore()->Get(chunk, object));
          WritePullNextStreamChunkReply(object, message_out);
          int store_fd = object->store_fd;
          int data_size = object->data_size;
          self->doWrite(
              message_out, [self, store_fd, data_size](const Status& status) {
                if (data_size > 0 &&
                    self->used_fds_.find(store_fd) == self->used_fds_.end()) {
                  self->used_fds_.emplace(store_fd);
                  send_fd(self->nativeHandle(), store_fd);
                }
                return Status::OK();
              });
        } else {
          LOG(ERROR) << status.ToString();
          WriteErrorReply(status, message_out);
          self->doWrite(message_out);
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doStopStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  bool failed;
  TRY_READ_REQUEST(ReadStopStreamRequest(root, stream_id, failed));
  // NB: don't erase the metadata from meta_service, since there's may
  // reader listen on this stream.
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Stop(stream_id, failed));
  std::string message_out;
  WriteStopStreamReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doPutName(const json& root) {
  auto self(shared_from_this());
  ObjectID object_id;
  std::string name;
  TRY_READ_REQUEST(ReadPutNameRequest(root, object_id, name));
  RESPONSE_ON_ERROR(
      server_ptr_->PutName(object_id, name, [self](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WritePutNameReply(message_out);
        } else {
          LOG(ERROR) << "Failed to put name: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doGetName(const json& root) {
  auto self(shared_from_this());
  std::string name;
  bool wait;
  TRY_READ_REQUEST(ReadGetNameRequest(root, name, wait));
  RESPONSE_ON_ERROR(server_ptr_->GetName(
      name, wait, [self]() { return self->running_.load(); },
      [self](const Status& status, const ObjectID& object_id) {
        std::string message_out;
        if (status.ok()) {
          WriteGetNameReply(object_id, message_out);
        } else {
          LOG(ERROR) << "Failed to get name: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doDropName(const json& root) {
  auto self(shared_from_this());
  std::string name;
  TRY_READ_REQUEST(ReadDropNameRequest(root, name));
  RESPONSE_ON_ERROR(server_ptr_->DropName(name, [self](const Status& status) {
    std::string message_out;
    LOG(INFO) << "drop name callback: " << status;
    if (status.ok()) {
      WriteDropNameReply(message_out);
    } else {
      LOG(ERROR) << "Failed to drop name: " << status.ToString();
      WriteErrorReply(status, message_out);
    }
    self->doWrite(message_out);
    return Status::OK();
  }));
  return false;
}

bool SocketConnection::doMigrateObject(const json& root) {
  auto self(shared_from_this());
  ObjectID object_id;
  bool local;
  bool is_stream;
  std::string peer, peer_rpc_endpoint;
  TRY_READ_REQUEST(ReadMigrateObjectRequest(root, object_id, local, is_stream,
                                            peer, peer_rpc_endpoint));
  if (is_stream) {
    RESPONSE_ON_ERROR(server_ptr_->MigrateStream(
        object_id, local, peer, peer_rpc_endpoint,
        [self](const Status& status, const ObjectID& target) {
          std::string message_out;
          if (status.ok()) {
            WriteMigrateObjectReply(target, message_out);
          } else {
            LOG(ERROR) << "Failed to start migrating stream: "
                       << status.ToString();
            WriteErrorReply(status, message_out);
          }
          self->doWrite(message_out);
          return Status::OK();
        }));
  } else {
    RESPONSE_ON_ERROR(server_ptr_->MigrateObject(
        object_id, local, peer, peer_rpc_endpoint,
        [self](const Status& status, const ObjectID& target) {
          std::string message_out;
          if (status.ok()) {
            WriteMigrateObjectReply(target, message_out);
          } else {
            LOG(ERROR) << "Failed to migrate object: " << status.ToString();
            WriteErrorReply(status, message_out);
          }
          self->doWrite(message_out);
          return Status::OK();
        }));
  }
  return false;
}

bool SocketConnection::doClusterMeta(const json& root) {
  auto self(shared_from_this());
  TRY_READ_REQUEST(ReadClusterMetaRequest(root));
  RESPONSE_ON_ERROR(
      server_ptr_->ClusterInfo([self](const Status& status, const json& tree) {
        std::string message_out;
        if (status.ok()) {
          WriteClusterMetaReply(tree, message_out);
        } else {
          LOG(ERROR) << "Check cluster meta: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doInstanceStatus(const json& root) {
  auto self(shared_from_this());
  TRY_READ_REQUEST(ReadInstanceStatusRequest(root));
  RESPONSE_ON_ERROR(server_ptr_->InstanceStatus(
      [self](const Status& status, const json& tree) {
        std::string message_out;
        if (status.ok()) {
          WriteInstanceStatusReply(tree, message_out);
        } else {
          LOG(ERROR) << "Check instance status: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doMakeArena(const json& root) {
  auto self(shared_from_this());
  size_t size;
  std::string message_out;

  TRY_READ_REQUEST(ReadMakeArenaRequest(root, size));
  if (size == std::numeric_limits<size_t>::max()) {
    size = server_ptr_->GetBulkStore()->FootprintLimit();
  }
  int store_fd = -1;
  uintptr_t base = reinterpret_cast<uintptr_t>(nullptr);
  RESPONSE_ON_ERROR(
      server_ptr_->GetBulkStore()->MakeArena(size, store_fd, base));
  WriteMakeArenaReply(store_fd, size, base, message_out);

  this->doWrite(message_out, [self, store_fd](const Status& status) {
    if (self->used_fds_.find(store_fd) == self->used_fds_.end()) {
      self->used_fds_.emplace(store_fd);
      send_fd(self->nativeHandle(), store_fd);
    }
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doFinalizeArena(const json& root) {
  auto self(shared_from_this());
  int fd = -1;
  std::vector<size_t> offsets, sizes;
  std::string message_out;

  TRY_READ_REQUEST(ReadFinalizeArenaRequest(root, fd, offsets, sizes));
  RESPONSE_ON_ERROR(
      server_ptr_->GetBulkStore()->FinalizeArena(fd, offsets, sizes));
  WriteFinalizeArenaReply(message_out);

  this->doWrite(message_out);
  return false;
}

void SocketConnection::doWrite(const std::string& buf) {
  std::string to_send;
  size_t length = buf.size();
  to_send.resize(length + sizeof(size_t));
  char* ptr = &to_send[0];
  memcpy(ptr, &length, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(ptr, buf.data(), length);
  bool write_in_progress = !write_msgs_.empty();
  write_msgs_.push_back(std::move(to_send));
  if (!write_in_progress) {
    doAsyncWrite();
  }
}

void SocketConnection::doWrite(const std::string& buf, callback_t<> callback) {
  std::string to_send;
  size_t length = buf.size();
  to_send.resize(length + sizeof(size_t));
  char* ptr = &to_send[0];
  memcpy(ptr, &length, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(ptr, buf.data(), length);
  bool write_in_progress = !write_msgs_.empty();
  write_msgs_.push_back(std::move(to_send));
  if (!write_in_progress) {
    doAsyncWrite(callback);
  }
}

void SocketConnection::doWrite(std::string&& buf) {
  bool write_in_progress = !write_msgs_.empty();
  write_msgs_.push_back(std::move(buf));
  if (!write_in_progress) {
    doAsyncWrite();
  }
}

void SocketConnection::doStop() {
  if (!running_.exchange(false)) {
    // already stopped, or haven't started
    return;
  }

  // do cleanup: clean up streams associated with this client
  for (auto stream_id : associated_streams_) {
    VINEYARD_SUPPRESS(server_ptr_->GetStreamStore()->Drop(stream_id));
  }

  // On Mac the state of socket may be "not connected" after the client has
  // already closed the socket, hence there will be an exception.
  boost::system::error_code ec;
  socket_.cancel(ec);
  socket_.shutdown(stream_protocol::socket::shutdown_both, ec);
  socket_.close(ec);

  // drop connection
  socket_server_ptr_->RemoveConnection(conn_id_);
}

void SocketConnection::doAsyncWrite() {
  auto self(shared_from_this());
  asio::async_write(socket_,
                    boost::asio::buffer(write_msgs_.front().data(),
                                        write_msgs_.front().length()),
                    [this, self](boost::system::error_code ec, std::size_t) {
                      if (!ec) {
                        write_msgs_.pop_front();
                        if (!write_msgs_.empty()) {
                          doAsyncWrite();
                        }
                      } else {
                        doStop();
                      }
                    });
}

void SocketConnection::doAsyncWrite(callback_t<> callback) {
  auto self(shared_from_this());
  asio::async_write(
      socket_,
      boost::asio::buffer(write_msgs_.front().data(),
                          write_msgs_.front().length()),
      [this, self, callback](boost::system::error_code ec, std::size_t) {
        if (!ec) {
          write_msgs_.pop_front();
          if (!write_msgs_.empty()) {
            doAsyncWrite(callback);
          } else {
            auto status = callback(Status::OK());
            if (!status.ok()) {
              doStop();
            }
          }
        } else {
          doStop();
        }
      });
}

SocketServer::SocketServer(vs_ptr_t vs_ptr)
    : vs_ptr_(vs_ptr), next_conn_id_(0) {}

void SocketServer::Start() {
  stopped_.store(false);
  doAccept();
}

void SocketServer::Stop() {
  if (stopped_.exchange(true)) {
    return;
  }

  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutx_);
  for (auto& pair : connections_) {
    pair.second->Stop();
  }
  connections_.clear();
}

bool SocketServer::ExistsConnection(int conn_id) const {
  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutx_);
  return connections_.find(conn_id) != connections_.end();
}

void SocketServer::RemoveConnection(int conn_id) {
  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutx_);
  auto conn = connections_.find(conn_id);
  if (conn != connections_.end()) {
    connections_.erase(conn);
  }
}

void SocketServer::CloseConnection(int conn_id) {
  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutx_);
  auto conn = connections_.find(conn_id);
  if (conn != connections_.end()) {
    conn->second->Stop();
    connections_.erase(conn);
  }
}

size_t SocketServer::AliveConnections() const {
  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutx_);
  return connections_.size();
}

}  // namespace vineyard
