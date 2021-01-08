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
      conn_id_(conn_id),
      running_(false) {}

void SocketConnection::Start() {
  running_ = true;
  doReadHeader();
}

void SocketConnection::Stop() {
  doStop();
  running_ = false;
}

void SocketConnection::doReadHeader() {
  auto self(this->shared_from_this());
  asio::async_read(socket_, asio::buffer(&read_msg_header_, sizeof(size_t)),
                   [this, self](boost::system::error_code ec, std::size_t) {
                     if (!ec && running_) {
                       doReadBody();
                     } else {
                       doStop();
                       socket_server_ptr_->RemoveConnection(conn_id_);
                       return;
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
                     if ((!ec || ec == asio::error::eof) && running_) {
                       bool exit = processMessage(read_msg_body_);
                       if (exit || ec == asio::error::eof) {
                         doStop();
                         socket_server_ptr_->RemoveConnection(conn_id_);
                         return;
                       }
                     } else {
                       doStop();
                       socket_server_ptr_->RemoveConnection(conn_id_);
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
  auto self(shared_from_this());
  switch (cmd) {
  case CommandType::RegisterRequest: {
    std::string client_version, message_out;
    TRY_READ_REQUEST(ReadRegisterRequest(root, client_version));
    WriteRegisterReply(server_ptr_->IPCSocket(), server_ptr_->RPCEndpoint(),
                       server_ptr_->instance_id(), message_out);
    doWrite(message_out);
  } break;
  case CommandType::GetBuffersRequest: {
    std::vector<ObjectID> ids;
    std::vector<std::shared_ptr<Payload>> objects;
    std::string message_out;

    TRY_READ_REQUEST(ReadGetBuffersRequest(root, ids));
    RESPONSE_ON_ERROR(
        server_ptr_->GetBulkStore()->ProcessGetRequest(ids, objects));
    WriteGetBuffersReply(objects, message_out);

    /* NOTE: Here we send the file descriptor after the objects.
     *       We are using sendmsg to send the file descriptor
     *       which is a sync method. In theory, this might cause
     *       the server to block, but currently this seems to be
     *       the only method that are widely used in practice, e.g.,
     *       boost and Plasma, and actually the file descriptor is
     *       a very short message.
     *       We will examine other methods later, such as using
     *       explicit file descritors.
     */
    auto self(shared_from_this());
    this->doWrite(message_out, [self, objects](const Status& status) {
      for (auto object : objects) {
        int store_fd = object->store_fd;
        if (self->used_fds_.find(store_fd) == self->used_fds_.end()) {
          self->used_fds_.emplace(store_fd);
          send_fd(self->nativeHandle(), store_fd);
        }
      }
      return Status::OK();
    });
  } break;
  case CommandType::CreateBufferRequest: {
    size_t size;
    std::shared_ptr<Payload> object;
    std::string message_out;

    TRY_READ_REQUEST(ReadCreateBufferRequest(root, size));
    ObjectID object_id;
    RESPONSE_ON_ERROR(server_ptr_->GetBulkStore()->ProcessCreateRequest(
        size, object_id, object));
    WriteCreateBufferReply(object_id, object, message_out);

    int store_fd = object->store_fd;
    this->doWrite(message_out, [self, store_fd](const Status& status) {
      if (self->used_fds_.find(store_fd) == self->used_fds_.end()) {
        self->used_fds_.emplace(store_fd);
        send_fd(self->nativeHandle(), store_fd);
      }
      return Status::OK();
    });
  } break;
  case CommandType::GetDataRequest: {
    std::vector<ObjectID> ids;
    bool sync_remote = false, wait = false;
    TRY_READ_REQUEST(ReadGetDataRequest(root, ids, sync_remote, wait));
    json tree;
    RESPONSE_ON_ERROR(server_ptr_->GetData(
        ids, sync_remote, wait, [self]() { return self->running_; },
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
  } break;
  case CommandType::ListDataRequest: {
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
  } break;
  case CommandType::CreateDataRequest: {
    json tree;
    TRY_READ_REQUEST(ReadCreateDataRequest(root, tree));
    RESPONSE_ON_ERROR(server_ptr_->CreateData(
        tree, [self](const Status& status, const ObjectID id,
                     const InstanceID instance_id) {
          std::string message_out;
          if (status.ok()) {
            WriteCreateDataReply(id, instance_id, message_out);
          } else {
            LOG(ERROR) << status.ToString();
            WriteErrorReply(status, message_out);
          }
          self->doWrite(message_out);
          return Status::OK();
        }));
  } break;
  case CommandType::PersistRequest: {
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
  } break;
  case CommandType::IfPersistRequest: {
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
  } break;
  case CommandType::ExistsRequest: {
    ObjectID id;
    TRY_READ_REQUEST(ReadExistsRequest(root, id));
    RESPONSE_ON_ERROR(server_ptr_->Exists(
        id, [self](const Status& status, bool const exists) {
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
  } break;
  case CommandType::ShallowCopyRequest: {
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
  } break;
  case CommandType::DelDataRequest: {
    std::vector<ObjectID> ids;
    bool force, deep;
    TRY_READ_REQUEST(ReadDelDataRequest(root, ids, force, deep));
    RESPONSE_ON_ERROR(
        server_ptr_->DelData(ids, force, deep, [self](const Status& status) {
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
  } break;
  case CommandType::CreateStreamRequest: {
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
  } break;
  case CommandType::MarkStreamRequest: {
    ObjectID stream_id;
    int64_t mark;
    TRY_READ_REQUEST(ReadMarkStreamRequest(root, stream_id, mark));
    auto status = server_ptr_->GetStreamStore()->Mark(stream_id, mark);
    std::string message_out;
    if (status.ok()) {
      WriteMarkStreamReply(message_out);
    } else {
      LOG(ERROR) << status.ToString();
      WriteErrorReply(status, message_out);
    }
    this->doWrite(message_out);
  } break;
  case CommandType::GetNextStreamChunkRequest: {
    ObjectID stream_id;
    size_t size;
    TRY_READ_REQUEST(ReadGetNextStreamChunkRequest(root, stream_id, size));
    RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Get(
        stream_id, size, [self](const Status& status, const ObjectID chunk) {
          std::string message_out;
          if (status.ok()) {
            std::shared_ptr<Payload> object;
            RETURN_ON_ERROR(
                self->server_ptr_->GetBulkStore()->ProcessGetRequest(chunk,
                                                                     object));
            WriteGetNextStreamChunkReply(object, message_out);
            int store_fd = object->store_fd;
            self->doWrite(message_out, [self, store_fd](const Status& status) {
              if (self->used_fds_.find(store_fd) == self->used_fds_.end()) {
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
  } break;
  case CommandType::PullNextStreamChunkRequest: {
    ObjectID stream_id;
    TRY_READ_REQUEST(ReadPullNextStreamChunkRequest(root, stream_id));
    this->associated_streams_.emplace(stream_id);
    RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Pull(
        stream_id, [self](const Status& status, const ObjectID chunk) {
          std::string message_out;
          if (status.ok()) {
            std::shared_ptr<Payload> object;
            RETURN_ON_ERROR(
                self->server_ptr_->GetBulkStore()->ProcessGetRequest(chunk,
                                                                     object));
            WritePullNextStreamChunkReply(object, message_out);
            int store_fd = object->store_fd;
            self->doWrite(message_out, [self, store_fd](const Status& status) {
              if (self->used_fds_.find(store_fd) == self->used_fds_.end()) {
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
  } break;
  case CommandType::StopStreamRequest: {
    ObjectID stream_id;
    bool failed;
    TRY_READ_REQUEST(ReadStopStreamRequest(root, stream_id, failed));
    // NB: don't erase the metadata from meta_service, since there's may
    // reader listen on this stream.
    RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Stop(stream_id, failed));
    std::string message_out;
    WriteStopStreamReply(message_out);
    this->doWrite(message_out);
  } break;
  case CommandType::PutNameRequest: {
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
  } break;
  case CommandType::GetNameRequest: {
    std::string name;
    bool wait;
    TRY_READ_REQUEST(ReadGetNameRequest(root, name, wait));
    RESPONSE_ON_ERROR(server_ptr_->GetName(
        name, wait, [self]() { return self->running_; },
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
  } break;
  case CommandType::DropNameRequest: {
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
  } break;
  case CommandType::ClusterMetaRequest: {
    TRY_READ_REQUEST(ReadClusterMetaRequest(root));
    RESPONSE_ON_ERROR(server_ptr_->ClusterInfo(
        [self](const Status& status, const json& tree) {
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
  } break;
  case CommandType::InstanceStatusRequest: {
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
  } break;
  case CommandType::ExitRequest:
    return true;
  default: {
    LOG(ERROR) << "Got unexpected command: " << type;
  }
  }
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
  // On Mac the state of socket may be "not connected" after the client has
  // already closed the socket, hence there will be an exception.
  boost::system::error_code ec;
  socket_.shutdown(stream_protocol::socket::shutdown_both, ec);
  socket_.close();
  // do cleanup: clean up streams associated with this client
  for (auto stream_id : associated_streams_) {
    VINEYARD_SUPPRESS(server_ptr_->GetStreamStore()->Drop(stream_id));
  }
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
                        socket_server_ptr_->RemoveConnection(conn_id_);
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
              socket_server_ptr_->RemoveConnection(conn_id_);
            }
          }
        } else {
          doStop();
          socket_server_ptr_->RemoveConnection(conn_id_);
        }
      });
}

SocketServer::SocketServer(vs_ptr_t vs_ptr)
    : vs_ptr_(vs_ptr), next_conn_id_(0) {}

void SocketServer::Start() { doAccept(); }

void SocketServer::Stop() {
  std::lock_guard<std::mutex> scope_lock(this->connections_mutx_);
  for (auto& pair : connections_) {
    pair.second->Stop();
  }
  connections_.clear();
}

bool SocketServer::ExistsConnection(int conn_id) const {
  std::lock_guard<std::mutex> scope_lock(this->connections_mutx_);
  return connections_.find(conn_id) != connections_.end();
}

void SocketServer::RemoveConnection(int conn_id) {
  std::lock_guard<std::mutex> scope_lock(this->connections_mutx_);
  connections_.erase(conn_id);
}

void SocketServer::CloseConnection(int conn_id) {
  std::lock_guard<std::mutex> scope_lock(this->connections_mutx_);
  connections_.at(conn_id)->Stop();
  RemoveConnection(conn_id);
}

size_t SocketServer::AliveConnections() const {
  std::lock_guard<std::mutex> scope_lock(this->connections_mutx_);
  return connections_.size();
}

}  // namespace vineyard
