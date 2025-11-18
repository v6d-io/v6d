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

#include "server/async/socket_server.h"

#include <fcntl.h>
#include <sys/mman.h>

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/memory/cuda_ipc.h"
#include "common/memory/fling.h"
#include "common/memory/memcpy.h"
#include "common/util/callback.h"
#include "common/util/functions.h"
#include "common/util/json.h"
#include "common/util/protocols.h"
#include "common/util/sidecar.h"
#include "server/server/vineyard_server.h"
#include "server/util/metrics.h"
#include "server/util/remote.h"

#include "thread-pool/thread_pool.h"

namespace vineyard {

// We set a hard limit for the message buffer size, since an evil client,
// e.g., telnet.
//
// We don't revise the structure of protocol, for backwards compatible, as
// we already released wheel packages on pypi.
constexpr size_t MESSAGE_HEADER_LIMIT = 256 * 1024 * 1024;  // 256M bytes

SocketConnection::SocketConnection(
    stream_protocol::socket socket, std::shared_ptr<VineyardServer> server_ptr,
    std::shared_ptr<SocketServer> socket_server_ptr, int conn_id,
    std::string host)
    : socket_(std::move(socket)),
      server_ptr_(server_ptr),
      socket_server_ptr_(socket_server_ptr),
      conn_id_(conn_id),
      peer_host(std::move(host)) {
  // hold the references of bulkstore using `shared_from_this()`.
  auto bulk_store = server_ptr_->GetBulkStore();
  if (bulk_store != nullptr) {
    bulk_store_ = bulk_store->shared_from_this();
  }
  auto plasma_bulk_store = server_ptr_->GetBulkStore<PlasmaID>();
  if (plasma_bulk_store != nullptr) {
    plasma_bulk_store_ = plasma_bulk_store->shared_from_this();
  }
  // initializing
  this->registered_.store(false);
  this->trace_log_level_ = server_ptr_->GetTraceLogLevel();
}

bool SocketConnection::sendFd(int fd) { return send_fd(nativeHandle(), fd); }

bool SocketConnection::Start() {
  running_.store(true);
  doReadHeader();

  return true;
}

bool SocketConnection::Stop() {
  if (!running_.exchange(false)) {
    // already stopped, or haven't started
    return false;
  }

  auto self(shared_from_this());
  if (server_ptr_->GetBulkStoreType() == StoreType::kDefault) {
    std::unordered_set<ObjectID> ids;
    auto status = bulk_store_->ReleaseConnection(this->getConnId());
    if (!status.ok() && !status.IsKeyError()) {
      LOG(WARNING) << "Failed to release the connection '" << this->getConnId()
                   << "' from object dependency: " << status.ToString();
    }
  }

  // do cleanup: clean up streams associated with this client
  for (auto stream_id : associated_streams_) {
    VINEYARD_SUPPRESS(server_ptr_->GetStreamStore()->Drop(stream_id));
  }

  // On Mac the state of socket may be "not connected" after the client has
  // already closed the socket, hence there will be an exception.
  boost::system::error_code ec;
  ec = socket_.cancel(ec);
  ec = socket_.shutdown(stream_protocol::socket::shutdown_both, ec);
  ec = socket_.close(ec);

  return true;
}

void SocketConnection::doReadHeader() {
  auto self(shared_from_this());
  if (!running_.load()) {  // don't read if stopped
    return;
  }
  asio::async_read(socket_, asio::buffer(&read_msg_header_, sizeof(size_t)),
                   [this, self](boost::system::error_code ec, std::size_t) {
                     if (!ec && running_.load()) {
                       doReadBody();
                     } else {
                       doStop();
                       //  ThrowException();
                     }
                   });
}

void SocketConnection::doReadBody() {
  if (read_msg_header_ > MESSAGE_HEADER_LIMIT) {
    VLOG(10) << "invalid message header value: " << read_msg_header_;
    doStop();
    // ThrowException();
    return;
  }
  read_msg_body_.resize(read_msg_header_ + 1);
  read_msg_body_[read_msg_header_] = '\0';
  auto self(shared_from_this());
  asio::async_read(socket_, asio::buffer(&read_msg_body_[0], read_msg_header_),
                   [this, self](boost::system::error_code ec, std::size_t) {
                     if ((!ec || ec == asio::error::eof) && running_.load()) {
                       bool exit = processMessage(read_msg_body_);
                       if (exit || ec == asio::error::eof) {
                         doStop();
                         if (!exit) {
                           //  ThrowException();
                         }
                         return;
                       }
                     } else {
                       doStop();
                       //  ThrowException();
                       return;
                     }
                   });
}

#ifndef __REPORT_JSON_ERROR
#ifndef NDEBUG
#define __REPORT_JSON_ERROR(err, data) \
  LOG(ERROR) << "json: " << err.what() << " when parsing" << data
#else
#define __REPORT_JSON_ERROR(err, data) \
  LOG(ERROR) << "json error: " << err.what()
#endif  // NDEBUG
#endif  // __REPORT_JSON_ERROR

#ifndef TRY_READ_FROM_JSON
#define TRY_READ_FROM_JSON(read_action, data)                              \
  try {                                                                    \
    read_action;                                                           \
  } catch (std::out_of_range const& err) {                                 \
    __REPORT_JSON_ERROR(err, data);                                        \
    std::stringstream err_message;                                         \
    err_message << std::string(                                            \
                       "Failed to parse the JSON message: out of range: ") \
                << err.what();                                             \
    err_message << ", at " << __FILE__ << ":" << __LINE__ << "@"           \
                << #read_action;                                           \
    err_message << ", input is '" << data << "'";                          \
    std::string message_out;                                               \
    WriteErrorReply(Status::Invalid(err_message.str()), message_out);      \
    this->doWrite(message_out);                                            \
    return false;                                                          \
  } catch (json::exception const& err) {                                   \
    __REPORT_JSON_ERROR(err, data);                                        \
    std::stringstream err_message;                                         \
    err_message << std::string(                                            \
                       "Failed to parse the JSON message: json error: ")   \
                << err.what();                                             \
    err_message << ", at " << __FILE__ << ":" << __LINE__ << "@"           \
                << #read_action;                                           \
    err_message << ", input is '" << data << "'";                          \
    std::string message_out;                                               \
    WriteErrorReply(Status::Invalid(err_message.str()), message_out);      \
    this->doWrite(message_out);                                            \
    return false;                                                          \
  } catch (std::exception const& err) {                                    \
    __REPORT_JSON_ERROR(err, data);                                        \
    std::stringstream err_message;                                         \
    err_message << std::string(                                            \
                       "Failed to parse the JSON message: exception: ")    \
                << err.what();                                             \
    err_message << ", at " << __FILE__ << ":" << __LINE__ << "@"           \
                << #read_action;                                           \
    err_message << ", input is '" << data << "'";                          \
    std::string message_out;                                               \
    WriteErrorReply(Status::Invalid(err_message.str()), message_out);      \
    this->doWrite(message_out);                                            \
    return false;                                                          \
  }

#endif  // TRY_READ_FROM_JSON

#ifndef TRY_READ_REQUEST
#define TRY_READ_REQUEST(operation, data, ...)                             \
  do {                                                                     \
    Status read_status;                                                    \
    TRY_READ_FROM_JSON(read_status = operation(data, ##__VA_ARGS__), data) \
    if (!read_status.ok()) {                                               \
      std::string error_message_out;                                       \
      WriteErrorReply(read_status, error_message_out);                     \
      self->doWrite(error_message_out);                                    \
      return false;                                                        \
    }                                                                      \
  } while (0)
#endif  // TRY_READ_REQUEST

#ifndef RESPONSE_ON_ERROR
#define RESPONSE_ON_ERROR(status)                                             \
  do {                                                                        \
    auto exec_status = (status);                                              \
    if (!exec_status.ok()) {                                                  \
      VLOG(100) << "Error: unexpected error occurs during message handling: " \
                << exec_status.ToString();                                    \
      std::string error_message_out;                                          \
      WriteErrorReply(exec_status, error_message_out);                        \
      self->doWrite(error_message_out);                                       \
      return false;                                                           \
    }                                                                         \
  } while (0)
#endif  // RESPONSE_ON_ERROR

bool SocketConnection::processMessage(const std::string& message_in) {
  json root;
  std::istringstream is(message_in);
  auto self(shared_from_this());

  // DON'T let vineyardd crash when the client is malicious.
  uint64_t start = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  TRY_READ_FROM_JSON(root = json::parse(message_in), message_in);
  uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  if (!root.contains("type")) {
    RESPONSE_ON_ERROR(Status::Invalid("Invalid message: no 'type' field"));
  }

  std::string const& cmd = root["type"].get_ref<std::string const&>();
  if (!registered_.load() && cmd != command_t::REGISTER_REQUEST) {
    RESPONSE_ON_ERROR(Status::Invalid(
        "The connection is not registered yet, command is: " + cmd));
  }

  if (end - start > 500) {
    LOG(INFO) << "parse cmd: " << cmd << " json time: " << (end - start)
              << " us";
  }

  if (cmd == command_t::REGISTER_REQUEST) {
    return doRegister(root);
  } else if (cmd == command_t::REQUIRE_EXTRA_REQUEST_MEMORY_REQUEST) {
    return doRequireExtraRequestMemory(root);
  } else if (cmd == command_t::EXIT_REQUEST) {
    return true;
  } else if (cmd == command_t::CREATE_BUFFER_REQUEST) {
    return doCreateBuffer(root);
  } else if (cmd == command_t::CREATE_BUFFERS_REQUEST) {
    return doCreateBuffers(root);
  } else if (cmd == command_t::CREATE_DISK_BUFFER_REQUEST) {
    return doCreateDiskBuffer(root);
  } else if (cmd == command_t::CREATE_GPU_BUFFER_REQUEST) {
    return doCreateGPUBuffer(root);
  } else if (cmd == command_t::SEAL_BUFFER_REQUEST) {
    return doSealBlob(root);
  } else if (cmd == command_t::GET_BUFFERS_REQUEST) {
    return doGetBuffers(root);
  } else if (cmd == command_t::GET_GPU_BUFFERS_REQUEST) {
    return doGetGPUBuffers(root);
  } else if (cmd == command_t::DROP_BUFFER_REQUEST) {
    return doDropBuffer(root);
  } else if (cmd == command_t::SHRINK_BUFFER_REQUEST) {
    return doShrinkBuffer(root);
  } else if (cmd == command_t::CREATE_REMOTE_BUFFER_REQUEST) {
    return doCreateRemoteBuffer(root);
  } else if (cmd == command_t::CREATE_REMOTE_BUFFERS_REQUEST) {
    return doCreateRemoteBuffers(root);
  } else if (cmd == command_t::GET_REMOTE_BUFFERS_REQUEST) {
    return doGetRemoteBuffers(root);
  } else if (cmd == command_t::INCREASE_REFERENCE_COUNT_REQUEST) {
    return doIncreaseReferenceCount(root);
  } else if (cmd == command_t::RELEASE_REQUEST) {
    return doRelease(root);
  } else if (cmd == command_t::DEL_DATA_WITH_FEEDBACKS_REQUEST) {
    return doDelDataWithFeedbacks(root);
  } else if (cmd == command_t::DEL_HUGE_DATA_REQUEST) {
    return doDelHugeData(root);
  } else if (cmd == command_t::CREATE_BUFFER_PLASMA_REQUEST) {
    return doCreateBufferByPlasma(root);
  } else if (cmd == command_t::GET_BUFFERS_PLASMA_REQUEST) {
    return doGetBuffersByPlasma(root);
  } else if (cmd == command_t::PLASMA_SEAL_REQUEST) {
    return doSealPlasmaBlob(root);
  } else if (cmd == command_t::PLASMA_RELEASE_REQUEST) {
    return doPlasmaRelease(root);
  } else if (cmd == command_t::PLASMA_DEL_DATA_REQUEST) {
    return doPlasmaDelData(root);
  } else if (cmd == command_t::CREATE_USER_BUFFERS_REQUEST) {
    return doCreateUserBuffers(root);
  } else if (cmd == command_t::GET_USER_BUFFERS_REQUEST) {
    return doGetUserBuffers(root);
  } else if (cmd == command_t::DELETE_USER_BUFFERS_REQUEST) {
    return doDeleteUserBuffers(root);
  } else if (cmd == command_t::CREATE_DATA_REQUEST) {
    return doCreateData(root);
  } else if (cmd == command_t::CREATE_DATAS_REQUEST) {
    return doCreateDatas(root);
  } else if (cmd == command_t::CREATE_HUGE_DATAS_REQUEST) {
    return doCreatehugeDatas(root);
  } else if (cmd == command_t::GET_DATA_REQUEST) {
    return doGetData(root);
  } else if (cmd == command_t::GET_HUGE_DATA_REQUEST) {
    return doGetHugeData(root);
  } else if (cmd == command_t::DELETE_DATA_REQUEST) {
    return doDelData(root);
  } else if (cmd == command_t::LIST_DATA_REQUEST) {
    return doListData(root);
  } else if (cmd == command_t::EXISTS_REQUEST) {
    return doExists(root);
  } else if (cmd == command_t::PERSIST_REQUEST) {
    return doPersist(root);
  } else if (cmd == command_t::IF_PERSIST_REQUEST) {
    return doIfPersist(root);
  } else if (cmd == command_t::BATCH_PERSIST_REQUEST) {
    return doBatchPersist(root);
  } else if (cmd == command_t::LABEL_REQUEST) {
    return doLabelObject(root);
  } else if (cmd == command_t::CLEAR_REQUEST) {
    return doClear(root);
  } else if (cmd == command_t::MEMORY_TRIM_REQUEST) {
    return doMemoryTrim(root);
  } else if (cmd == command_t::CREATE_STREAM_REQUEST) {
    return doCreateStream(root);
  } else if (cmd == command_t::CREATE_FIXED_STREAM_REQUEST) {
    return doCreateFixedStream(root);
  } else if (cmd == command_t::OPEN_STREAM_REQUEST) {
    return doOpenStream(root);
  } else if (cmd == command_t::GET_NEXT_STREAM_CHUNK_REQUEST) {
    return doGetNextStreamChunk(root);
  } else if (cmd == command_t::PUSH_NEXT_STREAM_CHUNK_REQUEST) {
    return doPushNextStreamChunk(root);
  } else if (cmd == command_t::PUSH_NEXT_STREAM_CHUNK_BY_OFFSET_REQUEST) {
    return doPushNextStreamChunkByOffset(root);
  } else if (cmd == command_t::PULL_NEXT_STREAM_CHUNK_REQUEST) {
    return doPullNextStreamChunk(root);
  } else if (cmd == command_t::CHECK_FIXED_STREAM_RECEIVED_REQUEST) {
    return doCheckFixedStreamReceived(root);
  } else if (cmd == command_t::STOP_STREAM_REQUEST) {
    return doStopStream(root);
  } else if (cmd == command_t::DROP_STREAM_REQUEST) {
    return doDropStream(root);
  } else if (cmd == command_t::ABORT_STREAM_REQUEST) {
    return doAbortStream(root);
  } else if (cmd == command_t::PUT_STREAM_NAME_REQUEST) {
    return doPutStreamName(root);
  } else if (cmd == command_t::GET_STREAM_ID_BY_NAME_REQUEST) {
    return doGetStreamIDByName(root);
  } else if (cmd == command_t::ACTIVATE_REMOTE_FIXED_STREAM_REQUEST) {
    return doActivateRemoteFixedStream(root);
  } else if (cmd ==
             command_t::
                 VINEYARD_ACTIVATE_REMOTE_FIXED_STREAM_WITH_OFFSET_REQUEST) {
    return doVineyardActivateRemoteFixedStreamWithOffset(root);
  } else if (cmd == command_t::OPEN_FIXED_STREAM_REQUEST) {
    return doOpenFixedStream(root);
  } else if (cmd == command_t::CLOSE_STREAM_REQUEST) {
    return doCloseStream(root);
  } else if (cmd == command_t::DELETE_STREAM_REQUEST) {
    return doDeleteStream(root);
  } else if (cmd == command_t::VINEYARD_OPEN_REMOTE_FIXED_STREAM_REQUEST) {
    return doVineyardOpenRemoteFixedStream(root);
  } else if (cmd == command_t::VINEYARD_STOP_STREAM_REQUEST) {
    return doVineyardStopStream(root);
  } else if (cmd == command_t::VINEYARD_ABORT_REMOTE_STREAM_REQUEST) {
    return doVineyardAbortRemoteStream(root);
  } else if (cmd == command_t::VINEYARD_DROP_STREAM_REQUEST) {
    return doVineyardDropStream(root);
  } else if (cmd == command_t::VINEYARD_CLOSE_REMOTE_FIXED_STREAM_REQUEST) {
    return doVineyardCloseRemoteFixedStream(root);
  } else if (cmd == command_t::VINEYARD_GET_METAS_BY_NAMES_REQUEST) {
    return doVineyardGetMetasByNames(root);
  } else if (cmd == command_t::VINEYARD_GET_REMOTE_BLOBS_WITH_RDMA_REQUEST) {
    return doVineyardGetRemoteBlobs(root);
  } else if (cmd == command_t::VINEYARD_GET_REMOTE_BLOBS_WITH_OFFSET_REQUEST) {
    return doVineyardGetRemoteBlobsWithOffset(root);
  } else if (cmd == command_t::PUT_NAME_REQUEST) {
    return doPutName(root);
  } else if (cmd == command_t::PUT_NAMES_REQUEST) {
    return doPutNames(root);
  } else if (cmd == command_t::GET_NAME_REQUEST) {
    return doGetName(root);
  } else if (cmd == command_t::GET_NAME_LOCATION_REQUEST) {
    return doGetObjectLocation(root);
  } else if (cmd == command_t::PUT_NAME_LOCATION_REQUEST) {
    return doPutObjectLocation(root);
  } else if (cmd == command_t::GET_NAMES_REQUEST) {
    return doGetNames(root);
  } else if (cmd == command_t::DROP_NAMES_REQUEST) {
    return doDropNames(root);
  } else if (cmd == command_t::LIST_NAME_REQUEST) {
    return doListName(root);
  } else if (cmd == command_t::DROP_NAME_REQUEST) {
    return doDropName(root);
  } else if (cmd == command_t::MAKE_ARENA_REQUEST) {
    return doMakeArena(root);
  } else if (cmd == command_t::FINALIZE_ARENA_REQUEST) {
    return doFinalizeArena(root);
  } else if (cmd == command_t::NEW_SESSION_REQUEST) {
    return doNewSession(root);
  } else if (cmd == command_t::DELETE_SESSION_REQUEST) {
    return doDeleteSession(root);
  } else if (cmd == command_t::MOVE_BUFFERS_OWNERSHIP_REQUEST) {
    return doMoveBuffersOwnership(root);
  } else if (cmd == command_t::EVICT_REQUEST) {
    return doEvictObjects(root);
  } else if (cmd == command_t::LOAD_REQUEST) {
    return doLoadObjects(root);
  } else if (cmd == command_t::UNPIN_REQUEST) {
    return doUnpinObjects(root);
  } else if (cmd == command_t::IS_SPILLED_REQUEST) {
    return doIsSpilled(root);
  } else if (cmd == command_t::IS_IN_USE_REQUEST) {
    return doIsInUse(root);
  } else if (cmd == command_t::GET_VINEYARD_MMAP_FD_REQUEST) {
    return doGetVineyardMmapFd(root);
  } else if (cmd == command_t::CLUSTER_META_REQUEST) {
    return doClusterMeta(root);
  } else if (cmd == command_t::INSTANCE_STATUS_REQUEST) {
    return doInstanceStatus(root);
  } else if (cmd == command_t::MIGRATE_OBJECT_REQUEST) {
    return doMigrateObject(root);
  } else if (cmd == command_t::SHALLOW_COPY_REQUEST) {
    return doShallowCopy(root);
  } else if (cmd == command_t::DEBUG_REQUEST) {
    return doDebug(root);
  } else if (cmd == command_t::ACQUIRE_LOCK_REQUEST) {
    return doAcquireLock(root);
  } else if (cmd == command_t::RELEASE_LOCK_REQUEST) {
    return doReleaseLock(root);
  } else if (cmd == command_t::RELEASE_BLOBS_WITH_RDMA_REQUEST) {
    return doReleaseBlobsWithRDMA(root);
  } else {
    RESPONSE_ON_ERROR(Status::Invalid("Got unexpected command: " + cmd));
    return false;
  }
}

bool SocketConnection::doRegister(const json& root) {
  auto self(shared_from_this());
  std::string client_version;
  StoreType bulk_store_type;
  SessionID session_id;
  std::string username, password;
  TRY_READ_REQUEST(ReadRegisterRequest, root, client_version, bulk_store_type,
                   session_id, username, password);
  RESPONSE_ON_ERROR(server_ptr_->Verify(
      username, password,
      [self, bulk_store_type, session_id](const Status& status) -> Status {
        std::string message_out;
        if (status.ok()) {
          Status s = self->socket_server_ptr_->Register(self, session_id);
          if (s.ok()) {
            WriteRegisterReply(
                self->server_ptr_->IPCSocket(),
                self->server_ptr_->RPCEndpoint(),
                self->server_ptr_->instance_id(),
                self->server_ptr_->session_id(),
                self->server_ptr_->store_matched(bulk_store_type),
                self->server_ptr_->compression_enabled(), message_out);
          } else {
            WriteErrorReply(s, message_out);
          }
        } else {
          WriteErrorReply(Status::ConnectionError(status.ToString()),
                          message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doRequireExtraRequestMemory(json const& root) {
  auto self(shared_from_this());
  size_t size = 0;
  TRY_READ_REQUEST(ReadRequireExtraRequestMemoryRequest, root, size);
  if (size == 0 || size > std::numeric_limits<size_t>::max() / 2) {
    RESPONSE_ON_ERROR(Status::Invalid(
        "require extra request memory: invalid size: " + std::to_string(size)));
  }

  int fd = -1;
  RESPONSE_ON_ERROR(socket_server_ptr_->RequireExtraRequestMemory(
      this->getConnId(), size, fd));
  std::string message_out;
  WriteRequireExtraRequestMemoryReply(message_out);
  this->doWrite(message_out);
  LOG(INFO) << "require extra request memory: size = " << size
            << ", fd = " << fd;
  sendFd(fd);

  return false;
}

bool SocketConnection::doCreateBuffer(const json& root) {
  auto self(shared_from_this());
  size_t size;
  std::shared_ptr<Payload> object;
  std::string message_out;

  TRY_READ_REQUEST(ReadCreateBufferRequest, root, size);
  ObjectID object_id;
  RESPONSE_ON_ERROR(bulk_store_->Create(size, object_id, object));

  int fd_to_send = -1;
  if (object->data_size > 0 &&
      self->used_fds_.find(object->store_fd) == self->used_fds_.end()) {
    this->used_fds_.emplace(object->store_fd);
    fd_to_send = object->store_fd;
  }

  WriteCreateBufferReply(object_id, object, fd_to_send, message_out);

  this->doWrite(message_out, [this, self, fd_to_send](const Status& status) {
    if (fd_to_send != -1) {
      sendFd(fd_to_send);
    }
    LOG_SUMMARY("instances_memory_usage_bytes", server_ptr_->instance_id(),
                bulk_store_->Footprint());
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doCreateBuffers(const json& root) {
  auto self(shared_from_this());
  std::vector<size_t> sizes;
  std::vector<ObjectID> object_ids;
  std::vector<std::shared_ptr<Payload>> objects;
  std::string message_out;

  TRY_READ_REQUEST(ReadCreateBuffersRequest, root, sizes);
  for (auto const& size : sizes) {
    ObjectID object_id;
    std::shared_ptr<Payload> object;
    Status status = bulk_store_->Create(size, object_id, object);
    if (!status.ok()) {
      for (auto const& object : objects) {
        bulk_store_->Delete(object->id());
      }
      RESPONSE_ON_ERROR(status);
    }
    object_ids.emplace_back(object_id);
    objects.emplace_back(object);
  }

  std::set<int> fds_to_send_set;
  for (auto const& object : objects) {
    if (object->data_size > 0 &&
        self->used_fds_.find(object->store_fd) == self->used_fds_.end()) {
      this->used_fds_.emplace(object->store_fd);
      fds_to_send_set.emplace(object->store_fd);
    }
  }
  std::vector<int> fds_to_send(fds_to_send_set.begin(), fds_to_send_set.end());

  WriteCreateBuffersReply(object_ids, objects, fds_to_send, message_out);

  this->doWrite(message_out, [this, self, fds_to_send](const Status& status) {
    for (auto const& fd_to_send : fds_to_send) {
      if (fd_to_send != -1) {
        sendFd(fd_to_send);
      }
    }
    LOG_SUMMARY("instances_memory_usage_bytes", server_ptr_->instance_id(),
                bulk_store_->Footprint());
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doCreateDiskBuffer(const json& root) {
  auto self(shared_from_this());
  size_t size = 0;
  std::string path;
  std::shared_ptr<Payload> object;
  std::string message_out;

  TRY_READ_REQUEST(ReadCreateDiskBufferRequest, root, size, path);

  if (size == 0 && path.empty()) {
    RESPONSE_ON_ERROR(Status::Invalid(
        "create disk buffer: one of 'size' and 'path' must be specified"));
  }

  ObjectID object_id;
  RESPONSE_ON_ERROR(bulk_store_->CreateDisk(size, path, object_id, object));

  int fd_to_send = -1;
  if (object->data_size > 0 &&
      self->used_fds_.find(object->store_fd) == self->used_fds_.end()) {
    this->used_fds_.emplace(object->store_fd);
    fd_to_send = object->store_fd;
  }

  WriteCreateDiskBufferReply(object_id, object, fd_to_send, message_out);

  this->doWrite(message_out, [this, self, fd_to_send](const Status& status) {
    if (fd_to_send != -1) {
      sendFd(fd_to_send);
    }
    LOG_SUMMARY("instances_memory_usage_bytes", server_ptr_->instance_id(),
                bulk_store_->Footprint());
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doCreateGPUBuffer(json const& root) {
  auto self(shared_from_this());

  size_t size;
  std::shared_ptr<Payload> object;
  std::string message_out;

  TRY_READ_REQUEST(ReadCreateGPUBufferRequest, root, size);

#ifndef ENABLE_CUDA
  WriteErrorReply(Status::Invalid("GPU support is not enabled"), message_out);
#else
  ObjectID object_id;
  RESPONSE_ON_ERROR(bulk_store_->CreateGPU(size, object_id, object));
  if (!object->IsGPU() || object->pointer == nullptr) {
    RESPONSE_ON_ERROR(Status::Invalid(
        "Failed to create GPU buffer: invalid GPU memory pointer"));
  }

  std::vector<int64_t> handle(8 /* CUDA_IPC_HANDLE_SIZE = 64 */);
  send_cuda_pointer(reinterpret_cast<void*>(object->pointer),
                    reinterpret_cast<uint8_t*>(handle.data()));
  WriteGPUCreateBufferReply(object_id, object, handle, message_out);
#endif

  this->doWrite(message_out, [this, self](const Status& status) {
    LOG_SUMMARY("instances_gpu_memory_usage_bytes", server_ptr_->instance_id(),
                bulk_store_->Footprint());
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doSealBlob(json const& root) {
  auto self(shared_from_this());
  ObjectID id;
  TRY_READ_REQUEST(ReadSealRequest, root, id);
  RESPONSE_ON_ERROR(bulk_store_->Seal(id));
  RESPONSE_ON_ERROR(bulk_store_->AddDependency(id, getConnId()));
  std::string message_out;
  WriteSealReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doGetBuffers(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool unsafe = false;
  std::vector<std::shared_ptr<Payload>> objects;
  std::string message_out;

  TRY_READ_REQUEST(ReadGetBuffersRequest, root, ids, unsafe);
  RESPONSE_ON_ERROR(bulk_store_->GetUnsafe(ids, unsafe, objects));
  RESPONSE_ON_ERROR(bulk_store_->AddDependency(
      std::unordered_set<ObjectID>(ids.begin(), ids.end()), this->getConnId()));
  for (size_t i = 0; i < objects.size(); ++i) {
    if (objects[i]->pointer == nullptr) {
      RESPONSE_ON_ERROR(
          bulk_store_->ReloadColdObject(ids[i], objects[i], false));
    }
  }

  std::vector<int> fd_to_send;
  for (auto object : objects) {
    if (object->data_size > 0 &&
        self->used_fds_.find(object->store_fd) == self->used_fds_.end()) {
      self->used_fds_.emplace(object->store_fd);
      fd_to_send.emplace_back(object->store_fd);
    }
  }
  WriteGetBuffersReply(objects, fd_to_send, false, message_out);

  /* NOTE: Here we send the file descriptor after the objects.
   *       We are using sendmsg to send the file descriptor
   *       which is a sync method. In theory, this might cause
   *       the server to block, but currently this seems to be
   *       the only method that are widely used in practice, e.g.,
   *       boost and Plasma, and actually the file descriptor is
   *       a very short message.
   *
   *       We will examine other methods later, such as using
   *       explicit file descriptors.
   */
  this->doWrite(message_out, [self, objects, fd_to_send](const Status& status) {
    for (int store_fd : fd_to_send) {
      self->sendFd(store_fd);
    }
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doGetGPUBuffers(json const& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool unsafe = false;
  std::vector<std::shared_ptr<Payload>> objects;
  std::vector<std::vector<int64_t>> handles;
  std::string message_out;

  TRY_READ_REQUEST(ReadGetGPUBuffersRequest, root, ids, unsafe);
#ifndef ENABLE_CUDA
  WriteErrorReply(Status::Invalid("GPU support is not enabled"), message_out);
#else
  RESPONSE_ON_ERROR(bulk_store_->GetUnsafe(ids, unsafe, objects));
  std::vector<int64_t> handle(8 /* CUDA_IPC_HANDLE_SIZE = 64 */);
  for (auto object : objects) {
    send_cuda_pointer(reinterpret_cast<void*>(object->pointer),
                      reinterpret_cast<uint8_t*>(handle.data()));
    handles.emplace_back(handle);
  }
  WriteGetGPUBuffersReply(objects, handles, message_out);
#endif

  this->doWrite(message_out, [self](const Status& status) {
    // no need for sending fds for GPU buffers
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doDropBuffer(const json& root) {
  auto self(shared_from_this());
  ObjectID object_id = InvalidObjectID();
  TRY_READ_REQUEST(ReadDropBufferRequest, root, object_id);
  // Delete ignore reference count.
  auto status = bulk_store_->OnDelete(object_id);
  std::string message_out;
  if (status.ok()) {
    WriteDropBufferReply(message_out);
  } else {
    WriteErrorReply(status, message_out);
  }
  this->doWrite(message_out);
  LOG_SUMMARY("instances_memory_usage_bytes", server_ptr_->instance_id(),
              bulk_store_->Footprint());
  return false;
}

bool SocketConnection::doShrinkBuffer(const json& root) {
  auto self(shared_from_this());
  ObjectID object_id = InvalidObjectID();
  size_t size = 0;
  TRY_READ_REQUEST(ReadShrinkBufferRequest, root, object_id, size);
  // Delete ignore reference count.
  auto status = bulk_store_->Shrink(object_id, size);
  std::string message_out;
  if (status.ok()) {
    WriteShrinkBufferReply(message_out);
  } else {
    WriteErrorReply(status, message_out);
  }
  this->doWrite(message_out);
  LOG_SUMMARY("instances_memory_usage_bytes", server_ptr_->instance_id(),
              bulk_store_->Footprint());
  return false;
}

bool SocketConnection::doCreateRemoteBuffer(const json& root) {
  auto self(shared_from_this());
  size_t size;
  bool compress = false;
  bool use_rdma = false;
  std::shared_ptr<Payload> object;

  TRY_READ_REQUEST(ReadCreateRemoteBufferRequest, root, size, compress,
                   use_rdma);
  ObjectID object_id;
  RESPONSE_ON_ERROR(bulk_store_->Create(size, object_id, object));
  RESPONSE_ON_ERROR(bulk_store_->Seal(object_id));
  RESPONSE_ON_ERROR(bulk_store_->AddDependency(object_id, this->getConnId()));

  if (use_rdma) {
    std::string message_out;
    WriteCreateBufferReply(object->object_id, object, -1, message_out);
    this->doWrite(message_out);
  } else {
    auto callback = [self, this, compress,
                     object](const Status& status) -> Status {
      ReceiveRemoteBuffers(
          socket_, {object}, compress,
          [self, object](const Status& status) -> Status {
            VINEYARD_DISCARD(self->bulk_store_->RemoveDependency(
                object->object_id, self->getConnId()));
            std::string message_out;
            if (status.ok()) {
              WriteCreateBufferReply(object->object_id, object, -1,
                                     message_out);
            } else {
              // cleanup
              VINEYARD_DISCARD(self->bulk_store_->Delete(object->object_id));
              WriteErrorReply(status, message_out);
            }
            self->doWrite(message_out);
            return Status::OK();
          });
      LOG_SUMMARY("instances_memory_usage_bytes",
                  self->server_ptr_->instance_id(),
                  self->bulk_store_->Footprint());
      return Status::OK();
    };

    // ok to continue
    std::string message_out;
    WriteCreateBufferReply(object->object_id, object, -1, message_out);
    self->doWrite(message_out, callback, true);
  }
  return false;
}

bool SocketConnection::doCreateRemoteBuffers(const json& root) {
  auto self(shared_from_this());
  std::vector<size_t> sizes;
  bool compress = false;
  bool use_rdma = false;
  std::vector<ObjectID> object_ids;
  std::vector<std::shared_ptr<Payload>> objects;

  TRY_READ_REQUEST(ReadCreateRemoteBuffersRequest, root, sizes, compress,
                   use_rdma);
  for (auto const& size : sizes) {
    ObjectID object_id;
    std::shared_ptr<Payload> object;
    RESPONSE_ON_ERROR(bulk_store_->Create(size, object_id, object));
    RESPONSE_ON_ERROR(bulk_store_->Seal(object_id));
    RESPONSE_ON_ERROR(bulk_store_->AddDependency(object_id, this->getConnId()));
    object_ids.emplace_back(object_id);
    objects.emplace_back(object);
  }

  if (use_rdma) {
    std::string message_out;
    WriteCreateBuffersReply(object_ids, objects, std::vector<int>{},
                            message_out);
    this->doWrite(message_out);
  } else {
    auto callback = [self, this, compress, object_ids,
                     objects](const Status& status) -> Status {
      ReceiveRemoteBuffers(
          socket_, objects, compress,
          [self, object_ids, objects](const Status& status) -> Status {
            VINEYARD_DISCARD(self->bulk_store_->RemoveDependency(
                std::unordered_set<ObjectID>(object_ids.begin(),
                                             object_ids.end()),
                self->getConnId()));
            std::string message_out;
            if (status.ok()) {
              WriteCreateBuffersReply(object_ids, objects, std::vector<int>{},
                                      message_out);
            } else {
              // cleanup
              for (auto const& object : objects) {
                VINEYARD_DISCARD(self->bulk_store_->Delete(object->object_id));
              }
              WriteErrorReply(status, message_out);
            }
            self->doWrite(message_out);
            return Status::OK();
          });
      LOG_SUMMARY("instances_memory_usage_bytes",
                  self->server_ptr_->instance_id(),
                  self->bulk_store_->Footprint());
      return Status::OK();
    };

    // ok to continue
    std::string message_out;
    WriteCreateBuffersReply(object_ids, objects, std::vector<int>{},
                            message_out);
    self->doWrite(message_out, callback, true);
  }
  return false;
}

bool SocketConnection::doGetRemoteBuffers(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool unsafe = false;
  bool compress = false;
  std::vector<std::shared_ptr<Payload>> objects;
  std::string message_out;
  bool use_rdma = false;

  TRY_READ_REQUEST(ReadGetRemoteBuffersRequest, root, ids, unsafe, compress,
                   use_rdma);
  this->LockTransmissionObjects(ids);
  if (!bulk_store_->GetUnsafe(ids, unsafe, objects).ok()) {
    this->UnlockTransmissionObjects(ids);
    WriteErrorReply(Status::KeyError("Failed to get objects"), message_out);
    this->doWrite(message_out);
    return false;
  }
  if (!bulk_store_
           ->AddDependency(std::unordered_set<ObjectID>(ids.begin(), ids.end()),
                           this->getConnId())
           .ok()) {
    this->UnlockTransmissionObjects(ids);
    WriteErrorReply(Status::KeyError("Failed to add dependency"), message_out);
    this->doWrite(message_out);
    return false;
  }
  WriteGetBuffersReply(objects, {}, compress, message_out);

  if (!use_rdma) {
    this->doWrite(message_out, [self, objects, compress,
                                ids](const Status& status) {
      SendRemoteBuffers(self->socket_, objects, 0, compress,
                        [self, ids = std::move(ids)](const Status& status) {
                          if (!status.ok()) {
                            VLOG(100)
                                << "Failed to send buffers to remote client: "
                                << status.ToString();
                          }
                          self->UnlockTransmissionObjects(ids);
                          return Status::OK();
                        });
      std::unordered_set<ObjectID> ids_set(ids.begin(), ids.end());
      VINEYARD_DISCARD(
          self->bulk_store_->RemoveDependency(ids_set, self->getConnId()));
      return Status::OK();
    });
  } else {
    this->doWrite(message_out);
  }
  return false;
}

bool SocketConnection::doIncreaseReferenceCount(json const& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  TRY_READ_REQUEST(ReadIncreaseReferenceCountRequest, root, ids);
  RESPONSE_ON_ERROR(bulk_store_->AddDependency(
      std::unordered_set<ObjectID>(ids.begin(), ids.end()), this->getConnId()));
  std::string message_out;
  WriteIncreaseReferenceCountReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doRelease(json const& root) {
  auto self(shared_from_this());
  ObjectID id;  // Must be a blob id.
  TRY_READ_REQUEST(ReadReleaseRequest, root, id);
  RESPONSE_ON_ERROR(bulk_store_->Release(id, getConnId()));
  std::string message_out;
  WriteReleaseReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doDelDataWithFeedbacks(json const& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool force, deep, memory_trim, fastpath;
  double startTime = GetCurrentTime();

  TRY_READ_REQUEST(ReadDelDataWithFeedbacksRequest, root, ids, force, deep,
                   memory_trim, fastpath);
  RESPONSE_ON_ERROR(server_ptr_->DelData(
      ids, force, deep, memory_trim, fastpath,
      [self, startTime](const Status& status,
                        std::vector<ObjectID> const& delete_ids) {
        std::string message_out;
        if (status.ok()) {
          std::vector<ObjectID> deleted_blob_ids;
          for (auto id : delete_ids) {
            if (IsBlob(id)) {
              deleted_blob_ids.emplace_back(id);
            }
          }
          WriteDelDataWithFeedbacksReply(deleted_blob_ids, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        double endTime = GetCurrentTime();
        LOG_SUMMARY("data_request_duration_microseconds", "delete",
                    (endTime - startTime) * 1000000);
        LOG_COUNTER("data_requests_total", "delete");
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doDelHugeData(json const& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool force, deep, memory_trim, fastpath;

  size_t total_id = 0;
  TRY_READ_REQUEST(ReadDelHugeDataRequest, root, total_id, force, deep,
                   memory_trim, fastpath);
  ids.resize(total_id);
  VINEYARD_DISCARD(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(
      self->socket_server_ptr_->GetClientAttributeMsg(self->conn_id_, attr));
  RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
      ids.data(), total_id * sizeof(ObjectID), this->getConnId()));

  RESPONSE_ON_ERROR(server_ptr_->DelData(
      ids, force, deep, memory_trim, fastpath,
      [self, attr](const Status& status) {
        VLOG(2) << "Delete huge data request completed. Request id: "
                << attr.req_name << ", status: " << status.ToString();
        std::string message_out;
        if (status.ok()) {
          WriteDelHugeDataReply(message_out);
        } else {
          LOG(ERROR) << "Error: " << status.ToString()
                     << ". Request id: " << attr.req_name;
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doCreateBufferByPlasma(json const& root) {
  auto self(shared_from_this());
  PlasmaID plasma_id;
  ObjectID object_id = InvalidObjectID();
  size_t size, plasma_size;
  std::shared_ptr<PlasmaPayload> plasma_object;

  TRY_READ_REQUEST(ReadCreateBufferByPlasmaRequest, root, plasma_id, size,
                   plasma_size);

  std::string message_out;
  RESPONSE_ON_ERROR(plasma_bulk_store_->Create(size, plasma_size, plasma_id,
                                               object_id, plasma_object));

  int store_fd = plasma_object->store_fd, fd_to_send = -1;
  int data_size = plasma_object->data_size;

  if (data_size > 0 &&
      self->used_fds_.find(store_fd) == self->used_fds_.end()) {
    self->used_fds_.emplace(store_fd);
    fd_to_send = store_fd;
  }

  WriteCreateBufferByPlasmaReply(object_id, plasma_object, fd_to_send,
                                 message_out);

  this->doWrite(message_out, [this, self, fd_to_send](const Status& status) {
    if (fd_to_send != -1) {
      self->sendFd(fd_to_send);
    }
    LOG_SUMMARY("instances_memory_usage_bytes", server_ptr_->instance_id(),
                plasma_bulk_store_->Footprint());
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doGetBuffersByPlasma(json const& root) {
  auto self(shared_from_this());
  std::vector<PlasmaID> plasma_ids;
  bool unsafe = false;
  std::vector<std::shared_ptr<PlasmaPayload>> plasma_objects;
  std::string message_out;

  TRY_READ_REQUEST(ReadGetBuffersByPlasmaRequest, root, plasma_ids, unsafe);
  RESPONSE_ON_ERROR(
      plasma_bulk_store_->GetUnsafe(plasma_ids, unsafe, plasma_objects));
  RESPONSE_ON_ERROR(plasma_bulk_store_->AddDependency(
      std::unordered_set<PlasmaID>(plasma_ids.begin(), plasma_ids.end()),
      getConnId()));
  WriteGetBuffersByPlasmaReply(plasma_objects, message_out);

  /* NOTE: Here we send the file descriptor after the objects.
   *       We are using sendmsg to send the file descriptor
   *       which is a sync method. In theory, this might cause
   *       the server to block, but currently this seems to be
   *       the only method that are widely used in practice, e.g.,
   *       boost and Plasma, and actually the file descriptor is
   *       a very short message.
   *
   *       We will examine other methods later, such as using
   *       explicit file descriptors.
   */
  this->doWrite(message_out, [self, plasma_objects](const Status& status) {
    for (auto object : plasma_objects) {
      int store_fd = object->store_fd;
      int data_size = object->data_size;
      if (data_size > 0 &&
          self->used_fds_.find(store_fd) == self->used_fds_.end()) {
        self->used_fds_.emplace(store_fd);
        self->sendFd(store_fd);
      }
    }
    return Status::OK();
  });
  return false;
}

bool SocketConnection::doSealPlasmaBlob(json const& root) {
  auto self(shared_from_this());
  PlasmaID id;
  TRY_READ_REQUEST(ReadPlasmaSealRequest, root, id);
  RESPONSE_ON_ERROR(plasma_bulk_store_->Seal(id));
  RESPONSE_ON_ERROR(plasma_bulk_store_->AddDependency(id, getConnId()));
  std::string message_out;
  WriteSealReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doPlasmaRelease(json const& root) {
  auto self(shared_from_this());
  PlasmaID id;
  TRY_READ_REQUEST(ReadPlasmaReleaseRequest, root, id);
  RESPONSE_ON_ERROR(plasma_bulk_store_->Release(id, getConnId()));
  std::string message_out;
  WritePlasmaReleaseReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doPlasmaDelData(json const& root) {
  auto self(shared_from_this());
  PlasmaID id;
  TRY_READ_REQUEST(ReadPlasmaDelDataRequest, root, id);

  /// Plasma Data are not composable, so we do not have to wrestle with meta.
  RESPONSE_ON_ERROR(plasma_bulk_store_->OnDelete(id));

  std::string message_out;
  WritePlasmaDelDataReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doCreateData(const json& root) {
  auto self(shared_from_this());
  json tree;
  double startTime = GetCurrentTime();
  TRY_READ_REQUEST(ReadCreateDataRequest, root, tree);
  RESPONSE_ON_ERROR(server_ptr_->CreateData(
      tree, [tree, self, startTime](const Status& status, const ObjectID id,
                                    const Signature signature,
                                    const InstanceID instance_id) {
        std::string message_out;
        if (status.ok()) {
          WriteCreateDataReply(id, signature, instance_id, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        double endTime = GetCurrentTime();
        LOG_SUMMARY("data_request_duration_microseconds", "create",
                    (endTime - startTime) * 1000000);
        LOG_COUNTER("data_requests_total", "create");
        LOG_SUMMARY("object",
                    std::to_string(instance_id) + " " +
                        tree.value("typename", json(nullptr)).dump(),
                    1);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doCreateUserBuffers(json const& root) {
  auto self(shared_from_this());
  size_t offsets_num = 0, sizes_num = 0;
  TRY_READ_REQUEST(ReadCreateUserBuffersRequest, root, offsets_num, sizes_num);

  boost::asio::post(
      server_ptr_->GetIOContext(), [self, offsets_num, sizes_num]() {
        std::vector<uint64_t> offsets(offsets_num);
        std::vector<size_t> sizes(sizes_num);
        uint64_t start = 0, end = 0;
        start = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();

        VINEYARD_DISCARD(
            self->socket_server_ptr_->LseekExtraMsgReadPos(0, self->conn_id_));
        ClientAttributes attr;
        RESPONSE_ON_ERROR(self->socket_server_ptr_->GetClientAttributeMsg(
            self->conn_id_, attr));
        RESPONSE_ON_ERROR(self->socket_server_ptr_->ReadExtraMessage(
            offsets.data(), offsets.size() * sizeof(size_t), self->conn_id_));
        RESPONSE_ON_ERROR(self->socket_server_ptr_->ReadExtraMessage(
            sizes.data(), sizes.size() * sizeof(size_t), self->conn_id_));
        end = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ", doCreateUserBuffers read extra message time consumed = "
            << (end - start) << " us";

        start = end;
        std::vector<ObjectID> object_ids;
        object_ids.reserve(offsets.size());
        for (size_t i = 0; i < offsets.size(); i++) {
          ObjectID object_id;
          std::shared_ptr<Payload> object;
          Status status = self->server_ptr_->GetBulkStore()->CreateUserBlob(
              offsets[i], sizes[i], object_id, object);
          if (!status.ok()) {
            for (auto const& object : object_ids) {
              self->server_ptr_->GetBulkStore()->Delete(object);
            }
            RESPONSE_ON_ERROR(status);
          }
          object_ids.emplace_back(object_id);
        }
        end = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name << " create " << object_ids.size()
            << " user buffers, time consumed = " << (end - start) << " us";

        start = end;
        std::string message_out;
        WriteCreateUserBuffersReply(object_ids, message_out);
        VINEYARD_DISCARD(
            self->socket_server_ptr_->LseekExtraMsgWritePos(0, self->conn_id_));
        RESPONSE_ON_ERROR(self->socket_server_ptr_->WriteExtraMessage(
            object_ids.data(), object_ids.size() * sizeof(ObjectID),
            self->conn_id_));
        end = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ", doCreateUserBuffers write extra message time consumed = "
            << (end - start) << " us";
        self->doWrite(message_out);
        return false;
      });
  return false;
}

bool SocketConnection::doGetUserBuffers(json const& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> object_ids;
  std::vector<std::shared_ptr<Payload>> objects;
  std::string message_out;

  TRY_READ_REQUEST(ReadGetUserBuffersRequest, root, object_ids);
  for (auto const& object_id : object_ids) {
    VLOG(2) << "GetUserBuffers: object_id = " << ObjectIDToString(object_id);
  }
  VLOG(2) << "GetUserBuffers: object_ids.size() = " << object_ids.size();
  RESPONSE_ON_ERROR(bulk_store_->GetUnsafe(object_ids, true, objects));
  WriteGetUserBuffersReply(objects, message_out);
  self->doWrite(message_out);
  return false;
}

bool SocketConnection::doDeleteUserBuffers(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> object_ids;
  TRY_READ_REQUEST(ReadDeleteUserBuffersRequest, root, object_ids);
  VINEYARD_DISCARD(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  VLOG(2) << "DeleteUserBuffers: object_ids.size() = " << object_ids.size();
  uint64_t start = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
      object_ids.data(), object_ids.size() * sizeof(ObjectID), this->conn_id_));
  boost::asio::post(self->server_ptr_->GetIOContext(), [self, object_ids,
                                                        start]() {
    std::string message_out;
    std::vector<ObjectID> transmissions, non_transmissions;
    std::vector<std::shared_ptr<Payload>> to_delete_user_buffers;
    self->server_ptr_->GetBulkStore()->GetUnsafe(object_ids, true,
                                                 to_delete_user_buffers);
    for (auto const& object_id : object_ids) {
      // make the user blob invisible.
      VINEYARD_DISCARD(
          self->server_ptr_->GetBulkStore()->DeleteUserBlob(object_id));
    }
    do {
      transmissions.clear();
      non_transmissions.clear();
      std::unique_lock<std::mutex> lock =
          self->server_ptr_->FindTransmissionObjects(object_ids, transmissions,
                                                     non_transmissions);
      if (!transmissions.empty()) {
        // sleep
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        self->server_ptr_->RemoveFromMigrationList(non_transmissions);
        WriteDeleteUserBuffersReply(message_out);
        self->doWrite(message_out);
        uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        VLOG(self->trace_log_level_)
            << "delete " << object_ids.size()
            << " user buffers, time consumed = " << (end - start) << " us";
      }
    } while (!transmissions.empty());
  });
  return false;
}

bool SocketConnection::doCreateDatas(const json& root) {
  auto self(shared_from_this());
  std::vector<json> tree;
  double startTime = GetCurrentTime();
  TRY_READ_REQUEST(ReadCreateDatasRequest, root, tree);
  RESPONSE_ON_ERROR(server_ptr_->CreateData(
      tree, [tree, self, startTime](
                const Status& status, const std::vector<ObjectID> ids,
                const std::vector<Signature> signatures,
                const std::vector<InstanceID> instance_ids) {
        std::string message_out;
        if (status.ok()) {
          WriteCreateDatasReply(ids, signatures, instance_ids, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        double endTime = GetCurrentTime();
        LOG_SUMMARY("data_request_duration_microseconds", "create",
                    (endTime - startTime) * 1000000);
        LOG_COUNTER("data_requests_total", "create");
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doCreatehugeDatas(json const& root) {
  auto self(shared_from_this());
  uint64_t start = 0, end = 0;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  std::vector<json> tree;
  size_t total_json;
  std::vector<size_t> json_lengths;
  TRY_READ_REQUEST(ReadCreateHugeDatasRequest, root, total_json);
  json_lengths.resize(total_json);
  VINEYARD_DISCARD(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
      json_lengths.data(), total_json * sizeof(size_t), this->getConnId()));
  tree.resize(total_json);
  for (size_t i = 0; i < total_json; i++) {
    std::string json_str;
    json_str.resize(json_lengths[i]);
    RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
        &json_str[0], json_lengths[i], this->getConnId()));
    tree[i] = json_from_buf(json_str.c_str(), json_str.size());
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name
      << ". Read doCreatehugeDatas IPC msg cost:" << (end - start) << " us.";

  start = end;
  RESPONSE_ON_ERROR(server_ptr_->CreateData(
      tree,
      [self, start, attr](const Status& status, const std::vector<ObjectID> ids,
                          const std::vector<Signature> signatures,
                          const std::vector<InstanceID> instance_ids) {
        uint64_t end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ". CreateData cost:" << (end_ - start) << " us.";

        uint64_t start_ = end_;
        std::string message_out;
        if (status.ok()) {
          VINEYARD_DISCARD(self->socket_server_ptr_->LseekExtraMsgWritePos(
              0, self->conn_id_));
          Status status_ = self->socket_server_ptr_->WriteExtraMessage(
              ids.data(), ids.size() * sizeof(ObjectID), self->getConnId());
          if (status_.ok()) {
            WriteCreateHugeDatasReply(ids.size(), signatures[0],
                                      instance_ids[0], message_out);
          } else {
            LOG(ERROR) << "Error: " << status_.ToString()
                       << ". Request: " << attr.req_name;
            WriteErrorReply(status_, message_out);
          }
        } else {
          LOG(ERROR) << "Error: " << status.ToString()
                     << ". Request: " << attr.req_name;
          WriteErrorReply(status, message_out);
        }
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ". Write doCreatehugeDatas IPC reply cost:" << (end_ - start_)
            << " us.";
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doGetData(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool sync_remote = false, wait = false;
  double startTime = GetCurrentTime();
  TRY_READ_REQUEST(ReadGetDataRequest, root, ids, sync_remote, wait);
  json tree;
  RESPONSE_ON_ERROR(server_ptr_->GetData(
      ids, sync_remote, wait, [self]() { return self->running_.load(); },
      [self, startTime](const Status& status, const json& tree) {
        std::string message_out;
        if (status.ok()) {
          WriteGetDataReply(tree, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        double endTime = GetCurrentTime();
        LOG_SUMMARY("data_request_duration_microseconds", "get",
                    (endTime - startTime) * 1000000);
        LOG_COUNTER("data_requests_total", "get");
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doGetHugeData(json const& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  size_t ids_num;
  bool sync_remote = false, wait = false;
  uint64_t start = 0, end = 0;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  TRY_READ_REQUEST(ReadGetHugeDataRequest, root, ids_num, sync_remote, wait);
  ids.resize(ids_num);
  json tree;
  VINEYARD_DISCARD(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
      ids.data(), ids_num * sizeof(ObjectID), this->getConnId()));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name
      << ". Read doGetHugeData IPC msg cost:" << (end - start) << " us.";

  start = end;
  RESPONSE_ON_ERROR(server_ptr_->GetData(
      ids, sync_remote, wait, [self]() { return self->running_.load(); },
      [self, start, attr](const Status& status, const json& tree) {
        uint64_t start_ = start, end_ = 0;
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ". GetData cost:" << (end_ - start_) << " us.";
        start_ = end_;
        std::string message_out;
        if (status.ok()) {
          std::string json_str = json_to_string(tree);
          VINEYARD_DISCARD(self->socket_server_ptr_->LseekExtraMsgWritePos(
              0, self->conn_id_));
          Status status_ = self->socket_server_ptr_->WriteExtraMessage(
              json_str.data(), json_str.size(), self->getConnId());
          if (status_.ok()) {
            WriteGetHugeDataReply(json_str.size(), message_out);
          } else {
            LOG(ERROR) << "Error: " << status_.ToString()
                       << ". Request: " << attr.req_name;
            WriteErrorReply(status_, message_out);
          }
        } else {
          LOG(ERROR) << "Error: " << status.ToString()
                     << ". Request: " << attr.req_name;
          WriteErrorReply(status, message_out);
        }
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ". Write doGetHugeData IPC reply cost:" << (end_ - start_)
            << " us.";

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
  TRY_READ_REQUEST(ReadListDataRequest, root, pattern, regex, limit);
  RESPONSE_ON_ERROR(server_ptr_->ListData(
      pattern, regex, limit, [self](const Status& status, const json& tree) {
        std::string message_out;
        if (status.ok()) {
          WriteGetDataReply(tree, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
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
  bool force, deep, memory_trim, fastpath;
  double startTime = GetCurrentTime();
  TRY_READ_REQUEST(ReadDelDataRequest, root, ids, force, deep, memory_trim,
                   fastpath);
  RESPONSE_ON_ERROR(server_ptr_->DelData(
      ids, force, deep, memory_trim, fastpath,
      [self, startTime](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WriteDelDataReply(message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        double endTime = GetCurrentTime();
        LOG_SUMMARY("data_request_duration_microseconds", "delete",
                    (endTime - startTime) * 1000000);
        LOG_COUNTER("data_requests_total", "delete");
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doExists(const json& root) {
  auto self(shared_from_this());
  ObjectID id;
  TRY_READ_REQUEST(ReadExistsRequest, root, id);
  RESPONSE_ON_ERROR(
      server_ptr_->Exists(id, [self](const Status& status, bool const exists) {
        std::string message_out;
        if (status.ok()) {
          WriteExistsReply(exists, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
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
  TRY_READ_REQUEST(ReadPersistRequest, root, id);
  RESPONSE_ON_ERROR(
      server_ptr_->Persist(id, [self, root](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WritePersistReply(message_out);
          self->doWrite(message_out);
        } else if (status.IsEtcdError()) {
          // retry on etcd error: reprocess the message
          VLOG(100) << "Warning: "
                    << "Retry persist on etcd error: " << status.ToString();
          self->server_ptr_->GetIOContext().post(
              [self, root]() { self->doPersist(root); });
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
          self->doWrite(message_out);
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doBatchPersist(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  TRY_READ_REQUEST(ReadBatchPersistRequest, root, ids);

  RESPONSE_ON_ERROR(
      self->server_ptr_->Persist(ids, [self, root](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WriteBatchPersistReply(message_out);
          self->doWrite(message_out);
        } else if (status.IsEtcdError()) {
          // retry on etcd error: reprocess the message
          VLOG(100) << "Warning: "
                    << "Retry persist on etcd error: " << status.ToString();
          self->server_ptr_->GetIOContext().post(
              [self, root]() { self->doPersist(root); });
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
          self->doWrite(message_out);
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doIfPersist(const json& root) {
  auto self(shared_from_this());
  ObjectID id;
  TRY_READ_REQUEST(ReadIfPersistRequest, root, id);
  RESPONSE_ON_ERROR(server_ptr_->IfPersist(
      id, [self](const Status& status, bool const persist) {
        std::string message_out;
        if (status.ok()) {
          WriteIfPersistReply(persist, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doLabelObject(const json& root) {
  auto self(shared_from_this());
  ObjectID object_id = InvalidObjectID();
  std::vector<std::string> keys;
  std::vector<std::string> values;
  std::string message_out;

  TRY_READ_REQUEST(ReadLabelRequest, root, object_id, keys, values);
  RESPONSE_ON_ERROR(server_ptr_->LabelObjects(
      object_id, keys, values, [self](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WriteLabelReply(message_out);
        } else {
          VLOG(100) << "Error: " << status;
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doClear(const json& root) {
  auto self(shared_from_this());
  TRY_READ_REQUEST(ReadClearRequest, root);
  // clear:
  //    step 1: list
  //    step 2: compute delete set
  //    step 3: do delete
  RESPONSE_ON_ERROR(server_ptr_->ListAllData(
      [self](const Status& status, const std::vector<ObjectID>& objects) {
        if (status.ok()) {
          auto s = self->server_ptr_->DelData(
              objects, true, true, true, false, [self](const Status& status) {
                std::string message_out;
                if (status.ok()) {
                  WriteClearReply(message_out);
                } else {
                  VLOG(100) << "Error: " << status;
                  WriteErrorReply(status, message_out);
                }
                self->doWrite(message_out);
                return Status::OK();
              });
          if (!s.ok()) {
            std::string message_out;
            VLOG(100) << "Error: " << s;
            WriteErrorReply(s, message_out);
            self->doWrite(message_out);
          }
        } else {
          std::string message_out;
          VLOG(100) << "Error: " << status;
          WriteErrorReply(status, message_out);
          self->doWrite(message_out);
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doMemoryTrim(const json& root) {
  auto self(shared_from_this());
  std::string message_out;

  TRY_READ_REQUEST(ReadMemoryTrimRequest, root);
  // deprecated, use `DelData(memory_trim=true)` instead.
  bool trimmed = false;
  WriteMemoryTrimReply(trimmed, message_out);
  self->doWrite(message_out);
  return false;
}

bool SocketConnection::doCreateStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  bool fixed_size = false;
  int nums = 0;
  size_t size = 0;
  TRY_READ_REQUEST(ReadCreateStreamRequest, root, stream_id);
  auto status =
      server_ptr_->GetStreamStore()->Create(stream_id, fixed_size, nums, size);
  std::string message_out;
  if (status.ok()) {
    WriteCreateStreamReply(message_out);
  } else {
    VLOG(100) << "Error: " << status.ToString();
    WriteErrorReply(status, message_out);
  }
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doCreateFixedStream(json const& root) {
  auto self(shared_from_this());
  std::string stream_name;
  int nums = 0;
  size_t size = 0;
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadCreateFixedStreamRequest, root, stream_name, nums, size);
  VLOG(2) << "Create fixed stream: " << stream_name << ", nums: " << nums
          << ", size: " << size;
  stream_id = GenerateObjectID(server_ptr_->instance_id());
  if (stream_id == InvalidObjectID()) {
    LOG(ERROR) << "Error: Failed to generate stream id";
    std::string message_out;
    WriteErrorReply(Status::Invalid("Failed to generate stream id"),
                    message_out);
    this->doWrite(message_out);
    return false;
  }

  Status status =
      server_ptr_->GetStreamStore()->Create(stream_id, true, nums, size);
  if (!status.ok()) {
    std::string message_out;
    LOG(ERROR) << "Error: " << status.ToString();
    WriteErrorReply(status, message_out);
    this->doWrite(message_out);
    return false;
  }

  if (!stream_name.empty()) {
    status = server_ptr_->GetStreamStore()->PutName(stream_name, stream_id);
  }
  std::string message_out;
  if (status.ok()) {
    WriteCreateFixedStreamReply(message_out, stream_id);
  } else {
    LOG(ERROR) << "Error: " << status.ToString();
    if (!server_ptr_->GetStreamStore()->Delete(stream_id).ok()) {
      LOG(ERROR) << "Failed to cleanup stream: " << stream_id;
    }
    WriteErrorReply(status, message_out);
  }
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doOpenStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  int64_t mode;
  std::string stream_name;
  bool wait = false;
  uint64_t timeout = 0;
  TRY_READ_REQUEST(ReadOpenStreamRequest, root, stream_id, stream_name, mode,
                   wait, timeout);
  VLOG(2) << "Open stream: " << stream_id << ", name: " << stream_name
          << ", mode: " << mode << ", wait: " << wait
          << ", timeout: " << timeout;
  std::string owner = StreamStore::BuildOwner(self->peer_host, self->conn_id_);
  if (stream_id != InvalidObjectID()) {
    RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Open(
        stream_id, mode, owner, wait, timeout,
        [self, stream_name, stream_id](Status& status, ObjectID id) {
          VLOG(2) << "doOpenStream callback, stream_id:" << stream_id
                  << " stream_name:" << stream_name
                  << " status:" << status.ToString();
          std::string message_out;
          if (status.ok()) {
            WriteOpenStreamReply(message_out, id);
          } else {
            LOG(ERROR) << "Error: " << status.ToString();
            WriteErrorReply(status, message_out);
          }
          self->doWrite(message_out);
        }));
  } else {
    RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Open(
        stream_name, mode, owner, wait, timeout,
        [self, stream_name, stream_id](Status& status, ObjectID id) {
          VLOG(2) << "doOpenStream callback, stream_id:" << stream_id
                  << " stream_name:" << stream_name
                  << " status:" << status.ToString();
          std::string message_out;
          if (status.ok()) {
            WriteOpenStreamReply(message_out, id);
          } else {
            LOG(ERROR) << "Error: " << status.ToString();
            WriteErrorReply(status, message_out);
          }
          self->doWrite(message_out);
        }));
  }

  return false;
}

bool SocketConnection::doGetNextStreamChunk(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  size_t size;
  TRY_READ_REQUEST(ReadGetNextStreamChunkRequest, root, stream_id, size);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Get(
      stream_id, size, [self](const Status& status, const ObjectID chunk) {
        std::string message_out;
        if (status.ok()) {
          std::shared_ptr<Payload> object;
          RETURN_ON_ERROR(self->bulk_store_->GetUnsafe(chunk, true, object));
          int store_fd = object->store_fd, fd_to_send = -1;
          int data_size = object->data_size;
          if (data_size > 0 &&
              self->used_fds_.find(store_fd) == self->used_fds_.end()) {
            self->used_fds_.emplace(store_fd);
            fd_to_send = store_fd;
          }

          WriteGetNextStreamChunkReply(object, fd_to_send, message_out);
          self->doWrite(message_out, [self, fd_to_send](const Status& status) {
            if (fd_to_send != -1) {
              self->sendFd(fd_to_send);
            }
            return Status::OK();
          });
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
          self->doWrite(message_out);
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doPushNextStreamChunk(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id, chunk;
  TRY_READ_REQUEST(ReadPushNextStreamChunkRequest, root, stream_id, chunk);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Push(
      stream_id, chunk, [self](const Status& status, const ObjectID) {
        std::string message_out;
        if (status.ok()) {
          WritePushNextStreamChunkReply(message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doPushNextStreamChunkByOffset(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  size_t offset;
  TRY_READ_REQUEST(ReadPushNextStreamChunkByOffsetRequest, root, stream_id,
                   offset);
  ObjectID blob_id;
  size_t size = 0;
  int blob_nums;
  std::shared_ptr<Payload> blob_payload;
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->GetFixedStreamSizeInfo(
      stream_id, size, blob_nums));
  RESPONSE_ON_ERROR(server_ptr_->GetBulkStore()->CreateUserBlob(
      offset, size, blob_id, blob_payload));
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Push(
      stream_id, blob_id, [self](const Status& status, const ObjectID) {
        std::string message_out;
        if (status.ok()) {
          WritePushNextStreamChunkByOffsetReply(message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doPullNextStreamChunk(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadPullNextStreamChunkRequest, root, stream_id);
  this->associated_streams_.emplace(stream_id);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Pull(
      stream_id, [self](const Status& status, const ObjectID chunk) {
        std::string message_out;
        if (status.ok()) {
          WritePullNextStreamChunkReply(chunk, message_out);
        } else {
          if (!status.IsStreamDrained()) {
            VLOG(100) << "Error: " << status.ToString();
          }
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doCheckFixedStreamReceived(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id = InvalidObjectID();
  int index = -1;
  TRY_READ_REQUEST(ReadCheckFixedStreamReceivedRequest, root, stream_id, index);
  bool finished = false;
  RESPONSE_ON_ERROR(self->server_ptr_->GetStreamStore()->CheckBlobReceived(
      stream_id, index, finished));
  std::string message_out;
  WriteCheckFixedStreamReceivedReply(finished, message_out);
  self->doWrite(message_out);

  return false;
}

bool SocketConnection::doStopStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  bool failed;
  TRY_READ_REQUEST(ReadStopStreamRequest, root, stream_id, failed);
  // NB: don't erase the metadata from meta_service, since there's may
  // reader listen on this stream.
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Stop(stream_id, failed));
  std::string message_out;
  WriteStopStreamReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doDropStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadDropStreamRequest, root, stream_id);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Drop(stream_id));
  std::string message_out;
  WriteDropStreamReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doAbortStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadAbortStreamRequest, root, stream_id);
  /*
   * Currently, abort only occurs before the stream is activated or after the
   * stream is transferred. So we don't need to wait the transfer to be done.
   */
  bool success = false;
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Abort(stream_id, success));
  std::string message_out;
  WriteAbortStreamReply(message_out, success);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doPutStreamName(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  std::string name;
  TRY_READ_REQUEST(ReadPutStreamNameRequest, root, stream_id, name);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->PutName(name, stream_id));
  std::string message_out;
  WritePutStreamNameReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doGetStreamIDByName(const json& root) {
  auto self(shared_from_this());
  std::string name;
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadGetStreamIDByNameRequest, root, name);
  RESPONSE_ON_ERROR(
      server_ptr_->GetStreamStore()->GetStreamIDByName(name, stream_id));

  std::string message_out;
  WriteGetStreamIDByNameReply(stream_id, message_out);
  this->doWrite(message_out);

  return false;
}

bool SocketConnection::doActivateRemoteFixedStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id = InvalidObjectID();
  std::vector<std::vector<uint64_t>> recv_addr_list;
  std::vector<std::vector<uint64_t>> rkeys_list;
  std::vector<std::vector<size_t>> sizes_list;
  std::string advice_device;
  int port;
  TRY_READ_REQUEST(ReadActivateRemoteFixedStreamRequest, root, stream_id,
                   recv_addr_list, rkeys_list, sizes_list, advice_device, port);
  VLOG(100) << "remote device:" << advice_device << " with port:" << port;

  std::string message_out;
  WriteActivateRemoteFixedStreamReply(message_out);
  this->doWrite(message_out);

  boost::asio::post(server_ptr_->GetIOContext(), [self, recv_addr_list,
                                                  rkeys_list, sizes_list,
                                                  advice_device, port,
                                                  stream_id]() {
    self->server_ptr_->GetStreamStore()->ActivateRemoteFixedStream(
        stream_id, recv_addr_list, rkeys_list, sizes_list,
        [self, advice_device, port, stream_id](
            const Status& status_, ObjectID chunk,
            std::vector<uint64_t> addr_list, std::vector<uint64_t> rkey_list,
            std::vector<size_t> size_list, int index) {
          std::string message_out;
          Status status = status_;
          if (!status.ok()) {
            VLOG(100) << "Error: " << status.ToString();
            self->server_ptr_->GetStreamStore()->SetErrorFlag(stream_id,
                                                              status);
            WriteErrorReply(status, message_out);
            self->doWriteWithoutRead(message_out);
            return status;
          }

          // RDMA write
          uint64_t local_addr;
          size_t size;
          std::shared_ptr<Payload> object;
          status = self->bulk_store_->GetUnsafe(chunk, true, object);
          if (!status.ok()) {
            VLOG(100) << "Error: failed to get object";
            self->server_ptr_->GetStreamStore()->SetErrorFlag(stream_id,
                                                              status);
            WriteErrorReply(Status::KeyError("Failed to get object"),
                            message_out);
            self->doWriteWithoutRead(message_out);
            return status;
          }
          self->server_ptr_->GetBulkStore()->DeleteUserBlob(chunk);
          local_addr = reinterpret_cast<uint64_t>(object->pointer);
          size = object->data_size;
          void* base_addr = nullptr;
          status = self->server_ptr_->GetBulkStoreMmapAddr(base_addr);
          if (!status.ok()) {
            VLOG(100) << "Error: failed to get bulk store mmap addr";
            self->server_ptr_->GetStreamStore()->SetErrorFlag(stream_id,
                                                              status);
            WriteErrorReply(
                Status::KeyError("Failed to get bulk store mmap addr"),
                message_out);
            self->doWriteWithoutRead(message_out);
            return status;
          }
          uint64_t offset = local_addr - reinterpret_cast<uint64_t>(base_addr);
          status = self->socket_server_ptr_->SendDataWithRDMA(
              addr_list, local_addr, offset, size_list, size, rkey_list,
              self->peer_host, port, advice_device);
          if (!status.ok()) {
            VLOG(100) << "Error: failed to send data with RDMA";
            self->server_ptr_->GetStreamStore()->SetErrorFlag(stream_id,
                                                              status);
            WriteErrorReply(Status::IOError("Failed to send data with RDMA"),
                            message_out);
            self->doWriteWithoutRead(message_out);
            return status;
          }

          VLOG(100) << "SendDataWithRDMA success, index:" << index;
          self->server_ptr_->GetStreamStore()->SetBlobReceived(stream_id,
                                                               index);
          WriteStreamReadyAckReply(message_out, index);
          bool finished = false;
          status = self->server_ptr_->GetStreamStore()
                       ->IsFixedStreamTransferFinished(stream_id, finished);
          if (!status.ok()) {
            VLOG(100)
                << "Error: failed to check fixed stream transfer finished";
            self->server_ptr_->GetStreamStore()->SetErrorFlag(stream_id,
                                                              status);
            WriteErrorReply(status, message_out);
            self->doWriteWithoutRead(message_out);
            return status;
          }

          self->doWriteWithoutRead(message_out);
          return Status::OK();
        });
  });

  return false;
}

bool SocketConnection::doOpenFixedStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  int64_t mode;
  TRY_READ_REQUEST(ReadOpenFixedStreamRequest, root, stream_id, mode);
  std::string owner = StreamStore::BuildOwner(self->peer_host, self->conn_id_);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Open(
      stream_id, mode, owner, false, 0, [&](Status& status, ObjectID id) {
        std::string message_out;
        VLOG(100) << "Open stream return!";
        int fd = -1;
        if (status.ok()) {
          if (!server_ptr_->GetStreamStore()->GetRecvFd(stream_id, fd).ok()) {
            WriteErrorReply(Status::KeyError("Failed to get recv fd"),
                            message_out);
            VINEYARD_DISCARD(
                server_ptr_->GetStreamStore()->Close(stream_id, owner));
            this->doWrite(message_out);
          } else {
            WriteOpenFixedStreamReply(message_out);
            this->doWrite(message_out);
            self->sendFd(fd);
          }
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
          this->doWrite(message_out);
        }
      }));

  return false;
}

bool SocketConnection::doCloseStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadCloseStreamRequest, root, stream_id);
  std::string owner = StreamStore::BuildOwner(self->peer_host, self->conn_id_);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Close(stream_id, owner));
  std::string message_out;
  WriteCloseStreamReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doDeleteStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadDeleteStreamRequest, root, stream_id);
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Delete(stream_id));
  std::string message_out;
  WriteDeleteStreamReply(message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doVineyardOpenRemoteFixedStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  std::string stream_name;
  ObjectID local_id;
  std::string endpoint;
  uint64_t mode;
  size_t size;
  int blob_num;
  bool wait = false;
  uint64_t timeout = 0;
  TRY_READ_REQUEST(ReadVineyardOpenRemoteFixedStreamRequest, root, stream_id,
                   stream_name, local_id, blob_num, size, endpoint, mode, wait,
                   timeout);
  VLOG(2) << "Stream ID: " << ObjectIDToString(stream_id)
          << ", Stream Name: " << stream_name
          << ", Local ID: " << ObjectIDToString(local_id)
          << ", Blob Num: " << blob_num << ", Size: " << size
          << ", Endpoint: " << endpoint << ", Mode: " << mode
          << ", Wait: " << wait << ", Timeout: " << timeout;
  std::string owner = StreamStore::BuildOwner(self->peer_host, self->conn_id_);
  RESPONSE_ON_ERROR(server_ptr_->VineyardOpenRemoteFixedStream(
      stream_id, stream_name, local_id, blob_num, size, endpoint, mode, owner,
      wait, timeout, [self, local_id](const Status& status) {
        VLOG(2) << "VineyardOpenRemoteFixedStream done callback, local_id:"
                << ObjectIDToString(local_id)
                << ", status: " << status.ToString();
        std::string message_out;
        if (status.ok()) {
          int fd = -1;
          if (!self->server_ptr_->GetStreamStore()
                   ->GetRecvFd(local_id, fd)
                   .ok()) {
            WriteErrorReply(Status::KeyError("Failed to get recv fd"),
                            message_out);
            VINEYARD_DISCARD(
                self->server_ptr_->GetStreamStore()->Close(local_id, ""));
            self->doWrite(message_out);
          } else {
            WriteVineyardOpenRemoteFixedStreamReply(message_out, local_id);
            self->doWrite(message_out);
            self->sendFd(fd);
          }
        } else {
          LOG(ERROR) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
          self->doWrite(message_out);
        }
        return Status::OK();
      }));

  return false;
}

bool SocketConnection::doVineyardActivateRemoteFixedStreamWithOffset(
    const json& root) {
  auto self(shared_from_this());
  std::vector<uint64_t> offset_list;
  ObjectID stream_id = InvalidObjectID();
  TRY_READ_REQUEST(ReadVineyardActivateRemoteFixedStreamWithOffsetRequest, root,
                   stream_id, offset_list);
  VLOG(2) << "Vineyard activate remote fixed stream with offset, local id: "
          << ObjectIDToString(stream_id);

  boost::asio::post(server_ptr_->GetIOContext(), [self, stream_id,
                                                  offset_list]() {
    VLOG(2) << "doVineyardActivateRemoteFixedStreamWithOffset post task";
    std::vector<ObjectID> blob_list;
    size_t size = 0;
    int blob_num = 0;
    RESPONSE_ON_ERROR(
        self->server_ptr_->GetStreamStore()->GetFixedStreamSizeInfo(
            stream_id, size, blob_num));
    for (uint64_t offset : offset_list) {
      ObjectID blob_id = InvalidObjectID();
      std::shared_ptr<Payload> blob_payload;
      Status status = self->server_ptr_->GetBulkStore()->CreateUserBlob(
          offset, size, blob_id, blob_payload);
      // currently, create user blob will never fail
      if (!status.ok()) {
        for (auto const& id : blob_list) {
          self->server_ptr_->GetBulkStore()->DeleteUserBlob(id);
        }
        RESPONSE_ON_ERROR(status);
      }
      blob_list.push_back(blob_id);
    }

    RESPONSE_ON_ERROR(self->server_ptr_->VineyardActivateRemoteFixedStream(
        stream_id, self->conn_id_, false, blob_list,
        [self, blob_list, stream_id](
            const Status& status,
            std::vector<std::shared_ptr<Payload>>& payload_list) {
          VLOG(2) << "VineyardActivateRemoteFixedStreamWithOffset done "
                  << "callback, local_id:" << ObjectIDToString(stream_id)
                  << ", status: " << status.ToString();
          for (auto blob_id : blob_list) {
            self->server_ptr_->GetBulkStore()->DeleteUserBlob(blob_id);
          }
          std::string message_out;
          if (status.ok()) {
            WriteVineyardActivateRemoteFixedStreamWithOffsetReply(message_out);
          } else {
            LOG(ERROR) << "Error: " << status.ToString();
            WriteErrorReply(status, message_out);
            // release the user blobs
          }
          self->doWrite(message_out);
        }));
    return false;
  });
  return false;
}

bool SocketConnection::doVineyardCloseRemoteFixedStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadVineyardCloseRemoteFixedStreamRequest, root, stream_id);
  VLOG(2) << "Vineyard close remote fixed stream, local id: "
          << ObjectIDToString(stream_id);
  RESPONSE_ON_ERROR(server_ptr_->VineyardCloseRemoteFixedStream(
      stream_id, [self, stream_id](const Status& status) {
        VLOG(2) << "VineyardCloseRemoteFixedStream done callback, local_id:"
                << ObjectIDToString(stream_id)
                << ", status: " << status.ToString();
        std::string message_out;
        if (status.ok() || status.IsObjectNotExists()) {
          // if the stream is not exists, it means that the remote node
          // is restarted. So we need to close the stream in local node.
          WriteVineyardCloseRemoteFixedStreamReply(message_out);
        } else {
          LOG(ERROR) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        std::string owner =
            StreamStore::BuildOwner(self->peer_host, self->conn_id_);
        self->server_ptr_->GetStreamStore()->UnbindRemoteStream(stream_id);
        self->server_ptr_->GetStreamStore()->Close(stream_id, owner);
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doVineyardGetMetasByNames(const json& root) {
  auto self(shared_from_this());
  uint64_t start = 0, end = 0;
  std::vector<std::string> names;
  std::string rpc_endpoint;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  TRY_READ_REQUEST(ReadVineyardGetMetasByNamesRequest, root, names,
                   rpc_endpoint);
  RESPONSE_ON_ERROR(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name
      << ". Read get metas by names request cost: " << (end - start) << " us.";

  start = end;
  RESPONSE_ON_ERROR(server_ptr_->VineyardGetMetasByNames(
      names, rpc_endpoint, attr,
      [self, attr, start](const Status& status,
                          const std::vector<json>& metas) {
        uint64_t start_ = start, end_ = 0;
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ", get metas size: " << metas.size()
            << ", get metas by names request cost: " << (end_ - start_)
            << " us.";

        VLOG(2) << "VineyardGetMetasByNames done callback"
                << ", request:" << attr.req_name
                << ", status: " << status.ToString()
                << ", request id: " << attr.req_name;
        start_ = end_;
        std::string message_out;
        if (status.ok()) {
          WriteVineyardGetMetasByNamesReply(metas, message_out);
        } else {
          LOG(ERROR) << "Error: " << status.ToString()
                     << ", request: " << attr.req_name;
          WriteErrorReply(status, message_out);
        }
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ", write reply cost: " << (end_ - start_) << " us.";
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doVineyardGetRemoteBlobs(const json& root) {
  auto self(shared_from_this());
  std::vector<std::vector<ObjectID>> local_ids;
  std::vector<std::vector<ObjectID>> remote_ids;
  std::string rpc_endpoint;

  uint64_t start = 0, end = 0;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  TRY_READ_REQUEST(ReadVineyardGetRemoteBlobsWithRDMARequest, root, local_ids,
                   remote_ids, rpc_endpoint);
  RESPONSE_ON_ERROR(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

  // VLOG(2) << "Vineyard get remote blobs, local ids size: " <<
  // local_ids.size()
  //         << ", remote ids size: " << remote_ids.size()
  //         << ", rpc endpoint: " << rpc_endpoint;
  // for (size_t i = 0; i < local_ids.size(); i++) {
  //   VLOG(3) << "layer " << i << " :";
  //   for (size_t j = 0; j < local_ids[i].size(); ++j) {
  //     VLOG(3) << "local id: " << ObjectIDToString(local_ids[i][j])
  //             << ", remote id: " << ObjectIDToString(remote_ids[i][j]);
  //   }
  // }
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name
      << ". Read get remote blobs request cost: " << (end - start) << " us.";
  start = end;
  RESPONSE_ON_ERROR(self->server_ptr_->VineyardGetRemoteBlobs(
      local_ids, remote_ids, rpc_endpoint, attr,
      [self, local_ids, start, attr](const Status& status, int fd) {
        uint64_t start_ = 0, end_ = 0;
        start_ = start;
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ", local_ids size: " << local_ids.size()
            << ". Start get remote blobs request cost: " << (end_ - start_)
            << " us.";
        VLOG(2) << "VineyardGetRemoteBlobs ready callback, status: "
                << status.ToString() << ", request: " << attr.req_name;

        std::string message_out;
        if (status.ok()) {
          WriteVineyardGetRemoteBlobsWithRDMAReply(message_out);
        } else {
          LOG(ERROR) << "Error: " << status.ToString()
                     << ", request: " << attr.req_name;
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        if (fd != -1) {
          VLOG(2) << "Send fd: " << fd;
          self->sendFd(fd);
        } else {
          VLOG(2) << "No fd to send";
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doVineyardGetRemoteBlobsWithOffset(const json& root) {
  auto self(shared_from_this());
  std::vector<std::vector<size_t>> local_offsets;
  std::vector<std::vector<ObjectID>> remote_ids;
  std::vector<std::vector<uint64_t>> sizes_vec;
  size_t batch_nums = 0, batch_size = 0;
  std::string rpc_endpoint;

  uint64_t start = 0, end = 0;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  TRY_READ_REQUEST(ReadVineyardGetRemoteBlobsWithOffsetRequest, root,
                   batch_nums, batch_size, rpc_endpoint);
  local_offsets.resize(batch_nums);
  remote_ids.resize(batch_nums);
  sizes_vec.resize(batch_nums);
  RESPONSE_ON_ERROR(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  for (size_t i = 0; i < batch_nums; ++i) {
    local_offsets[i].resize(batch_size);
    remote_ids[i].resize(batch_size);
    sizes_vec[i].resize(batch_size);
    RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
        reinterpret_cast<char*>(local_offsets[i].data()),
        sizeof(size_t) * batch_size, conn_id_));
    RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
        reinterpret_cast<char*>(remote_ids[i].data()),
        sizeof(ObjectID) * batch_size, conn_id_));
    RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
        reinterpret_cast<char*>(sizes_vec[i].data()),
        sizeof(uint64_t) * batch_size, conn_id_));
    for (size_t j = 0; j < batch_size; ++j) {
      local_offsets[i][j] =
          reinterpret_cast<size_t>(
              server_ptr_->GetBulkStore()->GetUserBasePointer()) +
          local_offsets[i][j];
    }
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name
      << ". Read get remote blobs request cost: " << (end - start) << " us.";
  start = end;
  RESPONSE_ON_ERROR(self->server_ptr_->VineyardGetRemoteBlobsWithOffset(
      local_offsets, remote_ids, sizes_vec, rpc_endpoint, attr,
      [self, batch_nums, batch_size, start, attr](const Status& status,
                                                  int fd) {
        uint64_t start_ = 0, end_ = 0;
        start_ = start;
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name
            << ", buffer nums: " << batch_nums * batch_size
            << ". Start get remote blobs request cost: " << (end_ - start_)
            << " us.";
        VLOG(2) << "VineyardGetRemoteBlobs ready callback, status: "
                << status.ToString();

        std::string message_out;
        if (status.ok()) {
          WriteVineyardGetRemoteBlobsWithOffsetReply(message_out);
        } else {
          LOG(ERROR) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        if (fd != -1) {
          VLOG(2) << "Send fd: " << fd << " for request: " << attr.req_name;
          self->sendFd(fd);
        } else {
          VLOG(2) << "No fd to send for request: " << attr.req_name;
        }
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doVineyardStopStream(const json& root) {
  // TBD
  return false;
}

bool SocketConnection::doVineyardDropStream(const json& root) {
  // TBD
  return false;
}

bool SocketConnection::doVineyardAbortRemoteStream(const json& root) {
  auto self(shared_from_this());
  ObjectID stream_id;
  TRY_READ_REQUEST(ReadVineyardAbortRemoteStreamRequest, root, stream_id);
  VLOG(2) << "Vineyard abort remote stream, local id: "
          << ObjectIDToString(stream_id);
  bool success;
  RESPONSE_ON_ERROR(server_ptr_->GetStreamStore()->Abort(stream_id, success));
  if (!success) {
    std::string message_out;
    WriteVineyardAbortRemoteStreamReply(message_out, success);
    this->doWrite(message_out);
    return false;
  }
  RESPONSE_ON_ERROR(server_ptr_->VineyardAbortRemoteStream(
      stream_id, [self, stream_id](const Status& status, bool success) {
        VLOG(2) << "VineyardAbortRemoteStream done callback, local_id:"
                << ObjectIDToString(stream_id)
                << ", status: " << status.ToString();
        std::string message_out;
        if (status.ok()) {
          WriteVineyardAbortRemoteStreamReply(message_out, success);
        } else {
          LOG(ERROR) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doPutName(const json& root) {
  auto self(shared_from_this());
  ObjectID object_id;
  std::string name;
  bool overwrite = true;
  TRY_READ_REQUEST(ReadPutNameRequest, root, object_id, name, overwrite);
  name = escape_json_pointer(name);
  RESPONSE_ON_ERROR(server_ptr_->PutName(
      object_id, name, overwrite, [self](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WritePutNameReply(message_out);
        } else {
          VLOG(100) << "Error: failed to put name: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doPutNames(const json& root) {
  auto self(shared_from_this());
  uint64_t start = 0, end = 0;
  std::vector<ObjectID> object_id_vec;
  std::vector<std::string> name_vec;
  bool overwrite = true;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  TRY_READ_REQUEST(ReadPutNamesRequest, root, object_id_vec, name_vec,
                   overwrite);
  VINEYARD_DISCARD(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  for (auto& name : name_vec) {
    name = escape_json_pointer(name);
  }

  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name << ". Read put names reqest "
      << "cost: " << (end - start) << " us.";
  start = end;
  RESPONSE_ON_ERROR(server_ptr_->PutNames(
      object_id_vec, name_vec, overwrite,
      [self, start, attr](const Status& status) {
        uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name << ". Process put names "
            << "request cost: " << (end - start) << " us.";
        std::string message_out;
        if (status.ok()) {
          WritePutNamesReply(message_out);
        } else {
          LOG(ERROR) << "Error: failed to put names: " << status.ToString()
                     << ", request: " << attr.req_name;
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
  TRY_READ_REQUEST(ReadGetNameRequest, root, name, wait);
  // n.b.: no need for escape for `get`, as the translation has been handled
  // by nlohmann/json when compare keys.
  //
  // name = escape_json_pointer(name);
  RESPONSE_ON_ERROR(server_ptr_->GetName(
      name, wait, [self]() { return self->running_.load(); },
      [self](const Status& status, const ObjectID& object_id) {
        std::string message_out;
        if (status.ok()) {
          WriteGetNameReply(object_id, message_out);
        } else {
          VLOG(100) << "Error: failed to get name: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doGetNames(const json& root) {
  auto self(shared_from_this());
  std::vector<std::string> name_vec;
  bool wait;
  uint64_t start = 0, end = 0;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  TRY_READ_REQUEST(ReadGetNamesRequest, root, name_vec, wait);
  // n.b.: no need for escape for `get`, as the translation has been handled
  // by nlohmann/json when compare keys.
  //
  RESPONSE_ON_ERROR(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name << ". Read get names request "
      << "cost: " << (end - start) << " us.";

  start = end;
  RESPONSE_ON_ERROR(server_ptr_->GetNames(
      name_vec, wait, [self]() { return self->running_.load(); },
      [self, start, attr](const Status& status,
                          const std::vector<ObjectID>& object_id_vec) {
        uint64_t start_ = start, end_ = 0;
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name << ". Process get names "
            << "request cost: " << (end_ - start_) << " us.";
        start_ = end_;
        std::string message_out;
        if (status.ok()) {
          WriteGetNamesReply(object_id_vec, message_out);
        } else {
          VLOG(100) << "Error: failed to get name: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        end_ = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name << ". Write get names "
            << "reply cost: " << (end_ - start_) << " us.";
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doGetObjectLocation(const json& root) {
  auto self(shared_from_this());
  std::vector<std::string> names;
  TRY_READ_REQUEST(ReadGetObjectLocationRequest, root, names);
  VINEYARD_DISCARD(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));

  boost::asio::post(server_ptr_->GetIOContext(), [self, names]() {
    self->server_ptr_->GetObjectLocation(
        names, [self](const Status& status,
                      std::vector<std::vector<std::string>>& locations_vector) {
          std::string message_out;
          if (status.ok()) {
            WriteGetObjectLocationReply(locations_vector, message_out);
          } else {
            VLOG(100) << "Error: " << status.ToString();
            WriteErrorReply(status, message_out);
          }
          self->doWrite(message_out);
          return Status::OK();
        });
  });
  return false;
}

bool SocketConnection::doPutObjectLocation(const json& root) {
  auto self(shared_from_this());
  std::vector<std::string> names;
  std::vector<std::string> locations;
  int ttl_seconds = 300;
  uint64_t start = 0, end = 0;

  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  TRY_READ_REQUEST(ReadPutObjectLocationRequest, root, names, locations,
                   ttl_seconds);
  RESPONSE_ON_ERROR(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name << ". Read put object location "
      << "request cost: " << (end - start) << " us.";
  start = end;
  boost::asio::post(server_ptr_->GetIOContext(), [self, names, locations,
                                                  ttl_seconds, start]() {
    self->server_ptr_->PutObjectLocation(
        names, locations, ttl_seconds, [self, start](Status status) {
          uint64_t end =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
          VLOG(self->trace_log_level_)
              << "Request: " << self->peer_host << ":" << self->conn_id_
              << ". Process put object location request cost: " << (end - start)
              << " us.";
          std::string message_out;
          if (status.ok()) {
            WritePutObjectLocationReply(message_out);
          } else {
            VLOG(2) << "Error: " << status.ToString();
            WriteErrorReply(status, message_out);
          }
          self->doWrite(message_out);
          return Status::OK();
        });
  });
  return false;
}

bool SocketConnection::doListName(const json& root) {
  auto self(shared_from_this());
  std::string pattern;
  bool regex;
  size_t limit;
  TRY_READ_REQUEST(ReadListNameRequest, root, pattern, regex, limit);
  RESPONSE_ON_ERROR(server_ptr_->ListName(
      pattern, regex, limit,
      [self](const Status& status,
             const std::map<std::string, ObjectID>& names) {
        std::map<std::string, ObjectID> unescaped_names;
        for (auto const& item : names) {
          std::string name = item.first;
          unescaped_names.emplace(unescape_json_pointer(name), item.second);
        }
        std::string message_out;
        if (status.ok()) {
          WriteListNameReply(names, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
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
  TRY_READ_REQUEST(ReadDropNameRequest, root, name);
  // n.b.: no need for escape for `drop` here, as the translation has been
  // handled by during composing` the `Del` op in vineyard_server.cc.
  //
  // name = escape_json_pointer(name);
  RESPONSE_ON_ERROR(server_ptr_->DropName(name, [self](const Status& status) {
    std::string message_out;
    if (status.ok()) {
      WriteDropNameReply(message_out);
    } else {
      VLOG(100) << "Error: failed to drop name: " << status.ToString();
      WriteErrorReply(status, message_out);
    }
    self->doWrite(message_out);
    return Status::OK();
  }));
  return false;
}

bool SocketConnection::doDropNames(const json& root) {
  auto self(shared_from_this());
  uint64_t start = 0, end = 0;
  start = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
  std::vector<std::string> name_vec;
  TRY_READ_REQUEST(ReadDropNamesRequest, root, name_vec);

  std::vector<size_t> name_lengths;
  name_lengths.resize(name_vec.size());
  VINEYARD_DISCARD(socket_server_ptr_->LseekExtraMsgReadPos(0, conn_id_));
  ClientAttributes attr;
  RESPONSE_ON_ERROR(socket_server_ptr_->GetClientAttributeMsg(conn_id_, attr));
  RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
      name_lengths.data(), name_lengths.size() * sizeof(size_t), conn_id_));
  for (size_t i = 0; i < name_vec.size(); ++i) {
    name_vec[i].resize(name_lengths[i]);
    RESPONSE_ON_ERROR(socket_server_ptr_->ReadExtraMessage(
        name_vec[i].data(), name_lengths[i], conn_id_));
  }
  end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  VLOG(self->trace_log_level_)
      << "Request: " << attr.req_name << ". Read drop names request "
      << "cost: " << (end - start) << " us.";

  start = end;
  RESPONSE_ON_ERROR(server_ptr_->DropNames(
      name_vec, [self, start, attr](const Status& status) {
        uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        VLOG(self->trace_log_level_)
            << "Request: " << attr.req_name << ". Process drop names "
            << "request cost: " << (end - start) << " us.";
        std::string message_out;
        if (status.ok()) {
          WriteDropNamesReply(message_out);
        } else {
          LOG(ERROR) << "Error: failed to drop name: " << status.ToString()
                     << " for request: " << attr.req_name;
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
  json extra_metadata;
  TRY_READ_REQUEST(ReadShallowCopyRequest, root, id, extra_metadata);
  RESPONSE_ON_ERROR(server_ptr_->ShallowCopy(
      id, extra_metadata, [self](const Status& status, const ObjectID target) {
        std::string message_out;
        if (status.ok()) {
          WriteShallowCopyReply(target, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
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

  TRY_READ_REQUEST(ReadMakeArenaRequest, root, size);
  if (size == std::numeric_limits<size_t>::max()) {
    size = bulk_store_->FootprintLimit();
  }
  int store_fd = -1, fd_to_send = -1;
  uintptr_t base = reinterpret_cast<uintptr_t>(nullptr);
  RESPONSE_ON_ERROR(bulk_store_->MakeArena(size, store_fd, base));
  WriteMakeArenaReply(store_fd, size, base, message_out);

  if (self->used_fds_.find(store_fd) == self->used_fds_.end()) {
    self->used_fds_.emplace(store_fd);
    fd_to_send = store_fd;
  }

  this->doWrite(message_out, [self, fd_to_send](const Status& status) {
    if (fd_to_send != -1) {
      self->sendFd(fd_to_send);
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

  TRY_READ_REQUEST(ReadFinalizeArenaRequest, root, fd, offsets, sizes);
  RESPONSE_ON_ERROR(bulk_store_->FinalizeArena(fd, offsets, sizes));
  WriteFinalizeArenaReply(message_out);

  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doNewSession(const json& root) {
  auto self(shared_from_this());
  StoreType bulk_store_type;
  TRY_READ_REQUEST(ReadNewSessionRequest, root, bulk_store_type);
  VINEYARD_CHECK_OK(server_ptr_->GetRunner()->CreateNewSession(
      bulk_store_type,
      [self](Status const& status, std::string const& ipc_socket) {
        std::string message_out;
        if (status.ok()) {
          WriteNewSessionReply(message_out, ipc_socket);
        } else {
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doDeleteSession(const json& root) {
  std::string message_out;
  WriteDeleteSessionReply(message_out);
  socket_server_ptr_->Close();
  this->doWrite(message_out);
  return true;
}

bool SocketConnection::doMoveBuffersOwnership(json const& root) {
  auto self(shared_from_this());
  std::map<ObjectID, ObjectID> id_to_id;
  std::map<PlasmaID, ObjectID> pid_to_id;
  std::map<ObjectID, PlasmaID> id_to_pid;
  std::map<PlasmaID, PlasmaID> pid_to_pid;
  SessionID session_id;
  TRY_READ_REQUEST(ReadMoveBuffersOwnershipRequest, root, id_to_id, pid_to_id,
                   id_to_pid, pid_to_pid, session_id);
  if (session_id == server_ptr_->session_id()) {
    return false;
  }

  std::shared_ptr<VineyardServer> source_session;
  RESPONSE_ON_ERROR(server_ptr_->GetRunner()->Get(session_id, source_session));

  if (source_session->GetBulkStoreType() == StoreType::kDefault) {
    if (server_ptr_->GetBulkStoreType() == StoreType::kDefault) {
      RESPONSE_ON_ERROR(MoveBuffers(id_to_id, source_session));
    } else {
      RESPONSE_ON_ERROR(MoveBuffers(id_to_pid, source_session));
    }
  } else {
    if (server_ptr_->GetBulkStoreType() == StoreType::kDefault) {
      RESPONSE_ON_ERROR(MoveBuffers(pid_to_id, source_session));
    } else {
      RESPONSE_ON_ERROR(MoveBuffers(pid_to_pid, source_session));
    }
  }

  std::string message_out;
  WriteMoveBuffersOwnershipReply(message_out);
  this->doWrite(message_out);
  return false;
}

template <typename FROM, typename TO>
Status SocketConnection::MoveBuffers(
    std::map<FROM, TO> mapping,
    std::shared_ptr<VineyardServer>& source_session) {
  std::set<FROM> ids;
  for (auto const& item : mapping) {
    ids.insert(item.first);
  }

  std::map<FROM, typename ID_traits<FROM>::P> succeeded_ids;
  RETURN_ON_ERROR(source_session->GetBulkStore<FROM>()->RemoveOwnership(
      ids, succeeded_ids));

  std::map<TO, typename ID_traits<TO>::P> to_process_ids;
  for (auto& item : succeeded_ids) {
    typename ID_traits<TO>::P payload(item.second);
    payload.Reset();
    to_process_ids.emplace(mapping.at(item.first), payload);
  }

  RETURN_ON_ERROR(
      server_ptr_->GetBulkStore<TO>()->MoveOwnership(to_process_ids));

  // FIXME: this is a hack to make sure Moved buffers will never be released.
  int64_t ref_cnt;
  for (auto const& item : mapping) {
    VINEYARD_CHECK_OK(source_session->GetBulkStore<FROM>()->FetchAndModify(
        item.first, ref_cnt, 1));
    VINEYARD_CHECK_OK(server_ptr_->GetBulkStore<TO>()->FetchAndModify(
        item.second, ref_cnt, 1));
  }

  return Status::OK();
}

bool SocketConnection::doEvictObjects(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  std::string message_out;

  TRY_READ_REQUEST(ReadEvictRequest, root, ids);
  RESPONSE_ON_ERROR(
      server_ptr_->EvictObjects(ids, [self](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WriteEvictReply(message_out);
        } else {
          VLOG(100) << "Error: " << status;
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doLoadObjects(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  bool pin = false;
  std::string message_out;

  TRY_READ_REQUEST(ReadLoadRequest, root, ids, pin);
  RESPONSE_ON_ERROR(
      server_ptr_->LoadObjects(ids, pin, [self](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WriteLoadReply(message_out);
        } else {
          VLOG(100) << "Error: " << status;
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doUnpinObjects(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  std::string message_out;

  TRY_READ_REQUEST(ReadUnpinRequest, root, ids);
  RESPONSE_ON_ERROR(
      server_ptr_->UnpinObjects(ids, [self](const Status& status) {
        std::string message_out;
        if (status.ok()) {
          WriteUnpinReply(message_out);
        } else {
          VLOG(100) << "Error: " << status;
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doIsSpilled(json const& root) {
  auto self(shared_from_this());
  ObjectID id;  // Must be a blob id.
  TRY_READ_REQUEST(ReadIsSpilledRequest, root, id);
  bool is_spilled = false;
  RESPONSE_ON_ERROR(bulk_store_->IsSpilled(id, is_spilled));
  std::string message_out;
  WriteIsSpilledReply(is_spilled, message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doIsInUse(json const& root) {
  auto self(shared_from_this());
  ObjectID id;  // Must be a blob id.
  TRY_READ_REQUEST(ReadIsInUseRequest, root, id);
  bool is_in_use = false;
  RESPONSE_ON_ERROR(bulk_store_->IsInUse(id, is_in_use));
  std::string message_out;
  WriteIsInUseReply(is_in_use, message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doGetVineyardMmapFd(const json& root) {
  auto self(shared_from_this());
  TRY_READ_REQUEST(ReadGetVineyardMmapFdRequest, root);
  int fd = -1;
  size_t size = 0;
  size_t offset;
  std::string message_out;
  fd = server_ptr_->GetBulkStore()->GetBaseFd();
  size = server_ptr_->GetBulkStore()->GetBaseSize();
  // offset = server_ptr_->GetBulkStore()->GetBaseOffset();
  offset = server_ptr_->GetBulkStore()->GetUserOffset();

  WriteGetVineyardMmapFdReply(size, offset, message_out);
  this->doWrite(message_out, [self, fd](const Status& status) {
    if (status.ok()) {
      self->sendFd(fd);
    }
    return Status::OK();
  });

  return false;
}

bool SocketConnection::doClusterMeta(const json& root) {
  auto self(shared_from_this());
  TRY_READ_REQUEST(ReadClusterMetaRequest, root);
  RESPONSE_ON_ERROR(server_ptr_->ClusterInfo([self](const Status& status,
                                                    const json& tree) {
    std::string message_out;
    if (status.ok()) {
      WriteClusterMetaReply(tree, message_out);
    } else {
      VLOG(100) << "Error: failed to check cluster meta: " << status.ToString();
      WriteErrorReply(status, message_out);
    }
    self->doWrite(message_out);
    return Status::OK();
  }));
  return false;
}

bool SocketConnection::doInstanceStatus(const json& root) {
  auto self(shared_from_this());
  TRY_READ_REQUEST(ReadInstanceStatusRequest, root);
  RESPONSE_ON_ERROR(server_ptr_->InstanceStatus(
      [self](const Status& status, const json& tree) {
        std::string message_out;
        if (status.ok()) {
          WriteInstanceStatusReply(tree, message_out);
        } else {
          VLOG(100) << "Error: failed to check instance status: "
                    << status.ToString();
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
  TRY_READ_REQUEST(ReadMigrateObjectRequest, root, object_id);

  RESPONSE_ON_ERROR(server_ptr_->MigrateObject(
      object_id, [self]() { return self->running_.load(); },
      [self](const Status& status, const ObjectID& target) {
        std::string message_out;
        if (status.ok()) {
          WriteMigrateObjectReply(target, message_out);
        } else {
          VLOG(100) << "Error: failed to migrate object: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doDebug(const json& root) {
  std::string message_out;
  json result;
  WriteDebugReply(result, message_out);
  this->doWrite(message_out);
  return false;
}

bool SocketConnection::doAcquireLock(const json& root) {
  auto self(shared_from_this());
  std::string key;
  TRY_READ_REQUEST(ReadTryAcquireLockRequest, root, key);

  RESPONSE_ON_ERROR(server_ptr_->TryAcquireLock(
      key, [self](const Status& status, bool result, std::string actual_key) {
        std::string message_out;
        if (status.ok()) {
          WriteTryAcquireLockReply(result, actual_key, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doReleaseLock(const json& root) {
  auto self(shared_from_this());
  std::string key;
  TRY_READ_REQUEST(ReadTryReleaseLockRequest, root, key);

  RESPONSE_ON_ERROR(server_ptr_->TryReleaseLock(
      key, [self](const Status& status, bool result) {
        std::string message_out;
        if (status.ok()) {
          WriteTryReleaseLockReply(result, message_out);
        } else {
          VLOG(100) << "Error: " << status.ToString();
          WriteErrorReply(status, message_out);
        }
        self->doWrite(message_out);
        return Status::OK();
      }));
  return false;
}

bool SocketConnection::doReleaseBlobsWithRDMA(const json& root) {
  auto self(shared_from_this());
  std::vector<ObjectID> ids;
  TRY_READ_REQUEST(ReadReleaseBlobsWithRDMARequest, root, ids);

  boost::asio::post(server_ptr_->GetIOContext(), [self, ids]() {
    self->server_ptr_->UnlockTransmissionObjects(ids);
    std::unordered_set<ObjectID> id_set(ids.begin(), ids.end());
    VINEYARD_DISCARD(
        self->bulk_store_->RemoveDependency(id_set, self->getConnId()));
    std::string message_out;
    WriteReleaseBlobsWithRDMAReply(message_out);
    self->doWrite(message_out);
  });

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
  doAsyncWrite(std::move(to_send));
}

void SocketConnection::doWrite(const std::string& buf, callback_t<> callback,
                               const bool partial) {
  std::string to_send;
  size_t length = buf.size();
  to_send.resize(length + sizeof(size_t));
  char* ptr = &to_send[0];
  memcpy(ptr, &length, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(ptr, buf.data(), length);
  doAsyncWrite(std::move(to_send), callback, partial);
}

void SocketConnection::doWrite(std::string&& buf) {
  doAsyncWrite(std::move(buf));
}

void SocketConnection::doWriteWithoutRead(std::string& buf) {
  std::string to_send;
  size_t length = buf.size();
  to_send.resize(length + sizeof(size_t));
  char* ptr = &to_send[0];
  memcpy(ptr, &length, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(ptr, buf.data(), length);
  doAsyncWriteWithoutRead(std::move(to_send));
}

void SocketConnection::doStop() {
  this->ClearLockedObjects();
  this->socket_server_ptr_->ReleaseExtraRequestMemory(conn_id_);
  this->server_ptr_->GetStreamStore()->CleanResource(
      StreamStore::BuildOwner(this->peer_host, conn_id_));
  if (this->Stop()) {
    // drop connection
    socket_server_ptr_->RemoveConnection(conn_id_);
  }
}

void SocketConnection::doAsyncWrite(std::string&& buf) {
  std::shared_ptr<std::string> payload =
      std::make_shared<std::string>(std::move(buf));
  auto self(shared_from_this());
  asio::async_write(
      socket_, boost::asio::buffer(payload->data(), payload->length()),
      [this, self, payload](boost::system::error_code ec, std::size_t) {
        if (!ec) {
          doReadHeader();
        } else {
          doStop();
          // ThrowException();
        }
      });
}

void SocketConnection::doAsyncWriteWithoutRead(std::string&& buf) {
  std::shared_ptr<std::string> payload =
      std::make_shared<std::string>(std::move(buf));
  auto self(shared_from_this());
  asio::async_write(
      socket_, boost::asio::buffer(payload->data(), payload->length()),
      [this, self, payload](boost::system::error_code ec, std::size_t) {
        if (ec) {
          doStop();
          // ThrowException();
        }
      });
}

void SocketConnection::doAsyncWrite(std::string&& buf, callback_t<> callback,
                                    const bool partial) {
  std::shared_ptr<std::string> payload =
      std::make_shared<std::string>(std::move(buf));
  auto self(shared_from_this());
  asio::async_write(socket_,
                    boost::asio::buffer(payload->data(), payload->length()),
                    [this, self, payload, callback, partial](
                        boost::system::error_code ec, std::size_t) {
                      if (!ec) {
                        if (callback(Status::OK()).ok()) {
                          if (!partial) {
                            doReadHeader();
                          }
                        } else {
                          doStop();
                          // ThrowException();
                        }
                      } else {
                        doStop();
                        // ThrowException();
                      }
                    });
}

void SocketConnection::LockTransmissionObjects(
    const std::vector<ObjectID>& ids) {
  {
    std::lock_guard<std::mutex> lock(locked_objects_mutex_);
    for (auto const& id : ids) {
      if (locked_objects_.find(id) == locked_objects_.end()) {
        locked_objects_[id] = 1;
      } else {
        ++locked_objects_[id];
      }
    }
  }
  server_ptr_->LockTransmissionObjects(ids);
}

void SocketConnection::UnlockTransmissionObjects(
    const std::vector<ObjectID>& ids) {
  {
    std::lock_guard<std::mutex> lock(locked_objects_mutex_);
    for (auto const& id : ids) {
      if (locked_objects_.find(id) != locked_objects_.end()) {
        if (--locked_objects_[id] == 0) {
          locked_objects_.erase(id);
        }
      }
    }
  }
  server_ptr_->UnlockTransmissionObjects(ids);
}

void SocketConnection::ClearLockedObjects() {
  std::vector<ObjectID> ids;
  {
    std::lock_guard<std::mutex> lock(locked_objects_mutex_);
    for (auto const& kv : locked_objects_) {
      for (int i = 0; i < kv.second; ++i) {
        ids.push_back(kv.first);
      }
    }
    locked_objects_.clear();
  }
  server_ptr_->UnlockTransmissionObjects(ids);
}

void SocketConnection::ThrowException() {
  LOG(ERROR) << "Connection closed unexpected from " << peer_host
             << ", conn_id: " << conn_id_;
  throw std::runtime_error("Connection closed unexpected from " + peer_host +
                           ", conn_id: " + std::to_string(conn_id_));
}

SocketServer::SocketServer(std::shared_ptr<VineyardServer> vs_ptr)
    : vs_ptr_(vs_ptr), next_conn_id_(0) {}

void SocketServer::Start() {
  stopped_.store(false);
  closable_.store(false);
  doAccept();
}

void SocketServer::Stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  // stop accepting further connections
  this->Close();

  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutex_);
  std::vector<int> connection_ids_;
  for (auto& pair : connections_) {
    pair.second->Stop();
  }
  connections_.clear();

  std::lock_guard<std::recursive_mutex> lock(
      conn_id_to_extra_request_mem_mutex_);
  for (auto& item : conn_id_to_extra_request_mem_) {
    munmap(item.second.addr_, item.second.size_);
    close(item.second.fd_);
  }
}

void SocketServer::Close() { closable_.store(true); }

bool SocketServer::ExistsConnection(int conn_id) const {
  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutex_);
  return connections_.find(conn_id) != connections_.end();
}

void SocketServer::RemoveConnection(int conn_id) {
  {
    std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutex_);
    auto conn = connections_.find(conn_id);
    if (conn != connections_.end()) {
      connections_.erase(conn);
    }

    if (AliveConnections() == 0 && closable_.load()) {
      VINEYARD_CHECK_OK(vs_ptr_->GetRunner()->Delete(vs_ptr_->session_id()));
    }
  }
}

void SocketServer::CloseConnection(int conn_id) {
  {
    std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutex_);
    auto conn = connections_.find(conn_id);
    if (conn != connections_.end()) {
      conn->second->Stop();
      connections_.erase(conn);
    }
  }

  if (AliveConnections() == 0 && closable_.load()) {
    VINEYARD_CHECK_OK(vs_ptr_->GetRunner()->Delete(vs_ptr_->session_id()));
  }
}

size_t SocketServer::AliveConnections() const {
  std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutex_);
  return connections_.size();
}

Status SocketServer::RequireExtraRequestMemory(int conn_id, size_t size,
                                               int& fd) {
  {
    std::lock_guard<std::recursive_mutex> scope_lock(this->connections_mutex_);
    auto conn = connections_.find(conn_id);
    if (conn == connections_.end()) {
      return Status::Invalid("connection id is not exists:" +
                             std::to_string(conn_id));
    }
  }

  std::lock_guard<std::recursive_mutex> lock(
      conn_id_to_extra_request_mem_mutex_);
  if (conn_id_to_extra_request_mem_.find(conn_id) ==
      conn_id_to_extra_request_mem_.end()) {
    std::string name = "/vineyard_big_request_" + std::to_string(conn_id);
    fd = memfd_create(name.c_str(), 0);
    if (fd == -1) {
      return Status::IOError("fail to create memfd for big request");
    }
    if (ftruncate(fd, size) == -1) {
      close(fd);
      return Status::IOError("fail to truncate memfd for big request");
    }
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
      close(fd);
      return Status::IOError("fail to mmap memfd for big request");
    }
    conn_id_to_extra_request_mem_[conn_id] = {addr, size, fd};
  } else {
    return Status::Invalid(
        "big request memory has been allocated for conn_id: " +
        std::to_string(conn_id));
  }
  return Status::OK();
}

Status SocketServer::ReleaseExtraRequestMemory(int conn_id) {
  std::lock_guard<std::recursive_mutex> lock(
      conn_id_to_extra_request_mem_mutex_);
  auto it = conn_id_to_extra_request_mem_.find(conn_id);
  if (it != conn_id_to_extra_request_mem_.end()) {
    munmap(it->second.addr_, it->second.size_);
    close(it->second.fd_);
    conn_id_to_extra_request_mem_.erase(it);
  }
  return Status::OK();
}

Status SocketServer::ReadExtraMessage(void* data, size_t size, int conn_id) {
  std::lock_guard<std::recursive_mutex> lock(
      conn_id_to_extra_request_mem_mutex_);
  auto it = conn_id_to_extra_request_mem_.find(conn_id);
  if (it == conn_id_to_extra_request_mem_.end()) {
    return Status::Invalid(
        "big request memory has not been allocated for conn_id: " +
        std::to_string(conn_id));
  }
  if (it->second.read_pos_ + size > it->second.size_) {
    return Status::Invalid("big request memory overflow for conn_id: " +
                           std::to_string(conn_id));
  }
  memory::concurrent_memcpy(
      data, reinterpret_cast<char*>(it->second.addr_) + it->second.read_pos_,
      size);
  it->second.read_pos_ += size;
  return Status::OK();
}

Status SocketServer::WriteExtraMessage(const void* data, size_t size,
                                       int conn_id) {
  std::lock_guard<std::recursive_mutex> lock(
      conn_id_to_extra_request_mem_mutex_);
  auto it = conn_id_to_extra_request_mem_.find(conn_id);
  if (it == conn_id_to_extra_request_mem_.end()) {
    return Status::Invalid(
        "big request memory has not been allocated for conn_id: " +
        std::to_string(conn_id));
  }
  if (it->second.write_pos_ + size > it->second.size_) {
    return Status::Invalid("big request memory overflow for conn_id: " +
                           std::to_string(conn_id));
  }
  memory::concurrent_memcpy(
      reinterpret_cast<char*>(it->second.addr_) + it->second.write_pos_, data,
      size);
  it->second.write_pos_ += size;
  return Status::OK();
}

Status SocketServer::LseekExtraMsgWritePos(uint64_t offset, int conn_id) {
  std::lock_guard<std::recursive_mutex> lock(
      conn_id_to_extra_request_mem_mutex_);
  auto it = conn_id_to_extra_request_mem_.find(conn_id);
  if (it == conn_id_to_extra_request_mem_.end()) {
    return Status::Invalid(
        "big request memory has not been allocated for conn_id: " +
        std::to_string(conn_id));
  }
  if (offset > it->second.size_) {
    return Status::Invalid("big request memory overflow for conn_id: " +
                           std::to_string(conn_id));
  }
  it->second.write_pos_ = offset;
  return Status::OK();
}

Status SocketServer::LseekExtraMsgReadPos(uint64_t offset, int conn_id) {
  std::lock_guard<std::recursive_mutex> lock(
      conn_id_to_extra_request_mem_mutex_);
  auto it = conn_id_to_extra_request_mem_.find(conn_id);
  if (it == conn_id_to_extra_request_mem_.end()) {
    return Status::Invalid(
        "big request memory has not been allocated for conn_id: " +
        std::to_string(conn_id));
  }
  if (offset > it->second.size_) {
    return Status::Invalid("big request memory overflow for conn_id: " +
                           std::to_string(conn_id));
  }
  it->second.read_pos_ = offset;
  return Status::OK();
}

Status SocketServer::GetClientAttributeMsg(uint64_t conn_id,
                                           ClientAttributes& attr) {
  size_t length;
  std::string attr_str;
  RETURN_ON_ERROR(ReadExtraMessage(&length, sizeof(size_t), conn_id));
  attr_str.resize(length);
  RETURN_ON_ERROR(ReadExtraMessage(attr_str.data(), length, conn_id));
  attr = ClientAttributes::FromJsonString(attr_str);
  return Status::OK();
}

}  // namespace vineyard
