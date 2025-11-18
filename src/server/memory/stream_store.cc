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

#include "server/memory/stream_store.h"

#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "common/util/callback.h"
#include "common/util/logging.h"            // IWYU pragma: keep
#include "server/server/vineyard_server.h"  // IWYU pragma: keep

namespace vineyard {

#ifndef CHECK_STREAM_STATE
#define CHECK_STREAM_STATE(condition)                                  \
  do {                                                                 \
    if (!(condition)) {                                                \
      LOG(ERROR) << "Stream state error(" __FILE__                     \
                    ":" VINEYARD_TO_STRING(__LINE__) "): " #condition; \
      return callback(Status::InvalidStreamState(#condition),          \
                      InvalidObjectID());                              \
    }                                                                  \
  } while (0)
#endif  // CHECK_STREAM_STATE

bool DeferredStream::Alive() const { return alive_fn_(); }

bool DeferredStream::TestThenCall() const {
  Status status = Status::IOError("Stream operation timeout.");
  ObjectID ret_id = InvalidObjectID();
  if (!Alive()) {
    VLOG(100) << "Timeout, rpc return!";
    call_fn_(status, ret_id);
    return true;
  } else if (test_fn_(status, ret_id)) {
    VLOG(100) << "Test and call return!";
    call_fn_(status, ret_id);
    return true;
  }
  return false;
}

// manage a pool of streams.
Status StreamStore::Create(ObjectID const stream_id, bool fixed_size, int nums,
                           size_t size) {
  VLOG(2) << "Create stream, id: " << ObjectIDToString(stream_id)
          << ", fixed_size: " << fixed_size << ", nums: " << nums
          << ", size: " << size;
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (stream_id == InvalidObjectID()) {
    LOG(ERROR) << "Failed to create stream with invalid id.";
    return Status::Invalid("Failed to create stream with invalid id.");
  }
  if (streams_.find(stream_id) != streams_.end()) {
    LOG(ERROR) << "Failed to create the stream as it is already exists: "
               << ObjectIDToString(stream_id);
    return Status::ObjectExists(
        "Failed to create the stream as it is already exists: " +
        ObjectIDToString(stream_id));
  }
  std::shared_ptr<StreamHolder> stream_holder =
      std::make_shared<StreamHolder>();
  if (fixed_size) {
    stream_holder->SetFixedBlobStream(nums, size);
    std::string stream_file_name = "/tmp/vineyard-stream-" +
                                   std::to_string(getpid()) + "-" +
                                   std::to_string(stream_id);
    stream_holder->recv_mem_fd =
        open(stream_file_name.c_str(), O_RDWR | O_CREAT | O_NONBLOCK, 0666);
    if (stream_holder->recv_mem_fd < 0) {
      LOG(ERROR) << "failed to create stream file '" << stream_file_name
                 << "', " << strerror(errno);
      return Status::IOError("failed to open file '" + stream_file_name +
                             "', " + strerror(errno));
    }

    unlink(stream_file_name.c_str());
    if (ftruncate(stream_holder->recv_mem_fd, (off_t) STREAM_PAGE_SIZE) != 0) {
      LOG(ERROR) << "failed to ftruncate file " << stream_file_name;
      close(stream_holder->recv_mem_fd);
      return Status::IOError("failed to ftruncate file " + stream_file_name);
    }

    stream_holder->recv_mem_base =
        mmap(0, STREAM_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
             stream_holder->recv_mem_fd, 0);
    if (stream_holder->recv_mem_base == MAP_FAILED) {
      LOG(ERROR) << "failed to mmap stream file '" << stream_file_name << "', "
                 << strerror(errno);
      close(stream_holder->recv_mem_fd);
      return Status::IOError("failed to mmap file '" + stream_file_name +
                             "', " + strerror(errno));
    }
    VLOG(2) << "Create stream file:" << stream_file_name
            << " fd:" << stream_holder->recv_mem_fd
            << " base:" << stream_holder->recv_mem_base;

    memset(stream_holder->recv_mem_base, 0, STREAM_PAGE_SIZE);
  }
  std::string ttl_str =
      read_env("VINEYARD_STREAM_TTL_S", std::to_string(UINT64_MAX));
  stream_holder->ttl = std::stoull(ttl_str);
  streams_.emplace(stream_id, stream_holder);
  return Status::OK();
}

Status StreamStore::PutName(std::string name, ObjectID stream_id) {
  VLOG(2) << "Put name to stream, name: " << name
          << ", stream_id: " << ObjectIDToString(stream_id);
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "Failed to put name to stream: " << name
               << ", stream not exists.";
    return Status::ObjectNotExists("failed to put name to stream: " + name);
  }

  if (stream_names_.find(name) != stream_names_.end()) {
    LOG(ERROR) << "Failed to put name to stream: " << name
               << ". Name already exists.";
    return Status::ObjectExists("failed to put name to stream: " + name +
                                ". Name already exists.");
  }

  stream_names_[name] = stream_id;
  streams_[stream_id]->name = name;
  return Status::OK();
}

Status StreamStore::BindRemoteStream(ObjectID local_stream_id,
                                     ObjectID remote_stream_id,
                                     std::string endpoint,
                                     std::shared_ptr<RemoteClient> client) {
  VLOG(2) << "BindRemoteStream";
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(local_stream_id) == streams_.end()) {
    LOG(ERROR) << "Failed to bind remote stream: "
               << ObjectIDToString(local_stream_id) << ", stream not exists.";
    return Status::ObjectNotExists("failed to bind remote stream: " +
                                   ObjectIDToString(local_stream_id));
  }

  VLOG(2) << "Bind local stream to remote stream, local_id:"
          << ObjectIDToString(local_stream_id)
          << ", remote_id:" << ObjectIDToString(remote_stream_id)
          << ", endpoint:" << endpoint << ", remote client:" << client.get();

  streams_[local_stream_id]->is_forked = true;
  streams_[local_stream_id]->bind_stream_id = remote_stream_id;
  streams_[local_stream_id]->endpoint = endpoint;
  streams_[local_stream_id]->remote_client = client;

  return Status::OK();
}

Status StreamStore::UnbindRemoteStream(ObjectID local_stream_id) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(local_stream_id) == streams_.end()) {
    LOG(ERROR) << "Failed to unbind remote stream: "
               << ObjectIDToString(local_stream_id) << ", stream not exists.";
    return Status::ObjectNotExists("failed to unbind remote stream: " +
                                   ObjectIDToString(local_stream_id));
  }

  VLOG(2) << "Unbind local stream, local_id:"
          << ObjectIDToString(local_stream_id) << ", remote_id:"
          << ObjectIDToString(streams_[local_stream_id]->bind_stream_id)
          << ", endpoint:" << streams_[local_stream_id]->endpoint
          << ", remote client:"
          << streams_[local_stream_id]->remote_client.get();

  streams_[local_stream_id]->bind_stream_id = InvalidObjectID();
  streams_[local_stream_id]->endpoint = "";
  streams_[local_stream_id]->remote_client = nullptr;

  return Status::OK();
}

Status StreamStore::GetStreamIDByName(std::string name, ObjectID& stream_id) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (stream_names_.find(name) == stream_names_.end()) {
    LOG(ERROR) << "Failed to get stream id by name: " << name;
    return Status::ObjectNotExists("failed to get stream id by name: " + name);
  }

  stream_id = stream_names_.at(name);
  return Status::OK();
}

Status StreamStore::Open(ObjectID const stream_id, int64_t const mode,
                         std::string owner) {
  VLOG(2) << "Try to open stream by id: " << ObjectIDToString(stream_id)
          << ", mode: " << mode << ", owner: " << owner;
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (stream_id == InvalidObjectID()) {
    LOG(ERROR) << "Failed to open stream by id: "
               << ObjectIDToString(stream_id);
    return Status::ObjectNotExists("failed to open stream by id: " +
                                   ObjectIDToString(stream_id));
  }

  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "Failed to open stream by id: "
               << ObjectIDToString(stream_id);
    return Status::ObjectNotExists("stream cannot be open: " +
                                   ObjectIDToString(stream_id));
  }

  if (streams_[stream_id]->abort) {
    LOG(ERROR) << "Failed to open stream by id: " << ObjectIDToString(stream_id)
               << ", stream is aborted.";
    return Status::InvalidStreamState("stream is aborted");
  }

  if (streams_[stream_id]->open_mark & mode) {
    LOG(ERROR) << "Failed to open stream by id: " << ObjectIDToString(stream_id)
               << ", stream already opened.";
    return Status::StreamOpened();
  }
  VLOG(100) << "owner:" << owner << " mode:" << mode
            << " read:" << (int64_t) StreamOpenMode::read
            << " write:" << (int64_t) StreamOpenMode::write;
  if (mode & (int64_t) StreamOpenMode::read) {
    streams_[stream_id]->reader_owner = owner;
  }
  if (mode & (int64_t) StreamOpenMode::write) {
    streams_[stream_id]->writer_owner = owner;
  }
  streams_[stream_id]->open_mark |= mode;
  return Status::OK();
}

Status StreamStore::Open(std::string name, ObjectID& ret_id, int64_t const mode,
                         std::string owner) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (stream_names_.find(name) == stream_names_.end()) {
    return Status::ObjectNotExists("failed to open stream by name: " + name);
  }

  Status status = Open(stream_names_.at(name), mode, owner);
  if (status.ok()) {
    ret_id = stream_names_.at(name);
  }
  return status;
}

Status StreamStore::Open(ObjectID const stream_id, int64_t const mode,
                         std::string owner, bool wait, uint64_t timeout,
                         void_callback_t<Status&, ObjectID> callback) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  VLOG(100) << "open owner:" << owner;
  auto self(shared_from_this());
  Status status = Open(stream_id, mode, owner);
  if (status.IsObjectNotExists() && wait) {
    LOG(INFO) << "Stream is not exist, waiting for it to be created";
    uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    auto alive = [start, timeout]() -> bool {
      uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
      return now - start < timeout;
    };
    auto test = [self, stream_id, mode, owner](Status& status,
                                               ObjectID& ret_id) -> bool {
      VLOG(100) << "Try to open stream: " << stream_id;
      status = self->Open(stream_id, mode, owner);
      if (status.IsObjectNotExists()) {
        VLOG(100) << "Stream is not exist, waiting for it to be created";
        return false;
      }
      VLOG(100) << "Status:" << status.ToString();
      return true;
    };
    auto call = [callback](Status& status, ObjectID id) {
      callback(status, id);
    };
    deferred_.emplace_back(alive, test, call);
    return Status::OK();
  } else {
    VLOG(100) << "Stream is already exist, call callback and return";
    callback(status, stream_id);
    return Status::OK();
  }
}

Status StreamStore::Open(std::string stream_name, int64_t const mode,
                         std::string owner, bool wait, uint64_t timeout,
                         void_callback_t<Status&, ObjectID> callback) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  auto self(shared_from_this());
  ObjectID stream_id = InvalidObjectID();
  Status status = Open(stream_name, stream_id, mode, owner);
  if (status.IsObjectNotExists() && wait) {
    VLOG(2) << "Stream is not exist, waiting for it to be created";
    uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    auto alive = [start, timeout]() -> bool {
      uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
      return now - start < timeout;
    };
    auto test = [self, stream_name, mode, owner](Status& status,
                                                 ObjectID& ret_id) -> bool {
      VLOG(100) << "Try to open stream: " << stream_name;
      status = self->Open(stream_name, ret_id, mode, owner);
      if (status.IsObjectNotExists()) {
        VLOG(100) << "Stream is not exist, waiting for it to be created";
        return false;
      }
      VLOG(100) << "Status:" << status.ToString();
      return true;
    };
    auto call = [callback](Status& status, ObjectID id) {
      callback(status, id);
    };
    deferred_.emplace_back(alive, test, call);
    return Status::OK();
  } else {
    VLOG(2) << "Stream is already exist, call callback and return";
    callback(status, stream_id);
    return Status::OK();
  }
}

// for producer: return the next chunk to write, and make current chunk
// available for consumer to read
Status StreamStore::Get(ObjectID const stream_id, size_t const size,
                        callback_t<const ObjectID> callback) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    return callback(Status::ObjectNotExists("failed to allocate from stream"),
                    InvalidObjectID());
  }
  auto stream = streams_.at(stream_id);

  // precondition: there's no unsatistified writer, and still running
  CHECK_STREAM_STATE(!stream->writer_);
  CHECK_STREAM_STATE(!stream->drained && !stream->failed);

  // seal current chunk
  if (stream->current_writing_) {
    VINEYARD_DISCARD(store_->Seal(stream->current_writing_.get()));
    stream->ready_chunks_.push(stream->current_writing_.get());
    stream->current_writing_ = boost::none;
  }
  // weak up the pending reader
  if (stream->reader_) {
    // should be no reading chunk
    CHECK_STREAM_STATE(!stream->current_reading_);
    if (!stream->ready_chunks_.empty()) {
      stream->current_reading_ = stream->ready_chunks_.front();
      stream->ready_chunks_.pop();
      VINEYARD_SUPPRESS(
          stream->reader_.get()(Status::OK(), stream->current_reading_.get()));
      stream->reader_ = boost::none;
    }
  }

  if (allocatable(stream, size)) {
    // do allocation
    ObjectID chunk;
    std::shared_ptr<Payload> object;
    auto status = store_->Create(size, chunk, object);
    if (!status.ok()) {
      return callback(status, InvalidObjectID());
    } else {
      stream->current_writing_ = chunk;
      return callback(Status::OK(), stream->current_writing_.get());
    }
  } else {
    // pending the writer
    stream->writer_ = std::make_pair(size, callback);
    return Status::OK();
  }
}

// for producer: return the next chunk to write, and make current chunk
// available for consumer to read
Status StreamStore::Push(ObjectID const stream_id, ObjectID const chunk,
                         callback_t<const ObjectID> callback) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "Failed to push to stream: " << ObjectIDToString(stream_id)
               << ", stream not exists.";
    return callback(Status::ObjectNotExists("failed to push to stream"),
                    InvalidObjectID());
  }
  auto stream = streams_.at(stream_id);

  // precondition: there's no unsatistified writer, and still running
  CHECK_STREAM_STATE(!stream->writer_);
  CHECK_STREAM_STATE(!stream->drained && !stream->failed);

  if (stream->type == STREAM_TYPE::FIXED_SIZE_STREAM) {
    if (stream->pushed_nums == stream->blob_nums) {
      LOG(ERROR) << "Stream is full.";
      return callback(Status::IOError("Stream is full."), InvalidObjectID());
    }
    if (stream->abort) {
      LOG(ERROR) << "Stream is aborted.";
      return callback(Status::IOError("Stream is aborted."), InvalidObjectID());
    }
  }

  // seal current chunk
  stream->ready_chunks_.push(chunk);
  stream->pushed_nums++;

  // weak up the pending reader
  if (stream->type == STREAM_TYPE::NOMAL_STREAM) {
    if (stream->reader_) {
      // should be no reading chunk
      CHECK_STREAM_STATE(!stream->current_reading_);
      if (!stream->ready_chunks_.empty()) {
        stream->current_reading_ = stream->ready_chunks_.front();
        stream->ready_chunks_.pop();
        VINEYARD_SUPPRESS(stream->reader_.get()(
            Status::OK(), stream->current_reading_.get()));
        stream->reader_ = boost::none;
      }
    } else {
      VLOG(100) << "No reader, push chunk to ready queue";
    }
  }

  // done
  return callback(Status::OK(), InvalidObjectID());
}

// for consumer: read current chunk
Status StreamStore::Pull(ObjectID const stream_id,
                         callback_t<const ObjectID> callback) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    return callback(Status::ObjectNotExists("failed to pull from stream"),
                    InvalidObjectID());
  }
  auto stream = streams_.at(stream_id);

  // precondition: there's no unsatistified reader
  CHECK_STREAM_STATE(!stream->reader_);

  // drop current reading
  if (stream->current_reading_) {
    auto target = stream->current_reading_.get();
    Status status;
    if (IsBlob(target)) {
      status = store_->Delete(target);
    } else {
      status = server_->DelData(
          {target}, false, true, false, false, [](Status const& status) {
            if (!status.ok()) {
              LOG(WARNING) << "failed to delete the stream chunk: "
                           << status.ToString();
            }
            return Status::OK();
          });
    }
    if (!status.ok()) {
      return callback(status, InvalidObjectID());
    }
    stream->current_reading_ = boost::none;
  }
  // wake up the pending writer
  if (stream->writer_) {
    // should be no writing chunk
    CHECK_STREAM_STATE(!stream->current_writing_);
    auto writer = stream->writer_.get();
    if (allocatable(stream, writer.first)) {
      ObjectID chunk;
      std::shared_ptr<Payload> object;
      auto status = store_->Create(writer.first, chunk, object);
      if (!status.ok()) {
        VINEYARD_SUPPRESS(writer.second(status, InvalidObjectID()));
      } else {
        stream->current_writing_ = chunk;
        VINEYARD_SUPPRESS(
            writer.second(Status::OK(), stream->current_writing_.get()));
        stream->writer_ = boost::none;
      }
    }
  }

  if (!stream->ready_chunks_.empty()) {
    stream->current_reading_ = stream->ready_chunks_.front();
    stream->ready_chunks_.pop();
    return callback(Status::OK(), stream->current_reading_.get());
  } else {
    // if stream has been stopped, return a proper status.
    if (stream->drained) {
      return callback(Status::StreamDrained(), InvalidObjectID());
    } else if (stream->failed) {
      return callback(Status::StreamFailed(), InvalidObjectID());
    } else {
      // pending the reader
      stream->reader_ = callback;
      return Status::OK();
    }
  }
}

void StreamStore::AutoRead(
    std::shared_ptr<StreamHolder> stream,
    callback_t<ObjectID, std::vector<uint64_t>, std::vector<uint64_t>,
               std::vector<size_t>, int>
        callback) {
  std::vector<ObjectID> local_buffers;
  std::vector<std::vector<uint64_t>> addr_list;
  std::vector<std::vector<uint64_t>> rkey_list;
  std::vector<std::vector<size_t>> size_list;
  std::vector<int> index_list;
  {
    std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
    if (stream == nullptr) {
      LOG(ERROR) << "Stream object invalid.";
      callback(Status::Invalid("Stream object invalid."), InvalidObjectID(),
               {0}, {0}, {0}, -1);
      return;
    }

    if (stream->abort) {
      LOG(ERROR) << "Stream is aborted.";
      stream->transfer_finished = true;
      callback(Status::IOError("Stream is aborted."), InvalidObjectID(), {0},
               {0}, {0}, -1);
      return;
    }

    while (!stream->ready_chunks_.empty()) {
      stream->current_reading_ = stream->ready_chunks_.front();
      stream->ready_chunks_.pop();
      local_buffers.push_back(stream->current_reading_.get());
      addr_list.push_back(stream->receive_addr_list[stream->read_index]);
      rkey_list.push_back(stream->rkeys_list[stream->read_index]);
      size_list.push_back(stream->sizes_list[stream->read_index]);
      index_list.push_back(stream->read_index);

      stream->read_index++;
      stream->current_reading_ = boost::none;
    }
  }

  for (size_t i = 0; i < local_buffers.size(); i++) {
    {
      std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
      if (stream->abort) {
        LOG(ERROR) << "Stream is aborted. Interrupt the transfer.";
        callback(Status::IOError("Stream is aborted."), InvalidObjectID(), {0},
                 {0}, {0}, -1);
        stream->transfer_finished = true;
        return;
      }
    }

    Status status = callback(Status::OK(), local_buffers[i], addr_list[i],
                             rkey_list[i], size_list[i], index_list[i]);

    {
      std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to send data to remote: " << status.ToString();
        stream->transfer_finished = true;
        return;
      }
    }
  }

  {
    std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
    if (stream->blob_received_nums == stream->blob_nums) {
      stream->transfer_finished = true;
    } else {
      boost::asio::post(
          this->server_->GetIOContext(),
          [this, stream, callback]() { this->AutoRead(stream, callback); });
    }
  }
}

void StreamStore::ActivateRemoteFixedStream(
    ObjectID stream_id, std::vector<std::vector<uint64_t>> recv_addr_list,
    std::vector<std::vector<uint64_t>> rkeys,
    std::vector<std::vector<size_t>> sizes_list,
    callback_t<ObjectID, std::vector<uint64_t>, std::vector<uint64_t>,
               std::vector<size_t>, int>
        callback) {
  VLOG(2) << "Activate remote fixed stream, stream_id: "
          << ObjectIDToString(stream_id);
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "failed to pull from stream, stream id: "
               << ObjectIDToString(stream_id);
    callback(Status::ObjectNotExists("failed to pull from stream, stream id: " +
                                     ObjectIDToString(stream_id)),
             InvalidObjectID(), {0}, {0}, {0}, -1);
    return;
  }
  auto stream = streams_.at(stream_id);

  if (stream->type != FIXED_SIZE_STREAM) {
    LOG(ERROR) << "Stream is not fixed size stream";
    callback(Status::InvalidStreamState("Stream is not fixed size stream"),
             InvalidObjectID(), {0}, {0}, {0}, -1);
    return;
  }

  if (stream->abort) {
    LOG(ERROR) << "Stream is aborted.";
    callback(Status::IOError("Stream is aborted."), InvalidObjectID(), {0}, {0},
             {0}, -1);
    return;
  }

  stream->receive_addr_list = std::move(recv_addr_list);
  stream->rkeys_list = std::move(rkeys);
  stream->sizes_list = std::move(sizes_list);

  stream->auto_reader_ = callback;
  stream->transfer_finished = false;

  VLOG(2) << "Post read task to IOContext, stream_id: "
          << ObjectIDToString(stream_id);
  auto self(shared_from_this());
  boost::asio::post(server_->GetIOContext(), [callback, stream, self]() {
    self->AutoRead(stream, callback);
  });

  // if stream has been stopped, return a proper status.
  if (stream->drained) {
    callback(Status::StreamDrained(), InvalidObjectID(), {0}, {0}, {0}, -1);
  } else if (stream->failed) {
    callback(Status::StreamFailed(), InvalidObjectID(), {0}, {0}, {0}, -1);
  }
}

void StreamStore::AutoRead(std::shared_ptr<StreamHolder> stream,
                           callback_t<ObjectID, int> callback) {
  std::vector<ObjectID> local_buffers;
  std::vector<int> index_list;
  {
    std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
    if (stream == nullptr) {
      LOG(ERROR) << "Stream object invalid.";
      callback(Status::Invalid("Stream object invalid."), InvalidObjectID(),
               -1);
      return;
    }

    if (stream->abort) {
      LOG(ERROR) << "Stream is aborted.";
      stream->transfer_finished = true;
      callback(Status::IOError("Stream is aborted."), InvalidObjectID(), -1);
      return;
    }

    while (!stream->ready_chunks_.empty()) {
      stream->current_reading_ = stream->ready_chunks_.front();
      stream->ready_chunks_.pop();
      local_buffers.push_back(stream->current_reading_.get());
      index_list.push_back(stream->read_index);

      stream->read_index++;
      stream->current_reading_ = boost::none;
    }
  }

  for (size_t i = 0; i < local_buffers.size(); i++) {
    {
      std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
      if (stream->abort) {
        callback(Status::IOError("Stream is aborted."), InvalidObjectID(), -1);
        stream->transfer_finished = true;
        return;
      }
    }

    Status status = callback(Status::OK(), local_buffers[i], index_list[i]);

    {
      std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to send data to remote: " << status.ToString();
        stream->transfer_finished = true;
        return;
      }
    }
  }

  {
    std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
    if (stream->blob_received_nums == stream->blob_nums) {
      stream->transfer_finished = true;
    } else {
      boost::asio::post(
          this->server_->GetIOContext(),
          [this, stream, callback]() { this->AutoRead(stream, callback); });
    }
  }
}

void StreamStore::ActivateRemoteFixedStream(
    ObjectID stream_id, callback_t<ObjectID, int> callback) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "failed to pull from stream, stream id: "
               << ObjectIDToString(stream_id);
    callback(Status::ObjectNotExists("failed to pull from stream, stream id: " +
                                     ObjectIDToString(stream_id)),
             InvalidObjectID(), -1);
    return;
  }
  auto stream = streams_.at(stream_id);

  if (stream->type != FIXED_SIZE_STREAM) {
    LOG(ERROR) << "Stream is not fixed size stream";
    callback(Status::InvalidStreamState("Stream is not fixed size stream"),
             InvalidObjectID(), -1);
    return;
  }

  if (stream->abort) {
    LOG(ERROR) << "Stream is aborted.";
    callback(Status::IOError("Stream is aborted."), InvalidObjectID(), -1);
    return;
  }

  stream->auto_reader_test_ = callback;
  stream->transfer_finished = false;

  auto self(shared_from_this());
  boost::asio::post(server_->GetIOContext(), [callback, stream, self]() {
    self->AutoRead(stream, callback);
  });

  // if stream has been stopped, return a proper status.
  if (stream->drained) {
    callback(Status::StreamDrained(), InvalidObjectID(), -1);
  } else if (stream->failed) {
    callback(Status::StreamFailed(), InvalidObjectID(), -1);
  }
}

Status StreamStore::CheckBlobReceived(ObjectID stream_id, int index,
                                      bool& finished) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "failed to check blob received, stream id: "
               << ObjectIDToString(stream_id);
    return Status::ObjectNotExists(
        "failed to check blob received, stream id: " +
        ObjectIDToString(stream_id));
  }
  auto stream = streams_.at(stream_id);

  if (stream->type != FIXED_SIZE_STREAM) {
    LOG(ERROR) << "Stream is not fixed size stream";
    return Status::InvalidStreamState("Stream is not fixed size stream");
  }

  if (index < 0) {
    for (int i = 0; i < stream->blob_nums; i++) {
      if (stream->blob_received.find(i) == stream->blob_received.end()) {
        finished = false;
        return Status::OK();
      }
    }
    finished = true;
  } else {
    if (index >= stream->blob_nums) {
      LOG(ERROR) << "Index out of range";
      return Status::InvalidStreamState("Index out of range");
    }

    if (stream->blob_received.find(index) == stream->blob_received.end()) {
      finished = false;
    } else {
      finished = true;
    }
  }

  return Status::OK();
}

Status StreamStore::SetBlobReceived(ObjectID stream_id, int index) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "failed to set blob received, stream id: "
               << ObjectIDToString(stream_id);
    return Status::ObjectNotExists("failed to set blob received, stream id: " +
                                   ObjectIDToString(stream_id));
  }
  auto stream = streams_.at(stream_id);

  if (stream->type != FIXED_SIZE_STREAM) {
    LOG(ERROR) << "Stream is not fixed size stream";
    return Status::InvalidStreamState("Stream is not fixed size stream");
  }

  if (index < 0 || index >= stream->blob_nums || index >= STREAM_PAGE_SIZE) {
    LOG(ERROR) << "Index out of range";
    return Status::InvalidStreamState("Index out of range");
  }

  stream->blob_received.insert(index);
  stream->blob_received_nums++;

  reinterpret_cast<char*>(stream->recv_mem_base)[index] = 1;

  return Status::OK();
}

Status StreamStore::Stop(ObjectID const stream_id, bool failed) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    return Status::ObjectNotExists("failed to stop stream: " +
                                   ObjectIDToString(stream_id));
  }
  auto stream = streams_.at(stream_id);
  // the stream is still running
  if (stream->drained || stream->failed) {
    return Status::InvalidStreamState("Stream already stopped");
  }
  // no pending writer
  if (stream->writer_) {
    return Status::InvalidStreamState("Still pending writer on stream");
  }
  // seal current writing chunk
  if (stream->current_writing_) {
    VINEYARD_DISCARD(store_->Seal(stream->current_writing_.get()));
    stream->ready_chunks_.push(stream->current_writing_.get());
    stream->current_writing_ = boost::none;
  }
  // stop
  if (failed) {
    stream->failed = true;
  } else {
    stream->drained = true;
  }
  // weak up the pending reader
  if (stream->reader_) {
    // should be no reading chunk
    if (stream->current_reading_) {
      auto err =
          Status::InvalidStreamState("Shouldn't exists a being read chunk");
      VINEYARD_SUPPRESS(stream->reader_.get()(err, InvalidObjectID()));
      stream->reader_ = boost::none;
      return err;
    }
    if (stream->failed) {
      VINEYARD_SUPPRESS(
          stream->reader_.get()(Status::StreamFailed(), InvalidObjectID()));
      stream->reader_ = boost::none;
    } else if (!stream->ready_chunks_.empty()) {
      stream->current_reading_ = stream->ready_chunks_.front();
      stream->ready_chunks_.pop();
      VINEYARD_SUPPRESS(
          stream->reader_.get()(Status::OK(), stream->current_reading_.get()));
      stream->reader_ = boost::none;
    } else if (stream->drained) {
      VINEYARD_SUPPRESS(
          stream->reader_.get()(Status::StreamDrained(), InvalidObjectID()));
      stream->reader_ = boost::none;
    } else {
      RETURN_ON_ASSERT(false, "Impossible!");
    }
  }
  return Status::OK();
}

Status StreamStore::Drop(ObjectID const stream_id) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    return Status::ObjectNotExists("failed to drop stream: " +
                                   ObjectIDToString(stream_id));
  }
  auto stream = streams_.at(stream_id);
  if (!stream->failed && !stream->drained) {
    stream->failed = true;
  }
  // weakup pending reader
  if (stream->reader_) {
    // should be no reading chunk
    if (stream->current_reading_) {
      return Status::InvalidStreamState("Shouldn't exists a being read chunk");
    }
    VINEYARD_SUPPRESS(
        stream->reader_.get()(Status::StreamFailed(), InvalidObjectID()));
    stream->reader_ = boost::none;
  }
  // drop all memory chunks in ready queue, but still keep the reading chunk
  // to avoid crash the reader
  while (!stream->ready_chunks_.empty()) {
    auto target = stream->ready_chunks_.front();
    Status status;
    if (IsBlob(target)) {
      status = store_->Delete(target);
    } else {
      status = server_->DelData(
          {target}, false, false, true, false, [](Status const& status) {
            if (!status.ok()) {
              LOG(WARNING) << "failed to delete the stream chunk: "
                           << status.ToString();
            }
            return Status::OK();
          });
    }
    VINEYARD_DISCARD(status);
    stream->ready_chunks_.pop();
  }
  {
    // erase, prevent throwing
    auto loc = streams_.find(stream_id);
    if (loc != streams_.end()) {
      streams_.erase(loc);
    }
  }
  return Status::OK();
}

Status StreamStore::Close(ObjectID const stream_id, std::string access_key) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "failed to close stream: " << ObjectIDToString(stream_id)
               << ", stream is not found.";
    return Status::ObjectNotExists("failed to close stream: " +
                                   ObjectIDToString(stream_id));
  }
  VLOG(100) << "Close stream: " << stream_id;
  auto stream = streams_.at(stream_id);
  if (stream->type == FIXED_SIZE_STREAM) {
    if (stream->reader_owner == access_key) {
      stream->reader_owner = "";
      stream->reader_ = boost::none;
      stream->open_mark &= ~(static_cast<uint64_t>(StreamOpenMode::read));
      VLOG(100) << "Close reader, open mode:" << stream->open_mark;
    } else if (stream->writer_owner == access_key) {
      stream->writer_owner = "";
      stream->open_mark &= ~(static_cast<uint64_t>(StreamOpenMode::write));
      VLOG(100) << "Close writer, open mode:" << stream->open_mark;
    } else {
      VLOG(100) << "access key: " << access_key
                << " reader: " << stream->reader_owner
                << " writer: " << stream->writer_owner;
      return Status::Invalid("Invalid access key, access denied.");
    }
  } else {
    LOG(ERROR) << "Close is not supported for this stream.";
    return Status::NotImplemented("Close is not supported for this stream.");
  }
  return Status::OK();
}

Status StreamStore::SetErrorFlag(ObjectID const stream_id, Status const error) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "failed to set error flag for stream: "
               << ObjectIDToString(stream_id) << ", stream not found.";
    return Status::ObjectNotExists("failed to set error flag for stream: " +
                                   ObjectIDToString(stream_id));
  }

  auto stream = streams_.at(stream_id);
  if (stream->type != FIXED_SIZE_STREAM) {
    LOG(ERROR) << "Set error flag is not supported for this stream.";
    return Status::NotImplemented(
        "Set error flag is not supported for this stream.");
  }

  std::string error_str = error.ToString().substr(0, STREAM_ERROR_LENGTH);
  memcpy(reinterpret_cast<unsigned char*>(stream->recv_mem_base) +
             (STREAM_PAGE_SIZE - STREAM_ERROR_LENGTH - sizeof(unsigned char)),
         error_str.c_str(), error_str.size());
  reinterpret_cast<unsigned char*>(
      stream->recv_mem_base)[STREAM_PAGE_SIZE - sizeof(unsigned char)] =
      static_cast<unsigned char>(error.code());
  return Status::OK();
}

Status StreamStore::Abort(ObjectID const stream_id, bool& success) {
  VLOG(2) << "Try to abort stream: " << ObjectIDToString(stream_id);
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(ERROR) << "failed to abort stream: " << ObjectIDToString(stream_id)
               << ", stream is not found.";
    return Status::ObjectNotExists("failed to abort stream: " +
                                   ObjectIDToString(stream_id));
  }
  auto stream = streams_.at(stream_id);
  stream->abort = true;
  if (stream->auto_reader_ == boost::none) {
    // means that the stream is not activated
    stream->transfer_finished = true;
  }
  // To prevent the stream is not activated. To prevent the sender from waiting
  // forever after push, we set the error flag here.
  SetErrorFlag(stream_id, Status::IOError("Stream is aborted."));
  success = stream->transfer_finished;
  VLOG(2) << "Stream id: " << ObjectIDToString(stream_id)
          << ", abort result:" << success;
  return Status::OK();
}

Status StreamStore::Delete(ObjectID const stream_id) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  auto self(shared_from_this());
  if (streams_.find(stream_id) == streams_.end()) {
    LOG(WARNING) << "Delete stream not found: " << stream_id;
    return Status::OK();
  }
  auto stream = streams_.at(stream_id);
  stream->start_time = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  VLOG(2) << "Try to delete stream:" << stream->name
          << ", stream id:" << ObjectIDToString(stream_id)
          << ", open mark: " << stream->open_mark << ", type: " << stream->type
          << ", ttl: " << stream->ttl;
  if (stream->type == FIXED_SIZE_STREAM) {
    if (stream->open_mark) {
      VLOG(2) << "Stream is still open, defered delete it.";
      auto alive_t = [self, stream_id]() -> bool { return true; };
      auto test_t = [self, stream_id](Status& status,
                                      ObjectID& ret_id) -> bool {
        std::lock_guard<std::recursive_mutex> __guard(self->mutex_);
        VLOG(100) << "Check if the stream can be deleted";
        if (self->streams_.find(stream_id) == self->streams_.end()) {
          status = Status::ObjectNotExists("stream not found");
          return true;
        } else if (self->streams_[stream_id]->open_mark) {
          if (std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                      .count() -
                  self->streams_[stream_id]->start_time >
              self->streams_[stream_id]->ttl) {
            LOG(WARNING) << "Stream is still open, but timeout, force to "
                            "delete it";
            LOG(WARNING) << "Stream name: " << self->streams_[stream_id]->name
                         << " open mark: "
                         << self->streams_[stream_id]->open_mark
                         << " ttl: " << self->streams_[stream_id]->ttl
                         << " start time: "
                         << self->streams_[stream_id]->start_time;
            return true;
          }
          VLOG(100)
              << "Stream is still open, waiting for it to be closed.open mark: "
              << self->streams_[stream_id]->open_mark
              << " name:" << self->streams_[stream_id]->name;
          VLOG(100) << "stream is abort: " << self->streams_[stream_id]->abort;
          return false;
        } else {
          VLOG(100) << "Stream can be deleted";
          return true;
        }
      };
      auto call_t = [self, stream_id](Status& status, ObjectID id) {
        std::lock_guard<std::recursive_mutex> __guard(self->mutex_);
        self->PrintStreamInfo();
        if (self->streams_.find(stream_id) == self->streams_.end()) {
          return;
        }
        auto stream_ = self->streams_[stream_id];
        if (stream_->type == FIXED_SIZE_STREAM) {
          if (stream_->recv_mem_base != nullptr) {
            munmap(stream_->recv_mem_base, STREAM_PAGE_SIZE);
            close(stream_->recv_mem_fd);
          }
          while (!stream_->ready_chunks_.empty()) {
            stream_->current_reading_ = stream_->ready_chunks_.front();
            stream_->ready_chunks_.pop();
            self->server_->GetBulkStore()->DeleteUserBlob(
                stream_->current_reading_.get());
          }
        }
        for (auto item : self->stream_names_) {
          if (item.second == stream_id) {
            VLOG(100) << "Delete stream name: " << item.first;
            self->stream_names_.erase(item.first);
            break;
          }
        }
        VLOG(100) << "Delete stream: " << stream_id;
        auto stream = self->streams_[stream_id];
        self->streams_.erase(stream_id);
        VLOG(100) << "Stream deleted: " << stream_id;
        self->PrintStreamInfo();
      };
      VLOG(100) << "Defered delete stream: " << stream_id;
      deferred_.emplace_back(alive_t, test_t, call_t);
    } else {
      VLOG(2) << "Stream closed, delete it directly.";
      PrintStreamInfo();
      if (stream->recv_mem_base != nullptr) {
        munmap(stream->recv_mem_base, STREAM_PAGE_SIZE);
        close(stream->recv_mem_fd);
      }
      while (!stream->ready_chunks_.empty()) {
        stream->current_reading_ = stream->ready_chunks_.front();
        stream->ready_chunks_.pop();
        self->server_->GetBulkStore()->DeleteUserBlob(
            stream->current_reading_.get());
      }
      for (auto item : stream_names_) {
        if (item.second == stream_id) {
          stream_names_.erase(item.first);
          VLOG(100) << "Delete stream name: " << item.first;
          break;
        }
      }
      streams_.erase(stream_id);
      VLOG(100) << "Stream deleted: " << stream_id;
      PrintStreamInfo();
    }
  } else {
    LOG(INFO) << "Delete is not supported for this stream.";
    return Status::NotImplemented("Delete is not supported for this stream.");
  }

  return Status::OK();
}

Status StreamStore::CleanResource(std::string owner) {
  VLOG(2) << "Clean resource for owner: " << owner;
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  for (auto iter = streams_.begin(); iter != streams_.end(); iter++) {
    auto stream = iter->second;
    if (stream->reader_owner == owner || stream->writer_owner == owner) {
      if (stream->type == FIXED_SIZE_STREAM) {
        LOG(INFO) << "Clean and abort stream: " << iter->first;
        stream->abort = true;
        Status status =
            SetErrorFlag(iter->first, Status::IOError("Stream is aborted."));
        LOG(INFO) << "set error status:" << status.ToString();
        Close(iter->first, owner);
      }
    }
  }
  return Status::OK();
}

bool StreamStore::allocatable(std::shared_ptr<StreamHolder> stream,
                              size_t size) {
  if (store_->Footprint() + size <
      store_->FootprintLimit() * threshold_ / 100.0) {
    return true;
  } else {
    return false;
  }
}

}  // namespace vineyard
