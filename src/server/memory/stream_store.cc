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

#include "server/memory/stream_store.h"

#include <memory>
#include <mutex>
#include <utility>

#include "common/util/callback.h"
#include "common/util/logging.h"
#include "server/memory/memory.h"
#include "server/server/vineyard_server.h"

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

// manage a pool of streams.
Status StreamStore::Create(ObjectID const stream_id) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) != streams_.end()) {
    return Status::ObjectExists();
  }
  streams_.emplace(stream_id, std::make_shared<StreamHolder>());
  return Status::OK();
}

Status StreamStore::Open(ObjectID const stream_id, int64_t const mode) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    return Status::ObjectNotExists("stream cannot be open: " +
                                   ObjectIDToString(stream_id));
  }
  if (streams_[stream_id]->open_mark & mode) {
    return Status::StreamOpened();
  }
  streams_[stream_id]->open_mark |= mode;
  return Status::OK();
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
    return callback(Status::ObjectNotExists("failed to push to stream"),
                    InvalidObjectID());
  }
  auto stream = streams_.at(stream_id);

  // precondition: there's no unsatistified writer, and still running
  CHECK_STREAM_STATE(!stream->writer_);
  CHECK_STREAM_STATE(!stream->drained && !stream->failed);

  // seal current chunk
  stream->ready_chunks_.push(chunk);

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
          {target}, false, true, false, [](Status const& status) {
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
    // if stream has been stoped, return a proper status.
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

Status StreamStore::Stop(ObjectID const stream_id, bool failed) {
  std::lock_guard<std::recursive_mutex> __guard(this->mutex_);
  if (streams_.find(stream_id) == streams_.end()) {
    return Status::ObjectNotExists("failed to stop stream: " +
                                   ObjectIDToString(stream_id));
  }
  auto stream = streams_.at(stream_id);
  // the stream is still running
  if (stream->drained || stream->failed) {
    return Status::InvalidStreamState("Stream already stoped");
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
  stream->failed = true;
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
          {target}, false, true, false, [](Status const& status) {
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
