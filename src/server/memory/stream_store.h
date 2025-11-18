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

#ifndef SRC_SERVER_MEMORY_STREAM_STORE_H_
#define SRC_SERVER_MEMORY_STREAM_STORE_H_

#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "boost/optional/optional.hpp"

#include "common/util/asio.h"
#include "common/util/callback.h"
#include "common/util/uuid.h"
#include "server/memory/memory.h"
#include "server/util/remote.h"
#include "server/util/utils.h"

namespace vineyard {

#define STREAM_PAGE_SIZE 4096
#define STREAM_ERROR_LENGTH 256

// forward declarations.
class VineyardServer;

enum STREAM_TYPE {
  NOMAL_STREAM = 0,
  FIXED_SIZE_STREAM = 1,
};

enum class StreamOpenMode {
  read = 1,
  write = 2,
};

/**
 * @brief StreamHolder aims to maintain all chunks for a single stream.
 * "Stream" is a special kind of "Object" in vineyard, which represents
 * a stream (especially for I/O) that connects two drivers and avoids
 * the overhead of immediate temporary data structures and objects.
 *
 */
struct StreamHolder {
  boost::optional<ObjectID> current_writing_, current_reading_;
  std::queue<ObjectID> ready_chunks_;
  boost::optional<callback_t<ObjectID>> reader_;
  boost::optional<callback_t<ObjectID, std::vector<uint64_t>,
                             std::vector<uint64_t>, std::vector<size_t>, int>>
      auto_reader_ = boost::none;

  boost::optional<callback_t<ObjectID, int>> auto_reader_test_ = boost::none;
  boost::optional<std::pair<size_t, callback_t<ObjectID>>> writer_;
  bool drained{false}, failed{false};
  int64_t open_mark{0};
  STREAM_TYPE type{STREAM_TYPE::NOMAL_STREAM};
  int blob_nums;
  int pushed_nums = 0;
  size_t blob_size;
  int read_index;
  bool abort = false;
  bool transfer_finished = true;
  std::string reader_owner = "";
  std::string writer_owner = "";
  std::vector<std::vector<uint64_t>> receive_addr_list;
  std::vector<std::vector<uint64_t>> rkeys_list;
  std::vector<std::vector<size_t>> sizes_list;

  ObjectID bind_stream_id;
  std::string name;
  std::string endpoint;
  std::shared_ptr<RemoteClient> remote_client;
  bool is_forked = false;
  std::set<int> blob_received;
  int blob_received_nums = 0;
  uint64_t ttl = UINT64_MAX;
  uint64_t start_time = 0;
  // pointer to the memory region of the received flag array
  int recv_mem_fd = -1;
  void* recv_mem_base;

  void SetFixedBlobStream(int nums, size_t size) {
    type = FIXED_SIZE_STREAM;
    blob_nums = nums;
    blob_size = size;
    read_index = 0;
  }
};

class DeferredStream {
 public:
  using alive_t = std::function<bool()>;
  using test_t = std::function<bool(Status& status, ObjectID& ret_id)>;
  using call_t = std::function<void(Status& status, ObjectID id)>;

  DeferredStream(alive_t alive_fn, test_t test_fn, call_t call_fn)
      : alive_fn_(alive_fn), test_fn_(test_fn), call_fn_(call_fn) {}

  bool Alive() const;

  bool TestThenCall() const;

 private:
  alive_t alive_fn_;
  test_t test_fn_;
  call_t call_fn_;
};

/**
 * @brief StreamStore manages a pool of streams.
 *
 */
class StreamStore : public std::enable_shared_from_this<StreamStore> {
 public:
  StreamStore(std::shared_ptr<VineyardServer> server,
              std::shared_ptr<BulkStore> store, size_t const stream_threshold,
              boost::asio::io_context& context)
      : server_(server),
        store_(store),
        threshold_(stream_threshold),
        timer_(context, boost::asio::chrono::milliseconds(timer_millseconds_)) {
    ProcessDefered();
  }

  ~StreamStore() {
    std::lock_guard<std::recursive_mutex> guard(deferred_mutex_);
    deferred_.clear();
  }

  Status Create(ObjectID const stream_id, bool fixed_size = false, int nums = 0,
                size_t size = 0);

  Status PutName(std::string name, ObjectID stream_id);

  Status GetStreamIDByName(std::string name, ObjectID& stream_id);

  Status Open(ObjectID const stream_id, int64_t const mode, std::string owner);

  Status Open(std::string name, ObjectID& ret_id, int64_t const mode,
              std::string owner);

  Status Open(ObjectID const stream_id, int64_t const mode, std::string owner,
              bool wait, uint64_t timeout,
              void_callback_t<Status&, ObjectID> callback);

  Status Open(std::string stream_name, int64_t const mode, std::string owner,
              bool wait, uint64_t timeout,
              void_callback_t<Status&, ObjectID> callback);

  Status BindRemoteStream(ObjectID local_stream_id, ObjectID remote_stream_id,
                          std::string endpoint,
                          std::shared_ptr<RemoteClient> client);

  Status UnbindRemoteStream(ObjectID local_stream_id);

  Status SetErrorFlag(ObjectID const stream_id, Status const error);

  void PrintStreamInfo() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    VLOG(100) << "-----------------";
    VLOG(100) << "stream_name_list:";
    for (auto item : stream_names_) {
      VLOG(100) << "stream_name: " << item.first
                << " stream_id: " << item.second;
    }
    VLOG(100) << "stream_list:";
    for (auto item : streams_) {
      VLOG(100) << "stream_id: " << item.first
                << " stream_name: " << item.second->name;
    }
    static uint64_t last_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    if (now - last_time > SECOND_TO_MILLISECOND(3)) {
      LOG(INFO) << "Currently activate stream size:" << streams_.size();
      last_time = now;
    }
  }

  /**
   * @brief This is called by the producer of the stream and it makes current
   * chunk available for the consumer to read
   *
   * @return the next chunk to write
   */
  Status Get(ObjectID const stream_id, size_t const size,
             callback_t<const ObjectID> callback);

  /**
   * @brief This is called by the producer of the stream to emplace a chunk to
   * the ready queue.
   */
  Status Push(ObjectID const stream_id, ObjectID const chunk,
              callback_t<const ObjectID> callback);

  /**
   * @brief The consumer invokes this function to read current chunk
   *
   */
  Status Pull(ObjectID const stream_id, callback_t<const ObjectID> callback);

  void ActivateRemoteFixedStream(
      ObjectID stream_id, std::vector<std::vector<uint64_t>> recv_addr_list,
      std::vector<std::vector<uint64_t>> rkeys,
      std::vector<std::vector<size_t>> sizes_list,
      callback_t<ObjectID, std::vector<uint64_t>, std::vector<uint64_t>,
                 std::vector<size_t>, int>
          callback);

  void ActivateRemoteFixedStream(ObjectID stream_id,
                                 callback_t<ObjectID, int> callback);

  Status CheckBlobReceived(ObjectID stream_id, int index, bool& finished);

  Status SetBlobReceived(ObjectID stream_id, int index);

  Status GetRemoteInfo(ObjectID stream_id, ObjectID& remote_stream_id,
                       std::string& endpoint,
                       std::shared_ptr<RemoteClient>& client) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    auto stream = streams_.find(stream_id);
    if (stream == streams_.end()) {
      return Status::ObjectNotExists("stream not found");
    }
    if (stream->second->is_forked) {
      remote_stream_id = stream->second->bind_stream_id;
      endpoint = stream->second->endpoint;
      client = stream->second->remote_client;
      return Status::OK();
    } else {
      return Status::Invalid("stream is not forked");
    }
  }

  Status GetFixedStreamSizeInfo(ObjectID stream_id, size_t& size, int& nums) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    auto stream = streams_.find(stream_id);
    if (stream == streams_.end()) {
      return Status::ObjectNotExists("stream not found");
    }
    if (stream->second->type == FIXED_SIZE_STREAM) {
      size = stream->second->blob_size;
      nums = stream->second->blob_nums;
      return Status::OK();
    } else {
      return Status::Invalid("stream is not fixed size stream");
    }
  }

  Status ProcessDefered() {
    std::lock_guard<std::recursive_mutex> guard(deferred_mutex_);
    auto iter = deferred_.begin();
    while (iter != deferred_.end()) {
      if (iter->TestThenCall()) {
        VLOG(100) << "Remove from defered stream";
        iter = deferred_.erase(iter);
      } else {
        VLOG(100) << "Keep defered stream";
        ++iter;
      }
    }

    timer_.expires_after(boost::asio::chrono::milliseconds(1));
    timer_.async_wait([this](const boost::system::error_code& ec) {
      if (ec) {
        LOG(ERROR) << "Timer error: " << ec.message();
        return;
      }
      this->ProcessDefered();
    });
    return Status::OK();
  }

  /**
   * @brief Function stop is called by the vineyard clients.
   *
   */
  Status Stop(ObjectID const stream_id, bool failed);

  /**
   * @brief Function Drop is called by vineyard when the clients loose
   * connections
   *
   */
  Status Drop(ObjectID const stream_id);

  Status Close(ObjectID const stream_id, std::string owner);

  Status Delete(ObjectID const stream_id);

  Status CleanResource(std::string owner);

  Status Abort(ObjectID const stream_id, bool& success);

  Status IsFixedStreamTransferFinished(ObjectID const stream_id,
                                       bool& finished) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (streams_.find(stream_id) == streams_.end()) {
      return Status::ObjectNotExists("stream not found");
    }
    auto stream = streams_.at(stream_id);
    if (stream->type == FIXED_SIZE_STREAM) {
      finished = stream->blob_received_nums == stream->blob_nums;
      return Status::OK();
    } else {
      return Status::Invalid("stream is not fixed size stream");
    }
  }

  std::string static BuildOwner(std::string host, int conn_id) {
    return host + ":" + std::to_string(conn_id);
  }

  Status GetRecvFd(ObjectID const stream_id, int& fd) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    auto stream = streams_.find(stream_id);
    if (stream == streams_.end()) {
      return Status::ObjectNotExists("stream not found");
    }
    if (stream->second->recv_mem_fd == -1) {
      return Status::Invalid("stream mmap not ready");
    }
    fd = stream->second->recv_mem_fd;
    return Status::OK();
  }

 private:
  bool allocatable(std::shared_ptr<StreamHolder> stream, size_t size);

  void AutoRead(std::shared_ptr<StreamHolder> stream,
                callback_t<ObjectID, std::vector<uint64_t>,
                           std::vector<uint64_t>, std::vector<size_t>, int>
                    callback);

  void AutoRead(std::shared_ptr<StreamHolder> stream,
                callback_t<ObjectID, int> callback);

  bool IsStreamFinished(std::shared_ptr<StreamHolder> stream) {
    return stream->blob_received_nums == stream->blob_nums;
  }

  // protect the stream store
  std::recursive_mutex mutex_;

  std::shared_ptr<VineyardServer> server_;
  std::shared_ptr<BulkStore> store_;
  size_t threshold_;
  std::unordered_map<ObjectID, std::shared_ptr<StreamHolder>> streams_;
  std::unordered_map<std::string, ObjectID> stream_names_;
  uint64_t timer_millseconds_ = 1;

  boost::asio::steady_timer timer_;

  std::list<DeferredStream> deferred_;
  std::recursive_mutex deferred_mutex_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_STREAM_STORE_H_
