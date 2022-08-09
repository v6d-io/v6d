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

#ifndef SRC_SERVER_MEMORY_STREAM_STORE_H_
#define SRC_SERVER_MEMORY_STREAM_STORE_H_

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <utility>

#include "boost/optional/optional.hpp"

#include "common/util/callback.h"
#include "server/memory/memory.h"

namespace vineyard {

// forward declarations.
class VineyardServer;

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
  boost::optional<std::pair<size_t, callback_t<ObjectID>>> writer_;
  bool drained{false}, failed{false};
  int64_t open_mark{0};
};

/**
 * @brief StreamStore manages a pool of streams.
 *
 */
class StreamStore {
 public:
  StreamStore(std::shared_ptr<VineyardServer> server,
              std::shared_ptr<BulkStore> store, size_t const stream_threshold)
      : server_(server), store_(store), threshold_(stream_threshold) {}

  Status Create(ObjectID const stream_id);

  Status Open(ObjectID const stream_id, int64_t const mode);

  /**
   * @brief This is called by the producer of the steram and it makes current
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

 private:
  bool allocatable(std::shared_ptr<StreamHolder> stream, size_t size);

  // protect the stream store
  std::recursive_mutex mutex_;

  std::shared_ptr<VineyardServer> server_;
  std::shared_ptr<BulkStore> store_;
  size_t threshold_;
  std::unordered_map<ObjectID, std::shared_ptr<StreamHolder>> streams_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_MEMORY_STREAM_STORE_H_
