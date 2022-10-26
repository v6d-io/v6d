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

#ifndef SRC_SERVER_UTIL_REMOTE_H_
#define SRC_SERVER_UTIL_REMOTE_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/memory/payload.h"
#include "common/util/asio.h"
#include "common/util/callback.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

class VineyardServer;

class RemoteClient : public std::enable_shared_from_this<RemoteClient> {
 public:
  explicit RemoteClient(const std::shared_ptr<VineyardServer> vs_ptr);
  ~RemoteClient();

  Status Connect(const std::string& rpc_endpoint, const SessionID session_id);

  Status Connect(const std::string& host, const uint32_t port,
                 const SessionID session_id);

  Status MigrateObject(const ObjectID object_id, const json& meta,
                       callback_t<const ObjectID> callback);

 private:
  Status migrateBuffers(
      const std::set<ObjectID> blobs,
      callback_t<const std::map<ObjectID, ObjectID>&> results);

  Status collectRemoteBlobs(const json& tree, std::set<ObjectID>& blobs);

  Status recreateMetadata(json const& metadata, json& target,
                          std::map<ObjectID, ObjectID> const& result_blobs);

 private:
  Status doWrite(const std::string& message_out);

  Status doRead(std::string& message_in);

  Status doRead(json& root);
  InstanceID remote_instance_id_;

  std::shared_ptr<VineyardServer> server_ptr_;
  asio::io_context& context_;
  asio::ip::tcp::socket remote_tcp_socket_;
  asio::generic::stream_protocol::socket socket_;
  bool connected_;
};

/**
 * Notes on [Transferring remote blobs]
 *
 * The protocol work as:
 *
 *  - if compression is disabled, each blob is sent one by one and blobs
 *    that are empty are skipped.
 *
 *  - if compression is enabled, each blob will be compressed as several
 *    chunks, and each chunk will be sent as [chunk_size, chunk], where
 *    the chunk_size is a `size_t`.
 */

void SendRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                       std::vector<std::shared_ptr<Payload>> const& objects,
                       size_t index, const bool compress,
                       callback_t<> callback_after_finish);

void ReceiveRemoteBuffers(asio::generic::stream_protocol::socket& socket,
                          std::vector<std::shared_ptr<Payload>> const& objects,
                          size_t index, size_t offset, const bool decompress,
                          callback_t<> callback_after_finish);

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_REMOTE_H_
