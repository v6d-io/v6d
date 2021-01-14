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

#ifndef SRC_COMMON_UTIL_PROTOCOLS_H_
#define SRC_COMMON_UTIL_PROTOCOLS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/memory/payload.h"
#include "common/util/boost.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

enum class CommandType {
  NullCommand = 0,
  ExitRequest = 1,
  ExitReply = 2,
  RegisterRequest = 3,
  RegisterReply = 4,
  GetDataRequest = 5,
  GetDataReply = 6,
  PersistRequest = 8,
  ExistsRequest = 9,
  DelDataRequest = 10,
  ClusterMetaRequest = 11,
  ListDataRequest = 12,
  CreateBufferRequest = 13,
  GetBuffersRequest = 14,
  CreateDataRequest = 15,
  PutNameRequest = 16,
  GetNameRequest = 17,
  DropNameRequest = 18,
  CreateStreamRequest = 19,
  GetNextStreamChunkRequest = 20,
  PullNextStreamChunkRequest = 21,
  StopStreamRequest = 22,
  IfPersistRequest = 25,
  InstanceStatusRequest = 26,
  ShallowCopyRequest = 27,
  OpenStreamRequest = 28,
};

CommandType ParseCommandType(const std::string& str_type);

void WriteErrorReply(Status const& status, std::string& msg);

void WriteRegisterRequest(std::string& msg);

Status ReadRegisterRequest(const json& msg, std::string& version);

void WriteRegisterReply(const std::string& ipc_socket,
                        const std::string& rpc_endpoint,
                        const InstanceID instance_id, std::string& msg);

Status ReadRegisterReply(const json& msg, std::string& ipc_socket,
                         std::string& rpc_endpoint, InstanceID& instance_id,
                         std::string& version);

void WriteExitRequest(std::string& msg);

void WriteGetDataRequest(const ObjectID id, const bool sync_remote,
                         const bool wait, std::string& msg);

void WriteGetDataRequest(const std::vector<ObjectID>& ids,
                         const bool sync_remote, const bool wait,
                         std::string& msg);

Status ReadGetDataRequest(const json& root, std::vector<ObjectID>& ids,
                          bool& sync_remote, bool& wait);

void WriteGetDataReply(const json& content, std::string& msg);

Status ReadGetDataReply(const json& root, json& content);

Status ReadGetDataReply(const json& root,
                        std::unordered_map<ObjectID, json>& content);

void WriteListDataRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg);

Status ReadListDataRequest(const json& root, std::string& pattern, bool& regex,
                           size_t& limit);

void WriteCreateDataRequest(const json& content, std::string& msg);

Status ReadCreateDataRequest(const json& root, json& content);

void WriteCreateDataReply(const ObjectID& id, const Signature& sigature,
                          const InstanceID& instance_id, std::string& msg);

Status ReadCreateDataReply(const json& root, ObjectID& id, Signature& sigature,
                           InstanceID& instance_id);

void WritePersistRequest(const ObjectID id, std::string& msg);

Status ReadPersistRequest(const json& root, ObjectID& id);

void WritePersistReply(std::string& msg);

Status ReadPersistReply(const json& root);

void WriteIfPersistRequest(const ObjectID id, std::string& msg);

Status ReadIfPersistRequest(const json& root, ObjectID& id);

void WriteIfPersistReply(bool exists, std::string& msg);

Status ReadIfPersistReply(const json& root, bool& persist);

void WriteExistsRequest(const ObjectID id, std::string& msg);

Status ReadExistsRequest(const json& root, ObjectID& id);

void WriteExistsReply(bool exists, std::string& msg);

Status ReadExistsReply(const json& root, bool& exists);

void WriteDelDataRequest(const ObjectID id, const bool force, const bool deep,
                         std::string& msg);

void WriteDelDataRequest(const std::vector<ObjectID>& id, const bool force,
                         const bool deep, std::string& msg);

Status ReadDelDataRequest(const json& root, std::vector<ObjectID>& id,
                          bool& force, bool& deep);

void WriteDelDataReply(std::string& msg);

Status ReadDelDataReply(const json& root);

void WriteClusterMetaRequest(std::string& msg);

Status ReadClusterMetaRequest(const json& root);

void WriteClusterMetaReply(const json& content, std::string& msg);

Status ReadClusterMetaReply(const json& root, json& content);

void WriteInstanceStatusRequest(std::string& msg);

Status ReadInstanceStatusRequest(const json& root);

void WriteInstanceStatusReply(const json& content, std::string& msg);

Status ReadInstanceStatusReply(const json& root, json& content);

void WriteCreateBufferRequest(const size_t size, std::string& msg);

Status ReadCreateBufferRequest(const json& root, size_t& size);

void WriteCreateBufferReply(const ObjectID id,
                            const std::shared_ptr<Payload>& object,
                            std::string& msg);

Status ReadCreateBufferReply(const json& root, ObjectID& id, Payload& object);

void WriteGetBuffersRequest(const std::unordered_set<ObjectID>& ids,
                            std::string& msg);

Status ReadGetBuffersRequest(const json& root, std::vector<ObjectID>& ids);

void WriteGetBuffersReply(const std::vector<std::shared_ptr<Payload>>& objects,
                          std::string& msg);

Status ReadGetBuffersReply(const json& root,
                           std::unordered_map<ObjectID, Payload>& objects);

void WritePutNameRequest(const ObjectID object_id, const std::string& name,
                         std::string& msg);

Status ReadPutNameRequest(const json& root, ObjectID& object_id,
                          std::string& name);

void WritePutNameReply(std::string& msg);

Status ReadPutNameReply(const json& root);

void WriteGetNameRequest(const std::string& name, const bool wait,
                         std::string& msg);

Status ReadGetNameRequest(const json& root, std::string& name, bool& wait);

void WriteGetNameReply(const ObjectID& object_id, std::string& msg);

Status ReadGetNameReply(const json& root, ObjectID& object_id);

void WriteDropNameRequest(const std::string& name, std::string& msg);

Status ReadDropNameRequest(const json& root, std::string& name);

void WriteDropNameReply(std::string& msg);

Status ReadDropNameReply(const json& root);

void WriteCreateStreamRequest(const ObjectID& object_id, std::string& msg);

Status ReadCreateStreamRequest(const json& root, ObjectID& object_id);

void WriteCreateStreamReply(std::string& msg);

Status ReadCreateStreamReply(const json& root);

void WriteOpenStreamRequest(const ObjectID& object_id, const int64_t& mode,
                            std::string& msg);

Status ReadOpenStreamRequest(const json& root, ObjectID& object_id,
                             int64_t& mode);

void WriteOpenStreamReply(std::string& msg);

Status ReadOpenStreamReply(const json& root);

void WriteGetNextStreamChunkRequest(const ObjectID stream_id, const size_t size,
                                    std::string& msg);

Status ReadGetNextStreamChunkRequest(const json& root, ObjectID& stream_id,
                                     size_t& size);

void WriteGetNextStreamChunkReply(std::shared_ptr<Payload>& object,
                                  std::string& msg);

Status ReadGetNextStreamChunkReply(const json& root, Payload& object);

void WritePullNextStreamChunkRequest(const ObjectID stream_id,
                                     std::string& msg);

Status ReadPullNextStreamChunkRequest(const json& root, ObjectID& stream_id);

void WritePullNextStreamChunkReply(std::shared_ptr<Payload>& object,
                                   std::string& msg);

Status ReadPullNextStreamChunkReply(const json& root, Payload& object);

void WriteStopStreamRequest(const ObjectID stream_id, const bool failed,
                            std::string& msg);

Status ReadStopStreamRequest(const json& root, ObjectID& stream_id,
                             bool& failed);

void WriteStopStreamReply(std::string& msg);

Status ReadStopStreamReply(const json& root);

void WriteShallowCopyRequest(const ObjectID id, std::string& msg);

Status ReadShallowCopyRequest(const json& root, ObjectID& id);

void WriteShallowCopyReply(const ObjectID target_id, std::string& msg);

Status ReadShallowCopyReply(const json& root, ObjectID& target_id);

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_PROTOCOLS_H_
