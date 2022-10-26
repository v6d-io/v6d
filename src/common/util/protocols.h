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

#ifndef SRC_COMMON_UTIL_PROTOCOLS_H_
#define SRC_COMMON_UTIL_PROTOCOLS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/memory/gpu/unified_memory.h"
#include "common/memory/payload.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

enum class CommandType {
  DebugCommand = -1,
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
  MigrateObjectRequest = 29,
  CreateRemoteBufferRequest = 30,
  GetRemoteBuffersRequest = 31,
  DropBufferRequest = 32,
  MakeArenaRequest = 33,
  FinalizeArenaRequest = 34,
  ClearRequest = 36,
  PushNextStreamChunkRequest = 37,
  NewSessionRequest = 38,
  NewSessionReply = 39,
  DeleteSessionRequest = 40,
  DeleteSessionReply = 41,
  SealRequest = 42,
  CreateBufferByPlasmaRequest = 43,
  CreateBufferByPlasmaReply = 44,
  GetBuffersByPlasmaRequest = 45,
  GetBuffersByPlasmaReply = 46,
  PlasmaSealRequest = 47,
  PlasmaReleaseRequest = 48,
  PlasmaDelDataRequest = 49,
  MoveBuffersOwnershipRequest = 50,
  ReleaseRequest = 51,
  DelDataWithFeedbacksRequest = 52,
  IsInUseRequest = 53,
  IncreaseReferenceCountRequest = 54,
  IsSpilledRequest = 55,
  CreateGPUBufferRequest = 56,
  GetGPUBuffersRequest = 57,
  CreateDiskBufferRequest = 58,
  ListNameRequest = 59,
};

enum class StoreType {
  kDefault = 1,
  kPlasma = 2,
};

CommandType ParseCommandType(const std::string& str_type);

void WriteErrorReply(Status const& status, std::string& msg);

void WriteRegisterRequest(std::string& msg, StoreType const& bulk_store_type);

void WriteRegisterRequest(std::string& msg, StoreType const& bulk_store_type,
                          const ObjectID& session_id);

Status ReadRegisterRequest(const json& msg, std::string& version,
                           StoreType& bulk_store_type, SessionID& session_id);

void WriteRegisterReply(const std::string& ipc_socket,
                        const std::string& rpc_endpoint,
                        const InstanceID instance_id,
                        const SessionID session_id, bool& store_match,
                        std::string& msg);

Status ReadRegisterReply(const json& msg, std::string& ipc_socket,
                         std::string& rpc_endpoint, InstanceID& instance_id,
                         SessionID& sessionid, std::string& version,
                         bool& store_match);

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

void WriteListNameRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg);

Status ReadListNameRequest(const json& root, std::string& pattern, bool& regex,
                           size_t& limit);

void WriteListNameReply(std::map<std::string, ObjectID> const& names,
                        std::string& msg);

Status ReadListNameReply(const json& root,
                         std::map<std::string, ObjectID>& names);

void WriteCreateDataRequest(const json& content, std::string& msg);

Status ReadCreateDataRequest(const json& root, json& content);

void WriteCreateDataReply(const ObjectID& id, const Signature& signature,
                          const InstanceID& instance_id, std::string& msg);

Status ReadCreateDataReply(const json& root, ObjectID& id, Signature& signature,
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
                         const bool fastpath, std::string& msg);

void WriteDelDataRequest(const std::vector<ObjectID>& id, const bool force,
                         const bool deep, const bool fastpath,
                         std::string& msg);

Status ReadDelDataRequest(const json& root, std::vector<ObjectID>& id,
                          bool& force, bool& deep, bool& fastpath);

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
                            const int fd_to_send, std::string& msg);

Status ReadCreateBufferReply(const json& root, ObjectID& id, Payload& object,
                             int& fd_sent);

void WriteCreateGPUBufferRequest(const size_t size, std::string& msg);

Status ReadCreateGPUBufferRequest(const json& root, size_t& size);

void WriteGPUCreateBufferReply(const ObjectID id,
                               const std::shared_ptr<Payload>& object,
                               GPUUnifiedAddress uva, std::string& msg);

Status ReadGPUCreateBufferReply(
    const json& root, ObjectID& id, Payload& Object,
    std::shared_ptr<vineyard::GPUUnifiedAddress> uva);

void WriteCreateDiskBufferRequest(const size_t size, const std::string& path,
                                  std::string& msg);

Status ReadCreateDiskBufferRequest(const json& root, size_t& size,
                                   std::string& path);

void WriteCreateDiskBufferReply(const ObjectID id,
                                const std::shared_ptr<Payload>& object,
                                const int fd_to_send, std::string& msg);

Status ReadCreateDiskBufferReply(const json& root, ObjectID& id,
                                 Payload& object, int& fd_sent);

void WriteGetGPUBuffersRequest(const std::set<ObjectID>& ids, const bool unsafe,
                               std::string& msg);

Status ReadGetGPUBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                                bool& unsafe);

void WriteGetGPUBuffersReply(
    const std::vector<std::shared_ptr<Payload>>& objects,
    const std::vector<std::vector<int64_t>>& handle_to_send, std::string& msg);

Status ReadGetGPUBuffersReply(const json& root, std::vector<Payload>& objects,
                              std::vector<GPUUnifiedAddress>& uva_sent);

void WriteCreateRemoteBufferRequest(const size_t size, std::string& msg);

void WriteCreateRemoteBufferRequest(const size_t size, const bool compress,
                                    std::string& msg);

Status ReadCreateRemoteBufferRequest(const json& root, size_t& size,
                                     bool& compress);

void WriteGetBuffersRequest(const std::set<ObjectID>& ids, const bool unsafe,
                            std::string& msg);

void WriteGetBuffersRequest(const std::unordered_set<ObjectID>& ids,
                            const bool unsafe, std::string& msg);

Status ReadGetBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                             bool& unsafe);

void WriteGetBuffersReply(const std::vector<std::shared_ptr<Payload>>& objects,
                          const std::vector<int>& fd_to_send,
                          const bool compress, std::string& msg);

Status ReadGetBuffersReply(const json& root, std::vector<Payload>& objects,
                           std::vector<int>& fd_sent);

Status ReadGetBuffersReply(const json& root, std::vector<Payload>& objects,
                           std::vector<int>& fd_sent, bool& compress);

void WriteGetRemoteBuffersRequest(const std::set<ObjectID>& ids,
                                  const bool unsafe, std::string& msg);

void WriteGetRemoteBuffersRequest(const std::unordered_set<ObjectID>& ids,
                                  const bool unsafe, std::string& msg);

void WriteGetRemoteBuffersRequest(const std::set<ObjectID>& ids,
                                  const bool unsafe, const bool compress,
                                  std::string& msg);

void WriteGetRemoteBuffersRequest(const std::unordered_set<ObjectID>& ids,
                                  const bool unsafe, const bool compress,
                                  std::string& msg);

Status ReadGetRemoteBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                                   bool& unsafe, bool& compress);

void WriteDropBufferRequest(const ObjectID id, std::string& msg);

Status ReadDropBufferRequest(const json& root, ObjectID& id);

void WriteDropBufferReply(std::string& msg);

Status ReadDropBufferReply(const json& root);

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

void WriteMigrateObjectRequest(const ObjectID object_id, std::string& msg);

Status ReadMigrateObjectRequest(const json& root, ObjectID& object_id);

void WriteMigrateObjectRequest(const ObjectID object_id, const bool local,
                               const bool is_stream, std::string const& peer,
                               std::string const& peer_rpc_endpoint,
                               std::string& msg);

Status ReadMigrateObjectRequest(const json& root, ObjectID& object_id,
                                bool& local, bool& is_stream, std::string& peer,
                                std::string& peer_rpc_endpoint);

void WriteMigrateObjectReply(const ObjectID& object_id, std::string& msg);

Status ReadMigrateObjectReply(const json& root, ObjectID& object_id);

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

void WriteGetNextStreamChunkReply(std::shared_ptr<Payload> const& object,
                                  int fd_to_send, std::string& msg);

Status ReadGetNextStreamChunkReply(const json& root, Payload& object,
                                   int& fd_sent);

void WritePushNextStreamChunkRequest(const ObjectID stream_id,
                                     const ObjectID chunk, std::string& msg);

Status ReadPushNextStreamChunkRequest(const json& root, ObjectID& stream_id,
                                      ObjectID& chunk);

void WritePushNextStreamChunkReply(std::string& msg);

Status ReadPushNextStreamChunkReply(const json& root);

void WritePullNextStreamChunkRequest(const ObjectID stream_id,
                                     std::string& msg);

Status ReadPullNextStreamChunkRequest(const json& root, ObjectID& stream_id);

void WritePullNextStreamChunkReply(ObjectID const chunk, std::string& msg);

Status ReadPullNextStreamChunkReply(const json& root, ObjectID& chunk);

void WriteStopStreamRequest(const ObjectID stream_id, const bool failed,
                            std::string& msg);

Status ReadStopStreamRequest(const json& root, ObjectID& stream_id,
                             bool& failed);

void WriteStopStreamReply(std::string& msg);

Status ReadStopStreamReply(const json& root);

void WriteShallowCopyRequest(const ObjectID id, std::string& msg);

void WriteShallowCopyRequest(const ObjectID id, json const& extra_metadata,
                             std::string& msg);

Status ReadShallowCopyRequest(const json& root, ObjectID& id,
                              json& extra_metadata);

void WriteShallowCopyReply(const ObjectID target_id, std::string& msg);

Status ReadShallowCopyReply(const json& root, ObjectID& target_id);

void WriteMakeArenaRequest(const size_t size, std::string& msg);

Status ReadMakeArenaRequest(const json& root, size_t& size);

void WriteMakeArenaReply(const int fd, const size_t size, const uintptr_t base,
                         std::string& msg);

Status ReadMakeArenaReply(const json& root, int& fd, size_t& size,
                          uintptr_t& base);

void WriteFinalizeArenaRequest(const int fd, std::vector<size_t> const& offsets,
                               std::vector<size_t> const& sizes,
                               std::string& msg);

Status ReadFinalizeArenaRequest(const json& root, int& fd,
                                std::vector<size_t>& offsets,
                                std::vector<size_t>& sizes);

void WriteFinalizeArenaReply(std::string& msg);

Status ReadFinalizeArenaReply(const json& root);

void WriteClearRequest(std::string& msg);

Status ReadClearRequest(const json& root);

void WriteClearReply(std::string& msg);

Status ReadClearReply(const json& root);

void WriteDebugRequest(const json& debug, std::string& msg);

Status ReadDebugRequest(const json& root, json& debug);

void WriteDebugReply(const json& result, std::string& msg);

Status ReadDebugReply(const json& root, json& result);

void WriteNewSessionRequest(std::string& msg, StoreType const& bulk_store_type);

Status ReadNewSessionRequest(json const& root, StoreType& bulk_store_type);

void WriteNewSessionReply(std::string& msg, std::string const& socket_path);

Status ReadNewSessionReply(json const& root, std::string& socket_path);

void WriteDeleteSessionRequest(std::string& msg);

void WriteDeleteSessionReply(std::string& msg);

void WriteCreateBufferByPlasmaRequest(PlasmaID const plasma_id,
                                      size_t const size,
                                      size_t const plasma_size,
                                      std::string& msg);

Status ReadCreateBufferByPlasmaRequest(json const& root, PlasmaID& plasma_id,
                                       size_t& size, size_t& plasma_size);

void WriteCreateBufferByPlasmaReply(
    ObjectID const object_id,
    const std::shared_ptr<PlasmaPayload>& plasma_object, int fd_to_send,
    std::string& msg);

Status ReadCreateBufferByPlasmaReply(json const& root, ObjectID& object_id,
                                     PlasmaPayload& plasma_object,
                                     int& fd_sent);

void WriteGetBuffersByPlasmaRequest(std::set<PlasmaID> const& plasma_ids,
                                    const bool unsafe, std::string& msg);

Status ReadGetBuffersByPlasmaRequest(json const& root,
                                     std::vector<PlasmaID>& plasma_ids,
                                     bool& unsafe);

void WriteGetBuffersByPlasmaReply(
    std::vector<std::shared_ptr<PlasmaPayload>> const& plasma_objects,
    std::string& msg);

Status ReadGetBuffersByPlasmaReply(json const& root,
                                   std::vector<PlasmaPayload>& plasma_objects);

void WriteSealRequest(ObjectID const& object_id, std::string& message_out);

Status ReadSealRequest(json const& root, ObjectID& object_id);

void WritePlasmaSealRequest(PlasmaID const& plasma_id,
                            std::string& message_out);

Status ReadPlasmaSealRequest(json const& root, PlasmaID& plasma_id);

void WriteSealReply(std::string& msg);

Status ReadSealReply(json const& root);

void WritePlasmaReleaseRequest(PlasmaID const& plasma_id,
                               std::string& message_out);

Status ReadPlasmaReleaseRequest(json const& root, PlasmaID& plasma_id);

void WritePlasmaReleaseReply(std::string& msg);

Status ReadPlasmaReleaseReply(json const& root);

void WritePlasmaReleaseReply(std::string& msg);

Status ReadPlasmaReleaseReply(json const& root);

void WritePlasmaDelDataRequest(PlasmaID const& plasma_id,
                               std::string& message_out);

Status ReadPlasmaDelDataRequest(json const& root, PlasmaID& plasma_id);

void WritePlasmaDelDataReply(std::string& msg);

Status ReadPlasmaDelDataReply(json const& root);

/// normal -> normal
void WriteMoveBuffersOwnershipRequest(
    std::map<ObjectID, ObjectID> const& id_to_id, SessionID const session_id,
    std::string& msg);

/// normal -> plasma
void WriteMoveBuffersOwnershipRequest(
    std::map<PlasmaID, ObjectID> const& pid_to_id, SessionID const session_id,
    std::string& msg);

/// plasma -> normal
void WriteMoveBuffersOwnershipRequest(
    std::map<ObjectID, PlasmaID> const& id_to_pid, SessionID const session_id,
    std::string& msg);

/// plasma -> plasma
void WriteMoveBuffersOwnershipRequest(
    std::map<PlasmaID, PlasmaID> const& pid_to_pid, SessionID const session_id,
    std::string& msg);

Status ReadMoveBuffersOwnershipRequest(json const& root,
                                       std::map<ObjectID, ObjectID>& id_to_id,
                                       std::map<PlasmaID, ObjectID>& pid_to_id,
                                       std::map<ObjectID, PlasmaID>& id_to_pid,
                                       std::map<PlasmaID, PlasmaID>& pid_to_pid,
                                       SessionID& session_id);

void WriteMoveBuffersOwnershipReply(std::string& msg);

Status ReadMoveBuffersOwnershipReply(json const& root);

void WriteReleaseRequest(ObjectID const& object_id, std::string& msg);

Status ReadReleaseRequest(json const& root, ObjectID& object_id);

void WriteReleaseReply(std::string& msg);

Status ReadReleaseReply(json const& root);

void WriteDelDataWithFeedbacksRequest(const std::vector<ObjectID>& id,
                                      const bool force, const bool deep,
                                      const bool fastpath, std::string& msg);

Status ReadDelDataWithFeedbacksRequest(json const& root,
                                       std::vector<ObjectID>& id, bool& force,
                                       bool& deep, bool& fastpath);

void WriteDelDataWithFeedbacksReply(const std::vector<ObjectID>& deleted_bids,
                                    std::string& msg);

Status ReadDelDataWithFeedbacksReply(json const& root,
                                     std::vector<ObjectID>& deleted_bids);

void WriteIsInUseRequest(const ObjectID& id, std::string& msg);

Status ReadIsInUseRequest(json const& root, ObjectID& id);

void WriteIsInUseReply(const bool is_in_use, std::string& msg);

Status ReadIsInUseReply(json const& root, bool& is_in_use);

void WriteIsSpilledRequest(const ObjectID& id, std::string& msg);

Status ReadIsSpilledRequest(json const& root, ObjectID& id);

void WriteIsSpilledReply(const bool is_spilled, std::string& msg);

Status ReadIsSpilledReply(json const& root, bool& is_spilled);

void WriteIncreaseReferenceCountRequest(const std::vector<ObjectID>& ids,
                                        std::string& msg);

Status ReadIncreaseReferenceCountRequest(json const& root,
                                         std::vector<ObjectID>& ids);

void WriteIncreaseReferenceCountReply(std::string& msg);

Status ReadIncreaseReferenceCountReply(json const& root);

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_PROTOCOLS_H_
