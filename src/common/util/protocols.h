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

#ifndef SRC_COMMON_UTIL_PROTOCOLS_H_
#define SRC_COMMON_UTIL_PROTOCOLS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/memory/payload.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

struct command_t {
  // Connecting APIs
  static const std::string REGISTER_REQUEST;
  static const std::string REGISTER_REPLY;
  static const std::string EXIT_REQUEST;
  static const std::string EXIT_REPLY;
  static const std::string REQUIRE_EXTRA_REQUEST_MEMORY_REQUEST;
  static const std::string REQUIRE_EXTRA_REQUEST_MEMORY_REPLY;

  // Blobs APIs
  static const std::string CREATE_BUFFER_REQUEST;
  static const std::string CREATE_BUFFER_REPLY;
  static const std::string CREATE_BUFFERS_REQUEST;
  static const std::string CREATE_BUFFERS_REPLY;
  static const std::string CREATE_DISK_BUFFER_REQUEST;
  static const std::string CREATE_DISK_BUFFER_REPLY;
  static const std::string CREATE_GPU_BUFFER_REQUEST;
  static const std::string CREATE_GPU_BUFFER_REPLY;
  static const std::string SEAL_BUFFER_REQUEST;
  static const std::string SEAL_BUFFER_REPLY;
  static const std::string GET_BUFFERS_REQUEST;
  static const std::string GET_BUFFERS_REPLY;
  static const std::string GET_GPU_BUFFERS_REQUEST;
  static const std::string GET_GPU_BUFFERS_REPLY;
  static const std::string DROP_BUFFER_REQUEST;
  static const std::string DROP_BUFFER_REPLY;
  static const std::string SHRINK_BUFFER_REQUEST;
  static const std::string SHRINK_BUFFER_REPLY;
  static const std::string GET_USER_BUFFERS_REQUEST;
  static const std::string GET_USER_BUFFERS_REPLY;

  static const std::string REQUEST_FD_REQUEST;
  static const std::string REQUEST_FD_REPLY;

  static const std::string CREATE_REMOTE_BUFFER_REQUEST;
  static const std::string CREATE_REMOTE_BUFFERS_REQUEST;
  static const std::string GET_REMOTE_BUFFERS_REQUEST;

  static const std::string INCREASE_REFERENCE_COUNT_REQUEST;
  static const std::string INCREASE_REFERENCE_COUNT_REPLY;
  static const std::string RELEASE_REQUEST;
  static const std::string RELEASE_REPLY;
  static const std::string DEL_DATA_WITH_FEEDBACKS_REQUEST;
  static const std::string DEL_DATA_WITH_FEEDBACKS_REPLY;
  static const std::string DEL_HUGE_DATA_REQUEST;
  static const std::string DEL_HUGE_DATA_REPLY;

  static const std::string CREATE_BUFFER_PLASMA_REQUEST;
  static const std::string CREATE_BUFFER_PLASMA_REPLY;
  static const std::string GET_BUFFERS_PLASMA_REQUEST;
  static const std::string GET_BUFFERS_PLASMA_REPLY;
  static const std::string PLASMA_SEAL_REQUEST;
  static const std::string PLASMA_SEAL_REPLY;
  static const std::string PLASMA_RELEASE_REQUEST;
  static const std::string PLASMA_RELEASE_REPLY;
  static const std::string PLASMA_DEL_DATA_REQUEST;
  static const std::string PLASMA_DEL_DATA_REPLY;

  static const std::string CREATE_USER_BUFFERS_REQUEST;
  static const std::string CREATE_USER_BUFFERS_REPLY;
  static const std::string DELETE_USER_BUFFERS_REQUEST;
  static const std::string DELETE_USER_BUFFERS_REPLY;

  static const std::string GET_REMOTE_BLOBS_WITH_RDMA_REQUEST;
  static const std::string GET_REMOTE_BLOBS_WITH_RDMA_REPLY;

  // Metadata APIs
  static const std::string CREATE_DATA_REQUEST;
  static const std::string CREATE_DATA_REPLY;
  static const std::string CREATE_DATAS_REQUEST;
  static const std::string CREATE_DATAS_REPLY;
  static const std::string GET_DATA_REQUEST;
  static const std::string GET_DATA_REPLY;
  static const std::string LIST_DATA_REQUEST;
  static const std::string LIST_DATA_REPLY;
  static const std::string DELETE_DATA_REQUEST;
  static const std::string DELETE_DATA_REPLY;
  static const std::string EXISTS_REQUEST;
  static const std::string EXISTS_REPLY;
  static const std::string PERSIST_REQUEST;
  static const std::string PERSIST_REPLY;
  static const std::string IF_PERSIST_REQUEST;
  static const std::string IF_PERSIST_REPLY;
  static const std::string LABEL_REQUEST;
  static const std::string LABEL_REPLY;
  static const std::string CLEAR_REQUEST;
  static const std::string CLEAR_REPLY;
  static const std::string MEMORY_TRIM_REQUEST;
  static const std::string MEMORY_TRIM_REPLY;
  static const std::string RELEASE_BLOBS_WITH_RDMA_REQUEST;
  static const std::string RELEASE_BLOBS_WITH_RDMA_REPLY;
  static const std::string BATCH_PERSIST_REQUEST;
  static const std::string BATCH_PERSIST_REPLY;
  static const std::string CREATE_HUGE_DATAS_REQUEST;
  static const std::string CREATE_HUGE_DATAS_REPLY;
  static const std::string GET_HUGE_DATA_REQUEST;
  static const std::string GET_HUGE_DATA_REPLY;

  // Stream APIs
  static const std::string CREATE_STREAM_REQUEST;
  static const std::string CREATE_STREAM_REPLY;
  static const std::string OPEN_STREAM_REQUEST;
  static const std::string OPEN_STREAM_REPLY;
  static const std::string GET_NEXT_STREAM_CHUNK_REQUEST;
  static const std::string GET_NEXT_STREAM_CHUNK_REPLY;
  static const std::string PUSH_NEXT_STREAM_CHUNK_REQUEST;
  static const std::string PUSH_NEXT_STREAM_CHUNK_REPLY;
  static const std::string PUSH_NEXT_STREAM_CHUNK_BY_OFFSET_REQUEST;
  static const std::string PUSH_NEXT_STREAM_CHUNK_BY_OFFSET_REPLY;
  static const std::string PULL_NEXT_STREAM_CHUNK_REQUEST;
  static const std::string PULL_NEXT_STREAM_CHUNK_REPLY;
  static const std::string STOP_STREAM_REQUEST;
  static const std::string STOP_STREAM_REPLY;
  static const std::string DROP_STREAM_REQUEST;
  static const std::string DROP_STREAM_REPLY;
  static const std::string ABORT_STREAM_REQUEST;
  static const std::string ABORT_STREAM_REPLY;
  static const std::string PUT_STREAM_NAME_REQUEST;
  static const std::string PUT_STREAM_NAME_REPLY;
  static const std::string GET_STREAM_ID_BY_NAME_REQUEST;
  static const std::string GET_STREAM_ID_BY_NAME_REPLY;
  static const std::string ACTIVATE_REMOTE_FIXED_STREAM_REQUEST;
  static const std::string ACTIVATE_REMOTE_FIXED_STREAM_REPLY;
  static const std::string STREAM_READY_ACK;
  static const std::string CREATE_FIXED_STREAM_REQUEST;
  static const std::string CREATE_FIXED_STREAM_REPLY;
  static const std::string OPEN_FIXED_STREAM_REQUEST;
  static const std::string OPEN_FIXED_STREAM_REPLY;
  static const std::string CLOSE_STREAM_REQUEST;
  static const std::string CLOSE_STREAM_REPLY;
  static const std::string DELETE_STREAM_REQUEST;
  static const std::string DELETE_STREAM_REPLY;
  static const std::string CHECK_FIXED_STREAM_RECEIVED_REQUEST;
  static const std::string CHECK_FIXED_STREAM_RECEIVED_REPLY;

  // sidecar api
  static const std::string VINEYARD_OPEN_REMOTE_FIXED_STREAM_REQUEST;
  static const std::string VINEYARD_OPEN_REMOTE_FIXED_STREAM_REPLY;
  static const std::string VINEYARD_ACTIVATE_REMOTE_FIXED_STREAM_REQUEST;
  static const std::string VINEYARD_ACTIVATE_REMOTE_FIXED_STREAM_REPLY;
  static const std::string
      VINEYARD_ACTIVATE_REMOTE_FIXED_STREAM_WITH_OFFSET_REQUEST;
  static const std::string
      VINEYARD_ACTIVATE_REMOTE_FIXED_STREAM_WITH_OFFSET_REPLY;
  static const std::string VINEYARD_STOP_STREAM_REQUEST;
  static const std::string VINEYARD_STOP_STREAM_REPLY;
  static const std::string VINEYARD_DROP_STREAM_REQUEST;
  static const std::string VINEYARD_DROP_STREAM_REPLY;
  static const std::string VINEYARD_ABORT_REMOTE_STREAM_REQUEST;
  static const std::string VINEYARD_ABORT_REMOTE_STREAM_REPLY;
  static const std::string VINEYARD_STREAM_READY_ACK;
  static const std::string VINEYARD_CLOSE_REMOTE_FIXED_STREAM_REQUEST;
  static const std::string VINEYARD_CLOSE_REMOTE_FIXED_STREAM_REPLY;
  static const std::string VINEYARD_GET_METAS_BY_NAMES_REQUEST;
  static const std::string VINEYARD_GET_METAS_BY_NAMES_REPLY;
  static const std::string VINEYARD_GET_REMOTE_BLOBS_WITH_RDMA_REQUEST;
  static const std::string VINEYARD_GET_REMOTE_BLOBS_WITH_RDMA_REPLY;
  static const std::string VINEYARD_GET_REMOTE_BLOBS_WITH_OFFSET_REQUEST;
  static const std::string VINEYARD_GET_REMOTE_BLOBS_WITH_OFFSET_REPLY;

  // Names APIs
  static const std::string PUT_NAME_REQUEST;
  static const std::string PUT_NAME_REPLY;
  static const std::string GET_NAME_REQUEST;
  static const std::string GET_NAME_REPLY;
  static const std::string LIST_NAME_REQUEST;
  static const std::string LIST_NAME_REPLY;
  static const std::string DROP_NAME_REQUEST;
  static const std::string DROP_NAME_REPLY;
  static const std::string GET_NAME_LOCATION_REQUEST;
  static const std::string GET_NAME_LOCATION_REPLY;
  static const std::string GET_METAS_BY_NAMES_REQUEST;
  static const std::string GET_METAS_BY_NAMES_REPLY;
  static const std::string PUT_NAME_LOCATION_REQUEST;
  static const std::string PUT_NAME_LOCATION_REPLY;
  static const std::string GET_NAMES_REQUEST;
  static const std::string GET_NAMES_REPLY;
  static const std::string DROP_NAMES_REQUEST;
  static const std::string DROP_NAMES_REPLY;
  static const std::string PUT_NAMES_REQUEST;
  static const std::string PUT_NAMES_REPLY;

  // Arena APIs
  static const std::string MAKE_ARENA_REQUEST;
  static const std::string MAKE_ARENA_REPLY;
  static const std::string FINALIZE_ARENA_REQUEST;
  static const std::string FINALIZE_ARENA_REPLY;

  // Session APIs
  static const std::string NEW_SESSION_REQUEST;
  static const std::string NEW_SESSION_REPLY;
  static const std::string DELETE_SESSION_REQUEST;
  static const std::string DELETE_SESSION_REPLY;

  static const std::string MOVE_BUFFERS_OWNERSHIP_REQUEST;
  static const std::string MOVE_BUFFERS_OWNERSHIP_REPLY;

  // Spill APIs
  static const std::string EVICT_REQUEST;
  static const std::string EVICT_REPLY;
  static const std::string LOAD_REQUEST;
  static const std::string LOAD_REPLY;
  static const std::string UNPIN_REQUEST;
  static const std::string UNPIN_REPLY;
  static const std::string IS_SPILLED_REQUEST;
  static const std::string IS_SPILLED_REPLY;
  static const std::string IS_IN_USE_REQUEST;
  static const std::string IS_IN_USE_REPLY;

  // Meta APIs
  static const std::string GET_VINEYARD_MMAP_FD_REQUEST;
  static const std::string GET_VINEYARD_MMAP_FD_REPLY;
  static const std::string CLUSTER_META_REQUEST;
  static const std::string CLUSTER_META_REPLY;
  static const std::string INSTANCE_STATUS_REQUEST;
  static const std::string INSTANCE_STATUS_REPLY;
  static const std::string MIGRATE_OBJECT_REQUEST;
  static const std::string MIGRATE_OBJECT_REPLY;
  static const std::string SHALLOW_COPY_REQUEST;
  static const std::string SHALLOW_COPY_REPLY;
  static const std::string DEBUG_REQUEST;
  static const std::string DEBUG_REPLY;

  // distributed lock
  static const std::string ACQUIRE_LOCK_REQUEST;
  static const std::string ACQUIRE_LOCK_REPLY;
  static const std::string RELEASE_LOCK_REQUEST;
  static const std::string RELEASE_LOCK_REPLY;
};

extern std::map<std::string, int> CommandMap;

enum class StoreType {
  kDefault = 1,
  kPlasma = 2,
};

void WriteErrorReply(Status const& status, std::string& msg);

void WriteRegisterRequest(
    std::string& msg, StoreType const& bulk_store_type = StoreType::kDefault,
    const std::string& username = "", const std::string& password = "");

void WriteRegisterRequest(
    std::string& msg, StoreType const& bulk_store_type = StoreType::kDefault,
    const ObjectID& session_id = RootSessionID(),
    const std::string& username = "", const std::string& password = "");

Status ReadRegisterRequest(const json& msg, std::string& version,
                           StoreType& bulk_store_type, SessionID& session_id,
                           std::string& username, std::string& password);

void WriteRegisterReply(const std::string& ipc_socket,
                        const std::string& rpc_endpoint,
                        const InstanceID instance_id,
                        const SessionID session_id, const bool store_match,
                        const bool support_rpc_compression, std::string& msg);

Status ReadRegisterReply(const json& msg, std::string& ipc_socket,
                         std::string& rpc_endpoint, InstanceID& instance_id,
                         SessionID& sessionid, std::string& version,
                         bool& store_match, bool& support_rpc_compression);

void WriteExitRequest(std::string& msg);

void WriteRequireExtraRequestMemoryRequest(const size_t size, std::string& msg);

Status ReadRequireExtraRequestMemoryRequest(const json& root, size_t& size);

void WriteRequireExtraRequestMemoryReply(std::string& msg);

Status ReadRequireExtraRequestMemoryReply(const json& root);

void WriteCreateBufferRequest(const size_t size, std::string& msg);

Status ReadCreateBufferRequest(const json& root, size_t& size);

void WriteCreateBufferReply(const ObjectID id,
                            const std::shared_ptr<Payload>& object,
                            const int fd_to_send, std::string& msg);

Status ReadCreateBufferReply(const json& root, ObjectID& id, Payload& object,
                             int& fd_sent);

void WriteCreateBuffersRequest(const std::vector<size_t>& sizes,
                               std::string& msg);

Status ReadCreateBuffersRequest(const json& root, std::vector<size_t>& sizes);

void WriteCreateBuffersReply(
    const std::vector<ObjectID>& ids,
    const std::vector<std::shared_ptr<Payload>>& objects,
    const std::vector<int>& fds_to_send, std::string& msg);

Status ReadCreateBuffersReply(const json& root, std::vector<ObjectID>& ids,
                              std::vector<Payload>& objects,
                              std::vector<int>& fds_sent);

void WriteCreateDiskBufferRequest(const size_t size, const std::string& path,
                                  std::string& msg);

Status ReadCreateDiskBufferRequest(const json& root, size_t& size,
                                   std::string& path);

void WriteCreateDiskBufferReply(const ObjectID id,
                                const std::shared_ptr<Payload>& object,
                                const int fd_to_send, std::string& msg);

Status ReadCreateDiskBufferReply(const json& root, ObjectID& id,
                                 Payload& object, int& fd_sent);

void WriteCreateGPUBufferRequest(const size_t size, std::string& msg);

Status ReadCreateGPUBufferRequest(const json& root, size_t& size);

void WriteGPUCreateBufferReply(const ObjectID id,
                               const std::shared_ptr<Payload>& object,
                               const std::vector<int64_t>& handle,
                               std::string& msg);

Status ReadGPUCreateBufferReply(const json& root, ObjectID& id, Payload& object,
                                std::vector<int64_t>& handle);

void WriteSealRequest(ObjectID const& object_id, std::string& message_out);

Status ReadSealRequest(json const& root, ObjectID& object_id);

void WriteSealReply(std::string& msg);

Status ReadSealReply(json const& root);

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

void WriteGetUserBuffersRequest(std::vector<ObjectID>& ids, std::string& msg);

Status ReadGetUserBuffersRequest(const json& root, std::vector<ObjectID>& ids);

void WriteGetUserBuffersReply(
    const std::vector<std::shared_ptr<Payload>>& objects, std::string& msg);

Status ReadGetUserBuffersReply(const json& root, std::vector<Payload>& objects);

void WriteGetGPUBuffersRequest(const std::set<ObjectID>& ids, const bool unsafe,
                               std::string& msg);

Status ReadGetGPUBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                                bool& unsafe);

void WriteGetGPUBuffersReply(
    const std::vector<std::shared_ptr<Payload>>& objects,
    const std::vector<std::vector<int64_t>>& handles, std::string& msg);

Status ReadGetGPUBuffersReply(const json& root, std::vector<Payload>& objects,
                              std::vector<std::vector<int64_t>>& handles);

void WriteDropBufferRequest(const ObjectID id, std::string& msg);

Status ReadDropBufferRequest(const json& root, ObjectID& id);

void WriteDropBufferReply(std::string& msg);

Status ReadDropBufferReply(const json& root);

void WriteShrinkBufferRequest(const ObjectID id, const size_t size,
                              std::string& msg);

Status ReadShrinkBufferRequest(const json& root, ObjectID& id, size_t& size);

void WriteShrinkBufferReply(std::string& msg);

Status ReadShrinkBufferReply(const json& root);

void WriteRequestFDRequest(std::vector<int> fds, std::string& msg);

Status ReadRequestFDRequest(const json& root, size_t& size, bool& skip_fd);

void WriteRequestFDReply(const ObjectID id,
                         const std::shared_ptr<Payload>& object,
                         const int fd_to_send, std::string& msg);

Status ReadRequestFDReply(const json& root, ObjectID& id, Payload& object,
                          int& fd_sent);

void WriteCreateRemoteBufferRequest(const size_t size, bool use_rdma,
                                    std::string& msg);

void WriteCreateRemoteBufferRequest(const size_t size, const bool compress,
                                    bool use_rdma, std::string& msg);

Status ReadCreateRemoteBufferRequest(const json& root, size_t& size,
                                     bool& compress, bool& use_rdma);

void WriteCreateRemoteBuffersRequest(const std::vector<size_t>& sizes,
                                     const bool compress, bool use_rdma,
                                     std::string& msg);

Status ReadCreateRemoteBuffersRequest(const json& root,
                                      std::vector<size_t>& size, bool& compress,
                                      bool& use_rdma);

void WriteGetRemoteBuffersRequest(const std::set<ObjectID>& ids,
                                  const bool unsafe, bool use_rdma,
                                  std::string& msg);

void WriteGetRemoteBuffersRequest(const std::unordered_set<ObjectID>& ids,
                                  const bool unsafe, bool use_rdma,
                                  std::string& msg);

void WriteGetRemoteBuffersRequest(const std::set<ObjectID>& ids,
                                  const bool unsafe, const bool compress,
                                  bool use_rdma, std::string& msg);

void WriteGetRemoteBuffersRequest(const std::unordered_set<ObjectID>& ids,
                                  const bool unsafe, const bool compress,
                                  bool use_rdma, std::string& msg);

Status ReadGetRemoteBuffersRequest(const json& root, std::vector<ObjectID>& ids,
                                   bool& unsafe, bool& compress,
                                   bool& use_rdma);

void WriteIncreaseReferenceCountRequest(const std::vector<ObjectID>& ids,
                                        std::string& msg);

Status ReadIncreaseReferenceCountRequest(json const& root,
                                         std::vector<ObjectID>& ids);

void WriteIncreaseReferenceCountReply(std::string& msg);

Status ReadIncreaseReferenceCountReply(json const& root);

void WriteReleaseRequest(ObjectID const& object_id, std::string& msg);

Status ReadReleaseRequest(json const& root, ObjectID& object_id);

void WriteReleaseReply(std::string& msg);

Status ReadReleaseReply(json const& root);

void WriteDelDataWithFeedbacksRequest(const std::vector<ObjectID>& id,
                                      const bool force, const bool deep,
                                      const bool memory_trim,
                                      const bool fastpath, std::string& msg);

Status ReadDelDataWithFeedbacksRequest(json const& root,
                                       std::vector<ObjectID>& id, bool& force,
                                       bool& deep, bool& memory_trim,
                                       bool& fastpath);

void WriteDelDataWithFeedbacksReply(const std::vector<ObjectID>& deleted_bids,
                                    std::string& msg);

Status ReadDelDataWithFeedbacksReply(json const& root,
                                     std::vector<ObjectID>& deleted_bids);

void WriteDelHugeDataRequest(const size_t id_num, const bool force,
                             const bool deep, const bool memory_trim,
                             const bool fastpath, std::string& msg);

Status ReadDelHugeDataRequest(json const& root, size_t& id_num, bool& force,
                              bool& deep, bool& memory_trim, bool& fastpath);

void WriteDelHugeDataReply(std::string& msg);

Status ReadDelHugeDataReply(json const& root);

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

void WritePlasmaSealRequest(PlasmaID const& plasma_id,
                            std::string& message_out);

Status ReadPlasmaSealRequest(json const& root, PlasmaID& plasma_id);

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

void WriteCreateUserBuffersRequest(const std::vector<uint64_t>& offsets,
                                   const std::vector<size_t>& sizes,
                                   std::string& msg);

Status ReadCreateUserBuffersRequest(const json& root, size_t& offsets_num,
                                    size_t& sizes_num);

void WriteCreateUserBuffersReply(const std::vector<ObjectID>& ids,
                                 std::string& msg);

Status ReadCreateUserBuffersReply(const json& root, std::vector<ObjectID>& ids);

void WriteDeleteUserBuffersRequest(const std::vector<uint64_t>& ids,
                                   std::string& msg);

Status ReadDeleteUserBuffersRequest(const json& root,
                                    std::vector<uint64_t>& ids);

void WriteDeleteUserBuffersReply(std::string& msg);

Status ReadDeleteUserBuffersReply(const json& root);

void WriteGetRemoteBlobsWithRDMARequest(
    std::vector<std::vector<ObjectID>>& remote_ids, std::string& msg);

Status ReadGetRemoteBlobsWithRDMARequest(
    const json& root, std::vector<std::vector<ObjectID>>& remote_ids);

void WriteGetRemoteBlobsWithRDMAReply(std::string& msg, int index);

Status ReadGetRemoteBlobsWithRDMAReply(const json& root, int& index);

void WriteCreateDataRequest(const json& content, std::string& msg);

Status ReadCreateDataRequest(const json& root, json& content);

void WriteCreateDataReply(const ObjectID& id, const Signature& signature,
                          const InstanceID& instance_id, std::string& msg);

Status ReadCreateDataReply(const json& root, ObjectID& id, Signature& signature,
                           InstanceID& instance_id);

void WriteCreateDatasRequest(const std::vector<json>& contents,
                             std::string& msg);

Status ReadCreateDatasRequest(const json& root, std::vector<json>& contents);

void WriteCreateDatasReply(const std::vector<ObjectID>& ids,
                           const std::vector<Signature>& signatures,
                           const std::vector<InstanceID>& instance_ids,
                           std::string& msg);

Status ReadCreateDatasReply(const json& root, std::vector<ObjectID>& ids,
                            std::vector<Signature>& signatures,
                            std::vector<InstanceID>& instance_ids);

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

void WriteCreateHugeDatasRequest(const size_t& json_num, std::string& msg);

Status ReadCreateHugeDatasRequest(const json& root, size_t& json_num);

void WriteCreateHugeDatasReply(const size_t& ids_num,
                               const Signature& signature,
                               const InstanceID& instance_id, std::string& msg);

Status ReadCreateHugeDatasReply(const json& root, size_t& ids_num,
                                Signature& signatures,
                                InstanceID& instance_ids);

void WriteGetHugeDataRequest(const size_t id_num, const bool sync_remote,
                             const bool wait, std::string& msg);

Status ReadGetHugeDataRequest(const json& root, size_t& id_num,
                              bool& sync_remote, bool& wait);

void WriteGetHugeDataReply(size_t json_length, std::string& msg);

Status ReadGetHugeDataReply(const json& root, size_t& json_length);

void WriteListDataRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg);

Status ReadListDataRequest(const json& root, std::string& pattern, bool& regex,
                           size_t& limit);

void WriteDelDataRequest(const ObjectID id, const bool force, const bool deep,
                         const bool memory_trim, const bool fastpath,
                         std::string& msg);

void WriteDelDataRequest(const std::vector<ObjectID>& id, const bool force,
                         const bool deep, const bool memory_trim,
                         const bool fastpath, std::string& msg);

Status ReadDelDataRequest(const json& root, std::vector<ObjectID>& id,
                          bool& force, bool& deep, bool& memory_trim,
                          bool& fastpath);

void WriteDelDataReply(std::string& msg);

Status ReadDelDataReply(const json& root);

void WriteExistsRequest(const ObjectID id, std::string& msg);

Status ReadExistsRequest(const json& root, ObjectID& id);

void WriteExistsReply(bool exists, std::string& msg);

Status ReadExistsReply(const json& root, bool& exists);

void WritePersistRequest(const ObjectID id, std::string& msg);

Status ReadPersistRequest(const json& root, ObjectID& id);

void WritePersistReply(std::string& msg);

Status ReadPersistReply(const json& root);

void WriteBatchPersistRequest(const std::vector<ObjectID>& ids,
                              std::string& msg);

Status ReadBatchPersistRequest(const json& root, std::vector<ObjectID>& ids);

void WriteBatchPersistReply(std::string& msg);

Status ReadBatchPersistReply(const json& root);

void WriteIfPersistRequest(const ObjectID id, std::string& msg);

Status ReadIfPersistRequest(const json& root, ObjectID& id);

void WriteIfPersistReply(bool exists, std::string& msg);

Status ReadIfPersistReply(const json& root, bool& persist);

void WriteLabelRequest(const ObjectID id, const std::string& key,
                       const std::string& value, std::string& msg);

void WriteLabelRequest(const ObjectID id, const std::vector<std::string>& keys,
                       const std::vector<std::string>& values,
                       std::string& msg);

void WriteLabelRequest(const ObjectID id,
                       const std::map<std::string, std::string>& kvs,
                       std::string& msg);

Status ReadLabelRequest(json const& root, ObjectID& id,
                        std::vector<std::string>& keys,
                        std::vector<std::string>& values);

void WriteLabelReply(std::string& msg);

Status ReadLabelReply(json const& root);

void WriteClearRequest(std::string& msg);

Status ReadClearRequest(const json& root);

void WriteClearReply(std::string& msg);

Status ReadClearReply(const json& root);

void WriteMemoryTrimRequest(std::string& msg);

Status ReadMemoryTrimRequest(const json& root);

void WriteMemoryTrimReply(const bool trimmed, std::string& msg);

Status ReadMemoryTrimReply(const json& root, bool& trimmed);

void WriteReleaseBlobsWithRDMARequest(const std::unordered_set<ObjectID>& ids,
                                      std::string& msg);

Status ReadReleaseBlobsWithRDMARequest(const json& root,
                                       std::vector<ObjectID>& ids);

void WriteReleaseBlobsWithRDMAReply(std::string& msg);

Status ReadReleaseBlobsWithRDMAReply(const json& root);

void WriteGetMetasByNamesRequest(const std::vector<std::string>& names,
                                 std::string& msg);

Status ReadGetMetasByNamesRequest(const json& root,
                                  std::vector<std::string>& names);

void WriteGetMetasByNamesReply(std::vector<ObjectID>& ids, json& contents,
                               std::string& msg);

Status ReadGetMetasByNamesReply(const json& root, std::vector<ObjectID>& ids,
                                json& contents);

void WriteCreateStreamRequest(const ObjectID& object_id, std::string& msg);

Status ReadCreateStreamRequest(const json& root, ObjectID& object_id);

void WriteCreateStreamReply(std::string& msg);

Status ReadCreateStreamReply(const json& root);

void WriteCreateFixedStreamRequest(std::string stream_name, int blob_nums,
                                   size_t blob_size, std::string& msg);

Status ReadCreateFixedStreamRequest(const json& root, std::string& stream_name,
                                    int& blob_nums, size_t& blob_size);

void WriteCreateFixedStreamReply(std::string& msg, ObjectID& stream_id);

Status ReadCreateFixedStreamReply(const json& root, ObjectID& stream_id);

void WriteOpenStreamRequest(const ObjectID& object_id,
                            const std::string stream_name, const int64_t& mode,
                            bool wait, uint64_t timeout, std::string& msg);

Status ReadOpenStreamRequest(const json& root, ObjectID& object_id,
                             std::string& stream_name, int64_t& mode,
                             bool& wait, uint64_t& timeout);

void WriteOpenStreamReply(std::string& msg, ObjectID& ret_id);

Status ReadOpenStreamReply(const json& root, ObjectID& ret_id);

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

void WritePushNextStreamChunkByOffsetRequest(const ObjectID stream_id,
                                             const size_t offset,
                                             std::string& msg);

Status ReadPushNextStreamChunkByOffsetRequest(const json& root,
                                              ObjectID& stream_id,
                                              size_t& offset);

void WritePushNextStreamChunkByOffsetReply(std::string& msg);

Status ReadPushNextStreamChunkByOffsetReply(const json& root);

void WritePullNextStreamChunkRequest(const ObjectID stream_id,
                                     std::string& msg);

Status ReadPullNextStreamChunkRequest(const json& root, ObjectID& stream_id);

void WritePullNextStreamChunkReply(ObjectID const chunk, std::string& msg);

Status ReadPullNextStreamChunkReply(const json& root, ObjectID& chunk);

void WriteCheckFixedStreamReceivedRequest(const ObjectID stream_id, int index,
                                          std::string& msg);

Status ReadCheckFixedStreamReceivedRequest(const json& root,
                                           ObjectID& stream_id, int& index);

void WriteCheckFixedStreamReceivedReply(bool finished, std::string& msg);

Status ReadCheckFixedStreamReceivedReply(bool& finished, const json& root);

void WriteStopStreamRequest(const ObjectID stream_id, const bool failed,
                            std::string& msg);

Status ReadStopStreamRequest(const json& root, ObjectID& stream_id,
                             bool& failed);

void WriteStopStreamReply(std::string& msg);

Status ReadStopStreamReply(const json& root);

void WriteDropStreamRequest(const ObjectID stream_id, std::string& msg);

Status ReadDropStreamRequest(const json& root, ObjectID& stream_id);

void WriteDropStreamReply(std::string& msg);

Status ReadDropStreamReply(const json& root);

void WriteAbortStreamRequest(const ObjectID stream_id, std::string& msg);

Status ReadAbortStreamRequest(const json& root, ObjectID& stream_id);

void WriteAbortStreamReply(std::string& msg, bool success);

Status ReadAbortStreamReply(const json& root, bool& success);

void WritePutStreamNameRequest(const ObjectID stream_id, std::string name,
                               std::string& msg);

Status ReadPutStreamNameRequest(const json& root, ObjectID& stream_id,
                                std::string& name);

void WritePutStreamNameReply(std::string& msg);

Status ReadPutStreamNameReply(const json& root);

void WriteGetStreamIDByNameRequest(const std::string name, std::string& msg);

Status ReadGetStreamIDByNameRequest(const json& root, std::string& name);

void WriteGetStreamIDByNameReply(const ObjectID stream_id, std::string& msg);

Status ReadGetStreamIDByNameReply(const json& root, ObjectID& stream_id);

void WriteActivateRemoteFixedStreamRequest(ObjectID stream_id,
                                           std::vector<uint64_t>& buffer_list,
                                           std::vector<uint64_t>& rkeys,
                                           std::string& msg);

Status ReadActivateRemoteFixedStreamRequest(const json& root,
                                            ObjectID& stream_id,
                                            std::vector<uint64_t>& buffer_list,
                                            std::vector<uint64_t>& rkeys);

void WriteActivateRemoteFixedStreamReply(std::string& msg);

Status ReadActivateRemoteFixedStreamReply(const json& root);

void WriteStreamReadyAckReply(std::string& msg, int index);

Status ReadStreamReadyAckReply(const json& root, int& index);

void WriteActivateRemoteFixedStreamRequest(
    ObjectID stream_id, std::vector<std::vector<uint64_t>>& buffer_list,
    std::vector<std::vector<uint64_t>>& rkeys_list,
    std::vector<std::vector<size_t>>& sizes, std::string advice_device,
    int port, std::string& msg);

Status ReadActivateRemoteFixedStreamRequest(
    const json& root, ObjectID& stream_id,
    std::vector<std::vector<uint64_t>>& buffer_list,
    std::vector<std::vector<uint64_t>>& rkeys,
    std::vector<std::vector<size_t>>& sizes_list, std::string& advice_device,
    int& port);

void WriteActivateRemoteFixedStreamRequest(ObjectID stream_id,
                                           std::vector<uint64_t>& buffer_list,
                                           std::vector<size_t>& sizes,
                                           std::string& msg);

Status ReadActivateRemoteFixedStreamRequest(const json& root,
                                            ObjectID& stream_id,
                                            std::vector<uint64_t>& buffer_list,
                                            std::vector<size_t>& sizes_list);

void WriteActivateRemoteFixedStreamReply(std::string& msg);

Status ReadActivateRemoteFixedStreamReply(const json& root);

Status WriteOpenFixedStreamRequest(const ObjectID stream_id,
                                   const uint64_t mode, std::string& msg);

Status ReadOpenFixedStreamRequest(const json& root, ObjectID& stream_id,
                                  int64_t& mode);

void WriteOpenFixedStreamReply(std::string& msg);

Status ReadOpenFixedStreamReply(const json& root);

void WriteCloseStreamRequest(ObjectID stream_id, std::string& msg);

Status ReadCloseStreamRequest(const json& root, ObjectID& stream_id);

void WriteCloseStreamReply(std::string& msg);

Status ReadCloseStreamReply(const json& root);

void WriteDeleteStreamRequest(ObjectID stream_id, std::string& msg);

Status ReadDeleteStreamRequest(const json& root, ObjectID& stream_id);

void WriteDeleteStreamReply(std::string& msg);

Status ReadDeleteStreamReply(const json& root);

void WriteVineyardOpenRemoteFixedStreamRequest(
    ObjectID const remote_id, std::string stream_name, ObjectID local_id,
    int blob_nums, size_t size, std::string remote_endpoint, uint64_t mode,
    bool wait, uint64_t timeout, std::string& msg);

Status ReadVineyardOpenRemoteFixedStreamRequest(
    const json& root, ObjectID& remote_id, std::string& remote_stream_name,
    ObjectID& local_id, int& blob_nums, size_t& size,
    std::string& remote_endpoint, uint64_t& mode, bool& wait,
    uint64_t& timeout);

void WriteVineyardOpenRemoteFixedStreamReply(std::string& msg,
                                             ObjectID const local_stream_id);

Status ReadVineyardOpenRemoteFixedStreamReply(const json& root,
                                              ObjectID& local_id);

void WriteVineyardActivateRemoteFixedStreamRequest(
    ObjectID stream_id, bool create, std::vector<ObjectID>& blob_list,
    std::string& msg);

Status ReadVineyardActivateRemoteFixedStreamRequest(
    const json& root, ObjectID& stream_id, bool& create,
    std::vector<ObjectID>& blob_list);

void WriteVineyardActivateRemoteFixedStreamReply(
    std::string& msg, std::vector<std::shared_ptr<Payload>>& payload_list,
    std::vector<int>& fds_to_send);

Status ReadVineyardActivateRemoteFixedStreamReply(const json& root,
                                                  std::vector<Payload>& objects,
                                                  std::vector<int>& fds_sent);

void WriteVineyardActivateRemoteFixedStreamWithOffsetRequest(
    ObjectID stream_id, std::vector<size_t>& offsets, std::string& msg);

Status ReadVineyardActivateRemoteFixedStreamWithOffsetRequest(
    const json& root, ObjectID& stream_id, std::vector<size_t>& offsets);

void WriteVineyardActivateRemoteFixedStreamWithOffsetReply(std::string& msg);

Status ReadVineyardActivateRemoteFixedStreamWithOffsetReply(const json& root);

void WriteVineyardCloseRemoteFixedStreamRequest(ObjectID stream_id,
                                                std::string& msg);

Status ReadVineyardCloseRemoteFixedStreamRequest(const json& root,
                                                 ObjectID& stream_id);

void WriteVineyardCloseRemoteFixedStreamReply(std::string& msg);

Status ReadVineyardCloseRemoteFixedStreamReply(const json& root);

void WriteVineyardGetMetasByNamesRequest(std::vector<std::string>& names,
                                         std::string rpc_endpoint,
                                         std::string& msg);

Status ReadVineyardGetMetasByNamesRequest(const json& root,
                                          std::vector<std::string>& names,
                                          std::string& rpc_endpoint);

void WriteVineyardGetMetasByNamesReply(const std::vector<json>& contents,
                                       std::string& msg);

Status ReadVineyardGetMetasByNamesReply(const json& root,
                                        std::vector<json>& contents);

void WriteVineyardGetRemoteBlobsWithRDMARequest(
    std::vector<std::vector<ObjectID>>& local_ids,
    std::vector<std::vector<ObjectID>>& remote_ids, std::string& rpc_endpoint,
    std::string& msg);

Status ReadVineyardGetRemoteBlobsWithRDMARequest(
    const json& root, std::vector<std::vector<ObjectID>>& local_ids,
    std::vector<std::vector<ObjectID>>& remote_ids, std::string& rpc_endpoint);

void WriteVineyardGetRemoteBlobsWithRDMAReply(std::string& msg);

Status ReadVineyardGetRemoteBlobsWithRDMAReply(const json& root);

void WriteVineyardGetRemoteBlobsWithOffsetRequest(size_t batch_nums,
                                                  size_t batch_size,
                                                  std::string& rpc_endpoint,
                                                  std::string& msg);

Status ReadVineyardGetRemoteBlobsWithOffsetRequest(const json& root,
                                                   size_t& batch_nums,
                                                   size_t& batch_size,
                                                   std::string& rpc_endpoint);

void WriteVineyardGetRemoteBlobsWithOffsetReply(std::string& msg);

Status ReadVineyardGetRemoteBlobsWithOffsetReply(const json& root);

void WriteVineyardStopStreamRequest(ObjectID stream_id, bool failed,
                                    std::string& msg);

Status ReadVineyardStopStreamRequest(const json& root, ObjectID& stream_id,
                                     bool& failed);

void WriteVineyardStopStreamReply(std::string& msg);

Status ReadVineyardStopStreamReply(const json& root);

void WriteVineyardDropStreamRequest(ObjectID stream_id, std::string& msg);

Status ReadVineyardDropStreamRequest(const json& root, ObjectID& stream_id);

void WriteVineyardDropStreamReply(std::string& msg);

Status ReadVineyardDropStreamReply(const json& root);

void WriteVineyardAbortRemoteStreamRequest(ObjectID stream_id,
                                           std::string& msg);

Status ReadVineyardAbortRemoteStreamRequest(const json& root,
                                            ObjectID& stream_id);

void WriteVineyardAbortRemoteStreamReply(std::string& msg, bool success);

Status ReadVineyardAbortRemoteStreamReply(const json& root, bool& success);

void WritePutNameRequest(const ObjectID object_id, const std::string& name,
                         bool overwrite, std::string& msg);

Status ReadPutNameRequest(const json& root, ObjectID& object_id,
                          std::string& name, bool& overwrite);

void WritePutNameReply(std::string& msg);

Status ReadPutNameReply(const json& root);

void WritePutNamesRequest(const std::vector<ObjectID> object_ids,
                          const std::vector<std::string>& names, bool overwrite,
                          std::string& msg);

Status ReadPutNamesRequest(const json& root, std::vector<ObjectID>& object_id,
                           std::vector<std::string>& name, bool& overwrite);

void WritePutNamesReply(std::string& msg);

Status ReadPutNamesReply(const json& root);

void WriteGetNameRequest(const std::string& name, const bool wait,
                         std::string& msg);

Status ReadGetNameRequest(const json& root, std::string& name, bool& wait);

void WriteGetNameReply(const ObjectID& object_id, std::string& msg);

Status ReadGetNameReply(const json& root, ObjectID& object_id);

void WriteGetNamesRequest(const std::vector<std::string>& name, const bool wait,
                          std::string& msg);

Status ReadGetNamesRequest(const json& root, std::vector<std::string>& name,
                           bool& wait);

void WriteGetNamesReply(const std::vector<ObjectID>& object_id,
                        std::string& msg);

Status ReadGetNamesReply(const json& root, std::vector<ObjectID>& object_id);

void WriteListNameRequest(std::string const& pattern, bool const regex,
                          size_t const limit, std::string& msg);

Status ReadListNameRequest(const json& root, std::string& pattern, bool& regex,
                           size_t& limit);

void WriteListNameReply(std::map<std::string, ObjectID> const& names,
                        std::string& msg);

Status ReadListNameReply(const json& root,
                         std::map<std::string, ObjectID>& names);

void WriteDropNameRequest(const std::string& name, std::string& msg);

Status ReadDropNameRequest(const json& root, std::string& name);

void WriteDropNameReply(std::string& msg);

Status ReadDropNameReply(const json& root);

void WriteDropNamesRequest(const std::vector<std::string>& name,
                           std::string& msg);

Status ReadDropNamesRequest(const json& root, std::vector<std::string>& name);

void WriteDropNamesReply(std::string& msg);

Status ReadDropNamesReply(const json& root);

void WriteGetObjectLocationRequest(const std::vector<std::string>& names,
                                   std::string& msg);

Status ReadGetObjectLocationRequest(const json& root,
                                    std::vector<std::string>& names);

void WriteGetObjectLocationReply(
    std::vector<std::vector<std::string>>& locations, std::string& msg);

Status ReadGetObjectLocationReply(
    const json& root, std::vector<std::vector<std::string>>& locations);

void WritePutObjectLocationRequest(const std::vector<std::string>& names,
                                   const std::vector<std::string>& locations,
                                   int ttl_seconds, std::string& msg);

Status ReadPutObjectLocationRequest(const json& root,
                                    std::vector<std::string>& names,
                                    std::vector<std::string>& locations,
                                    int& ttl_seconds);

void WritePutObjectLocationReply(std::string& msg);

Status ReadPutObjectLocationReply(const json& root);

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

void WriteNewSessionRequest(std::string& msg, StoreType const& bulk_store_type);

Status ReadNewSessionRequest(json const& root, StoreType& bulk_store_type);

void WriteNewSessionReply(std::string& msg, std::string const& socket_path);

Status ReadNewSessionReply(json const& root, std::string& socket_path);

void WriteDeleteSessionRequest(std::string& msg);

void WriteDeleteSessionReply(std::string& msg);

template <typename From, typename To>
void WriteMoveBuffersOwnershipRequest(std::map<From, To> const& id_to_id,
                                      SessionID const session_id,
                                      std::string& msg);

/// normal -> normal
/// normal -> plasma
/// plasma -> normal
/// plasma -> plasma
Status ReadMoveBuffersOwnershipRequest(json const& root,
                                       std::map<ObjectID, ObjectID>& id_to_id,
                                       std::map<PlasmaID, ObjectID>& pid_to_id,
                                       std::map<ObjectID, PlasmaID>& id_to_pid,
                                       std::map<PlasmaID, PlasmaID>& pid_to_pid,
                                       SessionID& session_id);

void WriteMoveBuffersOwnershipReply(std::string& msg);

Status ReadMoveBuffersOwnershipReply(json const& root);

void WriteEvictRequest(const std::vector<ObjectID>& ids, std::string& msg);

Status ReadEvictRequest(json const& root, std::vector<ObjectID>& ids);

void WriteEvictReply(std::string& msg);

Status ReadEvictReply(json const& root);

void WriteLoadRequest(const std::vector<ObjectID>& ids, const bool pin,
                      std::string& msg);

Status ReadLoadRequest(json const& root, std::vector<ObjectID>& ids, bool& pin);

void WriteLoadReply(std::string& msg);

Status ReadLoadReply(json const& root);

void WriteUnpinRequest(const std::vector<ObjectID>& ids, std::string& msg);

Status ReadUnpinRequest(json const& root, std::vector<ObjectID>& ids);

void WriteUnpinReply(std::string& msg);

Status ReadUnpinReply(json const& root);

void WriteIsSpilledRequest(const ObjectID& id, std::string& msg);

Status ReadIsSpilledRequest(json const& root, ObjectID& id);

void WriteIsSpilledReply(const bool is_spilled, std::string& msg);

Status ReadIsSpilledReply(json const& root, bool& is_spilled);

void WriteIsInUseRequest(const ObjectID& id, std::string& msg);

Status ReadIsInUseRequest(json const& root, ObjectID& id);

void WriteIsInUseReply(const bool is_in_use, std::string& msg);

Status ReadIsInUseReply(json const& root, bool& is_in_use);

void WriteGetVineyardMmapFdRequest(std::string& msg);

Status ReadGetVineyardMmapFdRequest(const json& root);

void WriteGetVineyardMmapFdReply(size_t size, size_t offset, std::string& msg);

Status ReadGetVineyardMmapFdReply(const json& root, size_t& size,
                                  size_t& offset);

void WriteClusterMetaRequest(std::string& msg);

Status ReadClusterMetaRequest(const json& root);

void WriteClusterMetaReply(const json& content, std::string& msg);

Status ReadClusterMetaReply(const json& root, json& content);

void WriteInstanceStatusRequest(std::string& msg);

Status ReadInstanceStatusRequest(const json& root);

void WriteInstanceStatusReply(const json& content, std::string& msg);

Status ReadInstanceStatusReply(const json& root, json& content);

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

void WriteShallowCopyRequest(const ObjectID id, std::string& msg);

void WriteShallowCopyRequest(const ObjectID id, json const& extra_metadata,
                             std::string& msg);

Status ReadShallowCopyRequest(const json& root, ObjectID& id,
                              json& extra_metadata);

void WriteShallowCopyReply(const ObjectID target_id, std::string& msg);

Status ReadShallowCopyReply(const json& root, ObjectID& target_id);

void WriteDebugRequest(const json& debug, std::string& msg);

Status ReadDebugRequest(const json& root, json& debug);

void WriteDebugReply(const json& result, std::string& msg);

Status ReadDebugReply(const json& root, json& result);

void WriteTryAcquireLockRequest(const std::string& key, std::string& msg);

Status ReadTryAcquireLockRequest(const json& root, std::string& key);

void WriteTryAcquireLockReply(const bool result, const std::string actual_key,
                              std::string& msg);

Status ReadTryAcquireLockReply(const json& root, bool& result,
                               std::string& key);

void WriteTryReleaseLockRequest(const std::string& key, std::string& msg);

Status ReadTryReleaseLockRequest(const json& root, std::string& key);

void WriteTryReleaseLockReply(const bool result, std::string& msg);

Status ReadTryReleaseLockReply(const json& root, bool& result);

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_PROTOCOLS_H_
