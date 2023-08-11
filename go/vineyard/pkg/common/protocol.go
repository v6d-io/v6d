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

package common

import (
	"fmt"

	"github.com/goccy/go-json"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

const (
	REGISTER_REQUEST = "register_request"
	REGISTER_REPLY   = "register_reply"
	EXIT_REQUEST     = "exit_request"
	EXIT_REPLY       = "exit_reply"

	CREATE_BUFFER_REQUEST      = "create_buffer_request"
	CREATE_BUFFER_REPLY        = "create_buffer_reply"
	CREATE_DISK_BUFFER_REQUEST = "create_disk_buffer_request"
	CREATE_DISK_BUFFER_REPLY   = "create_disk_buffer_reply"
	CREATE_GPU_BUFFER_REQUEST  = "create_gpu_buffer_request"
	CREATE_GPU_BUFFER_REPLY    = "create_gpu_buffer_reply"
	SEAL_BUFFER_REQUEST        = "seal_request"
	SEAL_BUFFER_REPLY          = "seal_reply"
	GET_BUFFERS_REQUEST        = "get_buffers_request"
	GET_BUFFERS_REPLY          = "get_buffers_reply"
	GET_GPU_BUFFERS_REQUEST    = "get_gpu_buffers_request"
	GET_GPU_BUFFERS_REPLY      = "get_gpu_buffers_reply"
	DROP_BUFFER_REQUEST        = "drop_buffer_request"
	DROP_BUFFER_REPLY          = "drop_buffer_reply"

	CREATE_REMOTE_BUFFERS_REQUEST = "create_remote_buffers_request"
	CREATE_REMOTE_BUFFERS_REPLY   = "create_remote_buffers_reply"
	GET_REMOTE_BUFFERS_REQUEST    = "get_remote_buffers_request"
	GET_REMOTE_BUFFERS_REPLY      = "get_remote_buffers_reply"

	INCREASE_REFERENCE_COUNT_REQUEST = "increase_reference_count_request"
	INCREASE_REFERENCE_COUNT_REPLY   = "increase_reference_count_reply"
	RELEASE_REQUEST                  = "release_request"
	RELEASE_REPLY                    = "release_reply"
	DEL_DATA_WITH_FEEDBACK_REQUEST   = "del_data_with_feedback_request"
	DEL_DATA_WITH_FEEDBACK_REPLY     = "del_data_with_feedback_reply"

	CREATE_BUFFER_PLASMA_REQUEST = "create_buffer_by_plasma_request"
	CREATE_BUFFER_PLASMA_REPLY   = "create_buffer_by_plasma_reply"
	GET_BUFFERS_PLASMA_REQUEST   = "get_buffers_by_plasma_request"
	GET_BUFFERS_PLASMA_REPLY     = "get_buffers_by_plasma_reply"
	PLASMA_SEAL_REQUEST          = "plasma_seal_request"
	PLASMA_SEAL_REPLY            = "plasma_seal_reply"
	PLASMA_RELEASE_REQUEST       = "plasma_release_request"
	PLASMA_RELEASE_REPLY         = "plasma_release_reply"
	PLASMA_DEL_DATA_REQUEST      = "plasma_del_data_request"

	CREATE_DATA_REQUEST = "create_data_request"
	CREATE_DATA_REPLY   = "create_data_reply"
	GET_DATA_REQUEST    = "get_data_request"
	GET_DATA_REPLY      = "get_data_reply"
	LIST_DATA_REQUEST   = "list_data_request"
	LIST_DATA_REPLY     = "list_data_reply"
	DELETE_DATA_REQUEST = "del_data_request"
	DELETE_DATA_REPLY   = "del_data_reply"
	EXISTS_REQUEST      = "exists_request"
	EXISTS_REPLY        = "exists_reply"
	PERSIST_REQUEST     = "persist_request"
	PERSIST_REPLY       = "persist_reply"
	IF_PERSIST_REQUEST  = "if_persist_request"
	IF_PERSIST_REPLY    = "if_persist_reply"
	LABEL_REQUEST       = "label_request"
	LABEL_REPLY         = "label_reply"
	CLEAR_REQUEST       = "clear_request"
	CLEAR_REPLY         = "clear_reply"

	CREATE_STREAM_REQUEST          = "create_stream_request"
	CREATE_STREAM_REPLY            = "create_stream_reply"
	OPEN_STREAM_REQUEST            = "open_stream_request"
	OPEN_STREAM_REPLY              = "open_stream_reply"
	GET_NEXT_STREAM_CHUNK_REQUEST  = "get_next_stream_chunk_request"
	GET_NEXT_STREAM_CHUNK_REPLY    = "get_next_stream_chunk_reply"
	PUSH_NEXT_STREAM_CHUNK_REQUEST = "push_next_stream_chunk_request"
	PUSH_NEXT_STREAM_CHUNK_REPLY   = "push_next_stream_chunk_reply"
	PULL_NEXT_STREAM_CHUNK_REQUEST = "pull_next_stream_chunk_request"
	PULL_NEXT_STREAM_CHUNK_REPLY   = "pull_next_stream_chunk_reply"
	STOP_STREAM_REQUEST            = "stop_stream_request"
	STOP_STREAM_REPLY              = "stop_stream_reply"
	DROP_STREAM_REQUEST            = "drop_stream_request"
	DROP_STREAM_REPLY              = "drop_stream_reply"

	PUT_NAME_REQUEST  = "put_name_request"
	PUT_NAME_REPLY    = "put_name_reply"
	GET_NAME_REQUEST  = "get_name_request"
	GET_NAME_REPLY    = "get_name_reply"
	LIST_NAME_REQUEST = "list_name_request"
	LIST_NAME_REPLY   = "list_name_reply"
	DROP_NAME_REQUEST = "drop_name_request"
	DROP_NAME_REPLY   = "drop_name_reply"

	MAKE_ARENA_REQUEST     = "make_arena_request"
	MAKE_ARENA_REPLY       = "make_arena_reply"
	FINALIZE_ARENA_REQUEST = "finalize_arena_request"
	FINALIZE_ARENA_REPLY   = "finalize_arena_reply"

	NEW_SESSION_REQUEST    = "new_session_request"
	NEW_SESSION_REPLY      = "new_session_reply"
	DELETE_SESSION_REQUEST = "delete_session_request"
	DELETE_SESSION_REPLY   = "delete_session_reply"

	MOVE_BUFFERS_OWNERSHIP_REQUEST = "move_buffers_ownership_request"
	MOVE_BUFFERS_OWNERSHIP_REPLY   = "move_buffers_ownership_reply"

	EVICT_REQUEST      = "evict_request"
	EVICT_REPLY        = "evict_reply"
	LOAD_REQUEST       = "load_request"
	LOAD_REPLY         = "load_reply"
	UNPIN_REQUEST      = "unpin_request"
	UNPIN_REPLY        = "unpin_reply"
	IS_SPILLED_REQUEST = "is_spilled_request"
	IS_SPILLED_REPLY   = "is_spilled_reply"
	IS_IN_USE_REQUEST  = "is_in_use_request"
	IS_IN_USE_REPLY    = "is_in_use_reply"

	CLUSTER_META_REQUEST    = "cluster_meta"
	CLUSTER_META_REPLY      = "cluster_meta"
	INSTANCE_STATUS_REQUEST = "instance_status_request"
	INSTANCE_STATUS_REPLY   = "instance_status_reply"
	MIGRATE_OBJECT_REQUEST  = "migrate_object_request"
	MIGRATE_OBJECT_REPLY    = "migrate_object_reply"
	SHALLOW_COPY_REQUEST    = "shallow_copy_request"
	SHALLOW_COPY_REPLY      = "shallow_copy_reply"
	DEBUG_COMMAND           = "debug_command"
	DEBUG_REPLY             = "debug_reply"

	STORE_TYPE_NORMAL = "Normal"
	STORE_TYPE_PLASMA = "Plasma"
)

type request struct {
	Type string `json:"type"`
}

type reply struct {
	Type    string `json:"type"`
	Code    int    `json:"code"`
	Message string `json:"message,omitempty"`
}

func (r *reply) check() error {
	if r.Code != 0 {
		return Error(r.Code, r.Message)
	}
	return nil
}

type Reply interface {
	Check() error
}

type RegisterRequest struct {
	request
	Version   string `json:"version"`
	StoreType string `json:"store_type"`
}

type RegisterReply struct {
	reply
	IPCSocket   string           `json:"ipc_socket"`
	RPCEndpoint string           `json:"rpc_endpoint"`
	InstanceID  types.InstanceID `json:"instance_id"`
	Version     string           `json:"version,omitempty"`
}

func WriteRegisterRequest(version string) []byte {
	var request RegisterRequest
	request.Type = REGISTER_REQUEST
	request.Version = version
	request.StoreType = "Normal"
	return encodeMsg(request)
}

func (reply *RegisterReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != REGISTER_REPLY {
		return ReplyTypeMismatch(REGISTER_REPLY)
	}
	return nil
}

type ExitRequest struct {
	request
}

func WriteExitRequest() []byte {
	var request ExitRequest
	request.Type = EXIT_REQUEST
	return encodeMsg(request)
}

type CreateBufferRequest struct {
	request
	Size uint64 `json:"size"`
}

type CreateBufferReply struct {
	reply
	Created types.Payload `json:"created"`
}

func WriteCreateBufferRequest(size uint64) []byte {
	var request CreateBufferRequest
	request.Type = CREATE_BUFFER_REQUEST
	request.Size = size
	return encodeMsg(request)
}

func (reply *CreateBufferReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != CREATE_BUFFER_REPLY {
		return ReplyTypeMismatch(CREATE_BUFFER_REPLY)
	}
	return nil
}

type SealRequest struct {
	request
	ID types.ObjectID `json:"object_id"`
}

type SealReply struct {
	reply
}

func WriteSealRequest(id types.ObjectID) []byte {
	var request SealRequest
	request.Type = SEAL_BUFFER_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *SealReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != SEAL_BUFFER_REPLY {
		return ReplyTypeMismatch(SEAL_BUFFER_REPLY)
	}
	return nil
}

type GetBuffersRequest struct {
	request
	IDs    []types.ObjectID `json:"ids"`
	Unsafe bool             `json:"unsafe"`
}

type GetBuffersReply struct {
	reply
	Fds      []int           `json:"fds"`
	Payloads []types.Payload `json:"payloads"`
	Compress bool            `json:"compress"`
}

func WriteGetBuffersRequest(ids []types.ObjectID, unsafe bool) []byte {
	var request GetBuffersRequest
	request.Type = GET_BUFFERS_REQUEST
	request.IDs = ids
	request.Unsafe = unsafe
	return encodeMsg(request)
}

func (reply *GetBuffersReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != GET_BUFFERS_REPLY {
		return ReplyTypeMismatch(GET_BUFFERS_REPLY)
	}
	return nil
}

type DropBufferRequest struct {
	request
	ID types.ObjectID `json:"id"`
}

type DropBufferReply struct {
	reply
}

func WriteDropBufferRequest(id types.ObjectID) []byte {
	var request DropBufferRequest
	request.Type = DROP_BUFFER_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *DropBufferReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != DROP_BUFFER_REPLY {
		return ReplyTypeMismatch(DROP_BUFFER_REPLY)
	}
	return nil
}

type IncreaseRefCountRequest struct {
	request
	IDs []types.ObjectID `json:"ids"`
}

type IncreaseRefCountReply struct {
	reply
}

func WriteIncreaseRefCountRequest(ids []types.ObjectID) []byte {
	var request IncreaseRefCountRequest
	request.Type = INCREASE_REFERENCE_COUNT_REQUEST
	request.IDs = ids
	return encodeMsg(request)
}

func (reply *IncreaseRefCountReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != INCREASE_REFERENCE_COUNT_REPLY {
		return ReplyTypeMismatch(INCREASE_REFERENCE_COUNT_REPLY)
	}
	return nil
}

type ReleaseRequest struct {
	request
	ID types.ObjectID `json:"id"`
}

type ReleaseReply struct {
	reply
}

func WriteReleaseRequest(id types.ObjectID) []byte {
	var request ReleaseRequest
	request.Type = RELEASE_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *ReleaseReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != RELEASE_REPLY {
		return ReplyTypeMismatch(RELEASE_REPLY)
	}
	return nil
}

type CreateDataRequest struct {
	request
	Content map[string]any `json:"content"`
}

type CreateDataReply struct {
	reply
	ID         types.ObjectID   `json:"id"`
	Signature  types.Signature  `json:"signature"`
	InstanceID types.InstanceID `json:"instance_id"`
}

func WriteCreateDataRequest(content map[string]any) []byte {
	var request CreateDataRequest
	request.Type = CREATE_DATA_REQUEST
	request.Content = content
	return encodeMsg(request)
}

func (reply *CreateDataReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != CREATE_DATA_REPLY {
		return ReplyTypeMismatch(CREATE_DATA_REPLY)
	}
	return nil
}

type GetDataRequest struct {
	request
	ID         []types.ObjectID `json:"id"`
	SyncRemote bool             `json:"sync_remote"`
	Wait       bool             `json:"wait"`
}

type GetDataReply struct {
	reply
	Content map[string]map[string]any `json:"content"`
}

func WriteGetDataRequest(id []types.ObjectID, syncRemote bool, wait bool) []byte {
	var request GetDataRequest
	request.Type = GET_DATA_REQUEST
	request.ID = id
	request.SyncRemote = syncRemote
	request.Wait = wait
	return encodeMsg(request)
}

func (reply *GetDataReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != GET_DATA_REPLY {
		return ReplyTypeMismatch(GET_DATA_REPLY)
	}
	return nil
}

type DeleteDataRequest struct {
	request
	ID       []types.ObjectID `json:"id"`
	Fastpath bool             `json:"fastpath"`
}

type DeleteDataReply struct {
	reply
}

func WriteDeleteDataRequest(id []types.ObjectID, fastpath bool) []byte {
	var request DeleteDataRequest
	request.Type = DELETE_DATA_REQUEST
	request.ID = id
	request.Fastpath = fastpath
	return encodeMsg(request)
}

func (reply *DeleteDataReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != DELETE_DATA_REPLY {
		return ReplyTypeMismatch(DELETE_DATA_REPLY)
	}
	return nil
}

type ExistsRequest struct {
	request
	ID types.ObjectID `json:"id"`
}

type ExistsReply struct {
	reply
	Exists bool `json:"exists"`
}

func WriteExistsRequest(id types.ObjectID) []byte {
	var request ExistsRequest
	request.Type = EXISTS_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *ExistsReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != EXISTS_REPLY {
		return ReplyTypeMismatch(EXISTS_REPLY)
	}
	return nil
}

type PersistRequest struct {
	request
	ID types.ObjectID `json:"id"`
}

type PersistReply struct {
	reply
}

func WritePersistRequest(id types.ObjectID) []byte {
	var request PersistRequest
	request.Type = PERSIST_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *PersistReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != PERSIST_REPLY {
		return ReplyTypeMismatch(PERSIST_REPLY)
	}
	return nil
}

type IfPersistRequest struct {
	request
	ID types.ObjectID `json:"id"`
}

type IfPersistReply struct {
	reply
	Persist bool `json:"persist"`
}

func WriteIfPersistRequest(id types.ObjectID) []byte {
	var request IfPersistRequest
	request.Type = IF_PERSIST_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *IfPersistReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != IF_PERSIST_REPLY {
		return ReplyTypeMismatch(IF_PERSIST_REPLY)
	}
	return nil
}

type LabelRequest struct {
	request
	ID     types.ObjectID `json:"id"`
	Keys   []string       `json:"keys"`
	Values []string       `json:"values"`
}

type LabelReply struct {
	reply
}

func WriteLabelRequest(id types.ObjectID, key string, value string) []byte {
	var request LabelRequest
	request.Type = LABEL_REQUEST
	request.ID = id
	request.Keys = []string{key}
	request.Values = []string{value}
	return encodeMsg(request)
}

func WriteLabelsRequest(id types.ObjectID, keys []string, values []string) []byte {
	var request LabelRequest
	request.Type = LABEL_REQUEST
	request.ID = id
	request.Keys = keys
	request.Values = values
	return encodeMsg(request)
}

func (reply *LabelReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != LABEL_REPLY {
		return ReplyTypeMismatch(LABEL_REPLY)
	}
	return nil
}

type ClearRequest struct {
	request
}

type ClearReply struct {
	reply
}

func WriteClearRequest() []byte {
	var request ClearRequest
	request.Type = CLEAR_REQUEST
	return encodeMsg(request)
}

func (reply *ClearReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != CLEAR_REPLY {
		return ReplyTypeMismatch(CLEAR_REPLY)
	}
	return nil
}

type PutNameRequest struct {
	request
	ID   types.ObjectID `json:"object_id"`
	Name string         `json:"name"`
}

type PutNameReply struct {
	reply
}

func WritePutNameRequest(id types.ObjectID, name string) []byte {
	var request PutNameRequest
	request.Type = PUT_NAME_REQUEST
	request.ID = id
	request.Name = name
	return encodeMsg(request)
}

func (reply *PutNameReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != PUT_NAME_REPLY {
		return ReplyTypeMismatch(PUT_NAME_REPLY)
	}
	return nil
}

type GetNameRequest struct {
	request
	Name string `json:"name"`
	Wait bool   `json:"wait"`
}

type GetNameReply struct {
	reply
	ID types.ObjectID `json:"object_id"`
}

func WriteGetNameRequest(name string, wait bool) []byte {
	var request GetNameRequest
	request.Type = GET_NAME_REQUEST
	request.Name = name
	request.Wait = wait
	return encodeMsg(request)
}

func (reply *GetNameReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != GET_NAME_REPLY {
		return ReplyTypeMismatch(GET_NAME_REPLY)
	}
	return nil
}

type ListNameRequest struct {
	request
	Pattern string `json:"pattern"`
	Regex   bool   `json:"regex"`
	Limit   int    `json:"limit"`
}

type ListNameReply struct {
	reply
	Names map[string]types.ObjectID `json:"names"`
}

func WriteListNameRequest(pattern string, regex bool, limit int) []byte {
	var request ListNameRequest
	request.Type = LIST_NAME_REQUEST
	request.Pattern = pattern
	request.Regex = regex
	request.Limit = limit
	return encodeMsg(request)
}

func (reply *ListNameReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != LIST_NAME_REPLY {
		return ReplyTypeMismatch(LIST_NAME_REPLY)
	}
	return nil
}

type ListDataRequest struct {
	request
	Pattern string `json:"pattern"`
	Regex   bool   `json:"regex"`
	Limit   int    `json:"limit"`
}

type ListDataReply struct {
	reply
	Content map[string]map[string]any `json:"content"`
}

func WriteListDataRequest(pattern string, regex bool, limit int) []byte {
	var request ListDataRequest
	request.Type = LIST_DATA_REQUEST
	request.Pattern = pattern
	request.Regex = regex
	request.Limit = limit
	return encodeMsg(request)
}

func (reply *ListDataReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}

	if reply.Type != GET_DATA_REPLY {
		return ReplyTypeMismatch(LIST_DATA_REPLY)
	}
	return nil
}

type DropNameRequest struct {
	request
	Name string `json:"name"`
}

type DropNameReply struct {
	reply
}

func WriteDropNameRequest(name string) []byte {
	var request DropNameRequest
	request.Type = DROP_NAME_REQUEST
	request.Name = name
	return encodeMsg(request)
}

func (reply *DropNameReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != DROP_NAME_REPLY {
		return ReplyTypeMismatch(DROP_NAME_REPLY)
	}
	return nil
}

type EvictRequest struct {
	request
	IDs []types.ObjectID `json:"ids"`
}

type EvictReply struct {
	reply
}

func WriteEvictRequest(ids []types.ObjectID) []byte {
	var request EvictRequest
	request.Type = EVICT_REQUEST
	request.IDs = ids
	return encodeMsg(request)
}

func (reply *EvictReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != EVICT_REPLY {
		return ReplyTypeMismatch(EVICT_REPLY)
	}
	return nil
}

type LoadRequest struct {
	request
	IDs []types.ObjectID `json:"ids"`
	Pin bool             `json:"pin"`
}

type LoadReply struct {
	reply
}

func WriteLoadRequest(ids []types.ObjectID, pin bool) []byte {
	var request LoadRequest
	request.Type = LOAD_REQUEST
	request.IDs = ids
	request.Pin = pin
	return encodeMsg(request)
}

func (reply *LoadReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != LOAD_REPLY {
		return ReplyTypeMismatch(LOAD_REPLY)
	}
	return nil
}

type UnpinRequest struct {
	request
	IDs []types.ObjectID `json:"ids"`
}

type UnpinReply struct {
	reply
}

func WriteUnpinRequest(ids []types.ObjectID) []byte {
	var request UnpinRequest
	request.Type = UNPIN_REQUEST
	request.IDs = ids
	return encodeMsg(request)
}

func (reply *UnpinReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != UNPIN_REPLY {
		return ReplyTypeMismatch(UNPIN_REPLY)
	}
	return nil
}

type IsSpilledRequest struct {
	request
	ID types.ObjectID `json:"id"`
}

type IsSpilledReply struct {
	reply
	Spilled bool `json:"is_spilled"`
}

func WriteIsSpilledRequest(id types.ObjectID) []byte {
	var request IsSpilledRequest
	request.Type = IS_SPILLED_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *IsSpilledReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != IS_SPILLED_REPLY {
		return ReplyTypeMismatch(IS_SPILLED_REPLY)
	}
	return nil
}

type IsInUseRequest struct {
	request
	ID types.ObjectID `json:"id"`
}

type IsInUseReply struct {
	reply
	InUse bool `json:"is_in_use"`
}

func WriteIsInUseRequest(id types.ObjectID) []byte {
	var request IsInUseRequest
	request.Type = IS_IN_USE_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *IsInUseReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != IS_IN_USE_REPLY {
		return ReplyTypeMismatch(IS_IN_USE_REPLY)
	}
	return nil
}

type ClusterMetaRequest struct {
	request
}

type ClusterMetaReply struct {
	reply
	Meta any `json:"meta"`
}

func WriteClusterMetaRequest() []byte {
	var request ClusterMetaRequest
	request.Type = CLUSTER_META_REQUEST
	return encodeMsg(request)
}

func (reply *ClusterMetaReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != CLUSTER_META_REPLY {
		return ReplyTypeMismatch(CLUSTER_META_REPLY)
	}
	return nil
}

type InstanceStatusRequest struct {
	request
}

type InstanceStatusReply struct {
	reply
	Meta any `json:"meta"`
}

func WriteInstanceStatusRequest() []byte {
	var request InstanceStatusRequest
	request.Type = INSTANCE_STATUS_REQUEST
	return encodeMsg(request)
}

func (reply *InstanceStatusReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != INSTANCE_STATUS_REPLY {
		return ReplyTypeMismatch(INSTANCE_STATUS_REPLY)
	}
	return nil
}

type MigrateObjectRequest struct {
	request
	ID types.ObjectID `json:"object_id"`
}

type MigrateObjectReply struct {
	reply
	ID types.ObjectID `json:"object_id"`
}

func WriteMigrateObjectRequest(id types.ObjectID) []byte {
	var request MigrateObjectRequest
	request.Type = MIGRATE_OBJECT_REQUEST
	request.ID = id
	return encodeMsg(request)
}

func (reply *MigrateObjectReply) Check() error {
	if err := reply.check(); err != nil {
		return err
	}
	if reply.Type != MIGRATE_OBJECT_REPLY {
		return ReplyTypeMismatch(MIGRATE_OBJECT_REPLY)
	}
	return nil
}

// Encoding should be fairly safe, so we choose to panic
func encodeMsg(data any) []byte {
	if bytes, err := json.Marshal(data); err != nil {
		panic(fmt.Sprintf("failed to marshal data to json: %+v", err))
	} else {
		return bytes
	}
}
