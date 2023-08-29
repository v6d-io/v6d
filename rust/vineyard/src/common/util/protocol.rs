// Copyright 2020-2023 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use num_traits::ToPrimitive;
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::super::memory::Payload;
use super::json::*;
use super::status::*;
use super::uuid::*;

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Command;

#[allow(dead_code)]
impl Command {
    pub const REGISTER_REQUEST: &'static str = "register_request";
    pub const REGISTER_REPLY: &'static str = "register_reply";
    pub const EXIT_REQUEST: &'static str = "exit_request";
    pub const EXIT_REPLY: &'static str = "exit_reply";

    // Blobs APIs
    pub const CREATE_BUFFER_REQUEST: &'static str = "create_buffer_request";
    pub const CREATE_BUFFER_REPLY: &'static str = "create_buffer_reply";
    pub const CREATE_DISK_BUFFER_REQUEST: &'static str = "create_disk_buffer_request";
    pub const CREATE_DISK_BUFFER_REPLY: &'static str = "create_disk_buffer_reply";
    pub const CREATE_GPU_BUFFER_REQUEST: &'static str = "create_gpu_buffer_request";
    pub const CREATE_GPU_BUFFER_REPLY: &'static str = "create_gpu_buffer_reply";
    pub const SEAL_BUFFER_REQUEST: &'static str = "seal_request";
    pub const SEAL_BUFFER_REPLY: &'static str = "seal_reply";
    pub const GET_BUFFERS_REQUEST: &'static str = "get_buffers_request";
    pub const GET_BUFFERS_REPLY: &'static str = "get_buffers_reply";
    pub const GET_GPU_BUFFERS_REQUEST: &'static str = "get_gpu_buffers_request";
    pub const GET_GPU_BUFFERS_REPLY: &'static str = "get_gpu_buffers_reply";
    pub const DROP_BUFFER_REQUEST: &'static str = "drop_buffer_request";
    pub const DROP_BUFFER_REPLY: &'static str = "drop_buffer_reply";

    pub const REQUEST_FD_REQUEST: &'static str = "request_fd_request";
    pub const REQUEST_FD_REPLY: &'static str = "request_fd_reply";

    pub const CREATE_REMOTE_BUFFER_REQUEST: &'static str = "create_remote_buffer_request";
    pub const GET_REMOTE_BUFFERS_REQUEST: &'static str = "get_remote_buffers_request";

    pub const INCREASE_REFERENCE_COUNT_REQUEST: &'static str = "increase_reference_count_request";
    pub const INCREASE_REFERENCE_COUNT_REPLY: &'static str = "increase_reference_count_reply";
    pub const RELEASE_REQUEST: &'static str = "release_request";
    pub const RELEASE_REPLY: &'static str = "release_reply";
    pub const DEL_DATA_WITH_FEEDBACKS_REQUEST: &'static str = "del_data_with_feedbacks_request";
    pub const DEL_DATA_WITH_FEEDBACKS_REPLY: &'static str = "del_data_with_feedbacks_reply";

    pub const CREATE_BUFFER_PLASMA_REQUEST: &'static str = "create_buffer_by_plasma_request";
    pub const CREATE_BUFFER_PLASMA_REPLY: &'static str = "create_buffer_by_plasma_reply";
    pub const GET_BUFFERS_PLASMA_REQUEST: &'static str = "get_buffers_by_plasma_request";
    pub const GET_BUFFERS_PLASMA_REPLY: &'static str = "get_buffers_by_plasma_reply";
    pub const PLASMA_SEAL_REQUEST: &'static str = "plasma_seal_request";
    pub const PLASMA_SEAL_REPLY: &'static str = "plasma_seal_reply";
    pub const PLASMA_RELEASE_REQUEST: &'static str = "plasma_release_request";
    pub const PLASMA_RELEASE_REPLY: &'static str = "plasma_release_reply";
    pub const PLASMA_DEL_DATA_REQUEST: &'static str = "plasma_delete_data_request";
    pub const PLASMA_DEL_DATA_REPLY: &'static str = "plasma_delete_data_reply";

    // Metadata APIs
    pub const CREATE_DATA_REQUEST: &'static str = "create_data_request";
    pub const CREATE_DATA_REPLY: &'static str = "create_data_reply";
    pub const GET_DATA_REQUEST: &'static str = "get_data_request";
    pub const GET_DATA_REPLY: &'static str = "get_data_reply";
    pub const LIST_DATA_REQUEST: &'static str = "list_data_request";
    pub const LIST_DATA_REPLY: &'static str = "list_data_reply";
    pub const DELETE_DATA_REQUEST: &'static str = "del_data_request";
    pub const DELETE_DATA_REPLY: &'static str = "del_data_reply";
    pub const EXISTS_REQUEST: &'static str = "exists_request";
    pub const EXISTS_REPLY: &'static str = "exists_reply";
    pub const PERSIST_REQUEST: &'static str = "persist_request";
    pub const PERSIST_REPLY: &'static str = "persist_reply";
    pub const IF_PERSIST_REQUEST: &'static str = "if_persist_request";
    pub const IF_PERSIST_REPLY: &'static str = "if_persist_reply";
    pub const LABEL_REQUEST: &'static str = "label_request";
    pub const LABEL_REPLY: &'static str = "label_reply";
    pub const CLEAR_REQUEST: &'static str = "clear_request";
    pub const CLEAR_REPLY: &'static str = "clear_reply";

    // Stream APIs
    pub const CREATE_STREAM_REQUEST: &'static str = "create_stream_request";
    pub const CREATE_STREAM_REPLY: &'static str = "create_stream_reply";
    pub const OPEN_STREAM_REQUEST: &'static str = "open_stream_request";
    pub const OPEN_STREAM_REPLY: &'static str = "open_stream_reply";
    pub const GET_NEXT_STREAM_CHUNK_REQUEST: &'static str = "get_next_stream_chunk_request";
    pub const GET_NEXT_STREAM_CHUNK_REPLY: &'static str = "get_next_stream_chunk_reply";
    pub const PUSH_NEXT_STREAM_CHUNK_REQUEST: &'static str = "push_next_stream_chunk_request";
    pub const PUSH_NEXT_STREAM_CHUNK_REPLY: &'static str = "push_next_stream_chunk_reply";
    pub const PULL_NEXT_STREAM_CHUNK_REQUEST: &'static str = "pull_next_stream_chunk_request";
    pub const PULL_NEXT_STREAM_CHUNK_REPLY: &'static str = "pull_next_stream_chunk_reply";
    pub const STOP_STREAM_REQUEST: &'static str = "stop_stream_request";
    pub const STOP_STREAM_REPLY: &'static str = "stop_stream_reply";
    pub const DROP_STREAM_REQUEST: &'static str = "drop_stream_request";
    pub const DROP_STREAM_REPLY: &'static str = "drop_stream_reply";

    // Names APIs
    pub const PUT_NAME_REQUEST: &'static str = "put_name_request";
    pub const PUT_NAME_REPLY: &'static str = "put_name_reply";
    pub const GET_NAME_REQUEST: &'static str = "get_name_request";
    pub const GET_NAME_REPLY: &'static str = "get_name_reply";
    pub const LIST_NAME_REQUEST: &'static str = "list_name_request";
    pub const LIST_NAME_REPLY: &'static str = "list_name_reply";
    pub const DROP_NAME_REQUEST: &'static str = "drop_name_request";
    pub const DROP_NAME_REPLY: &'static str = "drop_name_reply";

    // Arena APIs
    pub const MAKE_ARENA_REQUEST: &'static str = "make_arena_request";
    pub const MAKE_ARENA_REPLY: &'static str = "make_arena_reply";
    pub const FINALIZE_ARENA_REQUEST: &'static str = "finalize_arena_request";
    pub const FINALIZE_ARENA_REPLY: &'static str = "finalize_arena_reply";

    // Session APIs
    pub const NEW_SESSION_REQUEST: &'static str = "new_session_request";
    pub const NEW_SESSION_REPLY: &'static str = "new_session_reply";
    pub const DELETE_SESSION_REQUEST: &'static str = "delete_session_request";
    pub const DELETE_SESSION_REPLY: &'static str = "delete_session_reply";

    pub const MOVE_BUFFERS_OWNERSHIP_REQUEST: &'static str = "move_buffers_ownership_request";
    pub const MOVE_BUFFERS_OWNERSHIP_REPLY: &'static str = "move_buffers_ownership_reply";

    // Spill APIs
    pub const EVICT_REQUEST: &'static str = "evict_request";
    pub const EVICT_REPLY: &'static str = "evict_reply";
    pub const LOAD_REQUEST: &'static str = "load_request";
    pub const LOAD_REPLY: &'static str = "load_reply";
    pub const UNPIN_REQUEST: &'static str = "unpin_request";
    pub const UNPIN_REPLY: &'static str = "unpin_reply";
    pub const IS_SPILLED_REQUEST: &'static str = "is_spilled_request";
    pub const IS_SPILLED_REPLY: &'static str = "is_spilled_reply";
    pub const IS_IN_USE_REQUEST: &'static str = "is_in_use_request";
    pub const IS_IN_USE_REPLY: &'static str = "is_in_use_reply";

    // Meta APIs
    pub const CLUSTER_META_REQUEST: &'static str = "cluster_meta";
    pub const CLUSTER_META_REPLY: &'static str = "cluster_meta";
    pub const INSTANCE_STATUS_REQUEST: &'static str = "instance_status_request";
    pub const INSTANCE_STATUS_REPLY: &'static str = "instance_status_reply";
    pub const MIGRATE_OBJECT_REQUEST: &'static str = "migrate_object_request";
    pub const MIGRATE_OBJECT_REPLY: &'static str = "migrate_object_reply";
    pub const SHALLOW_COPY_REQUEST: &'static str = "shallow_copy_request";
    pub const SHALLOW_COPY_REPLY: &'static str = "shallow_copy_reply";
    pub const DEBUG_REQUEST: &'static str = "debug_command";
    pub const DEBUG_REPLY: &'static str = "debug_reply";
}

fn check_ipc_error<'a>(root: &'a JSON, reply_type: &str) -> Result<()> {
    if root.contains_key("code") {
        let code = root["code"].as_u64().unwrap_or(0);
        if code != 0 {
            let mut error_message: String = "unable to find error message in the response".into();
            if let Some(message) = root.get("message") {
                if let Some(message) = message.as_str() {
                    error_message = message.into();
                }
            }
            return Err(VineyardError::new(
                unsafe { std::mem::transmute(code as u8) },
                error_message,
            ));
        }
    }
    if let Some(message_type) = root.get("type") {
        return vineyard_assert(
            message_type.as_str().map_or(false, |t| t == reply_type),
            format!("unexpected reply type: '{}'", message_type),
        );
    } else {
        return vineyard_assert(false, "no 'type' field in the response");
    }
}

#[derive(Debug, Default)]
pub struct RegisterRequest {
    pub version: String,
    pub store_type: String,
    pub session_id: i64,
    pub username: String,
    pub password: String,
    pub support_rpc_compression: bool,
}

pub fn write_register_request(r: RegisterRequest) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::REGISTER_REQUEST,
        "version": r.version,
        "store_type": r.store_type,
        "session_id": r.session_id,
        "username": r.username,
        "password": r.password,
        "support_rpc_compression": r.support_rpc_compression,
    }));
}

#[derive(Debug, Default)]
pub struct RegisterReply {
    pub ipc_socket: String,
    pub rpc_endpoint: String,
    pub instance_id: InstanceID,
    pub version: String,
    pub support_rpc_compression: bool,
}

pub fn read_register_reply(message: &str) -> Result<RegisterReply> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::REGISTER_REPLY)?;

    return Ok(RegisterReply {
        ipc_socket: get_string(root, "ipc_socket")?.into(),
        rpc_endpoint: get_string(root, "rpc_endpoint")?.into(),
        instance_id: get_uint(root, "instance_id")?,
        version: get_string(root, "version")?.into(),
        support_rpc_compression: get_bool_or(root, "support_rpc_compression", false),
    });
}

pub fn write_exit_request() -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::EXIT_REQUEST,
    }));
}

#[derive(Debug, Default)]
pub struct CreateBufferReply {
    pub id: ObjectID,
    pub payload: Payload,
    pub fd: i32,
}

pub fn write_create_buffer_request(size: usize) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::CREATE_BUFFER_REQUEST,
        "size": size,
    }));
}

pub fn read_create_buffer_reply(message: &str) -> Result<CreateBufferReply> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::CREATE_BUFFER_REPLY)?;

    let created = parse_json_object(&root["created"])?;
    let payload = Payload::from_json(&created)?;

    return Ok(CreateBufferReply {
        id: get_uint(root, "id")?,
        payload: payload,
        fd: get_int32::<i32>(root, "fd")?,
    });
}

pub fn write_create_disk_buffer_request(size: usize, path: &str) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::CREATE_DISK_BUFFER_REQUEST,
        "size": size,
        "path": path,
    }));
}

pub fn read_create_disk_buffer_reply(message: &str) -> Result<CreateBufferReply> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::CREATE_DISK_BUFFER_REPLY)?;

    let created = parse_json_object(&root["created"])?;
    let payload = Payload::from_json(&created)?;

    return Ok(CreateBufferReply {
        id: get_uint(root, "id")?,
        payload: payload,
        fd: get_int::<i64>(root, "fd")?
            .to_i32()
            .ok_or(VineyardError::io_error(
                "fd received from server must be a 32-bit integer",
            ))?,
    });
}

#[derive(Debug, Default)]
pub struct CreateGPUBufferReply {
    pub id: ObjectID,
    pub payload: Payload,
    pub handle: Vec<i64>,
}

pub fn write_create_gpu_buffer_request(size: usize) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::CREATE_GPU_BUFFER_REQUEST,
        "size": size,
    }));
}

pub fn read_create_gpu_buffer_reply(message: &str) -> Result<CreateGPUBufferReply> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::CREATE_GPU_BUFFER_REPLY)?;

    let created = parse_json_object(&root["created"])?;
    let payload = Payload::from_json(&created)?;

    let handle = root["handle"]
        .as_array()
        .ok_or(VineyardError::io_error("handle is not an array"))?
        .iter()
        .map(|v| {
            v.as_i64()
                .ok_or(VineyardError::io_error("handle is not an integer"))
        })
        .collect::<Result<Vec<i64>>>()?;

    return Ok(CreateGPUBufferReply {
        id: get_uint(root, "id")?,
        payload: payload,
        handle: handle,
    });
}

pub fn write_seal_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::SEAL_BUFFER_REQUEST,
        "object_id": id,
    }));
}

pub fn read_seal_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::SEAL_BUFFER_REPLY)?;

    return Ok(());
}

#[derive(Debug, Default)]
pub struct GetBuffersReply {
    pub payloads: Vec<Payload>,
    pub fds: Vec<i32>,
    pub compress: bool,
}

pub fn write_get_buffers_request(ids: &[ObjectID], unsafe_: bool) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::GET_BUFFERS_REQUEST,
        "ids": ids,
        "unsafe": unsafe_,
    }));
}

pub fn read_get_buffers_reply(message: &str) -> Result<GetBuffersReply> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::GET_BUFFERS_REPLY)?;

    let mut reply = GetBuffersReply::default();

    if let Some(Value::Array(ref payloads)) = root.get("payloads") {
        for payload in payloads {
            reply
                .payloads
                .push(Payload::from_json(payload.as_object().ok_or(
                    VineyardError::io_error(
                        "invalid get_buffers reply: payload in message is not a JSON object",
                    ),
                )?)?);
        }
    } else {
        let num: i64 = get_int(root, "num")?;
        for i in 0..num {
            match root[&i.to_string()] {
                Value::Object(ref payload) => {
                    reply.payloads.push(Payload::from_json(payload)?);
                }
                _ => {
                    return Err(VineyardError::io_error(
                        "invalid get_buffers reply: payload in message is not a JSON object",
                    ));
                }
            }
        }
    }

    if let Some(Value::Array(ref fds)) = root.get("fds") {
        for fd in fds {
            reply.fds.push(
                fd.as_i64()
                    .ok_or(VineyardError::io_error("fd is not an integer"))?
                    .to_i32()
                    .ok_or(VineyardError::io_error(
                        "fd received from server must be a 32-bit integer",
                    ))?,
            );
        }
    }
    return Ok(reply);
}

pub fn write_drop_buffer_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::DROP_BUFFER_REQUEST,
        "id": id,
    }));
}

pub fn read_drop_buffer_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::DROP_BUFFER_REPLY)?;

    return Ok(());
}

pub fn write_create_remote_buffer_request(size: usize, compress: bool) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::CREATE_REMOTE_BUFFER_REQUEST,
        "size": size,
        "compress": compress,
    }));
}

pub fn read_create_remote_buffer_reply(message: &str) -> Result<CreateBufferReply> {
    return read_create_buffer_reply(message);
}

pub fn write_get_remote_buffers_request(ids: &[ObjectID]) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::GET_REMOTE_BUFFERS_REQUEST,
        "ids": ids,
    }));
}

pub fn read_get_remote_buffers_reply(message: &str) -> Result<GetBuffersReply> {
    return read_get_buffers_reply(message);
}

pub fn write_increase_reference_count_request(id: &[ObjectID]) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::INCREASE_REFERENCE_COUNT_REQUEST,
        "ids": id,
    }));
}

pub fn read_increase_reference_count_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::INCREASE_REFERENCE_COUNT_REPLY)?;

    return Ok(());
}

pub fn write_release_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::RELEASE_REQUEST,
        "object_id": id,
    }));
}

pub fn read_release_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::RELEASE_REPLY)?;

    return Ok(());
}

#[derive(Debug, Default)]
pub struct CreateDataReply {
    pub id: ObjectID,
    pub signature: Signature,
    pub instance_id: InstanceID,
}

pub fn write_create_data_request(content: &JSON) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::CREATE_DATA_REQUEST,
        "content": content,
    }));
}

pub fn read_create_data_reply(message: &str) -> Result<CreateDataReply> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, "create_data_reply")?;

    return Ok(CreateDataReply {
        id: get_uint(root, "id")?,
        signature: get_uint(root, "signature")?,
        instance_id: get_uint(root, "instance_id")?,
    });
}

pub fn write_get_data_request(id: ObjectID, sync_remote: bool, wait: bool) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::GET_DATA_REQUEST,
        "id": vec![id],
        "sync_remote": sync_remote,
        "wait": wait,
    }));
}

pub fn read_get_data_reply(message: &str) -> Result<JSON> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, "get_data_reply")?;

    match root["content"] {
        Value::Array(ref content) => {
            if content.len() != 1 {
                return Err(VineyardError::io_error(
                    "failed to read get_data reply: content array's length is not 1",
                ));
            }
            return Ok(parse_json_object(&content[0])?.clone());
        }
        Value::Object(ref content) => match content.iter().next() {
            None => {
                return Err(VineyardError::io_error(
                    "failed to read get_data reply: content dict's length is not 1",
                ));
            }
            Some((_, meta)) => {
                return Ok(parse_json_object(meta)?.clone());
            }
        },
        _ => {
            return Err(VineyardError::io_error(
                "failed to read get_data reply: content is not an array or a dict",
            ));
        }
    }
}

pub fn write_get_data_batch_request(
    ids: &[ObjectID],
    sync_remote: bool,
    wait: bool,
) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::GET_DATA_REQUEST,
        "id": ids,
        "sync_remote": sync_remote,
        "wait": wait,
    }));
}

pub fn read_get_data_batch_reply(message: &str) -> Result<HashMap<ObjectID, JSON>> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, "get_data_reply")?;

    match root["content"] {
        Value::Array(ref content) => {
            let mut data = HashMap::new();
            for item in content {
                let object = parse_json_object(&item)?;
                data.insert(
                    object_id_from_string(get_string(object, "id")?)?,
                    object.clone(),
                );
            }
            return Ok(data);
        }
        Value::Object(ref content) => {
            let mut data = HashMap::new();
            for (id, object) in content.iter() {
                data.insert(
                    object_id_from_string(id)?,
                    parse_json_object(object)?.clone(),
                );
            }
            return Ok(data);
        }
        _ => {
            return Err(VineyardError::io_error(
                "failed to read get_data reply: content is not an array or a dict",
            ));
        }
    }
}

pub fn write_list_data_request(pattern: &str, regex: bool, limit: usize) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::LIST_DATA_REQUEST,
        "pattern": pattern,
        "regex": regex,
        "limit": limit,
    }));
}

pub fn read_list_data_reply(message: &str) -> Result<Vec<JSON>> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, "list_data_reply")?;

    let mut reply = Vec::new();
    match root["content"] {
        Value::Array(ref data) => {
            for item in data {
                reply.push(parse_json_object(&item)?.clone());
            }
            return Ok(reply);
        }
        _ => {
            return Err(VineyardError::io_error(
                "failed to read list_data reply: data is not an array",
            ));
        }
    }
}

pub fn write_delete_data_request(
    id: ObjectID,
    force: bool,
    deep: bool,
    fastpath: bool,
) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::DELETE_DATA_REQUEST,
        "id": vec![id],
        "force": force,
        "deep:": deep,
        "fastpath": fastpath,
    }));
}

pub fn write_delete_data_batch_request(
    ids: &[ObjectID],
    force: bool,
    deep: bool,
    fastpath: bool,
) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::DELETE_DATA_REQUEST,
        "id": ids,
        "force": force,
        "deep:": deep,
        "fastpath": fastpath,
    }));
}

pub fn read_delete_data_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::DELETE_DATA_REPLY)?;

    return Ok(());
}

pub fn write_exists_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::EXISTS_REQUEST,
        "id": id,
    }));
}

pub fn read_exists_reply(message: &str) -> Result<bool> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::EXISTS_REPLY)?;

    return Ok(get_bool_or(root, "exists", false));
}

pub fn write_persist_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::PERSIST_REQUEST,
        "id": id,
    }));
}

pub fn read_persist_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::PERSIST_REPLY)?;

    return Ok(());
}

pub fn write_if_persist_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::IF_PERSIST_REQUEST,
        "id": id,
    }));
}

pub fn read_if_persist_reply(message: &str) -> Result<bool> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::IF_PERSIST_REPLY)?;

    return Ok(get_bool_or(root, "persist", false));
}

pub fn write_label_request(id: ObjectID, keys: &[String], values: &[String]) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::LABEL_REQUEST,
        "id": id,
        "keys": keys,
        "values": values,
    }));
}

pub fn read_label_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::LABEL_REPLY)?;

    return Ok(());
}

pub fn write_clear_request() -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::CLEAR_REQUEST,
    }));
}

pub fn read_clear_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::CLEAR_REPLY)?;

    return Ok(());
}

pub fn write_put_name_request(object_id: ObjectID, name: &str) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::PUT_NAME_REQUEST,
        "object_id": object_id,
        "name": name,
    }));
}

pub fn read_put_name_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::PUT_NAME_REPLY)?;

    return Ok(());
}

pub fn write_get_name_request(name: &str, wait: bool) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::GET_NAME_REQUEST,
        "name": name,
        "wait": wait,
    }));
}

pub fn read_get_name_reply(message: &str) -> Result<ObjectID> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::GET_NAME_REPLY)?;

    return get_uint(root, "object_id");
}

pub fn write_list_name_request(pattern: &str, regex: bool, limit: usize) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::LIST_NAME_REQUEST,
        "pattern": pattern,
        "regex": regex,
        "limit": limit,
    }));
}

pub fn read_list_name_reply(message: &str) -> Result<HashMap<String, ObjectID>> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::LIST_NAME_REPLY)?;

    let names = parse_json_object(
        root.get("names")
            .ok_or(VineyardError::io_error("message does not contain names"))?,
    )?;
    let mut result = HashMap::new();
    for (name, value) in names {
        match value.as_u64() {
            None => {}
            Some(id) => {
                result.insert(name.clone(), id);
            }
        }
    }
    return Ok(result);
}

pub fn write_drop_name_request(name: &str) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::DROP_NAME_REQUEST,
        "name": name,
    }));
}

pub fn read_drop_name_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::DROP_NAME_REPLY)?;

    return Ok(());
}

pub fn write_evict_request(ids: &[ObjectID]) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::EVICT_REQUEST,
        "ids": ids,
    }));
}

pub fn read_evict_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::EVICT_REPLY)?;

    return Ok(());
}

pub fn write_load_request(ids: &[ObjectID], pin: bool) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::LOAD_REQUEST,
        "ids": ids,
        "pin": pin,
    }));
}

pub fn read_load_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::LOAD_REPLY)?;

    return Ok(());
}

pub fn write_unpin_request(ids: &[ObjectID]) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::UNPIN_REQUEST,
        "ids": ids,
    }));
}

pub fn read_unpin_reply(message: &str) -> Result<()> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::UNPIN_REPLY)?;

    return Ok(());
}

pub fn write_is_spilled_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::IS_SPILLED_REQUEST,
        "id": id,
    }));
}

pub fn read_is_spilled_reply(message: &str) -> Result<bool> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::IS_SPILLED_REPLY)?;

    return Ok(get_bool_or(root, "is_spilled", false));
}

pub fn write_is_inuse_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::IS_IN_USE_REQUEST,
        "id": id,
    }));
}

pub fn read_is_inuse_reply(message: &str) -> Result<bool> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::IS_IN_USE_REPLY)?;

    return Ok(get_bool_or(root, "is_in_use", false));
}

pub fn write_migrate_object_request(id: ObjectID) -> JSONResult<String> {
    return serde_json::to_string(&json!({
        "type": Command::MIGRATE_OBJECT_REQUEST,
        "object_id": id,
    }));
}

pub fn read_migrate_object_reply(message: &str) -> Result<ObjectID> {
    let root: Value = serde_json::from_str(message)?;
    let root = parse_json_object(&root)?;
    check_ipc_error(&root, Command::MIGRATE_OBJECT_REPLY)?;

    return get_uint(root, "object_id");
}
