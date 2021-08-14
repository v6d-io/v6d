use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Map, Value};

use std::collections::{HashMap, HashSet};
use std::io::{self, Error, ErrorKind};
use std::ptr;

use super::{InstanceID, ObjectID};

enum CommandType {
    RegisterRequest,
    RegisterReply,
    ExitRequest,
    ExitReply,
}

#[derive(Debug)]
pub struct Payload {
    object_id: ObjectID,
    store_fd: i32,
    arena_fd: i32,
    data_offset: isize,
    data_size: i64,
    map_size: i64,
    pointer: *const u8, // TODO: Check if this is right for nullptr
}

impl Default for Payload {
    fn default() -> Self {
        Payload {
            object_id: 0,
            store_fd: -1,
            arena_fd: -1,
            data_offset: 0,
            data_size: 0,
            map_size: 0,
            pointer: ptr::null(), // nullptr
        }
    }
}

impl Payload {
    pub fn new() -> Payload {
        let ret: Payload = Default::default();
        ret
    }

    pub fn to_json(&self) -> Value {
        json!({
            "object_id": self.object_id, 
            "store_fd": self.store_fd, 
            "data_offset": self.data_offset, 
            "data_size": self.data_size,
            "map_size": self.map_size})
    }

    pub fn from_json(&mut self, tree: &Value) {
        self.object_id = tree["object_id"].as_u64().unwrap() as InstanceID;
        self.store_fd = tree["store_fd"].as_i64().unwrap() as i32;
        self.data_offset = tree["data_offset"].as_i64().unwrap() as isize;
        self.data_size = tree["data_size"].as_i64().unwrap();
        self.map_size = tree["map_size"].as_i64().unwrap();
        self.pointer = ptr::null(); //  nullptr
    }
}

pub fn RETURN_ON_ASSERT(b: bool) {
    if !b {
        panic!()
    }
}

pub fn CHECK_IPC_ERROR(tree: &Value, root_type: &str) {
    if tree.as_object().unwrap().contains_key("code") {
        tree["code"].as_u64().unwrap_or(0);
        tree["message"].as_str().unwrap_or("");
    }
    RETURN_ON_ASSERT(tree["type"].as_str().unwrap() == root_type);
}

// Question: use unsafe extern "C" fn strtoull?
pub fn objectid_from_string(s: String) -> ObjectID {
    0
}

// Convert JSON Value to a String
pub fn encode_msg(msg: Value) -> String {
    let ret = serde_json::to_string(&msg).unwrap();
    ret
}

// Write functions: Derive and write JSON message to a String
pub fn write_register_request() -> String {
    let msg = json!({"type": "register_request", "version": "0.2.6" });
    encode_msg(msg)
}

// Read functions: Read JSON root to variants of ipc instance
pub fn read_register_request(root: Value) -> Result<String, Error> {
    RETURN_ON_ASSERT(root["type"] == "register_request");
    Ok(root["version"].as_str().unwrap_or("0.0.0").to_string())
}

pub fn write_register_reply(
    ipc_socket: String,
    rpc_endpoint: String,
    instance_id: InstanceID,
) -> String {
    let msg = json!({
        "type": "register_reply",
        "ipc_socket": ipc_socket,
        "rpc_endpoint": rpc_endpoint,
        "instance_id": instance_id,
        "version": "0.2.6"
    });
    encode_msg(msg)
}

pub struct RegisterReply {
    ipc_socket: String,
    rpc_endpoint: String,
    instance_id: InstanceID,
    version: String,
}

pub fn read_register_reply(root: Value) -> Result<RegisterReply, Error> {
    CHECK_IPC_ERROR(&root, "register_reply");
    let ipc_socket = root["ipc_socket"].as_str().unwrap().to_string();
    let rpc_endpoint = root["rpc_endpoint"].as_str().unwrap().to_string();
    let instance_id = root["instance_id"].as_u64().unwrap() as InstanceID;
    let version = root["version"].as_str().unwrap_or("0.0.0").to_string();
    let ret: RegisterReply = RegisterReply {
        ipc_socket,
        rpc_endpoint,
        instance_id,
        version,
    };
    Ok(ret)
}

pub fn write_exit_request() -> String {
    let msg = json!({"type": "exit_request"});
    encode_msg(msg)
}

pub fn write_get_data_request(id: ObjectID, sync_remote: bool, wait: bool) -> String {
    let msg = json!({
        "type": "exit_request",
        "id": vec!(id),
        "sync_remote": sync_remote,
        "wait": wait
    });
    encode_msg(msg)
}

pub fn write_get_vec_data_request(ids: Vec<ObjectID>, sync_remote: bool, wait: bool) -> String {
    let msg = json!({
        "type": "get_data_request",
        "id": ids,
        "sync_remote": sync_remote,
        "wait": wait
    });
    encode_msg(msg)
}

pub struct GetDataRequest {
    ids: Vec<Value>,
    sync_remote: bool,
    wait: bool,
}

pub fn read_get_data_request(root: Value) -> Result<GetDataRequest, Error> {
    RETURN_ON_ASSERT(root["type"] == "get_data_request");
    let ids: Vec<Value> = root["id"].as_array().unwrap().to_vec();
    let sync_remote: bool = root["sync_remote"].as_bool().unwrap_or(false);
    let wait: bool = root["wait"].as_bool().unwrap_or(false);
    let ret = GetDataRequest {
        ids,
        sync_remote,
        wait,
    };
    Ok(ret)
}

pub fn write_get_data_reply(content: Value) -> String {
    let msg = json!({"type": "get_data_reply", "content": content});
    encode_msg(msg)
}

pub fn read_get_data_reply(root: Value) -> Result<Value, Error> {
    CHECK_IPC_ERROR(&root, "get_data_reply");
    let content_group = &root["content"];
    if content_group.as_array().unwrap().len() != 1 {
        panic!("Failed to read get_data reply: {:?}", root);
    }
    let content = content_group
        .as_array()
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .clone();
    Ok(content)
}

// Question: key value 0, 1, ...?
pub fn read_get_unordered_data_reply(root: Value) -> Result<HashMap<ObjectID, Value>, Error> {
    CHECK_IPC_ERROR(&root, "get_data_reply");
    let mut content: HashMap<ObjectID, Value> = HashMap::new();
    let content_group = &root["content"];
    let mut key: usize = 0;
    for kv in content_group.as_array().unwrap().into_iter() {
        content.insert(objectid_from_string(key.to_string()), kv.clone());
        key += 1;
    }
    Ok(content)
}

pub fn write_list_data_request(pattern: String, regex: bool, limit: usize) -> String {
    let msg = json!({
        "type": "list_data_request",
        "pattern": pattern,
        "regex": regex,
        "limit": limit,
    });
    encode_msg(msg)
}

pub struct ListDataRequest {
    pattern: String,
    regex: bool,
    limit: usize,
}

pub fn read_list_data_request(root: Value) -> Result<ListDataRequest, Error> {
    RETURN_ON_ASSERT(root["type"] == "list_data_request");
    let pattern = root["pattern"].as_str().unwrap().to_string();
    let regex: bool = root["regex"].as_bool().unwrap_or(false);
    let limit = root["limit"].as_u64().unwrap() as usize;
    let ret = ListDataRequest {
        pattern,
        regex,
        limit,
    };
    Ok(ret)
}

pub fn write_create_buffer_request(size: usize) -> String {
    let msg = json!({"type": "create_buffer_request", "size": size});
    encode_msg(msg)
}

pub fn read_create_buffer_request(root: Value) -> Result<usize, Error> {
    RETURN_ON_ASSERT(root["type"] == "create_buffer_request");
    let size = root["size"].as_u64().unwrap() as usize;
    Ok(size)
}

pub fn write_create_buffer_reply(id: ObjectID, object: Payload) -> String {
    let tree: Value = object.to_json();
    let msg = json!({"type": "create_buffer_reply", "id": id, "created": tree});
    encode_msg(msg)
}

pub fn read_create_buffer_reply(root: Value) -> Result<(ObjectID, Payload), Error> {
    CHECK_IPC_ERROR(&root, "create_buffer_reply");
    let tree: &Value = &root["created"];
    let id = root["id"].as_u64().unwrap() as ObjectID;
    let mut object = Payload::new();
    object.from_json(tree);
    Ok((id, object))
}

pub fn write_create_remote_buffer_request(size: usize) -> String {
    let msg = json!({"type": "create_remote_buffer_request", "size": size});
    encode_msg(msg)
}

pub fn read_create_remote_buffer_request(root: Value) -> Result<usize, Error> {
    RETURN_ON_ASSERT(root["type"] == "create_remote_buffer_request");
    let size = root["size"].as_u64().unwrap() as usize;
    Ok(size)
}

pub fn write_get_buffer_request(ids: HashSet<ObjectID>) -> String {
    let mut map = Map::new();
    let mut idx: usize = 0;
    for id in &ids {
        map.insert(
            idx.to_string(),
            Value::Number(serde_json::Number::from(*id)),
        );
        idx += 1;
    }
    map.insert(
        String::from("type"),
        Value::String("get_buffers_request".to_string()),
    );
    map.insert(
        String::from("num"),
        Value::Number(serde_json::Number::from(ids.len())),
    );
    let msg = Value::Object(map);
    encode_msg(msg)
}

pub fn read_get_buffer_request(root: Value) -> Result<Vec<ObjectID>, Error> {
    RETURN_ON_ASSERT(root["type"] == "get_buffers_request");
    let mut ids: Vec<ObjectID> = Vec::new();
    let num: usize = root["size"].as_u64().unwrap() as usize;
    for idx in 0..num {
        ids.push(root[idx.to_string()].as_u64().unwrap() as ObjectID)
    }
    Ok(ids)
}

pub fn write_get_buffer_reply(objects: Vec<Box<Payload>>) -> String {
    let mut map = Map::new();
    let num: usize = objects.len();
    for idx in 0..num {
        let tree: Value = objects[idx].to_json();
        map.insert(idx.to_string(), tree);
    }
    map.insert(
        String::from("type"),
        Value::String("get_buffers_reply".to_string()),
    );
    map.insert(
        String::from("num"),
        Value::Number(serde_json::Number::from(objects.len())),
    );
    let msg = Value::Object(map);
    encode_msg(msg)
}

pub fn read_get_buffer_reply(root: Value) -> Result<HashMap<ObjectID, Payload>, Error> {
    CHECK_IPC_ERROR(&root, "get_buffers_reply");
    let mut objects: HashMap<ObjectID, Payload> = HashMap::new();
    let num: usize = root["num"].as_u64().unwrap() as usize;
    for idx in 0..num {
        let tree: &Value = &root[idx.to_string()];
        let mut object = Payload::new();
        object.from_json(tree);
        objects.insert(object.object_id, object);
    }
    Ok(objects)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_register_reply_test() {
        let msg = json!({
            "type": "register_reply",
            "ipc_socket": "some_ipc_socket",
            "rpc_endpoint": "some_rpc_endpoint",
            "instance_id": 1 as InstanceID,
            "version": "0.2.6"
        });
        let reply = read_register_reply(msg).unwrap();
        let (a, b, c, d) = (
            reply.ipc_socket,
            reply.rpc_endpoint,
            reply.instance_id,
            reply.version,
        );
        println!("{:?},{:?},{},{:?}", a, b, c, d);
    }
}
