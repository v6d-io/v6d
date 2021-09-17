use std::collections::{HashMap, HashSet};
use std::io;


use serde_json::{json, Map, Value};

use super::payload::Payload;
use super::status;
use super::uuid::*;


pub fn CHECK_IPC_ERROR(tree: &Value, root_type: &str) {
    if tree.as_object().unwrap().contains_key("code") {
        tree["code"].as_u64().unwrap_or(0);
        tree["message"].as_str().unwrap_or("");
    }
    status::RETURN_ON_ASSERT(tree["type"].as_str().unwrap() == root_type);
}

pub fn ENSURE_CONNECTED(b: bool) {
    if !b {
        panic!()
    }
    // Question. TODO: mutex
}

// Convert JSON Value to a String
pub fn encode_msg(msg: Value) -> String {
    let ret = serde_json::to_string(&msg).unwrap();
    ret
}

// Write functions: Derive the JSON message and write it to a String
pub fn write_register_request() -> String {
    let msg = json!({"type": "register_request", "version": "0.2.6"});
    encode_msg(msg)
}

#[derive(Debug)]
pub struct RegisterReply {
    pub ipc_socket: String,
    pub rpc_endpoint: String,
    pub instance_id: InstanceID,
    pub version: String,
}

// Read functions: Read the JSON root to variants of ipc instance
pub fn read_register_reply(root: Value) -> io::Result<RegisterReply> {
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

pub fn read_get_data_reply(root: Value) -> io::Result<Value> {
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
pub fn read_get_unordered_data_reply(root: Value) -> io::Result<HashMap<ObjectID, Value>> {
    CHECK_IPC_ERROR(&root, "get_data_reply");
    let mut content: HashMap<ObjectID, Value> = HashMap::new();
    let content_group = &root["content"];
    let mut key: usize = 0;
    for kv in content_group.as_array().unwrap().into_iter() {
        content.insert(object_id_from_string(&key.to_string()), kv.clone());

        key += 1;
    }
    Ok(content)
}

pub fn write_list_data_request(pattern: &String, regex: bool, limit: usize) -> String {
    let msg = json!({
        "type": "list_data_request",
        "pattern": pattern,
        "regex": regex,
        "limit": limit,
    });
    encode_msg(msg)
}

pub fn write_create_buffer_request(size: usize) -> String {
    let msg = json!({"type": "create_buffer_request", "size": size});
    encode_msg(msg)
}

pub fn read_create_buffer_reply(root: Value) -> io::Result<(ObjectID, Payload)> {
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

pub fn read_get_buffer_reply(root: Value) -> io::Result<HashMap<ObjectID, Payload>> {
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

pub fn write_put_name_request(object_id: ObjectID, name: &String) -> String {
    let msg = json!({"type": "put_name_request", "object_id": object_id, "name": name});
    encode_msg(msg)
}

pub fn read_put_name_reply(root: Value) -> io::Result<()> {
    CHECK_IPC_ERROR(&root, "put_name_reply");
    Ok(())
}

pub fn write_get_name_request(name: &String, wait: bool) -> String {
    let msg = json!({"type": "get_name_request", "name": name, "wait": wait});
    encode_msg(msg)
}

pub fn read_get_name_reply(root: Value) -> io::Result<ObjectID> {
    CHECK_IPC_ERROR(&root, "get_name_reply");
    let object_id = root["object_id"].as_u64().unwrap() as ObjectID;
    Ok(object_id)
}

pub fn write_drop_name_request(name: &String) -> String {
    let msg = json!({"type": "drop_name_request", "name": name});
    encode_msg(msg)
}

pub fn read_drop_name_reply(root: Value) -> io::Result<()> {
    CHECK_IPC_ERROR(&root, "drop_name_reply");
    Ok(())
}

pub fn write_persist_request(id: ObjectID) -> String {
    let msg = json!({"type": "persist_request", "id": id});
    encode_msg(msg)
}

pub fn read_persist_reply(root: Value) -> io::Result<(())> {
    CHECK_IPC_ERROR(&root, "persist_reply");
    Ok(())
}

pub fn write_if_persist_request(id: ObjectID) -> String {
    let msg = json!({"type": "if_persist_request", "id": id});
    encode_msg(msg)
}

pub fn read_if_persist_reply(root: Value) -> io::Result<(bool)> {
    CHECK_IPC_ERROR(&root, "if_persist_reply");
    let persist = root["persist"].as_bool().unwrap_or(false);
    Ok(persist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_print_read_register_reply() {
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
