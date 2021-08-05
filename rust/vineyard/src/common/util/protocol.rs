use serde::{Deserialize, Serialize};
use serde_json::{Value, Map, json};
use serde_json::Result as JsonResult;

use std::io::{self, ErrorKind, Error};
use std::collections::{HashMap, HashSet};

use super::{InstanceID, ObjectID};

enum CommandType {
    RegisterRequest,
    RegisterReply,
    ExitRequest,
    ExitReply,
}

pub struct Payload{
    object_id: ObjectID,
}

impl Payload {
    pub fn new() -> Payload{
        Payload{
            object_id: 0,
        }
    }

    pub fn to_json(&self) -> Value {
        json!(null)
    }

    pub fn from_json(&mut self, tree: &Value) -> Payload {
        Payload::new()
        //*self = Payload::new()
    }
}

pub fn RETURN_ON_ASSERT(b: bool) {
    if !b {
        panic!()
    }
}

pub fn CHECK_IPC_ERROR(tree: &Value, root_type: &str) {
    if tree.as_object().unwrap().contains_key("code"){
        // Question!
    }
    RETURN_ON_ASSERT(tree["type"].as_str().unwrap()==root_type)
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
    ipc_socket: &String, 
    rpc_endpoint: &String, 
    instance_id: InstanceID
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

// Question: should I insist on writing changing the value in the inputs or return
// Result<(String,String,String,InstanceID), Error>
pub fn read_register_reply(
    root: Value,
    ipc_socket: &mut String, 
    rpc_endpoint: &mut String, 
    instance_id: &mut InstanceID, 
    version: &mut String
) -> Result<(), Error> {
    CHECK_IPC_ERROR(&root, "register_reply");
    *ipc_socket = root["ipc_socket"].as_str().unwrap().to_string();
    *rpc_endpoint = root["rpc_endpoint"].as_str().unwrap().to_string();
    *instance_id = root["instance_id"].as_u64().unwrap(); // Note, explicitly use u64
    *version = root["version"].as_str().unwrap_or("0.0.0").to_string();
    Ok(())
}

pub fn write_exit_request() -> String {
    let msg = json!({"type": "exit_request"});
    encode_msg(msg)
}

// Question: No overloading in Rust for const std::vector<ObjectID>& ids
pub fn write_get_data_request(id: ObjectID, sync_remote: bool, wait: bool) -> String {
    let msg = json!({
        "type": "exit_request", 
        "id": vec!(id), 
        "sync_remote": sync_remote, 
        "wait": wait
    });
    encode_msg(msg)
}

pub fn read_get_data_request(root: Value) -> Result<(Vec<Value>, bool, bool), Error> {
    RETURN_ON_ASSERT(root["type"] == "get_data_request");
    let ids: Vec<Value> = root["id"].as_array().unwrap().to_vec();
    // Question: sync_remote = root.value("sync_remote", false);
    let sync_remote: bool = root["sync_remote"].as_bool().unwrap_or(false);
    let wait: bool = root["wait"].as_bool().unwrap_or(false);
    Ok((ids, sync_remote, wait))
}

pub fn write_get_data_reply(content: Value) -> String{
    let msg = json!({"type": "get_data_reply", "content": content});
    encode_msg(msg)
}

// TODO: Overloading
pub fn read_get_data_reply(root: Value) -> Result<Value, Error> {
    CHECK_IPC_ERROR(&root, "get_data_reply");
    let content_group = root["content"].clone();
    // Question: if content_group  if (content_group.size() != 1) {
    // return Status::ObjectNotExists("failed to read get_data reply: " +
    // root.dump());
    Ok(content_group)
}

// Q: usize == size_t?
pub fn write_list_data_request(pattern: String, regex: bool, limit: usize) -> String {
    let msg = json!({
        "type": "list_data_request", 
        "pattern": pattern, 
        "regex": regex, 
        "limit": limit,
    });
    encode_msg(msg)
}

pub fn read_list_data_request(root: Value) -> Result<(String, bool, usize), Error> {
    RETURN_ON_ASSERT(root["type"] == "list_data_request");
    let pattern = root["pattern"].as_str().unwrap().to_string();
    let regex: bool = root["regex"].as_bool().unwrap_or(false);
    let limit = root["limit"].as_u64().unwrap() as usize; // Question: no as_usize()
    Ok((pattern, regex, limit))
}

// Q: why some of them only have request and no reply?

pub fn write_create_buffer_request(size: usize) -> String {
    let msg = json!({"type": "create_buffer_request", "size": size});
    encode_msg(msg)
}

pub fn read_create_buffer_request(root: Value) -> Result<usize, Error> {
    RETURN_ON_ASSERT(root["type"] == "create_buffer_request");
    let size = root["size"].as_u64().unwrap() as usize;
    Ok(size)
}

// TODO: Payload
pub fn write_create_buffer_reply(id: ObjectID, object: &Payload) -> String {
    let tree: Value = object.to_json();
    let msg = json!({"type": "create_buffer_reply", "id": id, "created": tree});
    encode_msg(msg)
}

pub fn read_create_buffer_reply(root: Value) -> Result<(ObjectID, Payload), Error> {
    CHECK_IPC_ERROR(&root, "create_buffer_reply");
    let tree: Value = root["created"].clone();
    let id = root["id"].as_u64().unwrap() as ObjectID;
    let mut object = Payload::new(); 
    object.from_json(&tree);
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
    // Q: Does root[std::to_string(idx++)] = id start from 0 right?
    for id in &ids{
        map.insert(idx.to_string(), Value::Number(serde_json::Number::from(*id)));
        idx += 1;
    }
    map.insert(String::from("type"), Value::String("get_buffers_request".to_string()));
    map.insert(String::from("num"), Value::Number(serde_json::Number::from(ids.len())));
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
    for idx in 0..num{
        let tree: Value = objects[idx].to_json();
        map.insert(idx.to_string(), tree);
    }
    map.insert(String::from("type"), Value::String("get_buffers_reply".to_string()));
    map.insert(String::from("num"), Value::Number(serde_json::Number::from(objects.len())));
    let msg = Value::Object(map);
    encode_msg(msg)
}

pub fn read_get_buffer_reply(root: Value) -> Result<HashMap<ObjectID, Payload>, Error>  {
    CHECK_IPC_ERROR(&root, "get_buffers_reply");
    let mut objects: HashMap<ObjectID, Payload> = HashMap::new();
    let num: usize = root["num"].as_u64().unwrap() as usize;
    for idx in 0..num{
        let tree: &Value = &root[idx.to_string()];
        let mut object = Payload::new();
        object.from_json(tree);
        objects.insert(object.object_id, object);
    }
    Ok(objects)
}

// // Write functions: Write JSON content to a String
// pub fn write_register_request() -> String {
//     let msg = json!({"type": "register_request", "version": "0.2.6" });
//     encode_msg(msg)
// }

// // Read functions: Read JSON root to variants of ipc instance
// pub fn read_register_request(root: Value) -> Result<String, Error> {
//     RETURN_ON_ASSERT(root["type"] == "register_request");
//     Ok(root["version"].as_str().unwrap_or("0.0.0").to_string())
// }


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_register_reply_test(){
        let msg = json!({
            "type": "register_reply", 
            "ipc_socket": "some_ipc_socket",
            "rpc_endpoint": "some_rpc_endpoint",
            "instance_id": 1 as InstanceID,
            "version": "0.2.6"});
        let (mut a, mut b, mut c, mut d) = (String::new(),String::new(),0 ,String::new());
        read_register_reply(msg, &mut a, &mut b, &mut c, &mut d);
        println!("{:?},{:?},{},{:?}",a,b,c,d);
    }
}
