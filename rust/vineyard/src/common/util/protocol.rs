use serde::{Deserialize, Serialize};
use serde_json::{Result, Value, json};
use std::io::{self, ErrorKind, Error};

use super::InstanceID;

enum CommandType {
    RegisterRequest,
    RegisterReply,
    ExitRequest,
    ExitReply,
}

pub fn RETURN_ON_ASSERT(b: bool) {
    if !b {
        panic!()
    }
}

// json value to String
fn encode_msg(msg: Value) -> String {
    let ret = serde_json::to_string(&msg).unwrap();
    ret
}

fn write_register_request() -> String {
    let msg = json!({"type": "register_request", "version": "0.2.6" });
    encode_msg(msg)
}

fn read_register_request(msg: Value) -> Result<String> {
    RETURN_ON_ASSERT(msg["type"] == "register_request");
    Ok(msg["version"].as_str().unwrap_or("0.0.0").to_string())
}

fn read_register_reply(
    root: Value, 
    ipc_socket: &String, 
    rpc_endpoint: String, 
    instance_id: InstanceID, 
    version: &String
) -> Result<u64> {
    panic!();

}

