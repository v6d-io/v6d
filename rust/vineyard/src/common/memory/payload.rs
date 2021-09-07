use std::collections::{HashMap, HashSet};
use std::io;
use std::ptr;

use serde_json::{json, Map, Value};

use crate::common::util::status::*;
use crate::common::util::uuid::*;


#[derive(Debug)]
pub struct Payload {
    pub object_id: ObjectID,
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
