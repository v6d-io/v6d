use std::collections::{HashMap, HashSet};
/** Copyright 2020-2021 Alibaba Group Holding Limited.

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
use std::io;
use std::rc::{Rc, Weak};

use arrow::buffer as arrow;

use super::object::Object;
use super::object_factory::ObjectFactory;
use super::payload::Payload;
use super::status::*;
use super::uuid::*;

#[derive(Debug)]
pub struct Blob {
    id: ObjectID,
    size: usize,
    buffer: Rc<arrow::Buffer>,
}

#[derive(Debug)]
pub struct BlobWriter {
    object_id: ObjectID,
    payload: Payload,
    buffer: Rc<arrow::MutableBuffer>,
    metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct BufferSet {
    buffer_ids: HashSet<ObjectID>,
    buffers: HashMap<ObjectID, Option<Rc<arrow::Buffer>>>, // Question
}

impl Default for BufferSet {
    fn default() -> BufferSet {
        BufferSet {
            buffer_ids: HashSet::new() as HashSet<ObjectID>,
            buffers: HashMap::new() as HashMap<ObjectID, Option<Rc<arrow::Buffer>>>,
        }
    }
}

impl BufferSet {
    pub fn all_buffers(&self) -> &HashMap<ObjectID, Option<Rc<arrow::Buffer>>> {
        &self.buffers
    }

    pub fn emplace_null_buffer(&mut self, id: ObjectID) -> io::Result<()> {
        if let Some(buf) = self.buffers.get(&id) {
            if let Some(_) = buf {
                panic!(
                    "Invalid internal state: the buffer shouldn't has been filled, id = {}",
                    object_id_to_string(id)
                );
            }
            
        }
        self.buffer_ids.insert(id);
        self.buffers.insert(id, None);
        Ok(())
    }

    pub fn emplace_buffer(
        &mut self,
        id: ObjectID,
        buffer: Option<Rc<arrow::Buffer>>,
    ) -> io::Result<()> {
        match self.buffers.get(&id) {
            None => panic!(
                "Invalid internal state: no such buffer defined, id = {}",
                object_id_to_string(id)
            ),
            Some(buf) => {
                if let Some(_) = buf {
                    panic!(
                        "Invalid internal state: duplicated buffer, id = {}",
                        object_id_to_string(id)
                    );
                }
                self.buffers.insert(id, buffer);
            },
        }
        Ok(())
    }

    pub fn extend(&mut self, others: &BufferSet) {
        for (key, value) in others.buffers.iter() {
            self.buffers.insert(key.clone(), Some(Rc::clone(value.as_ref().unwrap())));
        }
    }

    pub fn contains(&self, id: ObjectID) -> bool {
        if let None = self.buffers.get(&id) {
            return false;
        }
        true
    }

    pub fn get(&self, id: ObjectID) -> Option<Rc<arrow::Buffer>> {
        match self.buffers.get(&id) {
            None => None,
            Some(buf) => Some(Rc::clone(buf.as_ref().unwrap()))
        }
    }
}


// Mmap先不写
