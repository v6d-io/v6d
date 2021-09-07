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

use arrow;

use super::object::Object;
use super::object_factory::ObjectFactory;
use super::status::*;
use super::uuid::*;

#[derive(Debug)]
pub struct Blob {
    size: usize,
    buffer: Rc<ArrowBuffer>,
}

#[derive(Debug)]
pub struct BlobWriter {}

#[derive(Debug)]
pub struct BufferSet {
    buffer_ids: HashSet<ObjectID>,
    buffers: HashMap<ObjectID, Rc<ArrowBuffer>>,
}

impl Default for BufferSet {
    fn default() -> BufferSet {
        BufferSet {
            buffer_ids: HashSet::new() as HashSet<ObjectID>,
            buffers: HashMap::new() as HashMap<ObjectID, Rc<ArrowBuffer>>,
        }
    }
}

impl BufferSet {
    pub fn all_buffers(&self) -> &HashMap<ObjectID, Rc<ArrowBuffer>> {
        &self.buffers
    }

    pub fn extend(&mut self, others: &BufferSet) {
        for (key, value) in others.buffers.iter() {
            self.buffers.insert(key.clone(), value.clone());
        }
    }

    pub fn emplace_buffer_null(&self, id: ObjectID) -> io::Result<Rc<ArrowBuffer>> {
        // TODO
        panic!()
    }

    pub fn emplace_buffer(
        &self,
        id: ObjectID,
        buffer: &Rc<ArrowBuffer>,
    ) -> io::Result<Rc<ArrowBuffer>> {
        // TODO
        panic!()
    }

    pub fn contains(&self, id: ObjectID) -> bool {
        if let None = self.buffers.get(&id) {
            return false;
        }
        true
    }

    pub fn get(&self, id: ObjectID) -> io::Result<Rc<ArrowBuffer>> {
        // TODO
        panic!()
    }
}

#[derive(Debug)]
pub struct ArrowBuffer {} // TODO. arrow/buffer: dependencies

// Mmap先不写
