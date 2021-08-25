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
use std::collections::{HashMap, HashSet};

use super::uuid::*;
use super::status::*;
use super::object::Object;
use super::object_factory::ObjectFactory;

#[derive(Debug)]
pub struct Blob {}

#[derive(Debug)]
pub struct BlobWriter {}

#[derive(Debug)]
pub struct BufferSet {
    buffer_ids: HashSet<ObjectID>,
    buffers: HashMap<ObjectID, Rc<Buffer>>,
}

impl Default for BufferSet {
    fn default() -> BufferSet {
        BufferSet{
            buffer_ids: HashSet::new() as HashSet<ObjectID>,
            buffers: HashMap::new() as HashMap<ObjectID, Rc<Buffer>>,
        }
    }
}

impl BufferSet {
    pub fn all_buffers(&self) -> HashMap<ObjectID, Rc<Buffer>> {
        self.buffers
    }
}

#[derive(Debug)]
pub struct  Buffer {} // TODO. arrow/buffer