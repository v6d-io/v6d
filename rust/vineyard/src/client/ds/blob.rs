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

use std::collections::{HashMap, HashSet};

use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;

use arrow::buffer as arrow;

use crate::common::util::arrow::*;
use crate::common::util::status::*;
use crate::common::util::typename::*;
use crate::common::util::uuid::*;

use super::super::client::Client;
use super::super::ipc_client::IPCClient;
use super::object::{
    register_vineyard_object, Create, Object, ObjectBase, ObjectBuilder, ObjectMetaAttr,
};
use super::object_factory::ObjectFactory;
use super::object_meta::ObjectMeta;

#[derive(Debug, Clone)]
pub struct Blob {
    meta: ObjectMeta,
    size: usize,
    buffer: Option<Rc<arrow::Buffer>>,
}

impl_typename!(Blob, "vineyard::Blob");

impl Default for Blob {
    fn default() -> Self {
        Blob {
            meta: ObjectMeta::default(),
            size: usize::MAX,
            buffer: None as Option<Rc<arrow::Buffer>>,
        }
    }
}

impl Object for Blob {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(meta.get_typename()?, &typename::<Blob>())?;
        self.meta = meta;

        if let Some(_) = self.buffer {
            return Ok(());
        }
        if self.meta.get_id() == empty_blob_id() {
            self.size = 0;
            return Ok(());
        }
        if !self.meta.is_local() {
            return Ok(());
        }
        match self.meta.get_buffer(self.meta.get_id()) {
            Ok(buffer) => {
                self.buffer = buffer;
                return Ok(());
            }
            Err(err) => {
                return Err(VineyardError::invalid(format!("Invalid internal state: failed to construct local blob since payload is missing {}", err)));
            }
        }
    }
}

register_vineyard_object!(Blob);

impl Display for Blob {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Blob<id={}, size={}>", self.meta().get_id(), self.size)
    }
}

impl Blob {
    pub fn new(meta: ObjectMeta, size: usize, buffer: Option<Rc<arrow::Buffer>>) -> Self {
        Blob {
            meta: meta,
            size: size,
            buffer: buffer,
        }
    }

    pub fn allocated_size(&self) -> usize {
        self.size
    }

    pub fn empty(client: *mut IPCClient) -> Box<Blob> {
        let mut blob = Blob::default();
        blob.size = 0;
        blob.meta.set_id(empty_blob_id());
        blob.meta.set_signature(empty_blob_id() as Signature);
        blob.meta.set_typename(&typename::<Blob>());
        blob.meta.add_int("length", 0);
        blob.meta.set_nbytes(0);
        blob.meta
            .add_uint("instance_id", unsafe { &*client }.instance_id());
        blob.meta.add_bool("transient", true);
        blob.meta.set_client(client);
        return Box::new(blob);
    }

    pub fn as_ptr(&self) -> Result<*const u8> {
        let buffer = self.buffer()?;
        return Ok(buffer.as_ptr());
    }

    pub fn as_ptr_unchecked(&self) -> *const u8 {
        return self.buffer_unchecked().as_ptr();
    }

    pub fn as_slice(&self) -> Result<&[u8]> {
        return unsafe { Ok(std::slice::from_raw_parts(self.as_ptr()?, self.size)) };
    }

    pub fn as_slice_unchecked(&self) -> &[u8] {
        return unsafe { std::slice::from_raw_parts(self.as_ptr_unchecked(), self.size) };
    }

    pub fn buffer(&self) -> Result<Rc<arrow::Buffer>> {
        match &self.buffer {
            None => {
                if self.size > 0 {
                    return Err(VineyardError::invalid(format!(
                        "The object might be a (partially) remote object and the payload
                         data is not locally available: {}",
                        object_id_to_string(self.meta().get_id())
                    )));
                }
                let buffer = to_buffer_null();
                return Ok(Rc::new(buffer));
            }
            Some(buffer) => {
                if self.size > 0 && buffer.len() == 0 {
                    return Err(VineyardError::invalid(format!(
                        "The object might be a (partially) remote object and the payload data
                    is not locally available: {}",
                        object_id_to_string(self.meta().get_id())
                    )));
                }
                return Ok(buffer.clone());
            }
        }
    }

    pub fn buffer_unchecked(&self) -> Rc<arrow::Buffer> {
        match &self.buffer {
            None => {
                let buffer = to_buffer_null();
                return Rc::new(buffer);
            }
            Some(buffer) => {
                return buffer.clone();
            }
        }
    }
}

#[derive(Debug)]
pub struct BlobWriter {
    sealed: bool,
    object_id: ObjectID,
    buffer: Option<Rc<arrow::Buffer>>,
    metadata: HashMap<String, String>,
}

impl ObjectBuilder for BlobWriter {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl ObjectBase for BlobWriter {
    fn build(&mut self, _client: &mut IPCClient) -> Result<()> {
        if !self.sealed {
            self.set_sealed(true);
        }
        return Ok(());
    }

    fn seal(self: Self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        client.seal_buffer(self.object_id)?;
        let mut blob = Blob::default();
        blob.size = self.size();
        blob.meta.set_id(self.object_id);
        blob.meta.set_typename(&typename::<Blob>());
        blob.meta.set_nbytes(self.size());
        blob.meta
            .add_uint("length", TryInto::<u64>::try_into(self.size())?);
        blob.meta.add_uint("instance_id", client.instance_id());
        blob.meta.add_bool("transient", true);

        blob.buffer = Some(Rc::new(to_buffer(self.as_ptr(), self.size())));
        blob.meta
            .set_or_add_buffer(self.object_id, blob.buffer.clone())?;
        return Ok(Box::new(blob));
    }
}

impl BlobWriter {
    pub fn new(id: ObjectID, buffer: Option<Rc<arrow::Buffer>>) -> Self {
        BlobWriter {
            sealed: false,
            object_id: id,
            buffer: buffer,
            metadata: HashMap::new(),
        }
    }

    pub fn id(&self) -> ObjectID {
        self.object_id
    }

    pub fn size(&self) -> usize {
        match &self.buffer {
            None => 0,
            Some(buf) => buf.len(),
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        return match &self.buffer {
            None => std::ptr::null(),
            Some(buf) => buf.as_ptr(),
        };
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        return self.as_ptr() as *mut u8;
    }

    pub fn as_slice(&self) -> &[u8] {
        return unsafe { std::slice::from_raw_parts(self.as_ptr(), self.size()) };
    }

    pub fn as_mut_slice(&self) -> &mut [u8] {
        return unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.size()) };
    }

    pub fn buffer(&self) -> Option<Rc<arrow::Buffer>> {
        return self.buffer.clone();
    }

    pub fn abort(&self, mut client: IPCClient) -> Result<()> {
        if self.sealed {
            return Err(VineyardError::object_sealed(
                "The blob write has already been sealed and cannot be aborted".into(),
            ));
        }
        return client.drop_buffer(self.object_id);
    }

    pub fn add_key_value(&mut self, key: &String, value: &String) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

pub struct BufferSet {
    buffer_ids: HashSet<ObjectID>,
    buffers: HashMap<ObjectID, Option<Rc<arrow::Buffer>>>,
}

impl Default for BufferSet {
    fn default() -> BufferSet {
        BufferSet {
            buffer_ids: HashSet::new() as HashSet<ObjectID>,
            buffers: HashMap::new() as HashMap<ObjectID, Option<Rc<arrow::Buffer>>>,
        }
    }
}

impl Debug for BufferSet {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let buffers: Vec<_> = self
            .buffers
            .iter()
            .map(|(k, v)| {
                if v.is_some() {
                    return (k, v.as_ref().unwrap().len());
                } else {
                    return (k, 0);
                }
            })
            .collect();
        write!(
            f,
            "BufferSet {{ buffer_ids: {:?}, buffers: {:?} }}>",
            self.buffer_ids, buffers
        )
    }
}

impl BufferSet {
    pub fn buffer_ids(&self) -> &HashSet<ObjectID> {
        return &self.buffer_ids;
    }

    pub fn buffers(&self) -> &HashMap<ObjectID, Option<Rc<arrow::Buffer>>> {
        return &self.buffers;
    }

    pub fn buffers_mut(&mut self) -> &mut HashMap<ObjectID, Option<Rc<arrow::Buffer>>> {
        return &mut self.buffers;
    }

    pub fn emplace(&mut self, id: ObjectID) -> Result<()> {
        match self.buffers.get(&id) {
            Some(Some(_)) => {
                return Err(VineyardError::invalid(format!(
                    "Invalid internal state: duplicated buffer, id = {}",
                    object_id_to_string(id)
                )));
            }
            _ => {
                self.buffer_ids.insert(id);
                self.buffers.insert(id, None);
                return Ok(());
            }
        }
    }

    pub fn emplace_buffer(
        &mut self,
        id: ObjectID,
        buffer: Option<Rc<arrow::Buffer>>,
    ) -> Result<()> {
        match self.buffers.get(&id) {
            Some(Some(_)) => {
                return Err(VineyardError::invalid(format!(
                    "emplace buffer: invalid internal state: duplicated buffer, id = {}",
                    object_id_to_string(id)
                )));
            }
            None => {
                return Err(VineyardError::invalid(format!(
                    "emplace buffer: invalid internal state: no such buffer defined, id = {}",
                    object_id_to_string(id)
                )));
            }
            Some(None) => {
                self.buffers.insert(id, buffer);
                return Ok(());
            }
        }
    }

    pub fn extend(&mut self, others: &BufferSet) {
        for (key, value) in others.buffers.iter() {
            match value {
                None => {
                    self.buffer_ids.insert(key.clone());
                    self.buffers.insert(key.clone(), None);
                }
                Some(buffer) => {
                    self.buffer_ids.insert(key.clone());
                    self.buffers.insert(key.clone(), Some(Rc::clone(buffer)));
                }
            }
        }
    }

    pub fn contains(&self, id: ObjectID) -> bool {
        return self.buffers.get(&id).is_some();
    }

    pub fn get(&self, id: ObjectID) -> Result<Option<Rc<arrow::Buffer>>> {
        return self
            .buffers
            .get(&id)
            .ok_or(VineyardError::invalid(format!(
                "get buffer: invalid internal state: no such buffer defined, id = {}",
                object_id_to_string(id)
            )))
            .cloned();
    }
}
