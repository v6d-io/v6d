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
use std::sync::{Arc, Mutex};

use arrow::buffer as arrow;
use lazy_static::lazy_static;
use serde_json::json;

use super::object::{Object, ObjectBase, ObjectBuilder, Registered};

use super::object_factory::ObjectFactory;
use super::object_meta::ObjectMeta;
use super::payload::Payload;
use super::status::*;
use super::typename::type_name;
use super::uuid::*;
use super::Client;
use super::IPCClient;

#[derive(Debug, Clone)]
pub struct Blob {
    id: ObjectID,
    meta: ObjectMeta,
    size: usize,
    buffer: Option<Rc<arrow::Buffer>>,
}

unsafe impl Send for Blob {}

impl Default for Blob {
    fn default() -> Self {
        Blob {
            id: invalid_object_id(),
            meta: ObjectMeta::default(),

            size: usize::MAX,
            buffer: None as Option<Rc<arrow::Buffer>>,
        }
    }
}

impl Registered for Blob {}

impl Object for Blob {
    fn meta(&self) -> &ObjectMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ObjectMeta {
        &mut self.meta
    }

    fn id(&self) -> ObjectID {
        self.id
    }

    fn set_id(&mut self, id: ObjectID) {
        self.id = id;
    }

    fn set_meta(&mut self, meta: &ObjectMeta) {
        self.meta = meta.clone();
    }
}

impl ObjectBase for Blob {}

impl Blob {

    pub fn allocated_size(&self) -> usize {
        self.size
    }

    pub fn data(&self) -> *const u8 {

        if self.size > 0 {
            match &self.buffer {
                None => panic!(
                    "The object might be a (partially) remote object and the payload data 
                is not locally available: {}",
                    object_id_to_string(self.id)
                ),
                Some(buf) => {
                    if buf.len() == 0 {
                        panic!(
                            "The object might be a (partially) remote object and the payload data 
                        is not locally available: {}",
                            object_id_to_string(self.id)
                        );
                    }
                }
            }
        }
        self.buffer.as_ref().unwrap().as_ptr()
    }

    pub fn buffer(&self) -> Rc<arrow::Buffer> {

        if self.size > 0 {
            match &self.buffer {
                None => panic!(
                    "The object might be a (partially) remote object and the payload data 
                is not locally available: {}",
                    object_id_to_string(self.id)
                ),
                Some(buf) => {
                    if buf.len() == 0 {
                        panic!(
                            "The object might be a (partially) remote object and the payload data 
                        is not locally available: {}",
                            object_id_to_string(self.id)
                        );
                    }
                }
            }
        }
        Rc::clone(&self.buffer.as_ref().unwrap())
    }

    pub fn construct(&mut self, meta: &ObjectMeta) {
        let __type_name: String = type_name::<Blob>().to_string(); // Question: type_name<Blob>()
        CHECK(meta.get_type_name() == __type_name); // Question
        self.meta = meta.clone();
        self.id = meta.get_id();
        if let Some(_) = self.buffer {
            return;
        }
        if self.id == empty_blob_id() {
            self.size = 0;
            return;
        }
        if !meta.is_local() {
            return;
        }
        self.buffer = meta.get_buffer(meta.get_id()).expect(
            format!(
                "Invalid internal state: failed to construct local blob since payload is missing {}",
                object_id_to_string(meta.get_id())
            )
            .as_str(),
        );
        if let None = self.buffer {
            panic!(
                "Invalid internal state: local blob found bit it is nullptr: {}",
                object_id_to_string(meta.get_id())
            )
        }
    }

    pub fn dump() {} // Question: VLOG(); VLOG_IS_ON()

    // Question: It will consume a client since IPCClient cannot implement clone
    // trait(UnixStream).
    pub fn make_empty(client: IPCClient) -> Rc<Blob> {
        let mut empty_blob = Blob::default();
        empty_blob.id = empty_blob_id();
        empty_blob.size = 0;
        empty_blob.meta.set_id(empty_blob_id());
        empty_blob.meta.set_signature(empty_blob_id() as Signature);
        empty_blob
            .meta
            .set_type_name(&type_name::<Blob>().to_string());
        empty_blob
            .meta
            .add_json_key_value(&"length".to_string(), &json!(0));
        empty_blob.meta.set_nbytes(0);

        empty_blob
            .meta
            .add_json_key_value(&"instance_id".to_string(), &json!(client.instance_id()));
        empty_blob
            .meta
            .add_json_key_value(&"transient".to_string(), &json!(true));
        let tmp: Rc<dyn Client> = Rc::new(client); // Needs clone trait here
        empty_blob.meta.set_client(Some(Rc::downgrade(&tmp)));

        Rc::new(empty_blob)
    }

    // Question: const uintptr_t pointer
    pub fn from_buffer(
        client: IPCClient,
        object_id: ObjectID,
        size: usize,
        pointer: usize,
    ) -> Rc<Blob> {
        let mut blob = Blob::default();
        blob.id = object_id;
        blob.size = size;
        blob.meta.set_id(object_id);
        blob.meta.set_signature(object_id as Signature);
        blob.meta.set_type_name(&"blob".to_string());
        blob.meta
            .add_json_key_value(&"length".to_string(), &json!(size));
        blob.meta.set_nbytes(size);

        // Question:
        // blob->buffer_ =
        // arrow::Buffer::Wrap(reinterpret_cast<const uint8_t*>(pointer), size);
        // from_raw_parts() capacity=len

        VINEYARD_CHECK_OK(
            blob.meta
                .buffer_set
                .borrow_mut()
                .emplace_null_buffer(object_id),
        );
        VINEYARD_CHECK_OK(
            blob.meta
                .buffer_set
                .borrow_mut()
                .emplace_buffer(object_id, &blob.buffer),
        );

        blob.meta
            .add_json_key_value(&"instance_id".to_string(), &json!(client.instance_id()));
        blob.meta
            .add_json_key_value(&"transient".to_string(), &json!(true));
        let tmp: Rc<dyn Client> = Rc::new(client);
        blob.meta.set_client(Some(Rc::downgrade(&tmp)));

        Rc::new(blob)

    }
}

#[derive(Debug)]
pub struct BlobWriter {
    object_id: ObjectID,
    payload: Payload,
    buffer: Option<Rc<arrow::MutableBuffer>>,
    metadata: HashMap<String, String>,
    sealed: bool,
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
    fn build(&mut self, client: &IPCClient) -> io::Result<()> {
        Ok(())
    }

    fn seal(&mut self, client: &IPCClient) -> Rc<dyn Object> {
        panic!()
    } // TODO: mmap
}

impl BlobWriter {
    pub fn id(&self) -> ObjectID {
        self.object_id
    }

    pub fn size(&self) -> usize {
        match &self.buffer {
            None => 0,
            Some(buf) => buf.len(),
        }
    }

    pub fn data(&self) -> *const u8 {
        self.buffer.as_ref().unwrap().as_ptr()
    }

    pub fn buffer(&self) -> Rc<arrow::MutableBuffer> {
        Rc::clone(&self.buffer.as_ref().expect("The buffer is empty."))
    }

    pub fn abort(&self, mut client: IPCClient) -> Result<(), bool> {
        if self.sealed {
            return Err(false); // Question: return Status::ObjectSealed();
        }
        return client.drop_buffer(self.object_id, self.payload.store_fd); // TODO: mmap

    }

    pub fn add_key_value(&mut self, key: &String, value: &String) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    pub fn dump() {} // Question: VLOG; VLOG_IS_ON

}

#[derive(Debug)]
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
        buffer: &Option<Rc<arrow::Buffer>>,

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
                self.buffers.insert(id, buffer.clone());

            }
        }
        Ok(())
    }

    pub fn extend(&mut self, others: &BufferSet) {
        for (key, value) in others.buffers.iter() {
            self.buffers
                .insert(key.clone(), Some(Rc::clone(value.as_ref().unwrap())));
        }
    }

    pub fn contains(&self, id: ObjectID) -> bool {
        if let None = self.buffers.get(&id) {
            return false;
        }
        true
    }
    pub fn get(&self, id: ObjectID) -> Result<Option<Rc<arrow::Buffer>>, bool> {
        match self.buffers.get(&id) {
            None => Err(false),
            Some(buf) => Ok(buf.clone()),

        }
    }
}

// blobwriter Mmap先不写
