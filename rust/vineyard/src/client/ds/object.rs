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
use std::rc::Rc;

use super::blob::Blob;
use super::object_meta::ObjectMeta;
use super::uuid::ObjectID;
use super::Client;

pub trait ObjectBase {
    fn build(client: Box<dyn Client>) -> io::Result<Blob>;
    fn seal(client: Box<dyn Client>) -> Rc<Object>;
}

#[derive(Debug, Clone)]
pub struct Object {
    pub meta: ObjectMeta,
    pub id: ObjectID,
}

impl Default for Object {
    fn default() -> Object {
        Object {
            meta: ObjectMeta::default(),
            id: 0,
        }
    }
}

impl Object {
    pub fn id(&self) -> ObjectID {
        self.id
    }

    pub fn meta(&self) -> &ObjectMeta {
        &self.meta
    }
    
    pub fn nbytes(&self) -> usize {
        self.meta.get_nbytes()
    }

    pub fn construct(&mut self, meta: &ObjectMeta) {
        self.id = meta.get_id();
        self.meta = meta.clone();
    }

    pub fn persist() {} // TODO

    pub fn is_local(&self) -> bool {
        self.meta.is_local()
    }

    pub fn is_persist(&self) -> bool { // TODO
        false
    }

    pub fn is_global(&self) -> bool {
        self.meta.is_global()
    }
}

impl ObjectBase for Object {
    fn build(client: Box<dyn Client>) -> io::Result<()> {
        Ok(())
    }

    fn seal(client: Box<dyn Client>) -> Rc<Object> {
        panic!("") // Question: shared_from_this()
    }
}

#[derive(Debug)]
pub struct ObjectBuilder {
    sealed: bool,
}

impl ObjectBuilder {
    pub fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}


impl ObjectBase for ObjectBuilder {
    fn build(client: Box<dyn Client>) -> io::Result<Blob> {
        panic!("") // Question: override = 0
    }

    fn seal(client: Box<dyn Client>) -> Rc<Object> {
        panic!("")
    }
}
