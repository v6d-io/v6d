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
use std::io::{self, Error, ErrorKind};
use std::rc::Rc;

use super::blob::Blob;
use super::object_meta::ObjectMeta;
use super::Client;
use super::uuid::ObjectID;

pub trait ObjectBase {
    fn build(client: Box<dyn Client>) -> Result<Blob, Error>;
    fn seal(client: Box<dyn Client>) -> Rc<Object>;
}

#[derive(Debug)]
pub struct Object {
    pub meta: ObjectMeta,
    pub id: ObjectID,
}

impl Default for Object {
    fn default() -> Object {
        Object{
            meta: ObjectMeta::default(),
            id: 0,
        }
    }
}

impl Object {
    pub fn construct(&mut self, meta: ObjectMeta) {
        self.id = meta.get_id();
        self.meta = meta;
    }
}

impl ObjectBase for Object {
    fn build(client: Box<dyn Client>) -> Result<Blob, Error> {
        panic!("")
    }

    fn seal(client: Box<dyn Client>) -> Rc<Object> {
        panic!("")
    }
}

#[derive(Debug)]
pub struct ObjectBuilder {
    sealed: bool,
}

impl ObjectBase for ObjectBuilder {
    fn build(client: Box<dyn Client>) -> Result<Blob, Error> {
        panic!("")
    }

    fn seal(client: Box<dyn Client>) -> Rc<Object> {
        panic!("")
    }
}
