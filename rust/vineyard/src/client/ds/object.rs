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

use serde_json::json;
use dyn_clone::{clone_trait_object, DynClone};

use super::blob::Blob;
use super::object_meta::ObjectMeta;
use super::status::*;
use super::uuid::ObjectID;
use super::Client;
use super::IPCClient;

pub trait ObjectBase {
    fn build(&mut self, client: &IPCClient) -> io::Result<()> {
        Ok(())
    }
    fn seal(&mut self, client: &IPCClient) -> Rc<Object> {
        panic!()
    }
}


pub trait Object: Send + ObjectBase + DynClone {
    fn meta(&self) -> &ObjectMeta;

    fn meta_mut(&mut self) -> &mut ObjectMeta;

    fn id(&self) -> ObjectID;

    fn set_id(&mut self, id: ObjectID);

    fn set_meta(&mut self, meta: &ObjectMeta);
    
    fn nbytes(&self) -> usize {
        self.meta().get_nbytes()
    }

    fn construct(&mut self, meta: &ObjectMeta) {
        self.set_id(meta.get_id());
        self.set_meta(meta);
    }

    fn persist(&self, client: &mut dyn Client) -> io::Result<()> {
        client.persist(self.id())
    }

    fn is_local(&self) -> bool {
        self.meta().is_local()
    }

    fn is_persist(&mut self) -> bool {
        let persist = !(self
            .meta()
            .get_key_value(&"transient".to_string())
            .as_bool()
            .unwrap());
        if (!persist) {
            let client = self.meta().get_client().unwrap().upgrade().unwrap();
            VINEYARD_CHECK_OK(client.if_persist(self.id()));
            let persist = client.if_persist(self.id()).unwrap();
            if persist {
                self.meta_mut()
                    .add_json_key_value(&"transient".to_string(), &json!(false));
            }
        }
        persist
    }

    fn is_global(&self) -> bool {
        self.meta().is_global()
    }
}

clone_trait_object!(Object);

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

impl ObjectBase for ObjectBuilder {}


pub trait Registered: Object {
    fn registered() {}
}