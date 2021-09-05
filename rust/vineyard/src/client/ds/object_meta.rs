use std::cell::RefCell;
use std::fmt;
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
use std::ops;
use std::rc::{Rc, Weak};

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use super::blob::BufferSet;
use super::object::Object;
use super::object_factory::ObjectFactory;
use super::Client;

use super::status::*;
use super::uuid::*;

use super::blob::ArrowBuffer; // TODO. arrow/buffer: dependencies

#[derive(Debug, Clone)]
pub struct ObjectMeta {
    client: Option<Weak<dyn Client>>, // Question: Weak<dyn Client>
    meta: Value,
    buffer_set: Rc<RefCell<BufferSet>>,
    incomplete: bool,
    force_local: bool,
}

impl fmt::Debug for dyn Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instance ID: {}\n", self.instance_id())
    }
}

impl Default for ObjectMeta {
    fn default() -> Self {
        ObjectMeta {
            client: None,
            meta: json!({}),
            buffer_set: Rc::new(RefCell::new(BufferSet::default())),
            incomplete: false,
            force_local: false,
        }
    }
}

impl ObjectMeta {
    pub fn from(other: &ObjectMeta) -> ObjectMeta {
        ObjectMeta {
            client: other.client.clone(),
            meta: other.meta.clone(),
            buffer_set: Rc::clone(&other.buffer_set),
            incomplete: other.incomplete,
            force_local: other.force_local,
        }
    }

    pub fn set_client(&mut self, client: Option<Weak<dyn Client>>) {
        self.client = client;
    }

    pub fn get_client(&self) -> Option<Weak<dyn Client>> {
        self.client.clone()
    }

    pub fn set_id(&mut self, id: ObjectID) {
        self.meta = serde_json::from_str(&id.to_string()).unwrap();
    }

    pub fn get_id(&self) -> ObjectID {
        self.meta["id"].as_u64().unwrap() as ObjectID
    }

    pub fn get_signature(&self) -> Signature {
        self.meta["signature"].as_u64().unwrap() as Signature
    }

    pub fn reset_signature(&mut self) {
        self.reset_key(&String::from("signature"));
    }

    pub fn set_global(&mut self, global: bool) {
        self.meta
            .as_object_mut()
            .unwrap()
            .insert(String::from("global"), serde_json::Value::Bool(global));
    }

    pub fn is_global(&self) -> bool {
        self.meta["global"].as_bool().unwrap()
    }

    pub fn set_type_name(&mut self, type_name: &String) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("typename"),
            serde_json::Value::String(type_name.clone()),
        );
    }

    pub fn get_type_name(&self) -> String {
        self.meta["typename"].as_str().unwrap().to_string()
    }

    pub fn set_nbytes(&mut self, nbytes: usize) {
        self.meta
            .as_object_mut()
            .unwrap()
            .insert(String::from("nbytes"), serde_json::Value::from(nbytes));
    }

    pub fn get_nbytes(&self) -> usize {
        match self.meta["nbytes"].is_null() {
            true => return 0,
            false => self.meta["nbytes"].as_u64().unwrap() as usize,
        }
    }

    pub fn get_instance_id(&self) -> InstanceID {
        self.meta["instance_id"].as_u64().unwrap() as InstanceID
    }

    pub fn is_local(&self) -> bool {
        if self.force_local {
            return true;
        }
        if self.meta["instance_id"].is_null() {
            return true;
        } else {
            match &self.client {
                // Question: is it correct?
                Some(client) => {
                    let instance_id = client.upgrade().unwrap().as_ref().instance_id();
                    return instance_id == self.meta["instance_id"].as_u64().unwrap() as InstanceID;
                }
                None => return false,
            }
        }
    }

    pub fn force_local(&mut self) {
        self.force_local = true;
    }

    pub fn has_key(&self, key: &String) -> bool {
        self.meta.as_object().unwrap().contains_key(key)
    }

    pub fn reset_key(&mut self, key: &String) {
        if self.meta.as_object_mut().unwrap().contains_key(key) {
            self.meta.as_object_mut().unwrap().remove(key);
        }
    }

    pub fn add_string_key_value(&mut self, key: &String, value: &String) {
        self.meta
            .as_object_mut()
            .unwrap()
            .insert(key.clone(), serde_json::Value::String(value.clone()));
    }

    pub fn add_json_key_value(&mut self, key: &String, value: &Value) {
        self.meta
            .as_object_mut()
            .unwrap()
            .insert(key.clone(), value.clone());
    }

    pub fn get_key_value(&self, key: &String) -> &Value {
        match self.meta.get(key) {
            Some(value) => return value,
            None => panic!("Invalid json value at key {}", key),
        }
    }

    pub fn add_member_with_meta(&mut self, name: &String, member: &ObjectMeta) {
        VINEYARD_ASSERT(!self.meta.as_object().unwrap().contains_key(name));
        self.meta
            .as_object_mut()
            .unwrap()
            .insert(name.clone(), member.meta.clone());
        self.buffer_set
            .borrow_mut()
            .extend(&member.buffer_set.borrow());
    }

    pub fn add_member_with_object(&mut self, name: &String, member: &Object) {
        self.add_member_with_meta(name, &member.meta);
    }

    pub fn add_member_with_id(&mut self, name: &String, member_id: ObjectID) {
        VINEYARD_ASSERT(!self.meta.as_object().unwrap().contains_key(name));
        let member_node = json!({ "id": object_id_to_string(member_id) });
        self.meta
            .as_object_mut()
            .unwrap()
            .insert(name.clone(), member_node);
        self.incomplete = true;
    }

    pub fn get_member(&self, name: &String) -> Rc<Object> {
        let meta = self.get_member_meta(name);
        let object = match ObjectFactory::create(&meta.get_type_name()) {
            //TODO
            Err(_) => {
                let mut object = Box::new(Object::default());
                object.construct(&meta);
                return Rc::new(*object);
            }
            Ok(mut object) => {
                object.construct(&meta);
                return Rc::new(*object);
            }
        };
    }

    pub fn get_member_meta(&self, name: &String) -> ObjectMeta {
        let mut ret = ObjectMeta::default();
        let child_meta = &self.meta[name.as_str()];
        VINEYARD_ASSERT(!child_meta.is_null());
        ret.set_meta_data(self.client.as_ref().unwrap(), &child_meta);
        let all_buffer_set = self.buffer_set.borrow();
        let all_blobs = all_buffer_set.all_buffers();
        let ret_blobs = ret.buffer_set.borrow().all_buffers().clone(); // Warning: memeroy?
        for (key, _) in ret_blobs.iter() {
            if let Some(value) = all_blobs.get(key) {
                ret.set_buffer(*key, value);
            }
            if let true = self.force_local {
                ret.force_local();
            }
        }

        ret
    }

    pub fn get_buffer(&self, blob_id: ObjectID) -> io::Result<Rc<ArrowBuffer>> {
        match self.buffer_set.borrow().get(blob_id) {
            Ok(buffer) => return Ok(buffer),
            Err(_) => panic!(
                "The target blob {} doesn't exist.",
                object_id_to_string(blob_id)
            ),
        }
    }

    pub fn set_buffer(&mut self, id: ObjectID, buffer: &Rc<ArrowBuffer>) {
        VINEYARD_ASSERT(self.buffer_set.borrow().contains(id));
        VINEYARD_CHECK_OK(self.buffer_set.borrow().emplace_buffer(id, buffer)); // TODO
    }

    pub fn reset(&mut self) {
        self.client = None;
        self.meta = json!({});
        self.buffer_set = Rc::new(RefCell::new(BufferSet::default()));
        self.incomplete = false;
    }

    pub fn print_meta() {}

    pub fn incomplete(&self) -> bool {
        self.incomplete
    }

    pub fn meta_data(&self) -> &Value {
        &self.meta
    }

    pub fn mut_meta_data(&mut self) -> &mut Value {
        &mut self.meta
    }

    pub fn set_meta_data(&mut self, client: &Weak<dyn Client>, meta: &Value) {
        self.client = Some(client.clone());
        self.meta = meta.clone();
        self.find_all_blobs(&self.meta);
    }

    pub fn get_buffer_set(&self) -> &Rc<RefCell<BufferSet>> {
        &self.buffer_set
    }

    // Check logic
    pub fn find_all_blobs(&self, tree: &Value) {
        if tree.is_null() {
            return;
        }
        let member_id = object_id_from_string(&tree["id"].as_str().unwrap().to_string());
        if is_blob(member_id) {
            let instance_id = self
                .client
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .instance_id();
            let cond1: bool = instance_id == tree["instance_id"].as_u64().unwrap() as InstanceID;
            let mut cond2: bool = false;
            if let None = self.client {
                cond2 = true;
            }
            if cond2 || cond1 {
                // QUESTION // TODO
                VINEYARD_CHECK_OK(self.buffer_set.borrow().emplace_buffer_null(member_id));
            }
        } else {
            for item in tree.as_object().unwrap().values() {
                if item.is_object() {
                    self.find_all_blobs(item);
                }
            }
        }
    }

    fn set_instance_id(&mut self, instance_id: InstanceID) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("instance_id"),
            serde_json::Value::from(instance_id),
        );
    }

    fn set_signature(&mut self, signature: Signature) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("signature"),
            serde_json::Value::from(signature),
        );
    }
}
