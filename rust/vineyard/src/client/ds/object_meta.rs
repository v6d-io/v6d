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

use std::rc::{Rc, Weak};
use std::ops;

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use super::{Client, ClientKind};
use super::blob::BufferSet;
use super::object::Object;
use super::object_factory::ObjectFactory;

use super::uuid::*;
use super::status::*;

#[derive(Debug, Clone)]
pub struct ObjectMeta {
    client: Weak<ClientKind>, // Question: Weak<dyn Client>  
    meta: Value,
    buffer_set: Rc<BufferSet>, 
    incomplete: bool,
    force_local: bool,
}

impl Default for ObjectMeta {
    fn default() -> Self {
        ObjectMeta {
            client: Weak::new(),
            meta: json!({}),
            buffer_set: Rc::new(BufferSet::default()), // Question: empty struct?
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

    pub fn set_client(&mut self, client: Weak<ClientKind>) {
        self.client = client;
    }

    pub fn get_client(&self) -> Weak<ClientKind> {
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
        self.meta.as_object_mut().unwrap().insert(
            String::from("global"), serde_json::Value::Bool(global)
        );
    }

    pub fn is_global(&self) -> bool {
        self.meta["global"].as_bool().unwrap()
    }

    pub fn set_type_name(&mut self, type_name: &String) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("typename"), serde_json::Value::String(type_name.clone())
        );
    }

    pub fn get_type_name(&self) -> String {
        self.meta["typename"].as_str().unwrap().to_string()
    }

    pub fn set_nbytes(&mut self, nbytes: usize) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("nbytes"), serde_json::Value::from(nbytes)
        );
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
        }else {
            if self.client.weak_count()!=0 { // Question: is it correct?
                let instance_id = match self.client.upgrade().unwrap().as_ref(){
                    ClientKind::IPCClient(client) => client.instance_id(),
                    ClientKind::RPCClient(client) => client.instance_id(),
                };
                return instance_id == self.meta["instance_id"].as_u64().unwrap() as InstanceID;
            }else {
                return false;
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
        if self.meta.as_object_mut().unwrap().contains_key(key){
            self.meta.as_object_mut().unwrap().remove(key);
        }
    }

    // Question: clone or reference?
    // A bunch of functions. Which to implement?
    // Function name?
    pub fn add_key_value_string(&mut self, key: &String, value: &String) { 
        self.meta.as_object_mut().unwrap().insert(
            key.clone(), serde_json::Value::String(value.clone())
        );
    }

    pub fn add_key_value_json(&mut self, key: &String, value: &Value) {
        self.meta.as_object_mut().unwrap().insert(
            key.clone(), value.clone()
        );
    }

    pub fn get_key_value(&self, key: &String) -> Value {
        json!({})
    }

    pub fn add_member(&mut self) {}


    pub fn get_member(&self, name: &String) -> Rc<Object> {
        let meta = self.get_member_meta(name); //TODO
        let object = match ObjectFactory::create(&meta.get_type_name()) {
            Err(_) => { // Question: std::unique_ptr<Object>(new Object());
                let mut object = Box::new(Object::default());
                object.construct(meta);
                return Rc::new(*object);
            }, 
            Ok(mut object) => {
                object.construct(meta);
                return Rc::new(*object);
            }
        };
        
    }

    pub fn get_member_meta(&self, name: &String) -> ObjectMeta {
        let ret = ObjectMeta::default();
        let child_meta = &self.meta[name.as_str()];
        VINEYARD_ASSERT(!child_meta.is_null());
        ret.set_meta_data(Rc::clone(&self.client.upgrade().unwrap()), &child_meta);
        let all_blobs = self.buffer_set.all_buffers();

        ret
    }

    pub fn get_buffer(&self) {}

    pub fn set_buffer(&mut self) {}

    pub fn reset() {}

    pub fn print_meta() {}

    pub fn incomplete() {}
    
    pub fn meta_data() {}

    pub fn mut_meta_data() {}

    pub fn set_meta_data(&mut self, client: ClientKind, meta: &Value) {
        self.client = Rc::<ClientKind>::downgrade(&Rc::new(client));
        self.meta = meta.clone(); // Question: move or ref?
        self.find_all_blobs();
    }

    // Question: fn unsafe()

    pub fn find_all_blobs(&self) {
        let tree = &self.meta;
        if tree.is_null(){
            return;
        }
        let member_id = object_id_from_string(&tree["id"].as_str().unwrap().to_string());
        if is_blob(member_id) {

        }else {
            
        }

    }

    pub fn set_instance_id(&mut self, instance_id: InstanceID) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("instance_id"), serde_json::Value::from(instance_id)
        );
    }

    pub fn set_signature(&mut self, signature: Signature) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("signature"), serde_json::Value::from(signature)
        );
    }

}
