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

use super::uuid::*;
use super::{Client, ClientKind};
use super::blob::BufferSet;
use super::object::Object;
use super::object_factory::ObjectFactory;

#[derive(Debug)]
pub struct ObjectMeta {
    client: Weak<ClientKind>, // Question: Since W<T> doesn't have T:?Sized for Weak<dyn Client>  
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
            buffer_set: Rc::new(BufferSet{}),
            incomplete: false,
            force_local: false,
        }
    }
}

impl ObjectMeta {
    fn from(other: &ObjectMeta) -> ObjectMeta {
        ObjectMeta {
            client: other.client.clone(),
            meta: other.meta.clone(),
            buffer_set: Rc::clone(&other.buffer_set),
            incomplete: other.incomplete,
            force_local: other.force_local,
        }
    }

    fn set_client(&mut self, client: Weak<ClientKind>) {
        self.client = client;
    }

    fn get_client(&self) -> Weak<ClientKind> {
        self.client.clone()
    }

    fn set_id(&mut self, id: ObjectID) {
        self.meta = serde_json::from_str(&id.to_string()).unwrap();
    }

    fn get_id(&self) -> ObjectID {
        self.meta["id"].as_u64().unwrap()
    }

    fn get_signature(&self) -> Signature {
        self.meta["signature"].as_u64().unwrap()
    }

    fn reset_signature(&mut self) {
        self.reset_key(&String::from("signature"));
    }

    fn set_global(&mut self, global: bool) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("global"), serde_json::Value::Bool(global)
        );
    }

    fn is_global(&self) -> bool {
        self.meta["global"].as_bool().unwrap()
    }

    fn set_type_name(&mut self, type_name: &String) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("typename"), serde_json::Value::String(type_name.clone())
        );
    }

    fn get_type_name(&self) -> String {
        self.meta["typename"].as_str().unwrap().to_string()
    }

    fn set_nbytes(&mut self, nbytes: usize) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("nbytes"), serde_json::Value::from(nbytes)
        );
    }

    fn get_nbytes(&self) -> usize {
        match self.meta["nbytes"].is_null() {
            true => return 0,
            false => self.meta["nbytes"].as_u64().unwrap() as usize,
        }
    }

    fn get_instance_id(&self) -> InstanceID {
        self.meta["instance_id"].as_u64().unwrap() as InstanceID
    }

    fn is_local(&self) -> bool {
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

    fn force_local(&mut self) {
        self.force_local = true;
    }

    fn has_key(&self, key: &String) -> bool {
        self.meta.as_object().unwrap().contains_key(key)
    }

    fn reset_key(&mut self, key: &String) {
        if self.meta.as_object_mut().unwrap().contains_key(key){
            self.meta.as_object_mut().unwrap().remove(key);
        }
    }

    // Question: clone or reference?
    // A bunch of functions. Which to implement?
    // Function name?
    fn add_key_value_string(&mut self, key: &String, value: &String) { 
        self.meta.as_object_mut().unwrap().insert(
            key.clone(), serde_json::Value::String(value.clone())
        );
    }

    fn add_key_value_json(&mut self, key: &String, value: &Value) {
        self.meta.as_object_mut().unwrap().insert(
            key.clone(), value.clone()
        );
    }

    fn get_key_value(&self, key: &String) -> Value {
        json!({})
    }

    fn add_member(&mut self) {}

    fn get_member(&self, name: &String) {
        let meta = self.get_member_meta(name);
        let object = ObjectFactory::create(&meta);
    }

    // Question: VINEYARD_ASSERT?
    fn get_member_meta(&self, name: &String) -> ObjectMeta {
        let child_meta = &self.meta[name.as_str()];
        ObjectMeta::default()
    }

    fn get_buffer(&self) {}

    fn set_buffer(&mut self) {}

    fn reset() {}

    fn print_meta() {}

    fn incomplete() {}
    
    fn meta_data() {}

    fn mut_meta_data() {}

    fn set_meta_data(&mut self, client: ClientKind, meta: Value) {
        self.client = Rc::<ClientKind>::downgrade(&Rc::new(client));
        self.meta = meta; // Question: move or ref?
        self.find_all_blobs();
    }

    // Question: fn unsafe()

    fn find_all_blobs(&self) {
        let tree = &self.meta;
        if tree.is_null(){
            return;
        }
        let member_id = object_id_from_string(&tree["id"].as_str().unwrap().to_string());
        if is_blob(member_id) {

        }else {
            
        }

    }

    fn set_instance_id(&mut self, instance_id: InstanceID) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("instance_id"), serde_json::Value::from(instance_id)
        );
    }

    fn set_signature(&mut self, signature: Signature) {
        self.meta.as_object_mut().unwrap().insert(
            String::from("signature"), serde_json::Value::from(signature)
        );
    }

}
