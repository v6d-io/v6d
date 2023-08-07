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
use std::hash::Hash;
use std::rc::Rc;

use arrow_buffer::Buffer;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

use crate::common::util::json::*;
use crate::common::util::status::*;
use crate::common::util::uuid::*;

use super::super::client::Client;
use super::super::IPCClient;
use super::blob::BufferSet;
use super::object::{Create, Object};
use super::object_factory::ObjectFactory;

#[derive(Debug, Clone)]
pub struct ObjectMeta {
    id: ObjectID,
    meta: JSON,
    client: *mut IPCClient,
    incomplete: bool,
    force_local: bool,
    buffers: Rc<BufferSet>,
}

impl Default for ObjectMeta {
    fn default() -> Self {
        ObjectMeta {
            id: invalid_object_id(),
            meta: JSON::new(),
            client: std::ptr::null_mut(),
            incomplete: false,
            force_local: false,
            buffers: Rc::new(BufferSet::default()),
        }
    }
}

impl ObjectMeta {
    pub fn new(client: *mut IPCClient, metadata: JSON) -> Result<Self> {
        let mut meta = ObjectMeta::default();
        meta.set_meta_data(client, metadata)?;
        return Ok(meta);
    }

    pub fn new_from_typename(typename: &str) -> Self {
        let mut meta = ObjectMeta::default();
        meta.set_typename(typename);
        return meta;
    }

    pub fn new_from_metadata(metadata: JSON) -> Result<Self> {
        let mut meta = ObjectMeta::default();
        meta.set_meta_data(std::ptr::null_mut(), metadata)?;
        return Ok(meta);
    }

    pub fn set_client(&mut self, client: *mut IPCClient) {
        self.client = client;
    }

    pub fn get_client(&self) -> Result<&mut IPCClient> {
        if self.client.is_null() {
            return Err(VineyardError::invalid(
                "the associated client is not available",
            ));
        } else {
            return Ok(unsafe { &mut *self.client });
        }
    }

    pub fn get_client_unchecked(&self) -> &IPCClient {
        return unsafe { &*self.client };
    }

    pub fn set_id(&mut self, id: ObjectID) {
        self.id = id;
        self.meta
            .insert("id".into(), Value::from(object_id_to_string(id)));
    }

    pub fn get_id(&self) -> ObjectID {
        return self.id;
    }

    pub fn get_signature(&self) -> Result<Signature> {
        return get_uint(&self.meta, "signature");
    }

    pub fn set_signature(&mut self, signature: Signature) {
        self.meta.insert("signature".into(), Value::from(signature));
    }

    pub fn reset_signature(&mut self) {
        self.reset_key(&String::from("signature"));
    }

    pub fn set_global(&mut self, global: bool) {
        self.meta
            .insert(String::from("global"), serde_json::Value::Bool(global));
    }

    pub fn is_global(&self) -> bool {
        return get_bool_or(&self.meta, "global", false);
    }

    pub fn set_transient(&mut self, transient: bool) {
        self.meta.insert(
            String::from("transient"),
            serde_json::Value::Bool(transient),
        );
    }

    pub fn is_transient(&self) -> bool {
        return get_bool_or(&self.meta, "transient", true);
    }

    pub fn is_persistent(&self) -> bool {
        return !self.is_transient();
    }

    pub fn set_typename(&mut self, typename: &str) {
        self.meta.insert(
            "typename".into(),
            serde_json::Value::String(typename.into()),
        );
    }

    pub fn get_typename(&self) -> Result<&str> {
        return get_string(&self.meta, "typename");
    }

    pub fn set_nbytes(&mut self, nbytes: usize) {
        self.meta
            .insert(String::from("nbytes"), Value::from(nbytes));
    }

    pub fn get_nbytes(&self) -> usize {
        return get_uint_or(&self.meta, "nbytes", 0) as usize;
    }

    pub fn get_instance_id(&self) -> Result<InstanceID> {
        return get_uint(&self.meta, "instance_id");
    }

    pub fn set_instance_id(&mut self, instance_id: InstanceID) {
        self.meta
            .insert("instance_id".into(), Value::from(instance_id));
    }

    pub fn is_local(&self) -> bool {
        if self.force_local {
            return true;
        }
        if !self.meta.contains_key("instance_id") {
            return true;
        }
        match (self.get_client(), self.meta.get("instance_id")) {
            (Ok(client), Some(instance_id)) => match instance_id.as_u64() {
                Some(instance_id) => return client.instance_id() == instance_id as InstanceID,
                _ => return false,
            },
            _ => return false,
        }
    }

    pub fn force_local(&mut self) {
        self.force_local = true;
    }

    pub fn has_key(&self, key: &str) -> bool {
        self.meta.contains_key(key)
    }

    pub fn reset_key(&mut self, key: &str) {
        if self.meta.contains_key(key) {
            self.meta.remove(key);
        }
    }

    pub fn add_value(&mut self, key: &str, value: Value) {
        self.meta
            .insert(key.into(), Value::String(value.to_string()));
    }

    pub fn get_value(&self, key: &str) -> Result<Value> {
        return serde_json::from_str(get_string(&self.meta, key)?).map_err(|e| {
            VineyardError::invalid(format!("Invalid json value at key {}: {}", key, e))
        });
    }

    pub fn add_bool(&mut self, key: &str, value: bool) {
        self.meta.insert(key.into(), serde_json::Value::Bool(value));
    }

    pub fn get_bool(&self, key: &str) -> Result<bool> {
        return get_bool(&self.meta, key);
    }

    pub fn add_int<T: Into<i64>>(&mut self, key: &str, value: T) {
        self.meta
            .insert(key.into(), serde_json::Value::Number(value.into().into()));
    }

    pub fn get_int<T: From<i64>>(&self, key: &str) -> Result<T> {
        return get_int(&self.meta, key);
    }

    pub fn add_isize(&mut self, key: &str, value: isize) {
        self.meta
            .insert(key.into(), serde_json::Value::Number(value.into()));
    }

    pub fn get_isize(&self, key: &str) -> Result<isize> {
        return get_isize(&self.meta, key);
    }

    pub fn add_uint<T: Into<u64>>(&mut self, key: &str, value: T) {
        self.meta
            .insert(key.into(), serde_json::Value::Number(value.into().into()));
    }

    pub fn get_uint<T: From<u64>>(&self, key: &str) -> Result<T> {
        return get_uint(&self.meta, key);
    }

    pub fn add_usize(&mut self, key: &str, value: usize) {
        self.meta
            .insert(key.into(), serde_json::Value::Number(value.into()));
    }

    pub fn get_usize(&self, key: &str) -> Result<usize> {
        return get_usize(&self.meta, key);
    }

    pub fn add_string<T: Into<String>>(&mut self, key: &str, value: T) {
        self.meta
            .insert(key.into(), serde_json::Value::String(value.into()));
    }

    pub fn get_string(&self, key: &str) -> Result<&str> {
        return get_string(&self.meta, key);
    }

    pub fn add_vector<T: Serialize>(&mut self, key: &str, value: &[T]) -> Result<()> {
        self.add_value(key, serde_json::to_value(value)?);
        return Ok(());
    }

    pub fn get_vector<T: DeserializeOwned>(&self, key: &str) -> Result<Vec<T>> {
        return serde_json::from_value(self.get_value(key)?).map_err(|e| {
            VineyardError::invalid(format!("Invalid json value at key {}: {}", key, e))
        });
    }

    pub fn add_set<T: Eq + Serialize + std::hash::Hash>(
        &mut self,
        key: &str,
        value: &HashSet<T>,
    ) -> Result<()> {
        self.add_value(key, serde_json::to_value(value)?);
        return Ok(());
    }

    pub fn get_set<T: DeserializeOwned + Eq + Hash>(&self, key: &str) -> Result<HashSet<T>> {
        return serde_json::from_value(self.get_value(key)?).map_err(|e| {
            VineyardError::invalid(format!("Invalid json value at key {}: {}", key, e))
        });
    }

    pub fn add_map_key_value<T: Serialize>(
        &mut self,
        key: &str,
        value: HashMap<String, T>,
    ) -> Result<()> {
        self.add_value(key, serde_json::to_value(value)?);
        return Ok(());
    }

    pub fn get_map_key_value<T: DeserializeOwned>(&self, key: &str) -> Result<HashMap<String, T>> {
        return serde_json::from_value(self.get_value(key)?).map_err(|e| {
            VineyardError::invalid(format!("Invalid json value at key {}: {}", key, e))
        });
    }

    pub fn add_member_meta(&mut self, key: &str, member: &ObjectMeta) -> Result<()> {
        if self
            .meta
            .insert(key.into(), Value::Object(member.meta.clone()))
            .is_some()
        {
            return Err(VineyardError::invalid(format!(
                "key '{}' already exists",
                key
            )));
        }
        match Rc::<BufferSet>::get_mut(&mut self.buffers) {
            Some(buffers) => {
                buffers.extend(&member.buffers);
            }
            None => {
                warn!("Cannot extend buffers of a shared object meta.");
            }
        };
        return Ok(());
    }

    pub fn add_member(&mut self, name: &str, member: Box<dyn Object>) -> Result<()> {
        return self.add_member_meta(name, member.meta());
    }

    pub fn add_member_ref(&mut self, name: &str, member: &dyn Object) -> Result<()> {
        return self.add_member_meta(name, member.meta());
    }

    pub fn add_member_rc(&mut self, name: &str, member: Rc<dyn Object>) -> Result<()> {
        return self.add_member_meta(name, member.meta());
    }

    pub fn add_member_id(&mut self, name: &str, member: ObjectID) -> Result<()> {
        match self
            .meta
            .insert(name.into(), json!({ "id": object_id_to_string(member) }))
        {
            Some(_) => {
                return Err(VineyardError::invalid(format!(
                    "key '{}' already exists",
                    name
                )));
            }
            None => {
                self.incomplete = true;
                return Ok(());
            }
        }
    }

    pub fn get_member<T: Object + Create>(&self, name: &str) -> Result<Box<T>> {
        use crate::client::downcast_object;

        let meta = self.get_member_meta(name)?;
        let mut object = T::create();
        object.construct(meta)?;
        return downcast_object::<T>(object);
    }

    pub fn get_member_untyped(&self, name: &str) -> Result<Box<dyn Object>> {
        let meta = self.get_member_meta(name)?;
        return ObjectFactory::create_from_metadata(meta);
    }

    pub fn get_member_meta(&self, key: &str) -> Result<ObjectMeta> {
        match self.meta.get(key) {
            Some(Value::Object(value)) => {
                let mut meta = ObjectMeta::default();
                meta.set_meta_data(self.client, value.clone())?;

                let buffers = meta.get_buffers_mut()?;
                for (id, buffer) in buffers.buffers_mut() {
                    // for remote object, the blob may not present here
                    if let Ok(Some(buf)) = self.buffers.get(*id) {
                        let _ = buffer.insert(buf);
                    }
                }
                if self.force_local {
                    meta.force_local();
                }
                return Ok(meta);
            }
            Some(_) => {
                return Err(VineyardError::invalid(format!(
                    "Invalid json value at key {}: not an object",
                    key
                )));
            }
            _ => {
                return Err(VineyardError::invalid(format!(
                    "Invalid json value: key {} not found",
                    key
                )));
            }
        }
    }

    pub fn get_member_id(&self, key: &str) -> Result<ObjectID> {
        match self.meta.get(key) {
            Some(Value::Object(value)) => {
                let id = object_id_from_string(get_string(value, "id")?)?;
                return Ok(id);
            }
            Some(_) => {
                return Err(VineyardError::invalid(format!(
                    "Invalid json value at key {}: not an object",
                    key
                )));
            }
            _ => {
                return Err(VineyardError::invalid(format!(
                    "Invalid json value: key {} not found",
                    key
                )));
            }
        }
    }

    pub fn get_buffer(&self, blob_id: ObjectID) -> Result<Option<Buffer>> {
        return self.buffers.get(blob_id).map_err(|err| {
            VineyardError::invalid(format!(
                "The target blob {} doesn't exist: {}",
                object_id_to_string(blob_id),
                err,
            ))
        });
    }

    pub fn set_buffer(&mut self, id: ObjectID, buffer: Option<Buffer>) -> Result<()> {
        match self.get_buffers_mut() {
            Ok(buffers) => {
                buffers.emplace_buffer(id, buffer)?;
            }
            Err(_) => {
                warn!("Cannot extend buffers of a shared object meta.");
            }
        };
        return Ok(());
    }

    pub fn set_or_add_buffer(&mut self, id: ObjectID, buffer: Option<Buffer>) -> Result<()> {
        match self.get_buffers_mut() {
            Ok(buffers) => {
                let _ = buffers.emplace(id); // ensure the id exists
                buffers.emplace_buffer(id, buffer)?;
            }
            Err(_) => {
                warn!("Cannot extend buffers of a shared object meta.");
            }
        };
        return Ok(());
    }

    pub fn is_incomplete(&self) -> bool {
        return self.incomplete;
    }

    pub fn set_incomplete(&mut self, incomplete: bool) {
        self.incomplete = incomplete;
    }

    pub fn reset(&mut self) {
        self.meta = JSON::new();
        self.client = std::ptr::null_mut();
        self.incomplete = false;
        self.force_local = false;
        self.buffers = Rc::new(BufferSet::default());
    }

    pub fn meta_data(&self) -> &JSON {
        return &self.meta;
    }

    pub fn mut_meta_data(&mut self) -> &mut JSON {
        return &mut self.meta;
    }

    pub fn set_meta_data(&mut self, client: *mut IPCClient, meta: JSON) -> Result<()> {
        // the return result is just for fast failure in `find_all_blobs`
        self.client = client;
        self.find_all_blobs(&meta)?;

        // leave the move after `find_all_blobs` to avoid copy
        self.id = object_id_from_string(get_string(&meta, "id")?)?;
        self.meta = meta;
        return Ok(());
    }

    pub fn get_buffers(&self) -> &BufferSet {
        return &self.buffers;
    }

    pub fn get_buffers_mut(&mut self) -> Result<&mut BufferSet> {
        match Rc::<BufferSet>::get_mut(&mut self.buffers) {
            Some(buffers) => {
                return Ok(buffers);
            }
            None => {
                warn!("Cannot extend buffers of a shared object meta.");
                return Err(VineyardError::invalid(
                    "Cannot manipulate buffers of a shared object meta.",
                ));
            }
        };
    }

    fn find_all_blobs(&mut self, tree: &JSON) -> Result<()> {
        if tree.is_empty() {
            return Ok(());
        }
        let member_id: ObjectID = object_id_from_string(get_string(tree, "id")?)?;
        if is_blob(member_id) {
            let blob_instance_id: ObjectID = get_uint(tree, "instance_id")?;
            let instance_id;
            if let Ok(client) = self.get_client() {
                instance_id = client.instance_id();
            } else {
                // no client instance, ignore
                return Ok(());
            }
            if instance_id == blob_instance_id {
                match Rc::<BufferSet>::get_mut(&mut self.buffers) {
                    Some(buffers) => {
                        // duplicate is possible, and is ok state
                        let _ = buffers.emplace(member_id);
                    }
                    None => {
                        warn!("Cannot extend buffers of a shared object meta.");
                    }
                };
            }
            return Ok(());
        } else {
            for item in tree.values() {
                if let Value::Object(item) = item {
                    self.find_all_blobs(item)?;
                }
            }
            return Ok(());
        }
    }
}
