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

use std::collections::HashMap;

use parking_lot::ReentrantMutexGuard;

use crate::common::util::json::*;
use crate::common::util::protocol::*;
use crate::common::util::status::*;
use crate::common::util::uuid::invalid_object_id;
use crate::common::util::uuid::Signature;
use crate::common::util::uuid::{InstanceID, ObjectID};

use super::ds::ObjectMeta;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VINEYARD_IPC_SOCKET_KEY: &str = "VINEYARD_IPC_SOCKET";
pub const VINEYARD_RPC_ENDPOINT_KEY: &str = "VINEYARD_RPC_ENDPOINT";
pub const DEFAULT_RPC_PORT: u16 = 9600;

#[derive(Debug, Clone)]
pub struct InstanceStatus {
    pub instance_id: InstanceID,
    pub deployment: String,
    pub memory_usage: usize,
    pub memory_limit: usize,
    pub deferred_requests: usize,
    pub ipc_connections: usize,
    pub rpc_connections: usize,
}

impl InstanceStatus {
    pub fn from_json(status: &JSON) -> Result<Self> {
        return Ok(InstanceStatus {
            instance_id: get_uint(status, "instance_id")?,
            deployment: get_string(status, "deployment")?.into(),
            memory_usage: get_usize(status, "memory_usage")?,
            memory_limit: get_usize(status, "memory_limit")?,
            deferred_requests: get_usize(status, "deferred_requests")?,
            ipc_connections: get_usize(status, "ipc_connections")?,
            rpc_connections: get_usize(status, "rpc_connections")?,
        });
    }
}

pub trait Client {
    /// Disconnect this client.
    fn disconnect(&mut self);

    fn connected(&mut self) -> bool;

    fn instance_id(&self) -> InstanceID;

    fn do_read(&mut self) -> Result<String>;

    fn do_write(&mut self, message_out: &str) -> Result<()>;

    fn create_metadata(&mut self, metadata: &ObjectMeta) -> Result<ObjectMeta>;

    fn get_metadata(&mut self, id: ObjectID) -> Result<ObjectMeta>;

    fn get_metadata_batch(&mut self, ids: &[ObjectID]) -> Result<Vec<ObjectMeta>>;

    fn fetch_and_get_metadata(&mut self, id: ObjectID) -> Result<ObjectMeta> {
        let local_id = self.migrate(id)?;
        return self.get_metadata(local_id);
    }

    fn fetch_and_get_metadata_batch(&mut self, ids: &[ObjectID]) -> Result<Vec<ObjectMeta>> {
        let mut local_ids = Vec::new();
        for id in ids {
            local_ids.push(self.migrate(*id)?);
        }
        return self.get_metadata_batch(&local_ids);
    }

    fn drop_buffer(&mut self, id: ObjectID) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_drop_buffer_request(id)?;
        self.do_write(&message_out)?;
        return read_drop_buffer_reply(&self.do_read()?);
    }

    fn seal_buffer(&mut self, id: ObjectID) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_seal_request(id)?;
        self.do_write(&message_out)?;
        return read_seal_reply(&self.do_read()?);
    }

    fn get_data(&mut self, id: ObjectID, sync_remote: bool, wait: bool) -> Result<JSON> {
        let _ = self.ensure_connect()?;
        let message_out = write_get_data_request(id, sync_remote, wait)?;
        self.do_write(&message_out)?;
        return read_get_data_reply(&self.do_read()?);
    }

    fn get_data_batch(&mut self, ids: &[ObjectID]) -> Result<Vec<JSON>> {
        let _ = self.ensure_connect()?;
        let message_out = write_get_data_batch_request(&ids, false, false)?;
        self.do_write(&message_out)?;
        let reply = read_get_data_batch_reply(&self.do_read()?)?;
        let mut results = Vec::new();
        for id in ids {
            results.push(
                reply
                    .get(id)
                    .ok_or(VineyardError::object_not_exists(format!(
                        "failed to get the metadata for object '{}'",
                        id
                    )))?
                    .clone(),
            );
        }
        return Ok(results);
    }

    fn create_data(&mut self, data: &JSON) -> Result<(ObjectID, Signature, InstanceID)> {
        let _ = self.ensure_connect()?;
        let message_out = write_create_data_request(data)?;
        self.do_write(&message_out)?;
        let reply = read_create_data_reply(&self.do_read()?)?;
        return Ok((reply.id, reply.signature, reply.instance_id));
    }

    fn sync_metadata(&mut self) -> Result<()> {
        if let Err(e) = self.get_data(invalid_object_id(), true, false) {
            if !e.is_object_not_exists() {
                return Err(e);
            }
        }
        return Ok(());
    }

    fn delete(&mut self, id: ObjectID, force: bool, deep: bool) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_delete_data_request(id, force, deep, false)?;
        self.do_write(&message_out)?;
        return read_delete_data_reply(&self.do_read()?);
    }

    fn delete_batch(&mut self, ids: &[ObjectID], force: bool, deep: bool) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_delete_data_batch_request(ids, force, deep, false)?;
        self.do_write(&message_out)?;
        return read_delete_data_reply(&self.do_read()?);
    }

    /// @param pattern: The pattern of typename.
    fn list_data(&mut self, pattern: &str, regex: bool, limit: usize) -> Result<Vec<JSON>> {
        let _ = self.ensure_connect()?;
        let message_out = write_list_data_request(pattern, regex, limit)?;
        self.do_write(&message_out)?;
        return read_list_data_reply(&self.do_read()?);
    }

    /// @param pattern: The pattern of object name.
    fn list_name(
        &mut self,
        pattern: &str,
        regex: bool,
        limit: usize,
    ) -> Result<HashMap<String, ObjectID>> {
        let _ = self.ensure_connect()?;
        let message_out = write_list_name_request(pattern, regex, limit)?;
        self.do_write(&message_out)?;
        return read_list_name_reply(&self.do_read()?);
    }

    fn persist(&mut self, id: ObjectID) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_persist_request(id)?;
        self.do_write(&message_out)?;
        return read_persist_reply(&self.do_read()?);
    }

    fn if_persist(&mut self, id: ObjectID) -> Result<bool> {
        let _ = self.ensure_connect()?;
        let message_out = write_if_persist_request(id)?;
        self.do_write(&message_out)?;
        return read_if_persist_reply(&self.do_read()?);
    }

    fn exists(&mut self, id: ObjectID) -> Result<bool> {
        let _ = self.ensure_connect()?;
        let message_out = write_exists_request(id)?;
        self.do_write(&message_out)?;
        return read_exists_reply(&self.do_read()?);
    }

    fn put_name(&mut self, id: ObjectID, name: &str) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_put_name_request(id, name)?;
        self.do_write(&message_out)?;
        return read_put_name_reply(&self.do_read()?);
    }

    fn get_name(&mut self, name: &str, wait: bool) -> Result<ObjectID> {
        let _ = self.ensure_connect()?;
        let message_out = write_get_name_request(name, wait)?;
        self.do_write(&message_out)?;
        return read_get_name_reply(&self.do_read()?);
    }

    fn drop_name(&mut self, name: &str) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_drop_name_request(name)?;
        self.do_write(&message_out)?;
        return read_drop_name_reply(&self.do_read()?);
    }

    fn migrate(&mut self, id: ObjectID) -> Result<ObjectID> {
        let _ = self.ensure_connect()?;
        let message_out = write_migrate_object_request(id)?;
        self.do_write(&message_out)?;
        return read_migrate_object_reply(&self.do_read()?);
    }

    fn clear(&mut self) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_clear_request()?;
        self.do_write(&message_out)?;
        return read_clear_reply(&self.do_read()?);
    }

    fn label(&mut self, id: ObjectID, key: &str, value: &str) -> Result<()> {
        let _ = self.ensure_connect()?;
        let keys: Vec<String> = vec![key.into()];
        let values: Vec<String> = vec![value.into()];
        let message_out = write_label_request(id, &keys, &values)?;
        self.do_write(&message_out)?;
        return read_label_reply(&self.do_read()?);
    }

    fn evict(&mut self, id: ObjectID) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_evict_request(&[id])?;
        self.do_write(&message_out)?;
        return read_evict_reply(&self.do_read()?);
    }

    fn evict_batch(&mut self, ids: &[ObjectID]) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_evict_request(ids)?;
        self.do_write(&message_out)?;
        return read_evict_reply(&self.do_read()?);
    }

    fn load(&mut self, id: ObjectID, pin: bool) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_load_request(&[id], pin)?;
        self.do_write(&message_out)?;
        return read_load_reply(&self.do_read()?);
    }

    fn load_batch(&mut self, ids: &[ObjectID], pin: bool) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_load_request(ids, pin)?;
        self.do_write(&message_out)?;
        return read_load_reply(&self.do_read()?);
    }

    fn unpin(&mut self, id: ObjectID) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_unpin_request(&[id])?;
        self.do_write(&message_out)?;
        return read_unpin_reply(&self.do_read()?);
    }

    fn unpin_batch(&mut self, ids: &[ObjectID]) -> Result<()> {
        let _ = self.ensure_connect()?;
        let message_out = write_unpin_request(ids)?;
        self.do_write(&message_out)?;
        return read_unpin_reply(&self.do_read()?);
    }

    fn ensure_connect(&mut self) -> Result<ReentrantMutexGuard<'_, ()>>;
}
