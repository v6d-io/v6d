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
use std::io;
use std::net::{Shutdown, TcpStream};
use std::sync::{Arc, Mutex};

use parking_lot::{ReentrantMutex, ReentrantMutexGuard};

use crate::common::util::protocol::*;
use crate::common::util::status::*;
use crate::common::util::uuid::*;

use super::client::*;
use super::ds::object_meta::ObjectMeta;
use super::io::*;

#[derive(Debug)]
pub struct RPCClient {
    connected: bool,
    pub ipc_socket: String,
    pub rpc_endpoint: String,
    pub instance_id: InstanceID,
    pub server_version: String,
    pub support_rpc_compression: bool,

    stream: TcpStream,
    lock: ReentrantMutex<()>,
}

impl Drop for RPCClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}

impl Client for RPCClient {
    fn disconnect(&mut self) {
        if !self.connected() {
            return;
        }
        self.connected = false;
        if let Ok(message_out) = write_exit_request() {
            if let Err(err) = self.do_write(&message_out) {
                error!("Failed to disconnect the client: {}", err);
            }
        }
        self.stream.shutdown(Shutdown::Both).unwrap_or_else(|e| {
            error!("Failed to shutdown IPCClient stream: {}", e);
        });
    }

    #[cfg(not(feature = "nightly"))]
    fn connected(&mut self) -> bool {
        return self.connected;
    }

    #[cfg(feature = "nightly")]
    fn connected(&mut self) -> bool {
        if self.stream.set_nonblocking(true).is_err() {
            return false;
        }
        match self.stream.peek(&mut [0]) {
            Ok(_) => {
                let _ = self.stream.set_nonblocking(false);
                return true;
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                let _ = self.stream.set_nonblocking(false);
                return true;
            }
            Err(_) => {
                self.connected = false;
                return false;
            }
        }
    }

    fn ensure_connect(&mut self) -> Result<ReentrantMutexGuard<'_, ()>> {
        if !self.connected() {
            return Err(VineyardError::io_error("client not connected"));
        }
        return Ok(self.lock.lock());
    }

    fn do_read(&mut self) -> Result<String> {
        return do_read(&mut self.stream);
    }

    fn do_write(&mut self, message_out: &str) -> Result<()> {
        return do_write(&mut self.stream, message_out);
    }

    fn instance_id(&self) -> InstanceID {
        return self.instance_id;
    }

    fn create_metadata(&mut self, metadata: &ObjectMeta) -> Result<ObjectMeta> {
        let mut meta = metadata.clone();
        meta.set_instance_id(self.instance_id());
        meta.set_transient(true);
        if !meta.has_key("nbytes") {
            meta.set_nbytes(0usize);
        }
        if meta.is_incomplete() {
            let _ = self.sync_metadata();
        }
        let (id, signature, instance_id) = self.create_data(meta.meta_data())?;
        meta.set_id(id);
        meta.set_signature(signature);
        meta.set_instance_id(instance_id);
        if meta.is_incomplete() {
            meta = self.get_metadata(id)?;
        }
        return Ok(meta);
    }

    fn get_metadata(&mut self, id: ObjectID) -> Result<ObjectMeta> {
        let data = self.get_data(id, false, false)?;
        let meta = ObjectMeta::new_from_metadata(data)?;
        return Ok(meta);
    }

    fn get_metadata_batch(&mut self, ids: &[ObjectID]) -> Result<Vec<ObjectMeta>> {
        let data_vec = self.get_data_batch(ids)?;
        let mut metadatas = Vec::new();
        for data in data_vec {
            let meta = ObjectMeta::new_from_metadata(data)?;
            metadatas.push(meta);
        }
        return Ok(metadatas);
    }
}

unsafe impl Send for RPCClient {}
unsafe impl Sync for RPCClient {}

impl RPCClient {
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Result<RPCClient> {
        let rpc_endpoint = std::env::var(VINEYARD_RPC_ENDPOINT_KEY)?;
        return RPCClient::connect_with_endpoint(rpc_endpoint.as_str());
    }

    pub fn connect(host: &str, port: u16) -> Result<RPCClient> {
        let mut stream = connect_rpc_endpoint_retry(host, port)?;
        let message_out = write_register_request(RegisterRequest {
            version: VERSION.into(),
            store_type: "Normal".into(),
            session_id: 0,
            username: String::new(),
            password: String::new(),
            support_rpc_compression: false,
        })?;
        do_write(&mut stream, &message_out)?;
        let reply = read_register_reply(&do_read(&mut stream)?)?;
        return Ok(RPCClient {
            connected: true,
            ipc_socket: reply.ipc_socket,
            rpc_endpoint: reply.rpc_endpoint,
            instance_id: reply.instance_id,
            server_version: reply.version,
            support_rpc_compression: reply.support_rpc_compression,
            stream: stream,
            lock: ReentrantMutex::new(()),
        });
    }

    pub fn connect_with_endpoint(endpoint: &str) -> Result<RPCClient> {
        let (host, port) = match endpoint.rfind(':') {
            Some(idx) => (&endpoint[..idx], endpoint[idx + 1..].parse::<u16>()?),
            None => (endpoint, DEFAULT_RPC_PORT),
        };
        return RPCClient::connect(host, port);
    }
}

pub struct RPCClientManager {}

impl RPCClientManager {
    pub fn get_default() -> Result<Arc<Mutex<RPCClient>>> {
        let default_rpc_endpoint = std::env::var(VINEYARD_RPC_ENDPOINT_KEY)?;
        return RPCClientManager::get_with_endpoint(default_rpc_endpoint);
    }

    pub fn get<S: Into<String>>(host: &str, port: u16) -> Result<Arc<Mutex<RPCClient>>> {
        let endpoint = format!("{}:{}", host, port);
        return RPCClientManager::get_with_endpoint(endpoint);
    }

    pub fn get_with_endpoint<S: Into<String>>(endpoint: S) -> Result<Arc<Mutex<RPCClient>>> {
        let mut clients: std::sync::MutexGuard<'_, _> = RPCClientManager::get_clients().lock()?;
        let endpoint: String = endpoint.into();
        if let Some(client) = clients.get(endpoint.as_str()) {
            if client.lock()?.connected() {
                return Ok(client.clone());
            }
        }
        let client = Arc::new(Mutex::new(RPCClient::connect_with_endpoint(&endpoint)?));
        clients.insert(endpoint, client.clone());
        return Ok(client);
    }

    pub fn close<S: Into<String>>(host: &str, port: u16) -> Result<()> {
        let endpoint = format!("{}:{}", host, port);
        return RPCClientManager::close_with_endpoint(endpoint);
    }

    pub fn close_with_endpoint<S: Into<String>>(endpoint: S) -> Result<()> {
        let mut clients = RPCClientManager::get_clients().lock()?;
        let endpoint = endpoint.into();
        if let Some(client) = clients.get(&endpoint) {
            if Arc::strong_count(client) == 1 {
                clients.remove(&endpoint);
            }
            return Ok(());
        } else {
            return Err(VineyardError::invalid(format!(
                "Failed to close the client due to the unknown endpoint: {}",
                endpoint
            )));
        }
    }

    fn get_clients() -> &'static Arc<Mutex<HashMap<String, Arc<Mutex<RPCClient>>>>> {
        lazy_static! {
            static ref CONNECTED_CLIENTS: Arc<Mutex<HashMap<String, Arc<Mutex<RPCClient>>>>> =
                Arc::new(Mutex::new(HashMap::new()));
        }
        return &CONNECTED_CLIENTS;
    }
}
