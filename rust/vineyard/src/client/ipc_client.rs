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
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::Mutex;

use arrow_buffer::Buffer;
use parking_lot::ReentrantMutex;
use parking_lot::ReentrantMutexGuard;

use crate::common::util::arrow::*;
use crate::common::util::protocol::*;
use crate::common::util::status::*;
use crate::common::util::typename::*;
use crate::common::util::uuid::*;

use super::client::*;
use super::ds::blob::{Blob, BlobWriter};
use super::ds::object::*;
use super::ds::object_meta::ObjectMeta;
use super::io::*;

mod memory {

    use std::collections::{hash_map, HashMap, HashSet};
    use std::fs::File;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::net::UnixStream;

    use memmap2::{Mmap, MmapMut, MmapOptions};

    use crate::common::memory::fling::recv_fd;
    use crate::common::util::status::*;

    #[derive(Debug)]
    pub struct MmapEntry {
        fd: File,
        ro_pointer: *const u8,
        rw_pointer: *mut u8,
        length: usize,

        mmap_readonly: Option<Mmap>,
        mmap_readwrite: Option<MmapMut>,
    }

    impl MmapEntry {
        pub fn new(fd: i32, map_size: usize, realign: bool) -> Self {
            let size = if realign {
                map_size - std::mem::size_of::<usize>()
            } else {
                map_size
            };
            return MmapEntry {
                fd: unsafe { File::from_raw_fd(fd) },
                ro_pointer: std::ptr::null(),
                rw_pointer: std::ptr::null_mut(),
                length: size,
                mmap_readonly: None,
                mmap_readwrite: None,
            };
        }

        #[allow(dead_code)]
        pub fn fd(&self) -> i32 {
            return self.fd.as_raw_fd();
        }

        pub fn map(&mut self) -> Result<*const u8> {
            if self.ro_pointer.is_null() {
                let mmap = unsafe { MmapOptions::new().len(self.length).offset(0).map(&self.fd) }?;
                self.ro_pointer = mmap.as_ptr();
                self.mmap_readonly = Some(mmap);
            }
            return Ok(self.ro_pointer);
        }

        pub fn map_mut(&mut self) -> Result<*mut u8> {
            if self.rw_pointer.is_null() {
                let mut mmap = unsafe { MmapOptions::new().len(self.length).map_mut(&self.fd) }?;
                self.rw_pointer = mmap.as_mut_ptr();
                self.mmap_readwrite = Some(mmap);
            }
            return Ok(self.rw_pointer);
        }
    }

    #[derive(Debug)]
    pub struct MmapManager {
        entries: HashMap<i32, MmapEntry>,
    }

    impl MmapManager {
        pub fn new() -> Self {
            return MmapManager {
                entries: HashMap::new(),
            };
        }

        pub fn mmap(
            &mut self,
            stream: &UnixStream,
            fd: i32,
            map_size: usize,
            realign: bool,
        ) -> Result<*const u8> {
            if let hash_map::Entry::Vacant(entry) = self.entries.entry(fd) {
                entry.insert(MmapEntry::new(recv_fd(stream)?, map_size, realign));
            }
            match self.entries.get_mut(&fd) {
                Some(entry) => {
                    return entry.map();
                }
                None => {
                    return Err(VineyardError::invalid(format!(
                        "Failed to find mmap entry for fd even after insert: {}",
                        fd
                    )));
                }
            }
        }

        pub fn mmap_mut(
            &mut self,
            stream: &UnixStream,
            fd: i32,
            map_size: usize,
            realign: bool,
        ) -> Result<*mut u8> {
            if let hash_map::Entry::Vacant(entry) = self.entries.entry(fd) {
                entry.insert(MmapEntry::new(recv_fd(stream)?, map_size, realign));
            }
            match self.entries.get_mut(&fd) {
                Some(entry) => {
                    return entry.map_mut();
                }
                None => {
                    return Err(VineyardError::invalid(format!(
                        "Failed to find mmap entry for fd even after insert: {}",
                        fd
                    )));
                }
            }
        }

        #[allow(dead_code)]
        pub fn exists(&self, fd: i32) -> i32 {
            if self.entries.contains_key(&fd) {
                return -1;
            } else {
                return fd;
            }
        }

        #[allow(dead_code)]
        pub fn deduplicate(&self, fd: i32, fds: &mut Vec<i32>, dedup: &mut HashSet<i32>) {
            if !dedup.contains(&fd) && !self.entries.contains_key(&fd) {
                fds.push(fd);
                dedup.insert(fd);
            }
        }
    }
}

#[derive(Debug)]
pub struct IPCClient {
    connected: bool,
    pub ipc_socket: String,
    pub rpc_endpoint: String,
    pub instance_id: InstanceID,
    pub server_version: String,
    pub support_rpc_compression: bool,

    stream: UnixStream,
    lock: ReentrantMutex<()>,
    mmap: memory::MmapManager,
}

impl Drop for IPCClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}

unsafe impl Send for IPCClient {}
unsafe impl Sync for IPCClient {}

impl Client for IPCClient {
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
        meta.set_client(self);
        return Ok(meta);
    }

    fn get_metadata(&mut self, id: ObjectID) -> Result<ObjectMeta> {
        let data = self.get_data(id, false, false)?;
        let mut meta = ObjectMeta::new(self, data)?;

        let buffer_id_vec: Vec<ObjectID> =
            meta.get_buffers().buffer_ids().iter().cloned().collect();
        let buffers = self.get_buffers(&buffer_id_vec, false)?;
        for (buffer_id, buffer) in buffers {
            meta.set_buffer(buffer_id, buffer)?;
        }
        return Ok(meta);
    }

    fn get_metadata_batch(&mut self, ids: &[ObjectID]) -> Result<Vec<ObjectMeta>> {
        let data_vec = self.get_data_batch(ids)?;
        let mut metadatas = Vec::new();
        let mut buffer_id_vec: Vec<ObjectID> = Vec::new();
        for data in data_vec {
            let meta = ObjectMeta::new(self, data)?;
            buffer_id_vec.extend(meta.get_buffers().buffer_ids());
            metadatas.push(meta);
        }

        let buffers = self.get_buffers(&buffer_id_vec, false)?;
        for meta in metadatas.iter_mut() {
            for buffer_id in meta.get_buffers().buffer_ids().clone() {
                if let Some(buffer) = buffers.get(&buffer_id) {
                    meta.set_buffer(buffer_id, buffer.clone())?;
                }
            }
        }
        return Ok(metadatas);
    }
}

impl IPCClient {
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Result<IPCClient> {
        let default_ipc_socket = std::env::var(VINEYARD_IPC_SOCKET_KEY)?;
        return IPCClient::connect(&default_ipc_socket);
    }

    pub fn connect(socket: &str) -> Result<IPCClient> {
        let mut stream = connect_ipc_socket_retry(&socket)?;
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
        return Ok(IPCClient {
            connected: true,
            ipc_socket: reply.ipc_socket,
            rpc_endpoint: reply.rpc_endpoint,
            instance_id: reply.instance_id,
            server_version: reply.version,
            support_rpc_compression: reply.support_rpc_compression,
            stream: stream,
            lock: ReentrantMutex::new(()),
            mmap: memory::MmapManager::new(),
        });
    }

    pub fn create_blob(&mut self, size: usize) -> Result<BlobWriter> {
        let (id, buffer) = self.create_buffer(size)?;
        return Ok(BlobWriter::new(id, buffer));
    }

    pub fn get_blob(&mut self, id: ObjectID) -> Result<Blob> {
        let buffer = self.get_buffer(id, false)?;
        let size = match &buffer {
            Some(buffer) => buffer.len(),
            None => 0,
        };
        let mut meta = ObjectMeta::new_from_typename(typename::<Blob>());
        meta.set_id(id);
        meta.set_instance_id(self.instance_id());
        meta.set_or_add_buffer(id, buffer.clone())?;
        return Ok(Blob::new(meta, size, buffer));
    }

    fn create_buffer(&mut self, size: usize) -> Result<(ObjectID, Option<Buffer>)> {
        if size == 0 {
            return Ok((empty_blob_id(), Some(arrow_buffer_null())));
        }
        let _ = self.ensure_connect()?;
        let message_out = write_create_buffer_request(size)?;
        self.do_write(&message_out)?;
        let reply = read_create_buffer_reply(&self.do_read()?)?;
        if reply.payload.data_size == 0 {
            return Ok((reply.id, Some(arrow_buffer_null())));
        }
        let pointer = self.mmap.mmap_mut(
            &self.stream,
            reply.payload.store_fd,
            reply.payload.map_size,
            true,
        )?;
        let buffer =
            arrow_buffer_with_offset(pointer, reply.payload.data_offset, reply.payload.data_size);
        return Ok((reply.id, Some(buffer)));
    }

    fn get_buffer(&mut self, id: ObjectID, unsafe_: bool) -> Result<Option<Buffer>> {
        let buffers = self.get_buffers(&[id], unsafe_)?;
        return buffers
            .get(&id)
            .cloned()
            .ok_or(VineyardError::object_not_exists(format!(
                "buffer {} doesn't exist",
                id
            )));
    }

    fn get_buffers(
        &mut self,
        ids: &[ObjectID],
        unsafe_: bool,
    ) -> Result<HashMap<ObjectID, Option<Buffer>>> {
        let _ = self.ensure_connect()?;
        let message_out = write_get_buffers_request(&ids, unsafe_)?;
        self.do_write(&message_out)?;
        let reply = read_get_buffers_reply(&self.do_read()?)?;

        let mut buffers = HashMap::new();
        for payload in reply.payloads {
            if payload.data_size == 0 {
                buffers.insert(payload.object_id, Some(arrow_buffer_null()));
                continue;
            }
            let pointer = self
                .mmap
                .mmap(&self.stream, payload.store_fd, payload.map_size, true)?;
            let buffer = arrow_buffer_with_offset(pointer, payload.data_offset, payload.data_size);
            buffers.insert(payload.object_id, Some(buffer));
        }
        return Ok(buffers);
    }

    pub fn get<T: Object + Create>(&mut self, id: ObjectID) -> Result<Box<T>> {
        let meta = self.get_metadata(id)?;
        let mut object = T::create();
        object.construct(meta)?;
        return downcast_object(object);
    }

    pub fn fetch_and_get<T: Object + Create>(&mut self, id: ObjectID) -> Result<Box<dyn Object>> {
        let meta = self.fetch_and_get_metadata(id)?;
        let mut object = T::create();
        object.construct(meta)?;
        return Ok(object);
    }
}

pub struct IPCClientManager {}

impl IPCClientManager {
    pub fn get_default() -> Result<Arc<Mutex<IPCClient>>> {
        let default_ipc_socket = std::env::var(VINEYARD_IPC_SOCKET_KEY)?;
        return IPCClientManager::get(default_ipc_socket);
    }

    pub fn get<S: Into<String>>(socket: S) -> Result<Arc<Mutex<IPCClient>>> {
        let mut clients = IPCClientManager::get_clients().lock()?;
        let socket = socket.into();
        if let Some(client) = clients.get(&socket) {
            if client.lock()?.connected() {
                return Ok(client.clone());
            }
        }
        let client = Arc::new(Mutex::new(IPCClient::connect(&socket)?));
        clients.insert(socket, client.clone());
        return Ok(client);
    }

    pub fn close<S: Into<String>>(socket: S) -> Result<()> {
        let mut clients = IPCClientManager::get_clients().lock()?;
        let socket = socket.into();
        if let Some(client) = clients.get(&socket) {
            if Arc::strong_count(client) == 1 {
                clients.remove(&socket);
            }
            return Ok(());
        } else {
            return Err(VineyardError::invalid(format!(
                "Failed to close the client due to the unknown socket: {}",
                socket
            )));
        }
    }

    fn get_clients() -> &'static Arc<Mutex<HashMap<String, Arc<Mutex<IPCClient>>>>> {
        lazy_static! {
            static ref CONNECTED_CLIENTS: Arc<Mutex<HashMap<String, Arc<Mutex<IPCClient>>>>> =
                Arc::new(Mutex::new(HashMap::new()));
        }
        return &CONNECTED_CLIENTS;
    }
}

#[macro_export]
macro_rules! get {
    ($client: ident, $object_ty: ty, $object_id: expr) => {
        $client.get::<$object_ty>($object_id)
    };
}

#[macro_export]
macro_rules! put {
    ($client: expr, $builder_ty: ty, $($arg: expr),* $(,)?) => {
        match <$builder_ty>::new($client, $($arg),*) {
            Ok(builder) => builder.seal($client),
            Err(e) => Err(e),
        }
    };
}
