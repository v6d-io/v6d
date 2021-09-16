use std::cell::{RefCell, RefMut};
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
use std::io::prelude::*;
use std::net::TcpStream;
use std::os::unix::net::UnixStream;

use serde_json::Value;

use super::ipc_client::IPCClient;
use super::rpc_client::RPCClient;
use super::rust_io::*;
use super::ObjectMeta;

use super::protocol::*;
use super::uuid::{InstanceID, ObjectID};

#[derive(Debug)]
pub enum ConnInputKind<'a, 'b> {
    IPCConnInput(&'a str),      // socket
    RPCConnInput(&'b str, u16), // host, port
}

#[derive(Debug)]
pub enum StreamKind {
    IPCStream(UnixStream),
    RPCStream(TcpStream),
}

pub trait Client {
    fn connect(&mut self, conn_input: ConnInputKind) -> io::Result<()>;

    // Disconnect this client.
    fn disconnect(&self);

    // Question: recv function in sys/socket.h?
    // if self.connected && recv(vineyard_conn_, NULL, 1, MSG_PEEK | MSG_DONTWAIT) != -1
    fn connected(&self) -> bool;

    // Obtain multiple metadatas from vineyard server.
    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> io::Result<ObjectMeta>;

    fn get_stream(&self) -> io::Result<RefMut<'_, StreamKind>>;

    fn put_name(&self, id: ObjectID, name: &String) -> io::Result<()> {
        ENSURE_CONNECTED(self.connected());
        let mut stream = self.get_stream()?;
        let message_out = write_put_name_request(id, name);
        do_write(&mut stream, &message_out)?;
        let mut message_in = String::new();
        do_read(&mut stream, &mut message_in)?;
        let message_in: Value = serde_json::from_str(&message_in)?;
        read_put_name_reply(message_in)?;
        Ok(())
    }

    fn get_name(&self, name: &String, wait: bool) -> io::Result<ObjectID> {
        ENSURE_CONNECTED(self.connected());
        let mut stream = self.get_stream()?;
        let message_out = write_get_name_request(name, wait);
        do_write(&mut stream, &message_out)?;
        let mut message_in = String::new();
        do_read(&mut stream, &mut message_in)?;
        let message_in: Value = serde_json::from_str(&message_in)?;
        let id = read_get_name_reply(message_in)?;
        Ok(id)
    }

    fn drop_name(&self, name: &String) -> io::Result<()> {
        ENSURE_CONNECTED(self.connected());
        let mut stream = self.get_stream()?;
        let message_out = write_drop_name_request(name);
        do_write(&mut stream, &message_out)?;
        let mut message_in = String::new();
        do_read(&mut stream, &mut message_in)?;
        let message_in: Value = serde_json::from_str(&message_in)?;
        read_drop_name_reply(message_in)?;
        Ok(())
    }

    fn instance_id(&self) -> InstanceID;

    fn persist(&self, id: ObjectID) -> io::Result<()> {
        ENSURE_CONNECTED(self.connected());
        let mut stream = self.get_stream()?;
        let message_out = write_persist_request(id);
        do_write(&mut stream, &message_out)?;
        let mut message_in = String::new();
        do_read(&mut stream, &mut message_in)?;
        let message_in: Value = serde_json::from_str(&message_in)?;
        read_persist_reply(message_in)?;
        Ok(())
    }

    fn if_persist(&self, id: ObjectID) -> io::Result<bool> {
        ENSURE_CONNECTED(self.connected());
        let mut stream = self.get_stream()?;
        let message_out = write_if_persist_request(id);
        do_write(&mut stream, &message_out)?;
        let mut message_in = String::new();
        do_read(&mut stream, &mut message_in)?;
        let message_in: Value = serde_json::from_str(&message_in)?;
        let persist = read_if_persist_reply(message_in)?;
        Ok(persist)
    }
