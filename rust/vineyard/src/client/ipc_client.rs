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
use std::env;
use std::mem;
use std::io::prelude::*;
use std::io::{self, Error, ErrorKind};
use std::os::unix::net::UnixStream;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use super::rust_io::*;
use super::client::Client;
use super::client::ConnInputKind::{self, IPCConnInput};
use super::client::StreamKind::{self, IPCStream};
use super::{InstanceID, ObjectID, ObjectMeta};
use crate::common::util::protocol::*;

pub static SOCKET_PATH: &'static str = "/tmp/vineyard.sock";

#[derive(Debug)]
pub struct IPCClient {
    connected: bool,
    ipc_socket: String,
    rpc_endpoint: String,
    vineyard_conn: i64,
    instance_id: InstanceID,
    server_version: String,
    stream: Option<StreamKind>,
}

impl IPCClient {
    fn new_default() -> IPCClient {
        IPCClient {
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vineyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
            stream : None as Option<StreamKind>,
        }
    }
}

impl Client for IPCClient {
    fn connect(&mut self, conn_input: ConnInputKind) -> io::Result<()> {
        let socket = match conn_input {
            IPCConnInput(socket) => socket,
            _ => panic!("Unsuitable type of connect input."),
        };
        let ipc_socket: String = String::from(socket);
        // Panic when they have connected while assigning different ipc_socket
        RETURN_ON_ASSERT(!self.connected || ipc_socket == self.ipc_socket);
        if self.connected {
            return Ok(());
        } else {
            self.ipc_socket = ipc_socket;
            let mut stream = connect_ipc_socket(&self.ipc_socket, self.vineyard_conn).unwrap();
            let mut ipc_stream = IPCStream(stream);

            let message_out: String = write_register_request();
            if let Err(e) = do_write(&mut ipc_stream, &message_out){
                self.connected = false; 
                return Err(e);
            }

            let mut message_in = String::new();
            do_read(&mut ipc_stream, &mut message_in)?;

            let message_in: Value = serde_json::from_str(&message_in).expect("JSON was not well-formatted");
            let register_reply: RegisterReply = read_register_reply(message_in)?;
            //println!("Register reply:\n{:?}\n", register_reply);

            self.instance_id = register_reply.instance_id;
            self.server_version = register_reply.version;
            self.rpc_endpoint = register_reply.rpc_endpoint;
            self.stream = Some(ipc_stream);
            self.connected = true;

            // TODO： Compatable server

            Ok(())
        }
    }

    fn disconnect(&self) {}

    fn connected(&mut self) -> bool {
        self.connected
    }

    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> io::Result<ObjectMeta>{
        panic!();
    }

    fn get_stream(&mut self) -> io::Result<&mut StreamKind> {
        match &mut self.stream{
            Some(stream) => return Ok(&mut *stream),
            None => panic!(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    //#[ignore]
    fn test_ipc_connect() {
        let print = true;
        let ipc_client = &mut IPCClient::new_default();
        if print {println!("Ipc client:\n {:?}\n", ipc_client)}
        ipc_client.connect(IPCConnInput(SOCKET_PATH));
        if print {println!("Ipc client after connect:\n {:?}\n", ipc_client)}
    }

    #[test]
    //#[ignore]
    fn test_ipc_put_and_get_name() {
        let print = false;
        let ipc_client = &mut IPCClient::new_default();
        if print {println!("Ipc client:\n {:?}\n", ipc_client)}
        ipc_client.connect(IPCConnInput(SOCKET_PATH)).unwrap();
        let id1 = 1 as ObjectID;
        let name = String::from("put&get_test_name");
        ipc_client.put_name(id1, &name);
        let id2 = ipc_client.get_name(&name, false).unwrap();
        assert_eq!(id1, id2);
        if print {println!("Ipc client after connect:\n {:?}\n", ipc_client)}
    }

    #[test]
    #[should_panic]
    //#[ignore]
    fn test_ipc_drop_name() {
        let print = false;
        let ipc_client = &mut IPCClient::new_default();
        if print {println!("Ipc client:\n {:?}\n", ipc_client)}
        ipc_client.connect(IPCConnInput(SOCKET_PATH)).unwrap();
        let id = 1 as ObjectID;
        let name = String::from("drop_test_name");
        ipc_client.put_name(id, &name);
        ipc_client.drop_name(&name);
        let id = ipc_client.get_name(&name, false).unwrap();
        if print {println!("Ipc client after connect:\n {:?}\n", ipc_client)}
    }

}
