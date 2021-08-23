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
use std::io::prelude::*;
use std::io::{self, Error, ErrorKind};
use std::mem;

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use super::client::Client;
use super::client::ConnInputKind::{self, RPCConnInput};
use super::client::StreamKind::{self, RPCStream};
use super::rust_io::*;
use super::{InstanceID, ObjectID, ObjectMeta};
use crate::common::util::protocol::*;

#[derive(Debug)]
pub struct RPCClient {
    connected: bool,
    ipc_socket: String,
    rpc_endpoint: String,
    vineyard_conn: i64,
    instance_id: InstanceID,
    server_version: String,
    remote_instance_id: InstanceID,
    stream: Option<StreamKind>,
}

impl Default for RPCClient {
    fn default() -> Self {
        RPCClient {
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vineyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
            remote_instance_id: 0,
            stream: None as Option<StreamKind>,
        }
    }
}

impl Client for RPCClient {
    fn connect(&mut self, conn_input: ConnInputKind) -> io::Result<()> {
        let (host, port) = match conn_input {
            RPCConnInput(host, port) => (host, port),
            _ => panic!("Unsuitable type of connect input."),
        };
        let rpc_host = String::from(host);
        let rpc_endpoint = format!("{}:{}", host, port.to_string());


        // Panic when they have connected while assigning different rpc_endpoint
        RETURN_ON_ASSERT(!self.connected || rpc_endpoint == self.rpc_endpoint);
        if self.connected {
            return Ok(());
        } else {
            self.rpc_endpoint = rpc_endpoint;
            let stream = connect_rpc_socket(&self.rpc_endpoint, port, self.vineyard_conn)?;
            let mut rpc_stream = RPCStream(stream);

            let message_out: String = write_register_request();
            if let Err(e) = do_write(&mut rpc_stream, &message_out) {
                self.connected = false;
                return Err(e);
            }

            let mut message_in = String::new();
            do_read(&mut rpc_stream, &mut message_in)?;

            let message_in: Value =
                serde_json::from_str(&message_in).expect("JSON was not well-formatted");
            let register_reply: RegisterReply = read_register_reply(message_in)?;
            //println!("Register reply:\n{:?}\n ", register_reply);

            self.remote_instance_id = register_reply.instance_id;
            self.server_version = register_reply.version;
            self.ipc_socket = register_reply.ipc_socket;
            self.stream = Some(rpc_stream);
            self.connected = true;

            // TODO： Compatable server

            Ok(())
        }

    }

    fn disconnect(&self) {}

    fn connected(&mut self) -> bool {
        self.connected
    }

    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> io::Result<ObjectMeta> {
        Ok(ObjectMeta {
            client: None,
            meta: String::new(),
        })
    }

    fn get_stream(&mut self) -> io::Result<&mut StreamKind> {
        match &mut self.stream {
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
    fn test_rpc_connect() {
        let print = true;
        let rpc_client = &mut RPCClient::default();
        if print {
            println!("Rpc client:\n {:?}\n", rpc_client)
        }
        rpc_client.connect(RPCConnInput("0.0.0.0", 9600));
        if print {
            println!("Rpc client after connect:\n {:?}\n ", rpc_client)
        }
    }
}
