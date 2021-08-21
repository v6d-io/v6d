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

use super::client::conn_input::{self, ipc_conn_input};
use super::client::Client;
use super::InstanceID;
use super::ObjectID;
use super::ObjectMeta;
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
}

// socket_fd is used to assign vineyard_conn
pub fn connect_ipc_socket(pathname: &String, socket_fd: i64) -> Result<UnixStream, Error> {
    let socket = Path::new(pathname);
    let mut stream = match UnixStream::connect(&socket) {
        Err(e) => panic!("The server is not running because: {}.", e),
        Ok(stream) => stream,
    };
    Ok(stream)
}

fn do_write(stream: &mut UnixStream, message_out: &String) -> io::Result<()> {
    send_message(stream, message_out.as_str())?;
    Ok(())
}

fn do_read(stream: &mut UnixStream, message_in: &mut String) -> io::Result<()> {
    *message_in = recv_message(stream)?;
    Ok(())
}

fn send_bytes(stream: &mut UnixStream, data: &[u8], length: usize) -> io::Result<()> {
    let mut remaining = length;
    let mut offset = 0;
    while remaining > 0 {
        let n = stream.write(&data[offset..])?;
        remaining -= n;
        offset += n;
    }
    Ok(())
}

fn send_message(stream: &mut UnixStream, message: &str) -> io::Result<()> {
    let len = message.len();
    let bytes = len.to_le_bytes();
    send_bytes(stream, &bytes, mem::size_of::<usize>())?;
    send_bytes(stream, message.as_bytes(), len)?;
    Ok(())
}

fn recv_bytes(stream: &mut UnixStream, data: &mut [u8], length: usize) -> io::Result<()> {
    let mut remaining = length;
    let mut offset = 0;
    while remaining > 0 {
        let n = stream.read(&mut data[offset..])?;
        remaining -= n;
        offset += n;
    }
    Ok(())
}
fn recv_message(stream: &mut UnixStream) -> io::Result<String> {
    let mut size_buf = [0u8; mem::size_of::<usize>()];
    recv_bytes(stream, &mut size_buf, mem::size_of::<usize>())?;
    let size = usize::from_le_bytes(size_buf);
    let mut message_buf = vec![0u8; size];
    recv_bytes(stream, message_buf.as_mut_slice(), size)?;
    Ok(String::from_utf8(message_buf).unwrap())
}


impl Client for IPCClient {
    fn connect(&mut self, conn_input: conn_input) -> Result<(), Error> {
        let socket = match conn_input {
            ipc_conn_input(socket) => socket,
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

            let message_out: String = write_register_request();
            if let Err(e) = do_write(&mut stream, &message_out){
                self.connected = false; 
                return Err(e);
            }

            let mut message_in = String::new();
            do_read(&mut stream, &mut message_in).unwrap();

            let message_in: Value = serde_json::from_str(&message_in).expect("JSON was not well-formatted");
            let register_reply: RegisterReply = read_register_reply(message_in).unwrap();
            //println!("Register reply:\n{:?}\n", register_reply);

            self.instance_id = register_reply.instance_id;
            self.server_version = register_reply.version;
            self.rpc_endpoint = register_reply.rpc_endpoint;
            self.connected = true;

            // TODO： Compatable server

            Ok(())
        }
    }

    fn disconnect(&self) {}

    fn connected(&mut self) -> bool {
        // if self.connected && recv(vineyard_conn_, NULL, 1, MSG_PEEK | MSG_DONTWAIT) != -1
        // Question: recv function in sys/socket.h?
        self.connected
    }

    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> Result<ObjectMeta, Error> {
        panic!();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    //#[ignore]
    fn ipc_connect() {
        let print = true;
        let ipc_client = &mut IPCClient {
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vineyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
        };
        if print {println!("Ipc client:\n {:?}\n", ipc_client)}
        ipc_client.connect(ipc_conn_input(SOCKET_PATH));
        if print {println!("Ipc client after connect:\n {:?}\n", ipc_client)}
    }
}
