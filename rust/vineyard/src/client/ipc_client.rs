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
use std::io::{self, ErrorKind, Error};
use std::os::unix::net::UnixStream;
use std::io::prelude::*;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Result, Value, json};

use super::client::Client;
use super::ObjectID;
use super::InstanceID;
use super::ObjectMeta;
use crate::common::util::protocol::*;

pub static SOCKET_PATH: &'static str = "/tmp/sock";


#[derive(Debug)]
pub struct IPCClient {
    connected: bool,
    ipc_socket: String,
    rpc_endpoint: String,
    vinyard_conn: i64,
    instance_id: InstanceID,
    server_version: String,
}

// fn connect_ipc_socket(pathname: &String, socket_fd: i64) -> Result<u64> {
pub fn connect_ipc_socket(pathname: &String) -> Result<UnixStream> {
    let socket = Path::new(pathname);
    let mut stream = match UnixStream::connect(&socket) {
        Err(_) => panic!("Server is not running."),
        Ok(stream) => stream,
    };
    Ok(stream)
}

fn do_write(message_out: &String) -> Result<u64> {
    let args: Vec<String> = env::args().map(|x| x.to_string()).collect();
    panic!();
}

fn do_read(root: &json) -> Result<u64> {
    panic!();
}

impl Client for IPCClient {

    // Connect to vineyardd using the given UNIX domain socket `ipc_socket`
    fn connect(&mut self, socket: &str) -> Result<()> {
        // TODO: Create a socket
        // Write a request  ( You need to start the vineyardd server on the same socket)
        // Read the reply

        // Error when they have connected while assigning different sockets 
        RETURN_ON_ASSERT(!self.connected || socket == self.ipc_socket);
        if self.connected {
            return Ok(());
        }else{ //not connected yet
            self.ipc_socket = socket.to_string();
            //connect_ipc_socket(socket, self.vinyard_conn).unwrap();
            let stream = connect_ipc_socket(&socket.to_string()).unwrap();

            // let message_out: String = write_register_request();
            // do_write(&message_out).unwrap();

            // let message_in = json!();
            // do_read(&message_in).unwrap();

            // let ipc_socket_value: String;
            // let rpc_endpoint_value: String;
            // read_register_reply().unwrap();
            // self.rpc_endpoint = rpc_endpoint_value;
            // self.connect = true;

            // compatable server

            return Ok(());
            
        }

    }

    fn disconnect(&self) {}

    fn connected(&self) -> bool {
        true
    }

    fn get_meta_data(&self, 
        object_id: ObjectID, 
        sync_remote: bool
    ) -> Result<ObjectMeta>{
        // Ok(ObjectMeta {
        //     client: None,
        //     meta: String::new(),
        // })
        panic!();
    }
}
// TODO: Test the connect

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn it_works() {
    //     assert_eq!(2 + 2, 4);
    // }

    #[test]
    fn ipc_connect(){
        let ipc_client = &mut IPCClient{
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vinyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
        };
        ipc_client.connect(SOCKET_PATH);
    }
}
