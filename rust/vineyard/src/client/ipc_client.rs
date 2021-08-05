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
use serde_json::{Value, json};
use serde_json::Result as JsonResult;

use super::client::Client;
use super::ObjectID;
use super::InstanceID;
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

// Question: Should these functions be added into Client trait?
// fn connect_ipc_socket(pathname: &String, socket_fd: i64) -> Result<u64> {
pub fn connect_ipc_socket(pathname: &String, vineyard_conn: i64) -> Result<UnixStream, Error> {
    let socket = Path::new(pathname);
    let mut stream = match UnixStream::connect(&socket) {
        Err(_) => panic!("The server is not running."),
        Ok(stream) => stream,
    };
    Ok(stream)
}

fn do_write(stream: &mut UnixStream, message_out: &String) -> Result<(), Error> {
    //let message_out = b"blabla";
    match stream.write_all(message_out.as_bytes()) { 
        Err(error) => panic!("Couldn't send message because of:{}.", error),
        Ok(_) => {Ok(())},
    }
}

// TODO
// fn send_message(vineyard_conn: i64, message_out: &String) -> Result<(), Error> {}

fn do_read(stream: &mut UnixStream, message_in: &mut String) -> Result<(), Error> {
    match stream.read_to_string(message_in) {
        Err(error) => panic!("Couldn't receive message because of：{}.", error),
        Ok(_) => {Ok(())},
    }
    
}

impl Client for IPCClient {

    // Connect to vineyardd using the given UNIX domain socket `ipc_socket`
    fn connect(&mut self, socket: &str) -> Result<(), Error> {
        
        // Panic when they have connected while assigning different sockets 
        let ipc_socket = String::from(socket);
        RETURN_ON_ASSERT(!self.connected || ipc_socket == self.ipc_socket);
        if self.connected {
            return Ok(());
        }else{ // If not connected yet
            // Connect to an ipc socket
            self.ipc_socket = ipc_socket;
            let mut stream = connect_ipc_socket(
                &self.ipc_socket, 
                self.vineyard_conn)
                .unwrap();

            // Write a request  ( You need to start the vineyardd server on the same socket)
            let message_out: String = write_register_request();
            //let message_out: String = String::from("blabla");
            do_write(&mut stream, &message_out).unwrap();
            //stream.write(b"blabla")?;

            // Read the reply
            let mut message_in = String::new();
            do_read(&mut stream, &mut message_in).unwrap();
            //stream.read_to_string(&mut message_in)?;
            println!("{}",message_in);
            println!("hey");

            // TODO： Read register reply

            // TODO： Compatable server

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
    ) -> Result<ObjectMeta, Error>{
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

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn ipc_connect(){
        let ipc_client = &mut IPCClient{
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vineyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
        };
        ipc_client.connect(SOCKET_PATH);
    }
}
