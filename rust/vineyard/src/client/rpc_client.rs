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
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use super::client::Client;
use super::client::conn_input::{self, rpc_conn_input};
use super::InstanceID;
use super::ObjectID;
use super::ObjectMeta;
use crate::common::util::protocol::*;

#[derive(Debug)]
pub struct RPCClient {
    connected: bool,
    ipc_socket: String,
    rpc_endpoint: String,
    vineyard_conn: i64,
    instance_id: InstanceID,
    server_version: String,
}

// Question: the port is u16 from the material I saw while u32 in C++
pub fn connect_rpc_socket(host: &String, port: u16, socket_fd: i64) -> Result<TcpStream, Error> {
    let mut stream = match TcpStream::connect(&host[..]) { //"0.0.0.0:9600"
        Err(error) => panic!("The server is not running because: {}", error),
        Ok(stream) => stream,
    };
    assert_eq!(
        stream.peer_addr().unwrap(),
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 9600)));

    Ok(stream)
}


fn do_write(stream: &mut TcpStream, message_out: &String) -> Result<(), Error> {
    match stream.write_all(message_out.as_bytes()) {
        Err(error) => panic!("Couldn't send message because: {}.", error),
        Ok(_) => Ok(()),
    }
}

fn do_read(stream: &mut TcpStream, message_in: &mut String) -> Result<(), Error> {
    match stream.read_to_string(message_in) {
        Err(error) => panic!("Couldn't receive message because: {}.", error),
        Ok(_) => Ok(()),
    }
}


impl Client for RPCClient {
    fn connect(&mut self, conn_input: conn_input) -> Result<(), Error> {
        let (host, port) = match conn_input{
            rpc_conn_input(host, port) => (host, port),
            _ => panic!("Unsuitable type of connect input."), 
        };
        let rpc_host: String = String::from(host);
        let rpc_endpoint: String = format!("{}:{}", host, port.to_string());

        // Panic when they have connected while assigning different rpc_endpoint
        RETURN_ON_ASSERT(!self.connected || rpc_endpoint == self.rpc_endpoint);
        if self.connected {
            return Ok(());
        } else {
            self.rpc_endpoint = rpc_endpoint;
            let mut stream = connect_rpc_socket(&self.rpc_endpoint, port, self.vineyard_conn).unwrap();

            // Write a request  ( You need to start the vineyardd server on the same socket)
            let message_out: String = write_register_request();
            do_write(&mut stream, &message_out).unwrap();

            // Read the reply
            let mut message_in = String::new();
            do_read(&mut stream, &mut message_in).unwrap();
            println!("-----There should be content between here!-----");
            println!("{}", message_in);
            println!("-----There should be content between here!-----");

            // TODO： Read register reply

            // TODO： Compatable server

            return Ok(());
        };
        
    }

    fn disconnect(&self) {}

    fn connected(&self) -> bool {
        true
    }

    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> Result<ObjectMeta, Error> {
        Ok(ObjectMeta {
            client: None,
            meta: String::new(),
        })
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_connect() {
        let rpc_client = &mut RPCClient {
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vineyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
        };
        rpc_client.connect(rpc_conn_input("0.0.0.0",9600));
    }
}
