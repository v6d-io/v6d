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
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream, TcpListener};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use super::client::conn_input::{self, rpc_conn_input};
use super::client::Client;
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
    let mut stream = match TcpStream::connect(&host[..]) {
        //"0.0.0.0:9600"
        Err(error) => panic!("The server is not running because: {}", error),
        Ok(stream) => stream,
    };
    assert_eq!(
        stream.peer_addr().unwrap(),
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 9600))
    );

    Ok(stream)
}

fn do_write(stream: &mut TcpStream, message_out: &String) -> Result<(), Error> {
    println!("{:?}", &message_out.as_bytes());
    println!("{:?}", &message_out.as_bytes()[0..5]);
    println!("{:?}", &vec![1,2,3]);
    match stream.write(&message_out.as_bytes()) {
        Err(error) => panic!("Couldn't send message because: {}.", error),
        Ok(_) => Ok(()),
    }
}

fn do_read(stream: &mut TcpStream, message_in: &mut String) -> Result<(), Error> {
    //let mut buf = vec![0; 16];
    match stream.read_to_string(message_in) {
        Err(error) => panic!("Couldn't receive message because: {}.", error),
        Ok(_) => {Ok(())}
    }
    
    
}

impl Client for RPCClient {
    fn connect(&mut self, conn_input: conn_input) -> Result<(), Error> {
        let (host, port) = match conn_input {
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
            let mut stream =
                connect_rpc_socket(&self.rpc_endpoint, port, self.vineyard_conn).unwrap();

            // Write a request
            let message_out: String = write_register_request();
            do_write(&mut stream, &message_out).unwrap();

            // Read the reply
            let mut message_in = String::new();
            do_read(&mut stream, &mut message_in).unwrap();
            println!("-----There should be content between here!-----");
            println!("{}", message_in);
            println!("-----There should be content between here!-----");

            // // Read register reply
            // let message_in: Value = serde_json::from_str(&message_in).expect("JSON was not well-formatted");
            // let reg_rep: RegisterReply = read_register_reply(message_in).unwrap();

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
    //#[ignore]
    fn rpc_connect() {
        let rpc_client = &mut RPCClient {
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vineyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
        };
        rpc_client.connect(rpc_conn_input("0.0.0.0", 9600));
    }

    #[test]
    #[ignore]
    fn small_test() -> std::io::Result<()> {

        let mut stream = TcpStream::connect("0.0.0.0:9600").unwrap(); //0.0.0.0:9600
        let _bytes_written = stream.write_all(b"Hello").unwrap();
        stream.flush()?;

        // use std::time::Duration;
        // stream.set_read_timeout(Some(Duration::new(5, 0))).unwrap();
        
        let mut buf = String::new(); //vec![0;16] 
        stream.read_to_string(&mut buf).unwrap();
        println!("{}", buf);

        Ok(())
    }

    #[test]
    #[ignore]
    fn small_http_test() -> std::io::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
        for stream in listener.incoming() {
            let stream = stream.unwrap();
    
            handle_connection(stream);
        }

        Ok(())
    }

    fn handle_connection(mut stream: TcpStream) {
        let mut buffer = [0; 1024];
    
        stream.read(&mut buffer).unwrap();
        //println!("Request: {}", String::from_utf8_lossy(&buffer[..]));

        let response = "HTTP/1.1 200 OK\r\n\r\n";
        stream.write(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    }

}
