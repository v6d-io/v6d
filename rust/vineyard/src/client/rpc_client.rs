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
    remote_instance_id: InstanceID,
}

// Question: the port is u16 from the material I saw while u32 in C++
pub fn connect_rpc_socket(host: &String, port: u16, socket_fd: i64) -> io::Result<TcpStream> {
    let mut stream = match TcpStream::connect(&host[..]) {
        Err(e) => panic!("The server is not running because: {}", e),
        Ok(stream) => stream,
    };
    Ok(stream)
}

fn do_write(stream: &mut TcpStream, message_out: &String) -> io::Result<()> {
    send_message(stream, message_out.as_str())?;
    Ok(())
}

fn do_read(stream: &mut TcpStream, message_in: &mut String) -> io::Result<()> {
    *message_in = recv_message(stream)?;
    Ok(())
}

fn send_bytes(stream: &mut TcpStream, data: &[u8], length: usize) -> io::Result<()> {
    let mut remaining = length;
    let mut offset = 0;
    while remaining > 0 {
        let n = stream.write(&data[offset..])?;
        remaining -= n;
        offset += n;
    }
    Ok(())
}

fn send_message(stream: &mut TcpStream, message: &str) -> io::Result<()> {
    let len = message.len();
    let bytes = len.to_le_bytes();
    send_bytes(stream, &bytes, mem::size_of::<usize>())?;
    send_bytes(stream, message.as_bytes(), len)?;
    Ok(())
}

fn recv_bytes(stream: &mut TcpStream, data: &mut [u8], length: usize) -> io::Result<()> {
    let mut remaining = length;
    let mut offset = 0;
    while remaining > 0 {
        let n = stream.read(&mut data[offset..])?;
        remaining -= n;
        offset += n;
    }
    Ok(())
}
fn recv_message(stream: &mut TcpStream) -> io::Result<String> {
    let mut size_buf = [0u8; mem::size_of::<usize>()];
    recv_bytes(stream, &mut size_buf, mem::size_of::<usize>())?;
    let size = usize::from_le_bytes(size_buf);
    let mut message_buf = vec![0u8; size];
    recv_bytes(stream, message_buf.as_mut_slice(), size)?;
    Ok(String::from_utf8(message_buf).unwrap())
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

            let message_out: String = write_register_request();
            if let Err(e) = do_write(&mut stream, &message_out){
                self.connected = false; 
                return Err(e);
            }

            let mut message_in = String::new();
            do_read(&mut stream, &mut message_in).unwrap();

            let message_in: Value = serde_json::from_str(&message_in).expect("JSON was not well-formatted");
            let register_reply: RegisterReply = read_register_reply(message_in).unwrap();
            //println!("Register reply:\n{:?}\n ", register_reply);

            self.remote_instance_id = register_reply.instance_id;
            self.server_version = register_reply.version;
            self.ipc_socket = register_reply.ipc_socket;
            self.connected = true;

            // TODO： Compatable server

            Ok(())
        }
    }

    fn disconnect(&self) {}

    fn connected(&mut self) -> bool {
        self.connected
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
        let print = true;
        let rpc_client = &mut RPCClient {
            connected: false,
            ipc_socket: String::new(),
            rpc_endpoint: String::new(),
            vineyard_conn: 0,
            instance_id: 0,
            server_version: String::new(),
            remote_instance_id: 0,
        };
        if print {println!("Rpc client:\n {:?}\n", rpc_client)}
        rpc_client.connect(rpc_conn_input("0.0.0.0", 9600));
        if print {println!("Rpc client after connect:\n {:?}\n ", rpc_client)}
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
