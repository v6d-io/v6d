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
    vinyard_conn: i64,
    instance_id: InstanceID,
    server_version: String,
}

pub fn connect_rpc_socket(host: &String, port: u32, socket_fd: i64) -> Result<(), Error> {
    panic!();
}

impl Client for RPCClient {
    fn connect(&mut self, conn_input: conn_input) -> Result<(), Error> {
        let (host, port) = match conn_input{
            rpc_conn_input(host, port) => (host, port),
            _ => panic!("Insuitable type of connect input."), 
        };
        let rpc_host: String = String::from(host);
        let rpc_endpoint: String = format!("{}:{}", host, port.to_string());
        // Panic when they have connected while assigning different rpc_endpoint
        RETURN_ON_ASSERT(!self.connected || rpc_endpoint == self.rpc_endpoint);
        if self.connected {
            return Ok(());
        } else {
            self.rpc_endpoint = rpc_endpoint;
            //let mut stream = connect_rpc_socket(&rpc_host, port, self.vineyard_conn).unwrap();


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
