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
use super::ObjectID;
use super::ObjectMeta;
use std::io::{self, Error, ErrorKind};
use std::os::unix::net::UnixStream;

pub enum conn_input<'a, 'b> {
    ipc_conn_input(&'a str),      // socket
    rpc_conn_input(&'b str, u16), // host, port
}

pub trait Client {
    fn connect(&mut self, conn_input: conn_input) -> io::Result<()>;

    // Disconnect this client.
    fn disconnect(&self);

    fn connected(&mut self) -> bool;

    // Obtain multiple metadatas from vineyard server.
    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> io::Result<ObjectMeta>;
}
