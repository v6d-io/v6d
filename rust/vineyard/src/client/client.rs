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

pub enum conn_input<'a, 'b>{
    ipc_conn_input(&'a str), // socket
    rpc_conn_input(&'b str, u32), // host, port
}

pub trait Client {
    // Connect to vineyardd using the given UNIX domain socket ipc_socket.
    // Parameters: ipc_socket – Location of the UNIX domain socket.
    // Returns: Status that indicates whether the connect has succeeded.
    fn connect(&mut self, conn_input: conn_input) -> Result<(), Error>; // TODO: Check result types

    // Disconnect this client.
    fn disconnect(&self);

    // Check if the client still connects to the vineyard server.
    // Returns. True when the connection is still alive, otherwise false.
    fn connected(&self) -> bool;

    // Obtain multiple metadatas from vineyard server.
    // Parameters:
    // ids – The object ids to get.
    // meta_data – The result metadata will be store in meta_data as return value.
    // sync_remote – Whether to trigger an immediate remote metadata synchronization
    // before get specific metadata. Default is false.
    // Returns: Status that indicates whether the get action has succeeded.
    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> Result<ObjectMeta, Error>;
}
