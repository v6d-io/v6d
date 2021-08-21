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

use std::io::{self, Error, ErrorKind};
use std::os::unix::net::UnixStream;
use std::net::TcpStream;

use super::ObjectID;
use super::ObjectMeta;
use crate::common::util::protocol::*;

pub enum ConnInputKind<'a, 'b> {
    IPCConnInput(&'a str),      // socket
    RPCConnInput(&'b str, u16), // host, port
}

pub enum StreamKind{
    IPCStream(UnixStream),
    RPCStream(TcpStream),
}

pub trait Client {
    fn connect(&mut self, conn_input: ConnInputKind) -> io::Result<()>;

    // Disconnect this client.
    fn disconnect(&self);

    fn connected(&mut self) -> bool;

    // Obtain multiple metadatas from vineyard server.
    fn get_meta_data(&self, object_id: ObjectID, sync_remote: bool) -> io::Result<ObjectMeta>;

    fn put_name(&mut self, stream: StreamKind, id: ObjectID, name: &String) -> io::Result<()>{
        // ENSURE_CONNECTED(self.connected());
        // let message_out = write_put_name_request(id, name);
        // do_write(&mut stream, &message_out)?;
        // let mut message_in = String::new();
        // do_read(&mut stream, &mut message_in)?;


        Ok(())
    }
}
