use std::io;
/** Copyright 2020-2023 Alibaba Group Holding Limited.

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
use std::io::prelude::*;
use std::mem;
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::path::Path;

use super::client::StreamKind;

// socket_fd is used to assign vineyard_conn
pub fn connect_ipc_socket(pathname: &String, socket_fd: i64) -> io::Result<UnixStream> {
    let socket = Path::new(pathname);
    let stream = match UnixStream::connect(&socket) {
        Err(e) => panic!("The server is not running because: {}.", e),
        Ok(stream) => stream,
    };
    Ok(stream)
}

// Question: the port is u16 from the material I saw while u32 in C++
pub fn connect_rpc_socket(host: &String, port: u16, socket_fd: i64) -> io::Result<TcpStream> {
    let stream = match TcpStream::connect(&host[..]) {
        Err(e) => panic!("The server is not running because: {}", e),
        Ok(stream) => stream,
    };
    Ok(stream)
}

pub fn do_write(stream: &mut StreamKind, message_out: &String) -> io::Result<()> {
    match stream {
        StreamKind::IPCStream(ipc_stream) => {
            ipc_io::send_message(ipc_stream, message_out.as_str())?
        }
        StreamKind::RPCStream(rpc_stream) => {
            rpc_io::send_message(rpc_stream, message_out.as_str())?
        }
    }
    Ok(())
}

pub fn do_read(stream: &mut StreamKind, message_in: &mut String) -> io::Result<()> {
    match stream {
        StreamKind::IPCStream(ipc_stream) => *message_in = ipc_io::recv_message(ipc_stream)?,
        StreamKind::RPCStream(rpc_stream) => *message_in = rpc_io::recv_message(rpc_stream)?,
    }
    Ok(())
}

mod ipc_io {
    use super::*;

    pub fn send_bytes(stream: &mut UnixStream, data: &[u8], length: usize) -> io::Result<()> {
        let mut remaining = length;
        let mut offset = 0;
        while remaining > 0 {
            let n = stream.write(&data[offset..])?;
            remaining -= n;
            offset += n;
        }
        Ok(())
    }

    pub fn send_message(stream: &mut UnixStream, message: &str) -> io::Result<()> {
        let len = message.len();
        let bytes = len.to_le_bytes();
        send_bytes(stream, &bytes, mem::size_of::<usize>())?;
        send_bytes(stream, message.as_bytes(), len)?;
        Ok(())
    }

    pub fn recv_bytes(stream: &mut UnixStream, data: &mut [u8], length: usize) -> io::Result<()> {
        let mut remaining = length;
        let mut offset = 0;
        while remaining > 0 {
            let n = stream.read(&mut data[offset..])?;
            remaining -= n;
            offset += n;
        }
        Ok(())
    }

    pub fn recv_message(stream: &mut UnixStream) -> io::Result<String> {
        let mut size_buf = [0u8; mem::size_of::<usize>()];
        recv_bytes(stream, &mut size_buf, mem::size_of::<usize>())?;
        let size = usize::from_le_bytes(size_buf);
        let mut message_buf = vec![0u8; size];
        recv_bytes(stream, message_buf.as_mut_slice(), size)?;
        Ok(String::from_utf8(message_buf).unwrap())
    }
}

mod rpc_io {
    use super::*;

    pub fn send_bytes(stream: &mut TcpStream, data: &[u8], length: usize) -> io::Result<()> {
        let mut remaining = length;
        let mut offset = 0;
        while remaining > 0 {
            let n = stream.write(&data[offset..])?;
            remaining -= n;
            offset += n;
        }
        Ok(())
    }

    pub fn send_message(stream: &mut TcpStream, message: &str) -> io::Result<()> {
        let len = message.len();
        let bytes = len.to_le_bytes();
        send_bytes(stream, &bytes, mem::size_of::<usize>())?;
        send_bytes(stream, message.as_bytes(), len)?;
        Ok(())
    }

    pub fn recv_bytes(stream: &mut TcpStream, data: &mut [u8], length: usize) -> io::Result<()> {
        let mut remaining = length;
        let mut offset = 0;
        while remaining > 0 {
            let n = stream.read(&mut data[offset..])?;
            remaining -= n;
            offset += n;
        }
        Ok(())
    }

    pub fn recv_message(stream: &mut TcpStream) -> io::Result<String> {
        let mut size_buf = [0u8; mem::size_of::<usize>()];
        recv_bytes(stream, &mut size_buf, mem::size_of::<usize>())?;
        let size = usize::from_le_bytes(size_buf);
        let mut message_buf = vec![0u8; size];
        recv_bytes(stream, message_buf.as_mut_slice(), size)?;
        Ok(String::from_utf8(message_buf).unwrap())
    }
}
