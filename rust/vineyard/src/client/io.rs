// Copyright 2020-2023 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::io::{Read, Write};
use std::mem;
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::path::Path;

use crate::common::util::status::*;

const NUM_CONNECT_ATTEMPTS: usize = 10;
const CONNECT_TIMEOUT_MS: u64 = 1000;

fn connect_ipc_socket(pathname: &str) -> Result<UnixStream> {
    let socket = Path::new(pathname);
    match UnixStream::connect(&socket) {
        Err(e) => {
            return Err(VineyardError::io_error(format!(
                "Failed to connect to {:?}. {}",
                socket, e
            )));
        }
        Ok(stream) => {
            return Ok(stream);
        }
    };
}

fn connect_rpc_endpoint(host: &str, port: u16) -> Result<TcpStream> {
    match TcpStream::connect(format!("{}:{}", host, port)) {
        Err(e) => {
            return Err(VineyardError::io_error(format!(
                "Failed to connect to {}:{}. {}",
                host, port, e
            )));
        }
        Ok(stream) => {
            return Ok(stream);
        }
    };
}

pub fn connect_ipc_socket_retry(pathname: &str) -> Result<UnixStream> {
    let mut i = 0;
    loop {
        match connect_ipc_socket(pathname) {
            Err(e) => {
                if i < NUM_CONNECT_ATTEMPTS {
                    i += 1;
                    info!(
                        concat!(
                            "Connection to IPC socket failed for pathname {} ",
                            "with ret = {}, retrying {} more times."
                        ),
                        pathname,
                        e,
                        NUM_CONNECT_ATTEMPTS - i
                    );
                    std::thread::sleep(std::time::Duration::from_millis(CONNECT_TIMEOUT_MS));
                    continue;
                } else {
                    return Err(e);
                }
            }
            Ok(stream) => {
                return Ok(stream);
            }
        }
    }
}

pub fn connect_rpc_endpoint_retry(host: &str, port: u16) -> Result<TcpStream> {
    let mut i = 0;
    loop {
        match connect_rpc_endpoint(host, port) {
            Err(e) => {
                if i < NUM_CONNECT_ATTEMPTS {
                    i += 1;
                    info!(
                        concat!(
                            "Connection to RPC socket failed for endpoint {}:{} ",
                            "with ret = {}, retrying {} more times."
                        ),
                        host,
                        port,
                        e,
                        NUM_CONNECT_ATTEMPTS - i
                    );
                    std::thread::sleep(std::time::Duration::from_millis(CONNECT_TIMEOUT_MS));
                    continue;
                } else {
                    return Err(e);
                }
            }
            Ok(stream) => {
                return Ok(stream);
            }
        }
    }
}

fn recv_bytes<T: Read>(stream: &mut T, data: &mut [u8], length: usize) -> Result<()> {
    let mut remaining = length;
    let mut offset = 0;
    while remaining > 0 {
        let n = stream.read(&mut data[offset..])?;
        remaining -= n;
        offset += n;
    }
    Ok(())
}

fn recv_message<T: Read>(stream: &mut T) -> Result<String> {
    let mut size_buf = [0u8; mem::size_of::<usize>()];
    recv_bytes(stream, &mut size_buf, mem::size_of::<usize>())?;
    let size = usize::from_le_bytes(size_buf);
    let mut message_buf = vec![0u8; size];
    recv_bytes(stream, message_buf.as_mut_slice(), size)?;
    Ok(String::from_utf8(message_buf).unwrap())
}

fn send_bytes<T: Write>(stream: &mut T, data: &[u8], length: usize) -> Result<()> {
    let mut remaining = length;
    let mut offset = 0;
    while remaining > 0 {
        let n = stream.write(&data[offset..])?;
        remaining -= n;
        offset += n;
    }
    Ok(())
}

fn send_message<T: Write>(stream: &mut T, message: &str) -> Result<()> {
    let len = message.len();
    let bytes = len.to_le_bytes();
    send_bytes(stream, &bytes, mem::size_of::<usize>())?;
    send_bytes(stream, message.as_bytes(), len)?;
    Ok(())
}

pub fn do_read<T: Read>(stream: &mut T) -> Result<String> {
    return recv_message(stream);
}

pub fn do_write<T: Write>(stream: &mut T, message_out: &str) -> Result<()> {
    return send_message(stream, message_out);
}
