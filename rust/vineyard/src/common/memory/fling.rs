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

use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::fd::FromRawFd;
use std::os::unix::net::UnixStream;

use sendfd::{RecvWithFd, SendWithFd};

use super::super::util::status::*;

pub fn send_fd(conn: &UnixStream, fd: i32) -> Result<()> {
    conn.send_with_fd(&[], &[unsafe { File::from_raw_fd(fd) }.as_raw_fd()])?;
    Ok(())
}

pub fn recv_fd(conn: &UnixStream) -> Result<i32> {
    let mut buf = [0u8; 1];
    let mut fds = [0; 1];
    let (_, received) = conn.recv_with_fd(&mut buf, &mut fds)?;
    if received != 1 {
        return Err(VineyardError::io_error(format!(
            "Failed to receive fd, received: {}, buf: {:?}",
            received, fds,
        )));
    }
    return Ok(fds[0]);
}
