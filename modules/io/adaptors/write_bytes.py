#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import base64
import json
import sys

import fsspec
import pyarrow as pa
import vineyard
from vineyard.io.byte import ByteStreamBuilder

import ossfs

fsspec.register_implementation("oss", ossfs.OSSFileSystem)


def write_bytes(
    vineyard_socket, path, stream_id, storage_options, proc_num, proc_index
):
    client = vineyard.connect(vineyard_socket)
    streams = client.get(stream_id)
    if len(streams) != proc_num or streams[proc_index] is None:
        raise ValueError(
            f"Fetch stream error with proc_num={proc_num},proc_index={proc_index}"
        )
    instream = streams[proc_index]
    reader = instream.open_reader(client)

    # Write file distributively
    path += f"_{proc_index}"
    of = fsspec.open(path, "wb", **storage_options)
    with of as f:
        while True:
            try:
                buf = reader.next()
            except vineyard.StreamDrainedException:
                break
            f.write(bytes(memoryview(buf)))


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "usage: ./write_bytes <ipc_socket> <path> <stream_id> <storage_options> <proc_num> <proc_index>"
        )
        exit(1)
    ipc_socket = sys.argv[1]
    path = sys.argv[2]
    stream_id = sys.argv[3]
    storage_options = json.loads(
        base64.b64decode(sys.argv[4].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[5])
    proc_index = int(sys.argv[6])
    write_bytes(ipc_socket, path, stream_id, storage_options, proc_num, proc_index)
