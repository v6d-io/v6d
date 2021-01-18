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
from typing import Dict

import fsspec
import pyarrow as pa
import pyorc
import vineyard
from fsspec.utils import read_block
from vineyard.io.byte import ByteStreamBuilder

import ossfs

fsspec.register_implementation("oss", ossfs.OSSFileSystem)


def read_bytes(
    vineyard_socket: str,
    path: str,
    storage_options: Dict,
    read_options: Dict,
    proc_num: int,
    proc_index: int,
):
    client = vineyard.connect(vineyard_socket)
    builder = ByteStreamBuilder(client)

    header_row = read_options.get("header_row", False)
    for k, v in read_options.items():
        if k in ("header_row", "include_all_columns"):
            builder[k] = "1" if v else "0"
        elif k == "delimiter":
            builder[k] = bytes(v, "utf-8").decode("unicode_escape")
        else:
            builder[k] = v

    offset = 0
    chunk_size = 1024 * 1024 * 4
    of = fsspec.open(path, mode="rb", **storage_options)
    with of as f:
        header_line = read_block(f, 0, 1, b'\n')
        builder["header_line"] = header_line.decode("unicode_escape")
        if header_row:
            offset = len(header_line)
        stream = builder.seal(client)
        client.persist(stream)
        ret = {"type": "return", "content": repr(stream.id)}
        print(json.dumps(ret), flush=True)

        writer = stream.open_writer(client)
        try:
            total_size = f.size()
        except TypeError:
            total_size = f.size
        part_size = (total_size - offset) // proc_num
        begin = part_size * proc_index + offset
        end = min(begin + part_size, total_size)
        if proc_index == 0:
            begin -= int(header_row)

        while begin < end:
            buf = read_block(f, begin, min(chunk_size, end - begin), delimiter=b"\n")
            size = len(buf)
            if not size:
                break
            begin += size - 1
            chunk = writer.next(size)
            buf_writer = pa.FixedSizeBufferWriter(chunk)
            buf_writer.write(buf)
            buf_writer.close()

        writer.finish()


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "usage: ./read_bytes <ipc_socket> <path> <storage_options> <read_options> <proc_num> <proc_index>"
        )
        exit(1)
    ipc_socket = sys.argv[1]
    path = sys.argv[2]
    storage_options = json.loads(
        base64.b64decode(sys.argv[3].encode("utf-8")).decode("utf-8")
    )
    read_options = json.loads(
        base64.b64decode(sys.argv[4].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[5])
    proc_index = int(sys.argv[6])
    read_bytes(ipc_socket, path, storage_options, read_options, proc_num, proc_index)
