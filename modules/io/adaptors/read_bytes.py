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
from urllib.parse import urlparse
import vineyard
from fsspec.utils import read_block
from fsspec.core import split_protocol
from vineyard.io.byte import ByteStreamBuilder

from vineyard.drivers.io import ossfs

fsspec.register_implementation("oss", ossfs.OSSFileSystem)


def report_status(status, content):
    ret = {"type": status, "content": content}
    print(json.dumps(ret), flush=True)


def read_bytes(
    vineyard_socket: str,
    path: str,
    storage_options: Dict,
    read_options: Dict,
    proc_num: int,
    proc_index: int,
):
    """Read bytes from external storage and produce a ByteStream,
    which will later be assembled into a ParallelStream.

    Args:
        vineyard_socket (str): Ipc socket
        path (str): External storage path to write to
        storage_options (dict): Configurations of external storage
        read_options (dict): Additional options that could control the behavior of read
        proc_num (int): Total amount of process
        proc_index (int): The sequence of this process

    Raises:
        ValueError: If the stream is invalid.
    """
    client = vineyard.connect(vineyard_socket)
    builder = ByteStreamBuilder(client)

    serialization_mode = read_options.pop('serialization_mode', False)
    if serialization_mode:
        parsed = urlparse(path)
        try:
            fs = fsspec.filesystem(parsed.scheme)
        except ValueError as e:
            report_status("error", str(e))
            raise
        meta_file = f"{path}_{proc_index}.meta"
        blob_file = f"{path}_{proc_index}"
        if not fs.exists(meta_file) or not fs.exists(blob_file):
            report_status("error", f"Some serialization file cannot be found. Expected: {meta_file} and {blob_file}")
            raise FileNotFoundError('{}, {}'.format(meta_file, blob_file))
        # Used for read bytes of serialized graph
        meta_file = fsspec.open(meta_file, mode="rb", **storage_options)
        with meta_file as f:
            meta = f.read().decode('utf-8')
            meta = json.loads(meta)
        lengths = meta.pop("lengths")
        for k, v in meta.items():
            builder[k] = v
        stream = builder.seal(client)
        client.persist(stream)
        ret = {"type": "return", "content": repr(stream.id)}
        print(json.dumps(ret), flush=True)
        writer = stream.open_writer(client)
        of = fsspec.open(blob_file, mode="rb", **storage_options)
        with of as f:
            try:
                total_size = f.size()
            except TypeError:
                total_size = f.size
            assert total_size == sum(lengths), "Target file is corrupted"
            for length in lengths:
                buf = f.read(length)
                chunk = writer.next(length)
                buf_writer = pa.FixedSizeBufferWriter(pa.py_buffer(chunk))
                buf_writer.write(buf)
                buf_writer.close()
        writer.finish()
    else:
        # Used when reading tables from external storage.
        # Usually for load a property graph
        header_row = read_options.get("header_row", False)
        for k, v in read_options.items():
            if k in ("header_row", "include_all_columns"):
                builder[k] = "1" if v else "0"
            elif k == "delimiter":
                builder[k] = bytes(v, "utf-8").decode("unicode_escape")
            else:
                builder[k] = v

        try:
            protocol = split_protocol(path)[0]
            fs = fsspec.filesystem(protocol, **storage_options)
        except Exception as e:
            report_status("error", f"Cannot initialize such filesystem for '{path}'")
            raise

        if fs.isfile():
            files = [path]
        else:
            try:
                files = fs.glob(path + '*')
                assert files, f"Cannot find such files: {path}"
            except:
                report_status("error", f"Cannot find such files for '{path}'")
                raise

        ''' Note [Semantic of read_block with delimiter]:

        read_block(fp, begin, size, delimiter) will:

            - find the first `delimiter` from `begin`, then starts read
            - after `size`, go through util the next `delimiter` or EOF, then finishes read.
              Note that the returned size may exceed `size`.
        '''

        chunk_size = 1024 * 1024 * 4
        for index, file_path in enumerate(files):
            with fs.open(file_path, mode="rb") as f:
                offset = 0
                # Only process header line when processing first file
                # And open the writer when processing first file
                if index == 0:
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

                # See Note [Semantic of read_block with delimiter].
                if index == 0 and proc_index == 0:
                    begin -= int(header_row)

                while begin < end:
                    buf = read_block(f, begin, min(chunk_size, end - begin), delimiter=b"\n")
                    size = len(buf)
                    if not size:
                        break
                    begin += size - 1
                    chunk = writer.next(size)
                    buf_writer = pa.FixedSizeBufferWriter(pa.py_buffer(chunk))
                    buf_writer.write(buf)
                    buf_writer.close()
        writer.finish()


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("usage: ./read_bytes <ipc_socket> <path> <storage_options> <read_options> <proc_num> <proc_index>")
        exit(1)
    ipc_socket = sys.argv[1]
    path = sys.argv[2]
    storage_options = json.loads(base64.b64decode(sys.argv[3].encode("utf-8")).decode("utf-8"))
    read_options = json.loads(base64.b64decode(sys.argv[4].encode("utf-8")).decode("utf-8"))
    proc_num = int(sys.argv[5])
    proc_index = int(sys.argv[6])
    read_bytes(ipc_socket, path, storage_options, read_options, proc_num, proc_index)
