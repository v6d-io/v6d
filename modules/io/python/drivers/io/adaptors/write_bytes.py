#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Alibaba Group Holding Limited.
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

import vineyard
from vineyard.io.byte import ByteStream
from vineyard.io.utils import expand_full_path
from vineyard.io.utils import report_error
from vineyard.io.utils import report_exception

try:
    from vineyard.drivers.io import ossfs
except ImportError:
    ossfs = None

if ossfs:
    fsspec.register_implementation("oss", ossfs.OSSFileSystem)


def write_bytes(
    vineyard_socket,
    path,
    stream_id,
    storage_options,
    _write_options,
    proc_num,
    proc_index,
):
    """Read bytes from stream and write to external storage.

    Args:
        vineyard_socket (str): Ipc socket
        path (str): External storage path to write to
        stream_id (str): ObjectID of the stream to be read from, which is a
                         ParallelStream
        storage_options (dict): Configurations of external storage
        write_options (dict): Additional options that could control the behavior
                              of write
        proc_num (int): Total amount of process
        proc_index (int): The sequence of this process

    Raises:
        ValueError: If the stream is invalid.
    """
    client = vineyard.connect(vineyard_socket)
    streams = client.get(stream_id)
    if len(streams) != proc_num or streams[proc_index] is None:
        report_error(
            f"Fetch stream error with proc_num = {proc_num}, proc_index = {proc_index}"
        )
        sys.exit(-1)

    instream: ByteStream = streams[proc_index]
    try:
        reader = instream.open_reader(client)
        of = fsspec.open(f"{path}_{proc_index}", "wb", **storage_options)
    except Exception:  # pylint: disable=broad-except
        report_exception()
        sys.exit(-1)

    lengths = []  # store lengths of each chunk. may be unused
    with of as f:
        while True:
            try:
                chunk = reader.next()
            except (StopIteration, vineyard.StreamDrainedException):
                break
            lengths.append(len(chunk))
            f.write(bytes(chunk))


def main():
    if len(sys.argv) < 8:
        print(
            "usage: ./write_bytes <ipc_socket> <path> <stream_id> "
            "<storage_options> <write_options> <proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    path = expand_full_path(sys.argv[2])
    stream_id = sys.argv[3]
    storage_options = json.loads(
        base64.b64decode(sys.argv[4].encode("utf-8")).decode("utf-8")
    )
    write_options = json.loads(
        base64.b64decode(sys.argv[5].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[6])
    proc_index = int(sys.argv[7])
    write_bytes(
        ipc_socket,
        path,
        stream_id,
        storage_options,
        write_options,
        proc_num,
        proc_index,
    )


if __name__ == "__main__":
    main()
