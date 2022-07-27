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
import traceback
from typing import Dict

import fsspec
from fsspec.core import split_protocol
from fsspec.utils import read_block

import vineyard
from vineyard.io.byte import ByteStream
from vineyard.io.utils import expand_full_path
from vineyard.io.utils import parse_readable_size
from vineyard.io.utils import report_error
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success

try:
    from vineyard.drivers.io import ossfs
except ImportError:
    ossfs = None

if ossfs:
    fsspec.register_implementation("oss", ossfs.OSSFileSystem)


def read_bytes(  # noqa: C901, pylint: disable=too-many-statements
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
    params = dict()

    read_block_delimiter = read_options.pop('read_block_delimiter', '\n')
    if read_block_delimiter is not None:
        read_block_delimiter = read_block_delimiter.encode('utf-8')

    # Used when reading tables from external storage.
    # Usually for load a property graph
    header_row = read_options.get("header_row", False)
    for k, v in read_options.items():
        if k in ("header_row", "include_all_columns"):
            params[k] = "1" if v else "0"
        elif k == "delimiter":
            params[k] = bytes(v, "utf-8").decode("unicode_escape")
        else:
            params[k] = v

    try:
        protocol = split_protocol(path)[0]
        fs = fsspec.filesystem(protocol, **storage_options)
    except Exception:  # pylint: disable=broad-except
        report_error(
            f"Cannot initialize such filesystem for '{path}', "
            f"exception is:\n{traceback.format_exc()}"
        )
        sys.exit(-1)

    if fs.isfile(path):
        files = [path]
    else:
        try:
            files = fs.glob(path + '*')
            assert files, f"Cannot find such files: {path}"
        except Exception:  # pylint: disable=broad-except
            report_error(f"Cannot find such files for '{path}'")
            sys.exit(-1)

    # Note [Semantic of read_block with delimiter]:
    #
    # read_block(fp, begin, size, delimiter) will:
    #
    #    - find the first `delimiter` from `begin`, then starts read
    #    - after `size`, go through util the next `delimiter` or EOF,
    #      then finishes read.
    #
    # Note that the returned size may exceed `size`.
    #

    stream, writer = None, None
    if 'chunk_size' in storage_options:
        chunk_size = parse_readable_size(storage_options['chunk_size'])
    else:
        chunk_size = 1024 * 1024 * 64  # default: 64MB

    try:
        for index, file_path in enumerate(files):
            with fs.open(file_path, mode="rb") as f:
                offset = 0
                # Only process header line when processing first file
                # And open the writer when processing first file
                if index == 0:
                    if header_row:
                        header_line = read_block(f, 0, 1, read_block_delimiter)
                        params["header_line"] = header_line.decode("unicode_escape")
                        offset = len(header_line)
                    stream = ByteStream.new(client, params)
                    client.persist(stream.id)
                    report_success(stream.id)
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
                    buffer = read_block(
                        f,
                        begin,
                        min(chunk_size, end - begin),
                        delimiter=read_block_delimiter,
                    )
                    size = len(buffer)
                    if size <= 0:
                        break
                    begin += size - 1
                    if size > 0:
                        chunk = writer.next(size)
                        vineyard.memory_copy(chunk, 0, buffer)
        writer.finish()
    except Exception:  # pylint: disable=broad-except
        report_exception()
        if writer is not None:
            writer.fail()
        sys.exit(-1)


def main():
    if len(sys.argv) < 7:
        print(
            "usage: ./read_bytes <ipc_socket> <path> <storage_options> <read_options> "
            "<proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    path = expand_full_path(sys.argv[2])
    storage_options = json.loads(
        base64.b64decode(sys.argv[3].encode("utf-8")).decode("utf-8")
    )
    read_options = json.loads(
        base64.b64decode(sys.argv[4].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[5])
    proc_index = int(sys.argv[6])
    read_bytes(ipc_socket, path, storage_options, read_options, proc_num, proc_index)


if __name__ == "__main__":
    main()
