#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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
import logging
import sys
import traceback
from typing import Dict

import fsspec
from fsspec.core import get_fs_token_paths
from fsspec.utils import read_block

import vineyard
from vineyard.io.byte import ByteStream
from vineyard.io.utils import expand_full_path
from vineyard.io.utils import parse_readable_size
from vineyard.io.utils import report_error
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success

logger = logging.getLogger('vineyard')


try:
    from vineyard.drivers.io import fsspec_adaptors
except Exception:  # pylint: disable=broad-except
    logger.warning("Failed to import fsspec adaptors for hdfs, oss, etc")


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


def read_byte_blocks(
    fp,
    proc_num,
    proc_index,
    index,
    header_row,
    offset,
    chunk_size,
    read_block_delimiter,
    writer,
):
    try:
        total_size = fp.size()
    except TypeError:
        total_size = fp.size
    part_size = (total_size - offset) // proc_num
    begin = part_size * proc_index + offset
    end = total_size if proc_index == proc_num - 1 else begin + part_size

    # See Note [Semantic of read_block with delimiter].
    if index == 0 and proc_index == 0:
        begin -= int(header_row)

    first_chunk = True
    while begin < end:
        buffer = read_block(
            fp,
            begin,
            min(chunk_size, end - begin),
            delimiter=read_block_delimiter,
        )
        if first_chunk:
            # strip the UTF-8 BOM
            if buffer[0:3] == b'\xef\xbb\xbf':
                buffer = buffer[3:]
        first_chunk = False
        size = len(buffer)
        if size <= 0:
            break
        begin += size
        if size > 0:
            chunk = writer.next(size)
            vineyard.memory_copy(chunk, 0, buffer)


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
        read_options (dict): Additional options that could control the
                             behavior of read
        proc_num (int): Total amount of process
        proc_index (int): The sequence of this process

    Raises:
        ValueError: If the stream is invalid.
    """
    client = vineyard.connect(vineyard_socket)
    params = dict()

    # Cut strings after # and read potential kvs
    parts = path.split('#', 1)
    path = parts[0]
    if len(parts) > 1:
        options = parts[1]
        for split_by_hashtap in options.split('#'):
            for split_by_ampersand in split_by_hashtap.split('&'):
                k, v = split_by_ampersand.split('=')
                read_options[k] = v

    read_block_delimiter = read_options.pop('read_block_delimiter', '\n')
    if read_block_delimiter is not None:
        read_block_delimiter = read_block_delimiter.encode('utf-8')

    # Used when reading tables from external storage.
    # Usually for load a property graph

    # header_row: each file has a header_row when reading from directories
    header_row = bool(read_options.get("header_row", False))
    # first_header_row: only the first file (alphabetical ordered) has a header_row
    # when reading from directories
    first_header_row = read_options.get("first_header_row", False)
    for k, v in read_options.items():
        if k in ("header_row", "first_header_row", "include_all_columns", "accumulate"):
            params[k] = "1" if v else "0"
        elif k == "delimiter":
            params[k] = bytes(v, "utf-8").decode("unicode_escape")
        elif not isinstance(v, str):
            params[k] = repr(v)
        else:
            params[k] = v

    try:
        # files would be empty if it's a glob pattern and globbed nothing.
        fs, _, files = get_fs_token_paths(path, storage_options=storage_options)
    except Exception:  # pylint: disable=broad-except
        report_error(
            f"Cannot initialize such filesystem for '{path}', "
            f"exception is:\n{traceback.format_exc()}"
        )
        sys.exit(-1)
    try:
        assert files
    except Exception:  # pylint: disable=broad-except
        report_error(f"Cannot find such files for '{path}'")
        sys.exit(-1)
    files = sorted(files)

    stream, writer = None, None
    if 'chunk_size' in storage_options:
        chunk_size = parse_readable_size(storage_options['chunk_size'])
    else:
        chunk_size = 1024 * 1024 * 128  # default: 64MB

    try:
        for index, file_path in enumerate(files):
            with fs.open(file_path, mode="rb") as fp:
                offset = 0
                # Only process header line when processing first file
                # And open the writer when processing first file
                if header_row or (first_header_row and index == 0):
                    header_line = read_block(fp, 0, 1, read_block_delimiter)
                    params["header_line"] = header_line.decode("unicode_escape")
                    offset = len(header_line)
                    # strip the UTF-8 BOM
                    if header_line[0:3] == b'\xef\xbb\xbf':
                        header_line = header_line[3:]
                if index == 0:
                    stream = ByteStream.new(client, params)
                    client.persist(stream.id)
                    report_success(stream.id)
                    writer = stream.open_writer(client)

                read_byte_blocks(
                    fp,
                    proc_num,
                    proc_index,
                    index,
                    header_row,
                    offset,
                    chunk_size,
                    read_block_delimiter,
                    writer,
                )

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
