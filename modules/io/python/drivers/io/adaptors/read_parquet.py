#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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
import fsspec.implementations.arrow
from fsspec.core import split_protocol

import vineyard
from vineyard.io.dataframe import DataframeStream
from vineyard.io.utils import expand_full_path
from vineyard.io.utils import report_error
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success

logger = logging.getLogger('vineyard')


try:
    from vineyard.drivers.io import fsspec_adaptors
except Exception as e:  # pylint: disable=broad-except
    logger.warning("Failed to import fsspec adaptors for hdfs, oss, etc %s", e)


def make_empty_batch(schema):
    import pyarrow

    colmuns = [pyarrow.array([], t) for t in schema.types]
    return pyarrow.RecordBatch.from_arrays(colmuns, schema.names)


def read_parquet_blocks(fs, path, read_options, proc_num, proc_index, writer):
    import pyarrow.parquet

    columns = read_options.get('columns', None)
    kwargs = {}
    if columns:
        kwargs['columns'] = columns.split(',')

    with fs.open(path, 'rb') as f:
        reader = pyarrow.parquet.ParquetFile(f)
        row_groups_per_proc = reader.num_row_groups // proc_num
        if reader.num_row_groups % proc_num != 0:
            row_groups_per_proc += 1
        row_group_begin = row_groups_per_proc * proc_index
        row_group_end = min(
            row_groups_per_proc * (proc_index + 1), reader.num_row_groups
        )

        if row_group_begin < row_group_end:
            kwargs = {}
            for batch in reader.iter_batches(
                row_groups=range(row_group_begin, row_group_end),
                use_threads=False,
                **kwargs,
            ):
                writer.write(batch)
        else:
            writer.write(make_empty_batch(reader.schema_arrow))


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

    # Used when reading tables from external storage.
    # Usually for load a property graph
    #
    # possbile values:
    #
    #   - columns, seperated by ','
    for k, v in read_options.items():
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

    stream, writer = None, None
    try:
        for index, file_path in enumerate(files):
            if index == 0:
                stream = DataframeStream.new(client, {})
                client.persist(stream.id)
                report_success(stream.id)
                writer = stream.open_writer(client)
            read_parquet_blocks(
                fs, file_path, read_options, proc_num, proc_index, writer
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
            "usage: ./read_parquet <ipc_socket> <path> <storage_options> "
            "<read_options> <proc_num> <proc_index>"
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
