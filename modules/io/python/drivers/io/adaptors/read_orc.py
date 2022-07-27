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
from typing import Dict
from urllib.parse import urlparse

import pyarrow as pa

import fsspec
import fsspec.implementations.hdfs
import pyorc

import vineyard
from vineyard.io.dataframe import DataframeStream
from vineyard.io.utils import expand_full_path
from vineyard.io.utils import report_success

try:
    from vineyard.drivers.io import ossfs
except ImportError:
    ossfs = None

if ossfs:
    fsspec.register_implementation("oss", ossfs.OSSFileSystem)
fsspec.register_implementation("hive", fsspec.implementations.hdfs.PyArrowHDFS)


def arrow_type(field):
    if field.name == "decimal":
        return pa.decimal128(field.precision)
    elif field.name == "uniontype":
        return pa.union(field.cont_types)
    elif field.name == "array":
        return pa.list_(field.type)
    elif field.name == "map":
        return pa.map_(field.key, field.value)
    elif field.name == "struct":
        return pa.struct(field.fields)
    else:
        types = {
            "boolean": pa.bool_(),
            "tinyint": pa.int8(),
            "smallint": pa.int16(),
            "int": pa.int32(),
            "bigint": pa.int64(),
            "float": pa.float32(),
            "double": pa.float64(),
            "string": pa.string(),
            "char": pa.string(),
            "varchar": pa.string(),
            "binary": pa.binary(),
            "timestamp": pa.timestamp("ms"),
            "date": pa.date32(),
        }
        if field.name not in types:
            raise ValueError("Cannot convert to arrow type: " + field.name)
        return types[field.name]


def read_single_orc(path, fs, writer):
    chunk_rows = 1024 * 256

    with fs.open(path, "rb") as f:
        reader = pyorc.Reader(f)
        fields = reader.schema.fields  # pylint: disable=no-member
        schema = []
        for c in fields:
            schema.append((c, arrow_type(fields[c])))
        pa_struct = pa.struct(schema)
        while True:
            rows = reader.read(num=chunk_rows)
            if not rows:
                break
            batch = pa.RecordBatch.from_struct_array(pa.array(rows, type=pa_struct))
            writer.write(batch)


def read_orc(
    vineyard_socket,
    path,
    storage_options: Dict,
    read_options: Dict,
    _proc_num,
    proc_index,
):
    # This method is to read the data files of a specific hive table
    # that is stored as orc format in HDFS.
    #
    # In general, the data files of a hive table are stored at the hive
    # space in the HDFS with the table name as the directory,
    # e.g.,
    #
    # .. code:: python
    #
    #    '/user/hive/warehouse/sometable'
    #
    # To read the entire table, simply use 'hive://user/hive/warehouse/sometable'
    # as the path.
    #
    # In case the table is partitioned, use the sub-directory of a specific partition
    # to read only the data from that partition. For example, sometable is partitioned
    # by column date, we can read the data in a given date by giving path as
    #
    # .. code:: python
    #
    #    'hive://user/hive/warehouse/sometable/date=20201112'
    #
    if proc_index:
        raise ValueError("Parallel reading ORC hasn't been supported yet")
    if read_options:
        raise ValueError("Reading ORC doesn't support read options.")
    client = vineyard.connect(vineyard_socket)
    stream = DataframeStream.new(client)
    client.persist(stream.id)
    report_success(stream.id)

    writer = stream.open_writer(client)
    parsed = urlparse(path)

    fs = fsspec.filesystem(parsed.scheme, **storage_options)
    if fs.isfile(parsed.path):
        files = [parsed.path]
    else:
        files = [f for f in fs.ls(parsed.path, detail=False) if fs.isfile(f)]
    for file_path in files:
        read_single_orc(file_path, fs, writer)
    # hdfs = HDFileSystem(
    #     host=host, port=int(port), pars={"dfs.client.read.shortcircuit": "false"}
    # )

    writer.finish()


def main():
    if len(sys.argv) < 7:
        print(
            "usage: ./read_orc <ipc_socket> <path/directory> <storage_options> "
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
    read_orc(ipc_socket, path, storage_options, read_options, proc_num, proc_index)


if __name__ == "__main__":
    main()
