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
from urllib.parse import urlparse

import fsspec
import pyarrow as pa
import pyorc
import vineyard
from vineyard.io.dataframe import DataframeStreamBuilder


def orc_type(field):
    if pa.types.is_boolean(field):
        return pyorc.Boolean()
    elif pa.types.is_int8(field):
        return pyorc.TinyInt()
    elif pa.types.is_int16(field):
        return pyorc.SmallInt()
    elif pa.types.is_int32(field):
        return pyorc.Int()
    elif pa.types.is_int64(field):
        return pyorc.BigInt()
    elif pa.types.is_float32(field):
        return pyorc.Float()
    elif pa.types.is_float64(field):
        return pyorc.Double()
    elif pa.types.is_decimal(field):
        return pyorc.Decimal(field.precision, field.scale)
    elif pa.types.is_list(field):
        return pyorc.Array(orc_type(field.value_type))
    elif pa.types.is_timestamp(field):
        return pyorc.Timestamp()
    elif pa.types.is_date(field):
        return pyorc.Date()
    elif pa.types.is_binary(field):
        return pyorc.Binary()
    elif pa.types.is_string(field):
        return pyorc.String()
    else:
        raise ValueError("Cannot Convert %s" % field)


def write_orc(vineyard_socket, path, stream_id, storage_options, write_options, proc_num, proc_index):
    client = vineyard.connect(vineyard_socket)
    streams = client.get(stream_id)
    if len(streams) != proc_num or streams[proc_index] is None:
        raise ValueError(
            f"Fetch stream error with proc_num={proc_num},proc_index={proc_index}"
        )
    instream = streams[proc_index]
    reader = instream.open_reader(client)

    writer = None
    path += f"_{proc_index}"
    with fsspec.open(path, "wb", **storage_options) as f:
        while True:
            try:
                buf = reader.next()
            except vineyard.StreamDrainedException:
                writer.close()
                break
            buf_reader = pa.ipc.open_stream(buf)
            if writer is None:
                # get schema
                schema = {}
                for field in buf_reader.schema:
                    schema[field.name] = orc_type(field.type)
                writer = pyorc.Writer(f, pyorc.Struct(**schema))
            while True:
                try:
                    batch = buf_reader.read_next_batch()
                except StopIteration:
                    break
                df = batch.to_pandas()
                writer.writerows(df.itertuples(False, None))


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "usage: ./write_hdfs_orc <ipc_socket> <path> <stream id> <storage_options> <write_options> <proc_num> <proc_index>"
        )
        exit(1)
    ipc_socket = sys.argv[1]
    path = sys.argv[2]
    stream_id = sys.argv[3]
    storage_options = json.loads(
        base64.b64decode(sys.argv[4].encode("utf-8")).decode("utf-8")
    )
    write_options = json.loads(
        base64.b64decode(sys.argv[5].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[6])
    proc_index = int(sys.argv[7])
    write_orc(ipc_socket, path, stream_id, storage_options, write_options, proc_num, proc_index)
