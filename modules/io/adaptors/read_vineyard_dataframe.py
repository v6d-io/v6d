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
from urllib.parse import urlparse

import pyarrow as pa
import vineyard
from vineyard.io.dataframe import DataframeStreamBuilder


def read_vineyard_dataframe(
    vineyard_socket, path, storage_options, read_options, proc_num, proc_index
):
    client = vineyard.connect(vineyard_socket)
    builder = DataframeStreamBuilder(client)
    if storage_options:
        raise ValueError("Read vineyard current not support storage options")
    builder["header_row"] = "1" if read_options.get("header_row", False) else "0"
    builder["delimiter"] = bytes(read_options.get("delimiter", ","), "utf-8").decode(
        "unicode_escape"
    )

    stream = builder.seal(client)
    client.persist(stream)
    ret = {"type": "return", "content": repr(stream.id)}
    print(json.dumps(ret), flush=True)

    name = urlparse(path).netloc
    # the "name" part in URL can be a name, or an ObjectID for convenience.
    try:
        df_id = client.get_name(name)
    except:
        df_id = vineyard.ObjectID(name)
    dataframes = client.get(df_id)

    writer = stream.open_writer(client)
    for df in dataframes:
        rb = pa.RecordBatch.from_pandas(df)
        sink = pa.BufferOutputStream()
        rb_writer = pa.ipc.new_stream(sink, rb.schema)
        rb_writer.write_batch(rb)
        rb_writer.close()
        buf = sink.getvalue()
        chunk = writer.next(buf.size)
        buf_writer = pa.FixedSizeBufferWriter(pa.py_buffer(chunk))
        buf_writer.write(buf)
        buf_writer.close()

    writer.finish()


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "usage: ./read_vineyard_dataframe <ipc_socket> <vineyard_address> <storage_options> <read_options> <proc num> <proc index>"
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
    read_vineyard_dataframe(
        ipc_socket, path, storage_options, read_options, proc_num, proc_index
    )
