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

import json
import sys
from urllib.parse import urlparse

import vineyard

import pyorc
import pyarrow as pa

from vineyard.io.dataframe import DataframeStreamBuilder


def arrow_type(field):
    if field.name == 'decimal':
        return pa.decimal128(field.precision)
    elif field.name == 'uniontype':
        return pa.union(field.cont_types)
    elif field.name == 'array':
        return pa.list_(field.type)
    elif field.name == 'map':
        return pa.map_(field.key, field.value)
    elif field.name == 'struct':
        return pa.struct(field.fields)
    else:
        types = {
            'boolean': pa.bool_(),
            'tinyint': pa.int8(),
            'smallint': pa.int16(),
            'int': pa.int32(),
            'bigint': pa.int64(),
            'float': pa.float32(),
            'double': pa.float64(),
            'string': pa.string(),
            'char': pa.string(),
            'varchar': pa.string(),
            'binary': pa.binary(),
            'timestamp': pa.timestamp('ms'),
            'date': pa.date32(),
        }
        if field.name not in types:
            raise ValueError('Cannot convert to arrow type: ' + field.name)
        return types[field.name]


def read_local_orc(vineyard_socket, path, proc_num, proc_index):
    if proc_index:
        return 
    client = vineyard.connect(vineyard_socket)
    builder = DataframeStreamBuilder(client)

    stream = builder.seal(client)
    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    writer = stream.open_writer(client)

    with open(path, 'rb') as f:
        reader = pyorc.Reader(f)
        fields = reader.schema.fields
        schema = []
        for c in fields:
            schema.append((c, arrow_type(fields[c])))
        pa_struct = pa.struct(schema)
        while True:
            rows = reader.read(num=1024 * 1024)
            if not rows:
                break
            rb = pa.RecordBatch.from_struct_array(pa.array(rows, type=pa_struct))
            sink = pa.BufferOutputStream()
            rb_writer = pa.ipc.new_stream(sink, rb.schema)
            rb_writer.write_batch(rb)
            rb_writer.close()
            buf = sink.getvalue()
            chunk = writer.next(buf.size)
            buf_writer = pa.FixedSizeBufferWriter(chunk)
            buf_writer.write(buf)
            buf_writer.close()


    writer.finish()


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: ./read_local_orc <ipc_socket> <orc file path> <proc num> <proc index>')
        exit(1)
    ipc_socket = sys.argv[1]
    orc_path = sys.argv[2]
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    read_local_orc(ipc_socket, orc_path, proc_num, proc_index)
