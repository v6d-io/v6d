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

import vineyard

import pyarrow as pa

from urllib.parse import urlparse
from vineyard.io.dataframe import DataframeStreamBuilder


def read_vineyard_dataframe(vineyard_socket, path, proc_num, proc_index):
    client = vineyard.connect(vineyard_socket)
    builder = DataframeStreamBuilder(client)

    header_row = False
    fragments = urlparse(path).fragment.split('&')

    name = urlparse(path).netloc

    for frag in fragments:
        try:
            k, v = frag.split('=')
        except:
            pass
        else:
            if k == 'header_row':
                header_row = (v.upper() == 'TRUE')
                if header_row:
                    builder[k] = '1'
                else:
                    builder[k] = '0'
            elif k == 'delimiter':
                builder[k] = bytes(v, "utf-8").decode("unicode_escape")

    stream = builder.seal(client)
    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    writer = stream.open_writer(client)

    df_id = client.get_name(name)
    dataframes = client.get(df_id)

    for df in dataframes:
        rb = pa.RecordBatch.from_pandas(df)
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
        print('usage: ./read_vineyard_dataframe <ipc_socket> <vineyard_address> <proc num> <proc index>')
        exit(1)
    ipc_socket = sys.argv[1]
    vineyard_address = sys.argv[2]
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    read_vineyard_dataframe(ipc_socket, vineyard_address, proc_num, proc_index)
