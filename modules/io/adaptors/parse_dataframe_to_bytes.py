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

import vineyard
import pyarrow as pa
import sys
import io
import json

from urllib.parse import urlparse
from vineyard.io.byte import ByteStreamBuilder

def parse_dataframe(vineyard_socket, stream_id, path, proc_num, proc_index):
    client = vineyard.connect(vineyard_socket)
    streams = client.get(stream_id)
    if len(streams) != proc_num or streams[proc_index] is None:
        raise ValueError(f'Fetch stream error with proc_num={proc_num},proc_index={proc_index}')
    instream = streams[proc_index]
    stream_reader = instream.open_reader(client)

    header_row = False
    delimiter = ','
    fragments = urlparse(path).fragment.split('&')
    for frag in fragments:
        try:
            k, v = frag.split('=')
        except:
            pass
        else:
            if k == 'header_row':
                header_row = (v.upper() == 'TRUE')
            elif k == 'delimiter':
                delimiter = bytes(v, "utf-8").decode("unicode_escape")

    builder = ByteStreamBuilder(client)
    stream = builder.seal(client)
    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    stream_writer = stream.open_writer(client)
    first_write = header_row
    while True:
        try:
            content = stream_reader.next()
        except:
            stream_writer.finish()
            break
        buf_reader = pa.ipc.open_stream(content)
        for batch in buf_reader:
            df = batch.to_pandas()
            buf = df.to_csv(header=first_write, index=False, sep=delimiter).encode()
            first_write = False
            chunk = stream_writer.next(len(buf))
            buf_writer = pa.FixedSizeBufferWriter(chunk)
            buf_writer.write(buf)
            buf_writer.close()

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: ./parse_dataframe_to_bytes <ipc_socket> <stream_id> <file_path> <proc_num> <proc_index>')
        exit(1)
    ipc_socket = sys.argv[1]
    stream_id = sys.argv[2]
    file_path = sys.argv[3]
    proc_num = int(sys.argv[4])
    proc_index = int(sys.argv[5])
    parse_dataframe(ipc_socket, stream_id, file_path, proc_num, proc_index)

