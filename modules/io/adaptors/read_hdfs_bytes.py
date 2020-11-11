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

import sys
import json
from urllib.parse import urlparse

import pyarrow as pa
from hdfs3 import HDFileSystem

from vineyard.io.byte import ByteStreamBuilder


def read_hdfs_bytes(vineyard_socket, path, proc_num, proc_index):      
    if proc_index:
        return  
    client = vineyard.connect(vineyard_socket)
    builder = ByteStreamBuilder(client)

    host, port = urlparse(path).netloc.split(':')
    hdfs = HDFileSystem(host=host, port=int(port), pars={"dfs.client.read.shortcircuit": "false"})

    header_row = False
    fragments = urlparse(path).fragment.split('&')
    
    path = urlparse(path).path

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

    offset = 0
    length = 1024 * 1024

    if header_row:
        header_line = hdfs.read_block(path, 0, 1, b'\n')
        builder['header_line'] = header_line.decode('unicode_escape')
        offset = len(header_line)-1

    stream = builder.seal(client)
    
    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    writer = stream.open_writer(client)

    while True:
        buf = hdfs.read_block(path, offset, length, b'\n')
        size = len(buf)
        if not size:
            break
        offset += size
        chunk = writer.next(size)
        buf_writer = pa.FixedSizeBufferWriter(chunk)
        buf_writer.write(buf)
        buf_writer.close()

    writer.finish()


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: ./read_hdfs_bytes <ipc_socket> <hdfs path> <proc num> <proc index>')
        exit(1)
    ipc_socket = sys.argv[1]
    hdfs_path = sys.argv[2]
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    read_hdfs_bytes(ipc_socket, hdfs_path, proc_num, proc_index)
