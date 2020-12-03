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

from hdfs3 import HDFileSystem
import pyarrow as pa

from vineyard.io.byte import ByteStreamBuilder


def read_hdfs_bytes(vineyard_socket, path, proc_num, proc_index):
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
            elif k == 'include_all_columns':
                if v.upper() == 'TRUE':
                    builder[k] = '1'
                else:
                    builder[k] = '0'
            else:
                builder[k] = v

    offset = 0
    length = 1024 * 1024

    header_line = hdfs.read_block(path, 0, 1, b'\n')
    builder['header_line'] = header_line.decode('unicode_escape')
    if header_row:
        offset = len(header_line)

    stream = builder.seal(client)

    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    writer = stream.open_writer(client)

    total_size = hdfs.info(path)['size']
    begin = (total_size - offset) // proc_num * proc_index + offset
    end = (total_size - offset) // proc_num + begin
    if proc_index + 1 == proc_num:
        end = total_size
    if proc_index:
        begin = next_delimiter(hdfs, path, begin, end, b'\n')
    else:
        begin -= int(header_row)

    offset = begin
    while offset < end:
        buf = hdfs.read_block(path, offset, min(length, end - offset), b'\n')
        size = len(buf)
        if not size:
            break
        offset += size - 1
        chunk = writer.next(size)
        buf_writer = pa.FixedSizeBufferWriter(chunk)
        buf_writer.write(buf)
        buf_writer.close()

    writer.finish()


def next_delimiter(hdfs, path, begin, end, delimiter):
    length = 1024
    while begin < end:
        buf = hdfs.read_block(path, begin, length)
        if delimiter not in buf:
            begin += length
        else:
            begin += buf.find(delimiter)
            break
    return begin


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: ./read_hdfs_bytes <ipc_socket> <hdfs path> <proc num> <proc index>')
        exit(1)
    ipc_socket = sys.argv[1]
    hdfs_path = sys.argv[2]
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    read_hdfs_bytes(ipc_socket, hdfs_path, proc_num, proc_index)
