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
import pyarrow as pa

from urllib.parse import urlparse
from hdfs3 import HDFileSystem
from vineyard.io.byte import ByteStreamBuilder


def read_hdfs_bytes(path, vineyard_socket):
    client = vineyard.connect(vineyard_socket)
    builder = ByteStreamBuilder(client)
    stream = builder.seal(client)
    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    writer = stream.open_writer(client)

    host, port = urlparse(path).netloc.split(':')
    hdfs = HDFileSystem(host=host, port=int(port))
    path = urlparse(path).path
    blocks = hdfs.get_block_locations(path)

    for b in blocks:
        buf = hdfs.read_block(path, b['offset'], b['length'])
        chunk = writer.next(b['length'])
        buf_writer = pa.FixedSizeBufferWriter(chunk)
        buf_writer.write(buf)
        buf_writer.close()

    writer.finish()


if __name__ == '__main__':
    read_hdfs_bytes(sys.argv[1], sys.argv[2])
