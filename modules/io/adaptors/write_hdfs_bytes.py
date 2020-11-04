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


def write_hdfs_bytes(stream_id, path, vineyard_socket):
    client = vineyard.connect(vineyard_socket)
    stream = client.get(stream_id)[0]
    #stream = client.get_object(vineyard.ObjectID(stream_id))
    reader = stream.open_reader(client)

    host, port = urlparse(path).netloc.split(':')
    hdfs = HDFileSystem(host=host, port=int(port))
    path = urlparse(path).path

    with hdfs.open(path, 'wb') as f:
        while True:
            try:
                buf = reader.next()
            except:
                f.close()
                break
            f.write(bytes(memoryview(buf)))


if __name__ == '__main__':
    write_hdfs_bytes(sys.argv[1], sys.argv[2], sys.argv[3])
