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
import pyorc
import pyarrow as pa
import sys
import json

from vineyard.io.dataframe import DataframeStreamBuilder

def orc_type(field):
    if pa.is_int(field):
        return 'int'
    elif pa.is_string(field):
        return 'string'
    raise ValueError('Cannot Convert %s' % field)

def write_local_orc(stream_id, path, vineyard_socket):
    client = vineyard.connect(vineyard_socket)
    stream = client.get(stream_id)
    reader = stream.open_reader(client)

    writer = None
    with open(path, 'wb') as f:
        while buf := reader.next():
            buf_reader = pa.ipc.open_stream(buf)
            if writer is None:
                #get schema
                schema = {}
                for field in buf_reader.schema():
                    schema[field.name] = orc_type(field.type)
                writer = pyorc.Writer(f, pyorc.Struct(**schema))
            for batch in buf_reader:
                df = batch.to_pandas()
                writer.writerows(df.itertuples())


if __name__ == '__main__':
    write_local_orc(sys.argv[1], sys.argv[2], sys.argv[3])