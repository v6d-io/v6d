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
import io
import json

from vineyard.io.byte import ByteStreamBuilder


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
        raise ValueError('Cannot Convert %s' % field)


def parse_dataframe(vineyard_socket, stream_id, proc_num, proc_index):
    client = vineyard.connect(vineyard_socket)
    instream = client.get(stream_id)[0]
    stream_reader = instream.open_reader(client)

    builder = ByteStreamBuilder(client)
    stream = builder.seal(client)
    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    stream_writer = stream.open_writer(client)

    writer = None
    while True:
        try:
            content = stream_reader.next()
        except:
            stream_writer.finish()
            break
        buf_reader = pa.ipc.open_stream(content)
        b = io.BytesIO()
        schema = {}
        for field in buf_reader.schema:
            schema[field.name] = orc_type(field.type)
        writer = pyorc.Writer(b, pyorc.Struct(**schema))
        for batch in buf_reader:
            df = batch.to_pandas()
            writer.writerows(df.itertuples(False, None))
        writer.close()
        buf = b.getvalue()
        chunk = stream_writer.next(len(buf))
        buf_writer = pa.FixedSizeBufferWriter(chunk)
        buf_writer.write(buf)
        buf_writer.close()


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: ./parse_orc_to_dataframe <ipc_socket> <stream_id> <proc_num> <proc_index>')
        exit(0)
    ipc_socket = sys.argv[1]
    stream_id = sys.argv[2]
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    parse_dataframe(ipc_socket, stream_id, proc_num, proc_index)
