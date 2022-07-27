#! /usr/bin/env python
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

''' This module exposes support for DataframeStream, that use can used like:

.. code:: python

    # create a builder, then seal it as stream
    >>> stream = DataframeStream.new(client)
    >>> stream = builder.seal(client)
    >>> stream
    DataframeStream <o0001e09ddd98fd70>

    # use write to put chunks
    >>> writer = stream.open_writer(client)
    >>> writer.write_table(
            pa.Table.from_pandas(
                pd.DataFrame({"x": [1,2,3], "y": [4,5,6]})))

    # mark the stream as finished
    >>> writer.finish()

    # open a reader
    >>> reader = stream.open_reader(client)
    >>> batch = reader.next()
    >>> batch
    pyarrow.RecordBatch
    x: int64
    y: int64

    # the reader reaches the end of the stream
    >>> batch = reader.next()
    ---------------------------------------------------------------------------
    StreamDrainedException                    Traceback (most recent call last)
    ~/libvineyard/python/vineyard/io/dataframe.py in next(self)
        97             try:
    ---> 98                 buffer = self._client.next_buffer_chunk(self._stream)
        99                 with pa.ipc.open_stream(buffer) as reader:

    StreamDrainedException: Stream drain: Stream drained: no more chunks

    The above exception was the direct cause of the following exception:

    StopIteration                             Traceback (most recent call last)
    <ipython-input-11-10f09bf65f8a> in <module>
    ----> 1 batch = reader.next()

    ~/libvineyard/python/vineyard/io/dataframe.py in next(self)
        100                     return reader.read_next_batch()
        101             except StreamDrainedException as e:
    --> 102                 raise StopIteration('No more chunks') from e
        103
        104         def __str__(self) -> str:

    StopIteration: No more chunks
'''

import json
from io import BytesIO
from typing import Dict

import pyarrow as pa
import pyarrow.ipc  # pylint: disable=unused-import

from .._C import ObjectID
from .._C import ObjectMeta
from .._C import StreamDrainedException
from .._C import memory_copy
from .stream import BaseStream


class DataframeStream(BaseStream):
    def __init__(self, meta: ObjectMeta, params: Dict = None):
        super().__init__(meta)
        self._params = params

    @property
    def params(self):
        return self._params

    @staticmethod
    def new(client, params: Dict = None, meta: ObjectMeta = None) -> "DataframeStream":
        if meta is None:
            meta = ObjectMeta()
        meta['typename'] = 'vineyard::DataframeStream'
        if params is None:
            params = dict()
        meta['params_'] = params
        meta = client.create_metadata(meta)
        client.create_stream(meta.id)
        return DataframeStream(meta, params)

    class Reader(BaseStream.Reader):
        def __init__(self, client, stream: ObjectID):
            super().__init__(client, stream)

        def next(self) -> pa.RecordBatch:
            try:
                buffer = self._client.next_buffer_chunk(self._stream)
                with pa.ipc.open_stream(buffer) as reader:
                    return reader.read_next_batch()
            except StreamDrainedException as e:
                raise StopIteration('No more chunks') from e

        def read_table(self) -> pa.Table:
            batches = []
            while True:
                try:
                    batches.append(self.next())
                except StopIteration:
                    break
            return pa.Table.from_batches(batches)

    class Writer(BaseStream.Writer):
        def __init__(self, client, stream: ObjectID):
            super().__init__(client, stream)

            self._buffer = BytesIO()

        def next(self, size: int) -> memoryview:
            return self._client.new_buffer_chunk(self._stream, size)

        def write(self, batch: pa.RecordBatch):
            sink = BytesIO()
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write(batch)
            view = sink.getbuffer()
            if len(view) > 0:
                buffer = self.next(len(view))
                memory_copy(buffer, 0, view)

        def write_table(self, table: pa.Table):
            for batch in table.to_batches():
                self.write(batch)

        def finish(self):
            return self._client.stop_stream(self._stream, False)

    def _open_new_reader(self, client):
        return DataframeStream.Reader(client, self.id)

    def _open_new_writer(self, client):
        return DataframeStream.Writer(client, self.id)


def dataframe_stream_resolver(obj):
    meta = obj.meta
    if 'params_' in meta:
        params = json.loads(meta['params_'])
    else:
        params = dict
    return DataframeStream(obj.meta, params)


def register_dataframe_stream_types(_builder_ctx, resolver_ctx):
    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::DataframeStream', dataframe_stream_resolver)
