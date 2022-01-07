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

''' This module exposes support for ByteStream, that use can used like:

.. code:: python

    # create a builder, then seal it as stream
    >>> stream = ByteStream.new(client)
    >>> stream
    ByteStream <o0001e09ddd98fd70>

    # use write to put chunks
    >>> writer = stream.open_writer(client)
    >>> chunk = reader.next(1024)
    >>> chunk
    <memory at 0x136ca2ac0>
    >>> len(chunk)
    1024
    >>> chunk.readonly
    False
    >>> vineyard.memory_copy(chunk, offset=0, src=b'abcde')

    # mark the stream as finished
    >>> writer.finish()

    # open a reader
    >>> reader = stream.open_reader(client)
    >>> chunk = reader.next()
    >>> chunk
    <memory at 0x136d207c0>
    >>> len(chunk)
    1234
    >>> chunk.readonly
    True
    >>> bytes(chunk[:10])
    b'abcde\x00\x00\x00\x00\x00'

    # the reader reaches the end of the stream
    >>> chunk = reader.next()
    ---------------------------------------------------------------------------
    StreamDrainedException                    Traceback (most recent call last)
    ~/libvineyard/python/vineyard/io/byte.py in next(self)
        108
    --> 109         def next(self) -> memoryview:
        110             try:

    StreamDrainedException: Stream drain: Stream drained: no more chunks

    The above exception was the direct cause of the following exception:

    StopIteration                             Traceback (most recent call last)
    <ipython-input-11-d8809de11870> in <module>
    ----> 1 chunk = reader.next()

    ~/libvineyard/python/vineyard/io/byte.py in next(self)
        109         def next(self) -> memoryview:
        110             try:
    --> 111                 return self._client.next_buffer_chunk(self._stream)
        112             except StreamDrainedException as e:
        113                 raise StopIteration('No more chunks') from e

    StopIteration: No more chunks
'''

import json
from io import BytesIO
from typing import Dict

from .._C import ObjectID
from .._C import ObjectMeta
from .._C import StreamDrainedException
from .._C import memory_copy
from .stream import BaseStream


class ByteStream(BaseStream):
    def __init__(self, meta: ObjectMeta, params: Dict = None):
        super().__init__(meta)
        self._params = params

    @property
    def params(self):
        return self._params

    @staticmethod
    def new(client, params: Dict = None, meta: ObjectMeta = None) -> "ByteStream":
        if meta is None:
            meta = ObjectMeta()
        meta['typename'] = 'vineyard::ByteStream'
        if params is None:
            params = dict()
        meta['params_'] = params
        meta = client.create_metadata(meta)
        client.create_stream(meta.id)
        return ByteStream(meta, params)

    class Reader(BaseStream.Reader):
        def __init__(self, client, stream: ObjectID):
            self._client = client
            self._stream = stream
            self._client.open_stream(stream, 'r')

        def next(self) -> memoryview:
            try:
                return self._client.next_buffer_chunk(self._stream)
            except StreamDrainedException as e:
                raise StopIteration('No more chunks') from e

    class Writer(BaseStream.Writer):
        def __init__(self, client, stream: ObjectID):
            self._client = client
            self._stream = stream
            self._client.open_stream(stream, 'w')

            self._buffer_size_limit = 1024 * 1024 * 256
            self._buffer = BytesIO()

        @property
        def buffer_size_limit(self):
            return self._buffer_size_limit

        @buffer_size_limit.setter
        def buffer_size_limit(self, value: int):
            self._buffer_size_limit = value

        def next(self, size: int) -> memoryview:
            return self._client.new_buffer_chunk(self._stream, size)

        def write(self, data: bytes):
            self._buffer.write(data)
            self._try_flush_buffer()

        def _try_flush_buffer(self, force=False):
            view = self._buffer.getbuffer()
            if len(view) >= self._buffer_size_limit or (force and len(view) > 0):
                chunk = self.next(len(view))
                memory_copy(chunk, 0, view)
                self._buffer = BytesIO()

        def finish(self):
            self._try_flush_buffer(True)
            return self._client.stop_stream(self._stream, False)

    def _open_new_reader(self, client):
        return ByteStream.Reader(client, self.id)

    def _open_new_writer(self, client):
        return ByteStream.Writer(client, self.id)


def byte_stream_resolver(obj):
    meta = obj.meta
    if 'params_' in meta:
        params = json.loads(meta['params_'])
    else:
        params = dict
    return ByteStream(obj.meta, params)


def register_byte_stream_types(builder_ctx, resolver_ctx):
    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::ByteStream', byte_stream_resolver)
