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

import json
import logging
import traceback
from typing import Dict
from typing import List
from urllib.parse import urlparse

from .._C import ObjectID
from .._C import ObjectMeta
from .._C import StreamDrainedException
from ..core.driver import registerize
from ..core.resolver import resolver_context

logger = logging.getLogger('vineyard')


@registerize
def read(path, *args, handlers=None, **kwargs):
    """Open a path and read it as a single stream.

    Parameters
    ----------
    path: str
        Path to read, the last reader registered for the scheme of
        the path will be used.
    vineyard_ipc_socket: str
        The local or remote vineyard's IPC socket location that the
        remote readers will use to establish connections with the
        vineyard server.
    vineyard_endpoint: str, optional
        An optional address of vineyard's RPC socket, which will be
        used for retrieving server's information on the client side.
        If not provided, the `vineyard_ipc_socket` will be used, or
        it will tries to discovery vineyard's IPC or RPC endpoints
        from environment variables.
    """
    parsed = urlparse(path)
    if read.__factory and read.__factory.get(parsed.scheme):
        errors = []
        for reader in read.__factory[parsed.scheme][::-1]:
            try:
                proc_kwargs = kwargs.copy()
                r = reader(
                    path,
                    proc_kwargs.pop('vineyard_ipc_socket'),
                    *args,
                    handlers=handlers,
                    **proc_kwargs
                )
                if r is not None:
                    return r
            except Exception:  # pylint: disable=broad-except
                errors.append('%s: %s' % (reader.__name__, traceback.format_exc()))
        raise RuntimeError(
            'Unable to find a proper IO driver for %s, potential causes are:\n %s'
            % (path, '\n'.join(errors))
        )
    else:
        raise ValueError("No IO driver registered for %s" % path)


@registerize
def write(path, stream, *args, handlers=None, **kwargs):
    """Write the stream to a given path.

    Parameters
    ----------
    path: str
        Path to write, the last writer registered for the scheme of the path
        will be used.
    stream: vineyard stream
        Stream that produces the data to write.
    vineyard_ipc_socket: str
        The local or remote vineyard's IPC socket location that the remote
        readers will use to establish connections with the vineyard server.
    vineyard_endpoint: str, optional
        An optional address of vineyard's RPC socket, which will be used for
        retrieving server's information on the client side. If not provided,
        the `vineyard_ipc_socket` will be used, or it will tries to discovery
        vineyard's IPC or RPC endpoints from environment variables.
    """
    parsed = urlparse(path)
    if write.__factory and write.__factory.get(parsed.scheme):
        errors = []
        for writer in write.__factory[parsed.scheme][::-1]:
            try:
                proc_kwargs = kwargs.copy()
                writer(
                    path,
                    stream,
                    proc_kwargs.pop('vineyard_ipc_socket'),
                    *args,
                    handlers=handlers,
                    **proc_kwargs
                )
            except Exception:  # pylint: disable=broad-except
                errors.append('%s: %s' % (writer.__name__, traceback.format_exc()))
                continue
            else:
                return
        raise RuntimeError(
            'Unable to find a proper IO driver for %s, potential causes are:\n %s'
            % (path, '\n'.join(errors))
        )
    else:
        raise ValueError("No IO driver registered for %s" % path)


def open(path, *args, mode='r', handlers=None, **kwargs):
    """Open a path as a reader or writer, depends on the parameter :code:`mode`.
    If :code:`mode` is :code:`r`, it will open a stream for read, and open a
    stream for write when :code:`mode` is :code:`w`.

    Parameters
    ----------
    path: str
        Path to open.
    mode: char
        Mode about how to open the path, :code:`r` is for read and :code:`w` for write.
    vineyard_ipc_socket: str
        Vineyard's IPC socket location.
    vineyard_endpoint: str
        Vineyard's RPC socket address.
    handlers:
        A dict that will be filled with a :code:`handler` that contains the process
        handler of the underlying read/write process that can be joined using
        :code:`join` to capture the possible errors during the I/O proceeding.

    See Also
    --------
    vineyard.io.read
    vineyard.io.write
    """
    parsed = urlparse(path)
    if not parsed.scheme:
        path = 'file://' + path

    if mode == 'r':
        return read(path, *args, handlers=handlers, **kwargs)

    if mode == 'w':
        return write(path, *args, handlers=handlers, **kwargs)

    raise RuntimeError('Opening %s with mode %s is not supported' % (path, mode))


class BaseStream:
    class Reader:
        def __init__(self, client, stream: ObjectID, resolver=None):
            self._client = client
            self._stream = stream
            self._resolver = resolver
            self._client.open_stream(stream, 'r')

        def next(self) -> object:
            try:
                chunk = self._client.next_chunk(self._stream)
            except StreamDrainedException as e:
                raise StopIteration('No more chunks') from e

            if self._resolver is not None:
                return self._resolver(chunk)
            else:
                with resolver_context() as ctx:
                    return ctx.run(chunk)

        def next_metadata(self) -> ObjectMeta:
            try:
                return self._client.next_chunk_meta(self._stream)
            except StreamDrainedException as e:
                raise StopIteration('No more chunks') from e

        def __str__(self) -> str:
            return repr(self)

        def __repr__(self) -> str:
            return '%s of Stream <%r>' % (self.__class__, self._stream)

    class Writer:
        def __init__(self, client, stream: ObjectID):
            self._client = client
            self._stream = stream
            self._client.open_stream(stream, 'w')

        def next(self, size: int) -> memoryview:
            return self._client.new_buffer_chunk(self._stream, size)

        def append(self, chunk: ObjectID):
            return self._client.push_chunk(self._stream, chunk)

        def fail(self):
            return self._client.stop_stream(self._stream, True)

        def finish(self):
            return self._client.stop_stream(self._stream, False)

        def __str__(self) -> str:
            return repr(self)

        def __repr__(self) -> str:
            return '%s of Stream <%r>' % (self.__class__, self._stream)

    def __init__(self, meta: ObjectMeta, resolver=None):
        self._meta = meta
        self._stream = meta.id
        self._resolver = resolver

        self._reader = None
        self._writer = None

    @property
    def id(self) -> ObjectID:
        return self._stream

    @property
    def meta(self) -> ObjectMeta:
        return self._meta

    @property
    def reader(self) -> "BaseStream.Reader":
        return self.open_reader()

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return '%s <%r>' % (self.__class__.__name__, self._stream)

    def _open_new_reader(self, client) -> "BaseStream.Reader":
        '''Always open a new reader.'''
        return BaseStream.Reader(client, self.id, self._resolver)

    def open_reader(self, client=None) -> "BaseStream.Reader":
        if self._reader is None:
            if client is None:
                client = self._meta._client
            self._reader = self._open_new_reader(client)
        return self._reader

    @property
    def writer(self) -> "BaseStream.Writer":
        return self.open_writer()

    def _open_new_writer(self, client) -> "BaseStream.Writer":
        return BaseStream.Writer(client, self.id)

    def open_writer(self, client=None) -> "BaseStream.Writer":
        if self._writer is None:
            if client is None:
                client = self._meta._client
            self._writer = self._open_new_writer(client)
        return self._writer


class StreamCollection:
    """A stream collection is a set of stream, where each element is a stream, or,
    another stream collection.
    """

    KEY_OF_STREAMS = '__streams'
    KEY_OF_PATH = '__path'
    KEY_OF_GLOBAL = '__global'
    KEY_OF_OPTIONS = '__options'

    def __init__(self, meta: ObjectMeta, streams: List[ObjectID]):
        self._meta = meta
        self._streams = streams
        if StreamCollection.KEY_OF_GLOBAL in self._meta:
            self._global = self._meta[StreamCollection.KEY_OF_GLOBAL]
        else:
            self._global = False

    @staticmethod
    def new(
        client, metadata: Dict, streams: List[ObjectID], meta: ObjectMeta = None
    ) -> "StreamCollection":
        if meta is None:
            meta = ObjectMeta()
        meta['typename'] = 'vineyard::StreamCollection'
        for k, v in metadata.items():
            if k not in [
                'id',
                'signature',
                'instance_id',
                'transient',
                'global',
                'typename',
            ]:
                meta[k] = v
        meta[StreamCollection.KEY_OF_STREAMS] = [int(s) for s in streams]
        meta = client.create_metadata(meta)
        return StreamCollection(meta, streams)

    @property
    def id(self):
        return self.meta.id

    @property
    def meta(self):
        return self._meta

    @property
    def isglobal(self):
        return self._global

    @property
    def streams(self):
        return self._streams

    def __repr__(self) -> str:
        return "StreamCollection: %s [%s]" % (
            repr(self.id),
            [repr(s) for s in self.streams],
        )

    def __str__(self) -> str:
        return repr(self)


def stream_collection_resolver(obj):
    meta = obj.meta
    streams = json.loads(meta[StreamCollection.KEY_OF_STREAMS])
    return StreamCollection(meta, [ObjectID(s) for s in streams])


def register_stream_collection_types(_builder_ctx, resolver_ctx):
    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::StreamCollection', stream_collection_resolver)


__all__ = [
    'open',
    'read',
    'write',
    'BaseStream',
    'StreamCollection',
    'register_stream_collection_types',
]
