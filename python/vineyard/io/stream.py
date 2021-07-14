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

import logging
import traceback
from urllib.parse import urlparse

import vineyard
from ..core.driver import registerize

logger = logging.getLogger('vineyard')


@registerize
def read(path, *args, **kwargs):
    ''' Open a path and read it as a single stream.

        Parameters
        ----------
        path: str
            Path to read, the last reader registered for the scheme of the path will be used.
        vineyard_ipc_socket: str
            The local or remote vineyard's IPC socket location that the remote readers will
            use to establish connections with the vineyard server.
        vineyard_endpoint: str, optional
            An optional address of vineyard's RPC socket, which will be used for retrieving server's
            information on the client side. If not provided, the `vineyard_ipc_socket` will be used,
            or it will tries to discovery vineyard's IPC or RPC endpoints from environment variables.
    '''
    parsed = urlparse(path)
    if read.__factory and read.__factory.get(parsed.scheme):
        errors = []
        for reader in read.__factory[parsed.scheme][::-1]:
            try:
                proc_kwargs = kwargs.copy()
                r = reader(path, proc_kwargs.pop('vineyard_ipc_socket'), *args, **proc_kwargs)
                if r is not None:
                    return r
            except Exception as e:
                errors.append('%s: %s' % (reader.__name__, traceback.format_exc()))
        raise RuntimeError('Unable to find a proper IO driver for %s, potential causes are:\n %s' %
                           (path, '\n'.join(errors)))
    else:
        raise ValueError("No IO driver registered for %s" % path)


@registerize
def write(path, stream, *args, **kwargs):
    ''' Write the stream to a given path.

        Parameters
        ----------
        path: str
            Path to write, the last writer registered for the scheme of the path will be used.
        stream: vineyard stream
            Stream that produces the data to write.
        vineyard_ipc_socket: str
            The local or remote vineyard's IPC socket location that the remote readers will
            use to establish connections with the vineyard server.
        vineyard_endpoint: str, optional
            An optional address of vineyard's RPC socket, which will be used for retrieving server's
            information on the client side. If not provided, the `vineyard_ipc_socket` will be used,
            or it will tries to discovery vineyard's IPC or RPC endpoints from environment variables.
    '''
    parsed = urlparse(path)
    if write.__factory and write.__factory.get(parsed.scheme):
        errors = []
        for writer in write.__factory[parsed.scheme][::-1]:
            try:
                proc_kwargs = kwargs.copy()
                writer(path, stream, proc_kwargs.pop('vineyard_ipc_socket'), *args, **proc_kwargs)
            except Exception as e:
                errors.append('%s: %s' % (writer.__name__, traceback.format_exc()))
                continue
            else:
                return
        raise RuntimeError('Unable to find a proper IO driver for %s, potential causes are:\n %s' %
                           (path, '\n'.join(errors)))
    else:
        raise ValueError("No IO driver registered for %s" % path)


def open(path, *args, mode='r', **kwargs):
    ''' Open a path as a reader or writer, depends on the parameter :code:`mode`. If :code:`mode`
        is :code:`r`, it will open a stream for read, and open a stream for write when :code:`mode`
        is :code:`w`.

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

        See Also
        --------
        vineyard.io.read
        vineyard.io.write
    '''
    parsed = urlparse(path)
    if not parsed.scheme:
        path = 'file://' + path

    if mode == 'r':
        return read(path, *args, **kwargs)

    if mode == 'w':
        return write(path, *args, **kwargs)

    raise RuntimeError('Opening %s with mode %s is not supported' % (path, mode))


__all__ = ['open', 'read', 'write']
