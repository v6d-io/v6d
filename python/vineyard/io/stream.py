#! /usr/bin/env python
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

from urllib.parse import urlparse

import vineyard
from ..core.driver import registerize


@registerize
def read(path, *args, **kwargs):
    ''' Open a path and read it as a single stream.

        Parameter
        ---------
        path: str
            Path to read, the last reader registered for the scheme of the path will be used.
        vineyard_ipc_socket: str
            The local or remote vineyard's IPC socket location that the remote readers will
            use to establish connections with the vineyard server.
    '''
    parsed = urlparse(path)

    for reader in read.__factory[parsed.scheme][::-1]:
        try:
            r = reader(path[len(parsed.scheme) + 3:], kwargs.pop('vineyard_ipc_socket'), *args, **kwargs)
            if r is not None:
                return r
            else:
                continue
        except:
            continue

    raise RuntimeError('Unable to find a proper IO driver for %s' % path)


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
    '''
    parsed = urlparse(path)

    for writer in write.__factory[parsed.scheme][::-1]:
        try:
            writer(path[len(parsed.scheme) + 3:], stream, kwargs.pop('vineyard_ipc_socket'), *args, **kwargs)
        except:
            continue
        else:
            return

    raise RuntimeError('Unable to find a proper IO driver for %s' % path)


def open(path, *args, mode='r', **kwargs):
    ''' Open a path as a reader or writer, depends on the parameter :code:`mod`.

        Parameters
        ----------
        path: str
            Path to open.
        mode: char
            Mode about how to open the path, :code:`r` is for read and :code:`w` for write.
    '''
    if mode == 'r':
        return read(path, *args, **kwargs)

    if mode == 'w':
        return write(path, *args, **kwargs)

    raise RuntimeError('Opening %s with mode %s is not supported' % (path, mode))


__all__ = ['open', 'read', 'write']
