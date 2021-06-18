#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Alibaba Group Holding Limited.
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

'''
Pickle support for arbitrary vineyard objects.
'''

from io import BytesIO

import pickle

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle


class PickledReader:
    ''' Serialize a python object in zero-copy fashion and provides a bytes-like
        read interface.

        How do we keep the round-trip between the reader and writer:

        + for bytes/memoryview:

            - in reader: read it as the only chunk
            - in writer: write it as a blob

        + for other type of objects:

            - add a special `__VINEYARD__` tag at the beginning
    '''
    def __init__(self, value):
        self._value = value

        if isinstance(value, (bytes, memoryview)):
            self._buffers = [memoryview(value)]
            self._store_size = len(value)

        else:
            self._buffers = [None]
            self._store_size = 0

            buffers = []
            bs = pickle.dumps(value, protocol=5, fix_imports=True, buffer_callback=buffers.append)

            meta = BytesIO()
            meta.write(b'__VINEYARD__')
            self._poke_int(meta, len(buffers))
            ks = [len(buffers)]
            for buf in buffers:
                raw = buf.raw()
                self._buffers.append(raw)
                self._store_size += len(raw)
                self._poke_int(meta, len(raw))
                ks.append(len(raw))

            self._poke_int(meta, len(bs))
            meta.write(bs)
            self._buffers[0] = memoryview(meta.getbuffer())
            self._store_size += len(self._buffers[0])

        self._chunk_index = 0
        self._chunk_offset = 0

    @property
    def value(self):
        return self._value

    @property
    def store_size(self):
        return self._store_size

    def _poke_int(self, bs, value):
        bs.write(int.to_bytes(value, length=8, byteorder='big'))

    def read(self, block_size):
        if block_size == -1:
            block_size = 2**30
        if self._chunk_index >= len(self._buffers):  # may read again when it reach EOF.
            return b''
        if self._chunk_offset == len(self._buffers[self._chunk_index]):
            self._chunk_index += 1
            self._chunk_offset = 0
        if self._chunk_index >= len(self._buffers):
            return b''
        chunk = self._buffers[self._chunk_index]
        offset = self._chunk_offset
        next_offset = min(len(chunk), offset + block_size)
        result = self._buffers[self._chunk_index][offset:next_offset]
        self._chunk_offset += len(result)
        return result


class PickledWriter:
    ''' Deserialize a pickled bytes into a python object in zero-copy fashion.
    '''
    def __init__(self, store_size=-1):
        if store_size > 0:
            # optimization: reserve space for incoming contents
            self._buffer = BytesIO(initial_bytes=b'\x00' * store_size)
        else:
            self._buffer = BytesIO()
        self._buffer.seek(0)
        self._value = None

    @property
    def value(self):
        return self._value

    def write(self, bs):
        self._buffer.write(bs)

    def close(self):
        bs = self._buffer.getbuffer()
        buffers = []
        buffer_sizes = []
        try:
            offset, _ = self._peek_any(bs, 0, b'__VINEYARD__')
        except AssertionError:
            self._value = bs
            return
        offset, nbuffers = self._peek_int(bs, offset)
        for _ in range(nbuffers):
            offset, sz = self._peek_int(bs, offset)
            buffer_sizes.append(sz)
        offset, metalen = self._peek_int(bs, offset)
        offset, meta = self._peek_buffer(bs, offset, metalen)
        for nlen in buffer_sizes:
            offset, block = self._peek_buffer(bs, offset, nlen)
            buffers.append(block)
        self._value = pickle.loads(meta, fix_imports=True, buffers=buffers)

    def _peek_any(self, bs, offset, target):
        value = bs[offset:offset + len(target)]
        assert value == target, "Unexpected bytes: " + str(value)
        return offset + len(target), value

    def _peek_int(self, bs, offset):
        value = int.from_bytes(bs[offset:offset + 8], byteorder='big')
        return offset + 8, value

    def _peek_buffer(self, bs, offset, size):
        value = bs[offset:offset + size]
        return offset + size, value


__all__ = ['PickledReader', 'PickledWriter']
