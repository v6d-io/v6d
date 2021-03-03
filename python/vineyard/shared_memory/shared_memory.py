#! /usr/bin/env python3
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

'''
The shared_memory module provides similar interface like multiprocessing.shared_memory
for direct access shared memory backed by vineyard across processes.

The API is kept consistent with multiprocessing.shared_memory but the semantics is slightly
different. For vineyard, to make the shared memory visible for other process, a explictly
``seal`` or ``close`` operation is needed.

Refer to the documentation of multiprocessing.shared_memory for details.
'''

import multiprocessing.shared_memory
import struct
import types
import warnings

from vineyard._C import ObjectID


class SharedMemory:
    def __init__(self, vineyard_client, name=None, create=False, size=0):
        ''' Create or obtain a shared memory block that backed by vineyard.

            Parameters
            ----------
            vineyard_client:
                The vineyard IPC or RPC client.
            name:
                The vineyard ObjectID, could be vineyard.ObjectID, int or stringified ObjectID.
            create:
                Whether to create a new shared memory block or just obtain existing one.
            size:
                Size of the shared memory block.

            See Also
            --------
            multiprocessing.shared_memory.SharedMemory
        '''
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create:
            if size == 0:
                raise ValueError("'size' must be a positive number different from zero")
            if name is not None:
                raise ValueError("'name' can only be None if create=True")
        else:
            if size != 0:
                warnings.warn(
                    "'size' will take no effect if create=False",
                )
            if name is None:
                raise ValueError("'name' cannot be None if create=False")


        self._name = None
        self._size = None
        self._buf = None
        self._blob, self._blob_builder = None, None
        self._vineyard_client = vineyard_client

        if create:
            self._blob_builder = vineyard_client.create_blob(size)
            self._name = self._blob_builder.id
            self._size = size
            self._buf = memoryview(self._blob_builder)
        else:
            self._blob = vineyard_client.get_object(ObjectID(name))
            self._name = self._blob.id
            self._size = self._blob.size
            self._buf = memoryview(self._blob)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.name,
                False,
                self.size,
            ),
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

    @property
    def buf(self):
        "A memoryview of contents of the shared memory block."
        return self._buf

    @property
    def name(self):
        "Unique name that identifies the shared memory block."
        return repr(self._name)

    @property
    def size(self):
        "Size in bytes."
        return self._size

    def seal(self):
        "Seal the shared memory to make it visible for other processes."
        if self._blob_builder:
            self._blob = self._blob_builder.seal(self._vineyard_client)
            self._blob_builder = None
        return self

    def close(self):
        self.seal()

    def unlink(self):
        """Requests that the underlying shared memory block be destroyed."""
        if self._blob:
            self._vineyard_client.delete(self._blob.id)
            self._blob = None
        else:
            self._blob_builder.abort(self._vineyard_client)
            self._blob_builder = None
        self._buf = None


_encoding = "utf8"


class ShareableList(multiprocessing.shared_memory.ShareableList):
    '''
    Pattern for a mutable list-like object shareable via a shared
    memory block.  It differs from the built-in list type in that these
    lists can not change their overall length (i.e. no append, insert,
    etc.)

    Because values are packed into a memoryview as bytes, the struct
    packing format for any storable value must require no more than 8
    characters to describe its format.

    The ShareableList in vineyard differs slightly with its equivalent
    in the multiprocessing.shared_memory.ShareableList, as it becomes
    immutable after obtaining from the vineyard backend.

    See Also
    --------
    multiprocessing.shared_memory.ShareableList
    '''

    def __init__(self, vineyard_client, sequence=None, *, name=None):
        if name is None or sequence is not None:
            if name is not None:
                warnings.warn(
                    "'name' will take no effect as we are going to create a ShareableList",
                )

            sequence = sequence or ()
            _formats = [
                self._types_mapping[type(item)]
                    if not isinstance(item, (str, bytes))
                    else self._types_mapping[type(item)] % (
                        self._alignment * (len(item) // self._alignment + 1),
                    )
                for item in sequence
            ]
            self._list_len = len(_formats)
            assert sum(len(fmt) <= 8 for fmt in _formats) == self._list_len
            offset = 0
            # The offsets of each list element into the shared memory's
            # data area (0 meaning the start of the data area, not the start
            # of the shared memory area).
            self._allocated_offsets = [0]
            for fmt in _formats:
                offset += self._alignment if fmt[-1] != "s" else int(fmt[:-1])
                self._allocated_offsets.append(offset)
            _recreation_codes = [
                self._extract_recreation_code(item) for item in sequence
            ]
            requested_size = struct.calcsize(
                "q" + self._format_size_metainfo +
                "".join(_formats) +
                self._format_packing_metainfo +
                self._format_back_transform_codes
            )

            self.shm = SharedMemory(vineyard_client, create=True, size=requested_size)
        else:
            self.shm = SharedMemory(vineyard_client, name)

        if sequence is not None:
            _enc = _encoding
            struct.pack_into(
                "q" + self._format_size_metainfo,
                self.shm.buf,
                0,
                self._list_len,
                *(self._allocated_offsets)
            )
            struct.pack_into(
                "".join(_formats),
                self.shm.buf,
                self._offset_data_start,
                *(v.encode(_enc) if isinstance(v, str) else v for v in sequence)
            )
            struct.pack_into(
                self._format_packing_metainfo,
                self.shm.buf,
                self._offset_packing_formats,
                *(v.encode(_enc) for v in _formats)
            )
            struct.pack_into(
                self._format_back_transform_codes,
                self.shm.buf,
                self._offset_back_transform_codes,
                *(_recreation_codes)
            )

        else:
            self._list_len = len(self)  # Obtains size from offset 0 in buffer.
            self._allocated_offsets = list(
                struct.unpack_from(
                    self._format_size_metainfo,
                    self.shm.buf,
                    1 * 8
                )
            )

    def freeze(self):
        ''' Make the shareable list immutable and visible for other vineyard clients.
        '''
        self.shm.seal()

    __class_getitem__ = classmethod(types.GenericAlias)


__all__ = [ 'SharedMemory', 'ShareableList' ]
