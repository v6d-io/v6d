#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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

''' This module exposes support for FixedBlobStream.
'''

import contextlib
import mmap
import os
from typing import List
from typing import Optional

from vineyard._C import InvalidException
from vineyard._C import ObjectMeta
from vineyard._C import ObjectID
from vineyard.core import context
from vineyard.io.stream import BaseStream

class FixedBlobStream(BaseStream):
    def __init__(self, meta: ObjectMeta):
        super().__init__(meta)
        self.nums_ = meta['nums']
        self.size_ = meta['size']
        self.is_remote_ = meta['is_remote']
        self.rpc_endpoint_ = meta['rpc_endpoint']
        self.stream_name_ = meta['stream_name']
        self.mmap_size = 4096
        self.error_msg_len = 256

    @staticmethod
    def new(client,
            stream_name: str,
            nums: int,
            size: int,
            is_remote: bool = False,
            rpc_endpoint: Optional[str] = "") -> "FixedBlobStream":
        meta = ObjectMeta()
        meta['typename'] = 'vineyard::FixedBlobStream'
        meta['nums'] = nums
        meta['size'] = size
        meta['is_remote'] = is_remote
        meta['rpc_endpoint'] = rpc_endpoint
        meta['stream_name'] = stream_name

        meta.id = client.create_fixed_stream(stream_name, nums, size)
        return FixedBlobStream(meta)

    class Reader(BaseStream.Reader):
        def __init__(self, stream: "FixedBlobStream"):
            self.stream_ = stream
        
        def next(self) -> object:
            raise NotImplementedError("FixedBlobStream does not support read yet.")
        
        def next_metadata(self) -> ObjectMeta:
            raise NotImplementedError("FixedBlobStream does not support read yet.")

        def activate_stream_with_offset(self, offsets: List[int]):
            self.stream_.activate_stream_with_offset(offsets)

        def abort(self) -> bool:
            return self.stream_.abort()

        def finish(self):
            self.stream_.close()
        
        def finish_and_delete(self):
            client_ = self.stream_.client_
            self.stream_.close()
            FixedBlobStream.delete(client_, self.stream_)
        
        def check_block_received(self, index:int) -> bool:
            return self.stream_.check_block_received(index)
    
    class Writer(BaseStream.Writer):
        def __init__(self, stream: "FixedBlobStream"):
            self.stream_ = stream
        
        def next(self, size: int) -> memoryview:
            raise NotImplementedError("FixedBlobStream does not support write yet.")
    
        def append(self, offset: int):
            self.stream_.push_offset_block(offset)

        def fail(self):
            raise NotImplementedError("FixedBlobStream does not support write yet.")
    
        def abort(self) -> bool:
            return self.stream_.abort()
        
        def finish(self):
            self.stream_.close()
        
        def finish_and_delete(self):
            client_ = self.stream_.client_
            self.stream_.close()
            FixedBlobStream.delete(client_, self.stream_)
        
        def check_block_received(self, index:int) -> bool:
            return self.stream_.check_block_received(index)

    def open_reader(self, client, wait: bool = False, timeout: int = 0):
        self.open(client, "r", wait, timeout)
        return FixedBlobStream.Reader(self)
    
    def open_writer(self, client):
        self.open(client, "w")
        return FixedBlobStream.Writer(self)

    def open(self,
             client,
             mode,
             wait: bool = False,
             timeout: int = 0):
        self.client_ = client
        if (self.is_remote_):
            self.recv_mem_fd_ = self.client_.vineyard_open_remote_fixed_stream_with_name(self.stream_name_, self.meta.id, self.nums_, self.size_, self.rpc_endpoint_, mode, wait, timeout)
        else:
            self.recv_mem_fd_ = self.client_.open_fixed_stream(self.meta.id, mode)
        if (self.recv_mem_fd_ < 0):
            raise ValueError("Failed to open remote fixed stream")
        try:
            self.recv_mem_ = mmap.mmap(self.recv_mem_fd_, self.mmap_size, access=mmap.ACCESS_READ)
        except Exception as e:
            self.close()
            raise e

    def activate_stream_with_offset(self, offsets: List[int]):
        if (not self.is_remote_):
            raise ValueError("The stream is not remote stream")
        self.client_.vineyard_activate_remote_fixed_stream_with_offset(self.meta.id, offsets)

    def push_offset_block(self, offsets: int):
        self.client_.push_next_stream_chunk_by_offset(self.meta.id, offsets)

    def check_block_received(self, index:int) -> bool:
        if (self.recv_mem_[self.mmap_size - 1] != 0):
            self.recv_mem_.seek(self.mmap_size - self.error_msg_len - 1)
            error_msg = self.recv_mem_.read(self.error_msg_len)
            null_byte_index = error_msg.find(b'\0')
            if null_byte_index != -1:
                error_msg = error_msg[:null_byte_index]
            else:
                error_msg = error_msg
            raise InvalidException(error_msg.decode('ascii'))

        if (index == -1):
            ret = True
            for i in range(self.nums_):
                if self.recv_mem_[i] == 0:
                    ret = False
                    break
            return ret
        elif (index < 0 or index >= self.nums_):
            raise ValueError("Invalid index")
        else:
            return self.recv_mem_[index] == 1

        
    def close(self):
        try:
            if (self.is_remote_):
                self.client_.vineyard_close_remote_fixed_stream(self.meta.id)
            else:
                self.client_.close_stream(self.meta.id)
        except Exception as e:
            print("error:", e)

        os.close(self.recv_mem_fd_)
        self.recv_mem_.close()
        self.client_ = None
    
    def abort(self) -> bool:
        if (self.is_remote_):
            return self.client_.vineyard_abort_remote_stream(self.meta.id)
        else:
            return self.client_.abort_stream(self.meta.id)

    @staticmethod
    def delete(client, fixed_blob_stream: "FixedBlobStream"):
        client.delete_stream(fixed_blob_stream.meta.id)

def fixed_blob_stream_resolver(obj, resolver):  # pylint: disable=unused-argument
    meta = obj.meta
    return FixedBlobStream(meta)


def register_fixed_blob_stream_types(_builder_ctx, resolver_ctx):
    if resolver_ctx is not None:
        resolver_ctx.register(
            'vineyard::FixedBlobStream', fixed_blob_stream_resolver
        )


@contextlib.contextmanager
def recordbatch_stream_context():
    with context() as (builder_ctx, resolver_ctx):
        register_fixed_blob_stream_types(builder_ctx, resolver_ctx)
        yield builder_ctx, resolver_ctx
