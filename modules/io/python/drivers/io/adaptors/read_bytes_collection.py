#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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

import base64
import json
import logging
import multiprocessing
import os
import sys
from queue import Empty as QueueEmptyException
from queue import Queue as ConcurrentQueue
from typing import Dict
from typing import Tuple  # pylint: disable=unused-import

import fsspec
from fsspec.core import split_protocol
from fsspec.spec import AbstractFileSystem
from fsspec.utils import read_block

import vineyard
from vineyard.io.byte import ByteStream
from vineyard.io.stream import StreamCollection
from vineyard.io.utils import BaseStreamExecutor
from vineyard.io.utils import ThreadStreamExecutor
from vineyard.io.utils import expand_full_path
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success

try:
    from vineyard.drivers.io import ossfs
except ImportError:
    ossfs = None

if ossfs:
    fsspec.register_implementation("oss", ossfs.OSSFileSystem)

logger = logging.getLogger('vineyard')

CHUNK_SIZE = 1024 * 1024 * 128


def read_metadata(fs: AbstractFileSystem, path: str) -> Dict:
    logger.info('start reading metadata at %s', path)
    with fs.open(path, mode="rb") as f:
        return json.loads(f.read().decode('utf-8', errors='ignore'))


def read_byte_stream(
    client,
    fs: AbstractFileSystem,
    stream: ByteStream,
    path: str,
    chunk_size: int = CHUNK_SIZE,
):
    logger.info('start reading blob at %s', path)
    with fs.open(path, mode="rb") as f:
        try:
            total_size = f.size()
        except TypeError:
            total_size = f.size

        writer = stream.open_writer(client)
        try:
            begin, end = 0, total_size
            while begin < end:
                buffer = read_block(f, begin, min(chunk_size, end - begin))
                if len(buffer) > 0:
                    chunk = writer.next(len(buffer))
                    vineyard.memory_copy(chunk, 0, buffer)
                    begin += len(buffer)
        except Exception:  # pylint: disable=broad-except
            report_exception()
            writer.fail()
            sys.exit(-1)

        writer.finish()
        return total_size


class ReadToByteStreamExecutor(BaseStreamExecutor):
    def __init__(
        self,
        client,
        fs: AbstractFileSystem,
        task_queue: "ConcurrentQueue[Tuple[ByteStream, str]]",
        chunk_size: int = CHUNK_SIZE,
    ):
        self._client = client
        self._fs = fs
        self._task_queue = task_queue
        self._chunk_size = chunk_size

    def execute(self):
        processed_blobs, processed_bytes = 0, 0
        while True:
            try:
                stream, path = self._task_queue.get(block=False)
            except QueueEmptyException:
                break

            total_size = read_byte_stream(
                self._client, self._fs, stream, path, self._chunk_size
            )
            processed_blobs += 1
            processed_bytes += total_size

        return processed_blobs, processed_bytes


def read_stream_collections(
    client,
    fs: AbstractFileSystem,
    queue: "ConcurrentQueue[Tuple[ByteStream, str]]",
    base_prefix: str,
    prefix: str,
):
    metadata_path = os.path.join(prefix, 'metadata.json')
    blob_path = os.path.join(prefix, 'blob')
    if fs.exists(metadata_path):
        metadata = read_metadata(fs, metadata_path)
        streams = []
        for path in fs.listdir(prefix):
            if path['type'] == 'directory':
                streams.append(
                    read_stream_collections(
                        client, fs, queue, base_prefix, path['name']
                    )
                )
        stream_collection = StreamCollection.new(client, metadata, streams)
        return stream_collection.id
    else:
        # make a blob
        with fs.open(blob_path, 'rb') as f:
            options_path = blob_path + '.meta.json'
            if fs.exists(options_path):
                with fs.open(options_path, 'r') as f:
                    params = json.loads(f.read())
                total_size = params['length']
                options = params.get(StreamCollection.KEY_OF_OPTIONS, '{}')
            else:
                try:
                    total_size = f.size()
                except TypeError:
                    total_size = f.size
                options = '{}'
            # create a stream
            stream = ByteStream.new(
                client,
                params={
                    StreamCollection.KEY_OF_PATH: os.path.relpath(
                        blob_path, base_prefix
                    ),
                    'length': total_size,
                    StreamCollection.KEY_OF_OPTIONS: options,
                },
            )
            queue.put((stream, blob_path))
            return stream.id


def read_bytes_collection(
    vineyard_socket, prefix, storage_options, proc_num, proc_index
):
    """Read a set of files as a collection of ByteStreams."""
    client = vineyard.connect(vineyard_socket)

    protocol, prefix_path = split_protocol(prefix)
    fs = fsspec.filesystem(protocol, **storage_options)

    worker_prefix = os.path.join(prefix_path, '%s-%s' % (proc_num, proc_index))

    logger.info("start creating blobs ...")
    queue: "ConcurrentQueue[Tuple[ByteStream, str]]" = ConcurrentQueue()
    stream_id = read_stream_collections(client, fs, queue, worker_prefix, worker_prefix)

    client.persist(stream_id)
    report_success(stream_id)

    logger.info("start reading blobs ...")
    executor = ThreadStreamExecutor(
        ReadToByteStreamExecutor,
        parallism=multiprocessing.cpu_count(),
        client=client,
        fs=fs,
        task_queue=queue,
        chunk_size=CHUNK_SIZE,
    )
    executor.execute()


def main():
    if len(sys.argv) < 6:
        print(
            "usage: ./read_bytes_collection <ipc_socket> <prefix> <storage_options> "
            "<proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    prefix = expand_full_path(sys.argv[2])
    storage_options = json.loads(
        base64.b64decode(sys.argv[3].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[4])
    proc_index = int(sys.argv[5])
    read_bytes_collection(ipc_socket, prefix, storage_options, proc_num, proc_index)


if __name__ == "__main__":
    main()
