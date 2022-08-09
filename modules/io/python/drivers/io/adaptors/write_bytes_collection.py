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

import fsspec

import vineyard
from vineyard._C import ObjectID
from vineyard.io.byte import ByteStream
from vineyard.io.stream import StreamCollection
from vineyard.io.utils import BaseStreamExecutor
from vineyard.io.utils import ThreadStreamExecutor
from vineyard.io.utils import expand_full_path
from vineyard.io.utils import report_error
from vineyard.io.utils import report_exception

try:
    from vineyard.drivers.io import ossfs
except ImportError:
    ossfs = None

if ossfs:
    fsspec.register_implementation("oss", ossfs.OSSFileSystem)

logger = logging.getLogger('vineyard')


def write_metadata(streams: StreamCollection, prefix: str, storage_options: Dict):
    metadata = dict()
    for k, v in streams.meta.items():
        metadata[k] = v
    metadata_path = os.path.join(
        prefix, metadata[StreamCollection.KEY_OF_PATH], 'metadata.json'
    )
    logger.info('creating metadata for %r ...', metadata_path)
    with fsspec.open(metadata_path, 'wb', **storage_options) as fp:
        fp.write(json.dumps(metadata).encode('utf-8'))


def write_byte_stream(client, stream: ByteStream, prefix: str, storage_options: Dict):
    path = stream.params[StreamCollection.KEY_OF_PATH]
    try:
        reader = stream.open_reader(client)
        of = fsspec.open(os.path.join(prefix, path), "wb", **storage_options)
    except Exception:  # pylint: disable=broad-except
        report_exception()
        sys.exit(-1)

    with of as f:
        while True:
            try:
                chunk = reader.next()
            except (StopIteration, vineyard.StreamDrainedException):
                break
            f.write(bytes(chunk))

    options_path = path + '.meta.json'
    with fsspec.open(os.path.join(prefix, options_path), "w", **storage_options) as f:
        f.write(json.dumps(stream.params))


class WriteBytesExecutor(BaseStreamExecutor):
    def __init__(
        self,
        client,
        prefix,
        storage_options: Dict,
        task_queue: "ConcurrentQueue[ObjectID]",
    ):
        self._client = client
        self._prefix = prefix
        self._storage_options = storage_options
        self._task_queue = task_queue

    def execute(self):
        processed_blobs, processed_bytes = 0, 0
        while True:
            try:
                s = self._task_queue.get(block=False)
            except QueueEmptyException:
                break
            stream: ByteStream = self._client.get(s)
            length = stream.params['length']
            processed_blobs += 1
            processed_bytes += length

            write_byte_stream(self._client, stream, self._prefix, self._storage_options)

        return processed_blobs, processed_bytes


def write_stream_collections(
    client,
    stream_id: ObjectID,
    blob_queue: "ConcurrentQueue[ObjectID]",
    worker_prefix: str,
    storage_options: Dict,
):
    streams = client.get(stream_id)
    if isinstance(streams, StreamCollection):
        write_metadata(streams, worker_prefix, storage_options)
        for stream in streams.streams:
            write_stream_collections(
                client, stream, blob_queue, worker_prefix, storage_options
            )
    else:
        blob_queue.put(stream_id)


def write_bytes_collection(
    vineyard_socket, prefix, stream_id, storage_options, proc_num, proc_index
):
    """Read bytes from stream and write to external storage.

    Raises:
        ValueError: If the stream is invalid.
    """
    client = vineyard.connect(vineyard_socket)
    streams = client.get(stream_id)

    if len(streams) != proc_num:
        report_error("Expected: %s stream partitions" % proc_num)
        sys.exit(-1)

    worker_prefix = os.path.join(prefix, '%s-%s' % (proc_num, proc_index))

    # collect all blobs, and prepare metadata
    queue: "ConcurrentQueue[ObjectID]" = ConcurrentQueue()
    write_stream_collections(
        client, streams[proc_index].id, queue, worker_prefix, storage_options
    )

    # write streams to file
    executor = ThreadStreamExecutor(
        WriteBytesExecutor,
        parallism=multiprocessing.cpu_count(),
        client=client,
        prefix=worker_prefix,
        storage_options=storage_options,
        task_queue=queue,
    )
    executor.execute()


def main():
    if len(sys.argv) < 7:
        print(
            "usage: ./write_bytes_collection <ipc_socket> <prefix> <stream_id> "
            "<storage_options> <proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    prefix = expand_full_path(sys.argv[2])
    stream_id = sys.argv[3]
    storage_options = json.loads(
        base64.b64decode(sys.argv[4].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[5])
    proc_index = int(sys.argv[6])
    write_bytes_collection(
        ipc_socket, prefix, stream_id, storage_options, proc_num, proc_index
    )


if __name__ == "__main__":
    main()
